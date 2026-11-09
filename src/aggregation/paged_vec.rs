//! A grow-only, index-addressed vector backed by fixed-size pages.
//!
//! Several bucket-indexed aggregation structures are dense vectors indexed by a
//! [`BucketId`](crate::aggregation::BucketId): they grow one element at a time as
//! new buckets appear, and a plain `Vec` therefore reallocates (and copies) the
//! whole buffer with amortized doubling. When the bucket count reaches millions,
//! that reallocation dominates the query's CPU (jemalloc `large_ralloc`).
//!
//! `PagedVec` keeps the same O(1) indexed access but stores the elements in
//! fixed-size pages allocated on demand. Growing past the current end allocates
//! one page; no existing element is ever moved, so growth never triggers a large
//! reallocation. This mirrors the paging already used by
//! [`PagedTermMap`](crate::aggregation::bucket::PagedTermMap) and
//! `PagedBitset` (see `metric/cardinality/term_ord_accumulator.rs`).
//!
//! Pages are allocated lazily on the first `push` into them, so the element type
//! needs no `Default` and a fresh `PagedVec` allocates nothing. Only sequential
//! `push` and indexed access to `i < len` are supported. `PAGE_SHIFT` is log2 of
//! the page size in elements: a small page bounds the memory a mostly-empty
//! collector allocates, a large one reduces the number of allocations for a
//! multi-million-element structure.

use std::ops::{Index, IndexMut};

/// A grow-only vector with O(1) indexed access, stored as fixed-size pages.
///
/// `PAGE_SHIFT` is log2 of the number of elements per page.
#[derive(Debug)]
pub(crate) struct PagedVec<T, const PAGE_SHIFT: u32> {
    pages: Vec<Vec<T>>,
    len: usize,
}

impl<T, const PAGE_SHIFT: u32> PagedVec<T, PAGE_SHIFT> {
    /// Number of elements per page.
    const PAGE_SIZE: usize = 1usize << PAGE_SHIFT;
    /// Mask selecting the offset within a page.
    const PAGE_MASK: usize = Self::PAGE_SIZE - 1;

    /// Creates an empty `PagedVec` without allocating any page.
    pub(crate) fn new() -> Self {
        Self {
            pages: Vec::new(),
            len: 0,
        }
    }

    /// Creates a `PagedVec` holding `first` as its only element.
    pub(crate) fn singleton(first: T) -> Self {
        let mut vec = Self::new();
        vec.push(first);
        vec
    }

    /// Number of addressable elements.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Returns a reference to the element at `idx`, or `None` when out of bounds.
    #[inline]
    pub(crate) fn get(&self, idx: usize) -> Option<&T> {
        if idx < self.len {
            Some(&self.pages[idx >> PAGE_SHIFT][idx & Self::PAGE_MASK])
        } else {
            None
        }
    }

    /// Appends `value` at the end, allocating a new page only the first time it
    /// is crossed. Existing elements are never moved.
    #[inline]
    pub(crate) fn push(&mut self, value: T) {
        let page_idx = self.len >> PAGE_SHIFT;
        if page_idx == self.pages.len() {
            self.pages.push(Vec::with_capacity(Self::PAGE_SIZE));
        }
        self.pages[page_idx].push(value);
        self.len += 1;
    }

    /// Grows the vector to at least `new_len`, filling new slots with clones of
    /// `value`.
    ///
    /// Never shrinks: the paged layout is grow-only. Whole new pages are filled
    /// in one bulk allocation, so growth does not reallocate a growing buffer.
    pub(crate) fn resize(&mut self, new_len: usize, value: T)
    where T: Clone {
        while self.len < new_len {
            let page_idx = self.len >> PAGE_SHIFT;
            let end = new_len.min((page_idx + 1) << PAGE_SHIFT);
            if page_idx == self.pages.len() {
                if end - self.len == Self::PAGE_SIZE {
                    self.pages.push(vec![value.clone(); Self::PAGE_SIZE]);
                    self.len = end;
                    continue;
                }
                self.pages.push(Vec::with_capacity(Self::PAGE_SIZE));
            }
            let page = &mut self.pages[page_idx];
            debug_assert_eq!(page.len(), self.len & Self::PAGE_MASK);
            page.extend(std::iter::repeat_with(|| value.clone()).take(end - self.len));
            self.len = end;
        }
    }
}

impl<T, const PAGE_SHIFT: u32> Index<usize> for PagedVec<T, PAGE_SHIFT> {
    type Output = T;

    #[inline]
    fn index(&self, idx: usize) -> &T {
        &self.pages[idx >> PAGE_SHIFT][idx & Self::PAGE_MASK]
    }
}

impl<T, const PAGE_SHIFT: u32> IndexMut<usize> for PagedVec<T, PAGE_SHIFT> {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut T {
        &mut self.pages[idx >> PAGE_SHIFT][idx & Self::PAGE_MASK]
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::aggregation::bucket::SmallTermBuckets;
    use crate::aggregation::metric::IntermediateStats;
    use crate::aggregation::BucketId;

    /// The element type must not need `Default`: the pages are lazily allocated
    /// and only ever read for `idx < len`.
    #[derive(Debug, Clone, PartialEq)]
    struct NoDefault(u32);

    #[test]
    fn paged_vec_indexes_across_page_boundaries() {
        let mut v: PagedVec<NoDefault, 2> = PagedVec::new();
        assert_eq!(v.len(), 0);
        assert_eq!(v.get(0), None);

        v.push(NoDefault(1));
        v.push(NoDefault(2));
        v.push(NoDefault(3));
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], NoDefault(1));
        assert_eq!(v[2], NoDefault(3));
        assert_eq!(v.get(3), None);

        v[2] = NoDefault(30);
        assert_eq!(v[2], NoDefault(30));

        v.resize(5, NoDefault(0));
        assert_eq!(v.len(), 5);
        assert_eq!(v[3], NoDefault(0));
        assert_eq!(v[4], NoDefault(0));
        assert_eq!(v[2], NoDefault(30));

        // Shrinking is a no-op.
        v.resize(1, NoDefault(9));
        assert_eq!(v.len(), 5);
    }

    const PRECHECK_TARGET: usize = 6_250_000;
    const PRECHECK_REPEATS: usize = 5;

    fn best_of(mut f: impl FnMut()) -> f64 {
        let mut best = f64::MAX;
        for _ in 0..PRECHECK_REPEATS {
            let start = Instant::now();
            f();
            best = best.min(start.elapsed().as_secs_f64());
        }
        best
    }

    fn bench_paged_term_pushes<const SHIFT: u32>(term: &SmallTermBuckets<BucketId>) -> f64 {
        best_of(|| {
            let mut p: PagedVec<SmallTermBuckets<BucketId>, SHIFT> = PagedVec::new();
            for _ in 0..PRECHECK_TARGET {
                p.push(term.clone());
            }
            std::hint::black_box(p.len());
        })
    }

    /// E006-lesson pre-check: the growth path must be measurably cheaper than a
    /// `Vec` grown the same way, or the reallocation is not the lever.
    ///
    /// Run manually: `cargo +1.90 test --release --lib aggregation::paged_vec
    /// -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing pre-check, run manually"]
    fn precheck_growth_speedups() {
        // (1) `SegmentStatsCollector::buckets`: `resize_with` in amortized-doubling
        // steps, as `prepare_max_bucket` grows to the current max bucket.
        let vec_stats = best_of(|| {
            let mut v: Vec<IntermediateStats> = Vec::new();
            let mut target = 1;
            while target < PRECHECK_TARGET {
                target = (target * 2).min(PRECHECK_TARGET);
                v.resize_with(target, IntermediateStats::default);
            }
            std::hint::black_box(v.len());
        });
        let paged_stats = best_of(|| {
            let mut p: PagedVec<IntermediateStats, 12> = PagedVec::new();
            let mut target = 1;
            while target < PRECHECK_TARGET {
                target = (target * 2).min(PRECHECK_TARGET);
                p.resize(target, IntermediateStats::default());
            }
            std::hint::black_box(p.len());
        });
        println!(
            "precheck stats (shift 12): vec {vec_stats:.4}s, paged {paged_stats:.4}s, speedup \
             {:.2}x",
            vec_stats / paged_stats
        );

        // (2) `SegmentTermCollector::parent_buckets`: sequential `push` of a
        // `TermMap` (here the `SmallTermBuckets` the nested q33 collector uses).
        let term = SmallTermBuckets::<BucketId>::default();
        let vec_terms = best_of(|| {
            let mut v: Vec<SmallTermBuckets<BucketId>> = Vec::new();
            for _ in 0..PRECHECK_TARGET {
                v.push(term.clone());
            }
            std::hint::black_box(v.len());
        });
        println!("precheck terms: vec {vec_terms:.4}s");
        for (shift, paged) in [
            (6u32, bench_paged_term_pushes::<6>(&term)),
            (8, bench_paged_term_pushes::<8>(&term)),
            (12, bench_paged_term_pushes::<12>(&term)),
        ] {
            println!(
                "precheck terms (shift {shift}): paged {paged:.4}s, speedup {:.2}x",
                vec_terms / paged
            );
        }
    }
}
