use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use binggan::{InputGroup, black_box};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tantivy_columnar::ColumnValues;
use tantivy_columnar::column_values::{CodecType, serialize_and_load_u64_based_column_values};

fn get_data() -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(2u64);
    let mut data: Vec<_> = (100..55_000_u64)
        .map(|num| num + rng.random::<u8>() as u64)
        .collect();
    data.push(99_000);
    data.insert(1000, 2000);
    data.insert(2000, 100);
    data.insert(3000, 4100);
    data.insert(4000, 100);
    data.insert(5000, 800);
    data
}

#[inline(never)]
fn value_iter() -> impl Iterator<Item = u64> {
    0..20_000
}

type Col = Arc<dyn ColumnValues<u64>>;

/// Number of values of the synthetic columns used for the bulk-decode benchmarks.
const LEN: usize = 512 * 64;
/// Residual bit widths the blockwise-linear codec can store; they span the bitpacker's
/// amortization regimes (8 values/load at <= 8 bits, 4 at <= 14, 2 at <= 28, else 1).
const WIDTHS: [u8; 9] = [8, 12, 16, 20, 24, 27, 31, 48, 64];

/// Values whose per-block residual after the codec's line fit has about `width` bits.
fn residual_width_data(width: u8) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(u64::from(width) + 7);
    let noise = 1u64 << width;
    (0..LEN as u64)
        .map(|i| 1_000_000_000u64 + i * 7 + rng.random::<u64>() % noise)
        .collect()
}

/// `count` strictly ascending indexes spread over `LEN`, with gaps in `1..=gap`. A small gap
/// models a dense bucket, a large one the sparse `row_ids_for_docs` output of a term bucket.
fn ascending_indexes(count: usize, gap: u32) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(u64::from(gap) + 11);
    let mut out = Vec::with_capacity(count);
    let mut next = 0u32;
    while out.len() < count && (next as usize) < LEN {
        out.push(next);
        next += 1 + rng.random::<u32>() % gap;
    }
    out
}

#[derive(Clone)]
struct OutBuf(Rc<RefCell<Vec<u64>>>);

impl OutBuf {
    fn new(len: usize) -> Self {
        OutBuf(Rc::new(RefCell::new(vec![0u64; len])))
    }
}

struct BenchInput {
    col: Col,
    dense: Vec<u32>,
    sparse: Vec<u32>,
}

fn main() {
    let data = get_data();
    let mut inputs: Vec<(String, BenchInput)> = vec![
        (
            "bitpacked".to_string(),
            bench_input(&data, CodecType::Bitpacked),
        ),
        ("linear".to_string(), bench_input(&data, CodecType::Linear)),
        (
            "blockwise_linear".to_string(),
            bench_input(&data, CodecType::BlockwiseLinear),
        ),
    ];
    for width in WIDTHS {
        let data = residual_width_data(width);
        inputs.push((
            format!("blockwise_w{width}"),
            bench_input(&data, CodecType::BlockwiseLinear),
        ));
    }

    let mut group: InputGroup<BenchInput> = InputGroup::new_with_inputs(inputs);

    group.register("fastfield_get", |input: &BenchInput| {
        let mut sum = 0u64;
        for pos in value_iter() {
            sum = sum.wrapping_add(input.col.get_val(pos as u32));
        }
        black_box(sum);
    });

    let range_out = OutBuf::new(64);
    group.register("get_range_64", {
        let out = range_out.clone();
        move |input: &BenchInput| {
            let mut sum = 0u64;
            let mut buf = out.0.borrow_mut();
            for k in 0..LEN / 64 {
                input.col.get_range((k * 64) as u64, &mut buf[..64]);
                sum = sum.wrapping_add(buf[0]);
            }
            black_box(sum);
        }
    });
    group.register("manual_64_get_val", {
        let out = range_out;
        move |input: &BenchInput| {
            let mut sum = 0u64;
            let mut buf = out.0.borrow_mut();
            for k in 0..LEN / 64 {
                let base = (k * 64) as u32;
                for (i, slot) in buf[..64].iter_mut().enumerate() {
                    *slot = input.col.get_val(base + i as u32);
                }
                sum = sum.wrapping_add(buf[0]);
            }
            black_box(sum);
        }
    });

    let vals_out = OutBuf::new(LEN);
    group.register("get_vals_ascending_dense", {
        let out = vals_out.clone();
        move |input: &BenchInput| {
            let mut buf = out.0.borrow_mut();
            let n = input.dense.len();
            input.col.get_vals(&input.dense, &mut buf[..n]);
            black_box(buf[0]);
        }
    });
    group.register("manual_get_vals_ascending_dense", {
        let out = vals_out.clone();
        move |input: &BenchInput| {
            let mut sum = 0u64;
            let mut buf = out.0.borrow_mut();
            for (slot, &idx) in buf[..input.dense.len()].iter_mut().zip(&input.dense) {
                *slot = input.col.get_val(idx);
                sum = sum.wrapping_add(*slot);
            }
            black_box(sum);
        }
    });
    group.register("get_vals_ascending_sparse", {
        let out = vals_out.clone();
        move |input: &BenchInput| {
            let mut buf = out.0.borrow_mut();
            let n = input.sparse.len();
            input.col.get_vals(&input.sparse, &mut buf[..n]);
            black_box(buf[0]);
        }
    });
    group.register("manual_get_vals_ascending_sparse", {
        let out = vals_out;
        move |input: &BenchInput| {
            let mut sum = 0u64;
            let mut buf = out.0.borrow_mut();
            for (slot, &idx) in buf[..input.sparse.len()].iter_mut().zip(&input.sparse) {
                *slot = input.col.get_val(idx);
                sum = sum.wrapping_add(*slot);
            }
            black_box(sum);
        }
    });

    group.run();
}

fn bench_input(data: &[u64], codec: CodecType) -> BenchInput {
    BenchInput {
        col: serialize_and_load_u64_based_column_values(&data, &[codec]),
        dense: ascending_indexes(4096, 2),
        sparse: ascending_indexes(4096, 64),
    }
}
