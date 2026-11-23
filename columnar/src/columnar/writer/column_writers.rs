use std::cmp::Ordering;

use stacker::{ExpUnrolledLinkedList, MemoryArena};

use crate::columnar::writer::column_operation::{ColumnOperation, SymbolValue};
use crate::dictionary::{DictionaryBuilder, UnorderedId};
use crate::{Cardinality, NumericalType, NumericalValue, RowId};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
enum DocumentStep {
    Same = 0,
    Next = 1,
    Skipped = 2,
}

#[inline(always)]
fn delta_with_last_doc(last_doc_opt: Option<u32>, doc: u32) -> DocumentStep {
    let expected_next_doc = last_doc_opt.map(|last_doc| last_doc + 1).unwrap_or(0u32);
    match doc.cmp(&expected_next_doc) {
        Ordering::Less => DocumentStep::Same,
        Ordering::Equal => DocumentStep::Next,
        Ordering::Greater => DocumentStep::Skipped,
    }
}

/// Iterator returned by [`ColumnWriter::operation_iterator`].
///
/// Either lazily deserializes operations from the scratch buffer, or yields the
/// remapped document groups held in memory.
enum ColumnOperationIterator<'a, V> {
    Deserialize { cursor: &'a [u8] },
    Remapped(RemappedColumnOperationIterator<V>),
}

impl<V: SymbolValue> Iterator for ColumnOperationIterator<'_, V> {
    type Item = ColumnOperation<V>;

    fn next(&mut self) -> Option<ColumnOperation<V>> {
        match self {
            ColumnOperationIterator::Deserialize { cursor } => {
                ColumnOperation::<V>::deserialize(cursor)
            }
            ColumnOperationIterator::Remapped(remapped) => remapped.next(),
        }
    }
}

/// Holds a column's operations and yields them ordered by remapped document id.
///
/// `groups` holds one `(new_doc, value_start, value_end)` per document, sorted by
/// `new_doc`; the range covers the document's values, excluding its original
/// `NewDoc` operation (which is replaced by `NewDoc(new_doc)`).
struct RemappedColumnOperationIterator<V> {
    ops: Vec<ColumnOperation<V>>,
    groups: Vec<(RowId, u32, u32)>,
    group_idx: usize,
    next_value: u32,
    emit_new_doc: bool,
}

impl<V: SymbolValue> RemappedColumnOperationIterator<V> {
    fn new(ops: Vec<ColumnOperation<V>>, old_to_new_ids: &[RowId]) -> Self {
        let mut groups: Vec<(RowId, u32, u32)> = Vec::new();
        let mut group_new_doc: Option<RowId> = None;
        let mut group_value_start = 0u32;
        for (op_idx, op) in ops.iter().enumerate() {
            if let ColumnOperation::NewDoc(old_doc) = op {
                let new_doc = old_to_new_ids[*old_doc as usize];
                if let Some(previous_new_doc) = group_new_doc.replace(new_doc) {
                    groups.push((previous_new_doc, group_value_start, op_idx as u32));
                }
                group_value_start = op_idx as u32 + 1;
            }
        }
        if let Some(new_doc) = group_new_doc {
            groups.push((new_doc, group_value_start, ops.len() as u32));
        }
        // `old_to_new_ids` is a permutation, so the sort keys are unique and an
        // unstable sort yields the same order as the stable per-operation sort it
        // replaces.
        groups.sort_unstable_by_key(|(new_doc, _, _)| *new_doc);
        RemappedColumnOperationIterator {
            ops,
            groups,
            group_idx: 0,
            next_value: 0,
            emit_new_doc: true,
        }
    }
}

impl<V: SymbolValue> Iterator for RemappedColumnOperationIterator<V> {
    type Item = ColumnOperation<V>;

    fn next(&mut self) -> Option<ColumnOperation<V>> {
        loop {
            let (new_doc, value_start, value_end) = *self.groups.get(self.group_idx)?;
            if self.emit_new_doc {
                self.emit_new_doc = false;
                self.next_value = value_start;
                return Some(ColumnOperation::NewDoc(new_doc));
            }
            if self.next_value < value_end {
                let op = self.ops[self.next_value as usize];
                self.next_value += 1;
                return Some(op);
            }
            self.group_idx += 1;
            self.emit_new_doc = true;
        }
    }
}

#[derive(Copy, Clone, Default)]
pub struct ColumnWriter {
    // Detected cardinality of the column so far.
    cardinality: Cardinality,
    // Last document inserted.
    // None if no doc has been added yet.
    last_doc_opt: Option<u32>,
    // Buffer containing the serialized values.
    values: ExpUnrolledLinkedList,
}

impl ColumnWriter {
    /// Returns an iterator over the Symbol that have been recorded
    /// for the given column.
    /// When `old_to_new_ids_opt` is `Some`, the operations are reordered so that the
    /// documents appear in ascending new-doc order. `record` writes one `NewDoc`
    /// followed by the document's values, so the operations of a document are
    /// contiguous; and `old_to_new_ids` is a permutation. Remapping each document
    /// group and sorting the groups therefore matches the per-operation stable sort
    /// it replaces, without serializing and re-parsing every operation.
    pub(super) fn operation_iterator<'a, V: SymbolValue + 'a>(
        &self,
        arena: &MemoryArena,
        old_to_new_ids_opt: Option<&[RowId]>,
        buffer: &'a mut Vec<u8>,
    ) -> impl Iterator<Item = ColumnOperation<V>> + 'a + use<'a, V> {
        buffer.clear();
        self.values.read_to_end(arena, buffer);
        let Some(old_to_new_ids) = old_to_new_ids_opt else {
            return ColumnOperationIterator::Deserialize {
                cursor: &buffer[..],
            };
        };
        let mut ops: Vec<ColumnOperation<V>> = Vec::new();
        let mut cursor: &[u8] = &buffer[..];
        while let Some(op) = ColumnOperation::<V>::deserialize(&mut cursor) {
            ops.push(op);
        }
        ColumnOperationIterator::Remapped(RemappedColumnOperationIterator::new(ops, old_to_new_ids))
    }

    /// Records a change of the document being recorded.
    ///
    /// This function will also update the cardinality of the column
    /// if necessary.
    pub(super) fn record<S: SymbolValue>(&mut self, doc: RowId, value: S, arena: &mut MemoryArena) {
        // Difference between `doc` and the last doc.
        match delta_with_last_doc(self.last_doc_opt, doc) {
            DocumentStep::Same => {
                // This is the last encounterred document.
                self.cardinality = Cardinality::Multivalued;
            }
            DocumentStep::Next => {
                self.last_doc_opt = Some(doc);
                self.write_symbol::<S>(ColumnOperation::NewDoc(doc), arena);
            }
            DocumentStep::Skipped => {
                self.cardinality = self.cardinality.max(Cardinality::Optional);
                self.last_doc_opt = Some(doc);
                self.write_symbol::<S>(ColumnOperation::NewDoc(doc), arena);
            }
        }
        self.write_symbol(ColumnOperation::Value(value), arena);
    }

    // Get the cardinality.
    // The overall number of docs in the column is necessary to
    // deal with the case where the all docs contain 1 value, except some documents
    // at the end of the column.
    pub(crate) fn get_cardinality(&self, num_docs: RowId) -> Cardinality {
        match delta_with_last_doc(self.last_doc_opt, num_docs) {
            DocumentStep::Same | DocumentStep::Next => self.cardinality,
            DocumentStep::Skipped => self.cardinality.max(Cardinality::Optional),
        }
    }

    /// Appends a new symbol to the `ColumnWriter`.
    fn write_symbol<V: SymbolValue>(
        &mut self,
        column_operation: ColumnOperation<V>,
        arena: &mut MemoryArena,
    ) {
        self.values
            .writer(arena)
            .extend_from_slice(column_operation.serialize().as_ref());
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct NumericalColumnWriter {
    compatible_numerical_types: CompatibleNumericalTypes,
    column_writer: ColumnWriter,
}

impl NumericalColumnWriter {
    pub fn force_numerical_type(&mut self, numerical_type: NumericalType) {
        assert!(
            self.compatible_numerical_types
                .is_type_accepted(numerical_type)
        );
        self.compatible_numerical_types = CompatibleNumericalTypes::StaticType(numerical_type);
    }
}

/// State used to store what types are still acceptable
/// after having seen a set of numerical values.
#[derive(Clone, Copy)]
pub(crate) enum CompatibleNumericalTypes {
    Dynamic {
        all_values_within_i64_range: bool,
        all_values_within_u64_range: bool,
    },
    StaticType(NumericalType),
}

impl Default for CompatibleNumericalTypes {
    fn default() -> CompatibleNumericalTypes {
        CompatibleNumericalTypes::Dynamic {
            all_values_within_i64_range: true,
            all_values_within_u64_range: true,
        }
    }
}

impl CompatibleNumericalTypes {
    pub fn is_type_accepted(&self, numerical_type: NumericalType) -> bool {
        match self {
            CompatibleNumericalTypes::Dynamic {
                all_values_within_i64_range,
                all_values_within_u64_range,
            } => match numerical_type {
                NumericalType::I64 => *all_values_within_i64_range,
                NumericalType::U64 => *all_values_within_u64_range,
                NumericalType::F64 => true,
            },
            CompatibleNumericalTypes::StaticType(static_numerical_type) => {
                *static_numerical_type == numerical_type
            }
        }
    }

    pub fn accept_value(&mut self, numerical_value: NumericalValue) {
        match self {
            CompatibleNumericalTypes::Dynamic {
                all_values_within_i64_range,
                all_values_within_u64_range,
            } => match numerical_value {
                NumericalValue::I64(val_i64) => {
                    let value_within_u64_range = val_i64 >= 0i64;
                    *all_values_within_u64_range &= value_within_u64_range;
                }
                NumericalValue::U64(val_u64) => {
                    let value_within_i64_range = val_u64 < i64::MAX as u64;
                    *all_values_within_i64_range &= value_within_i64_range;
                }
                NumericalValue::F64(_) => {
                    *all_values_within_i64_range = false;
                    *all_values_within_u64_range = false;
                }
            },
            CompatibleNumericalTypes::StaticType(typ) => {
                assert_eq!(
                    numerical_value.numerical_type(),
                    *typ,
                    "Input type forbidden. This column has been forced to type {typ:?}, received \
                     {numerical_value:?}"
                );
            }
        }
    }

    pub fn to_numerical_type(self) -> NumericalType {
        for numerical_type in [NumericalType::I64, NumericalType::U64] {
            if self.is_type_accepted(numerical_type) {
                return numerical_type;
            }
        }
        NumericalType::F64
    }
}

impl NumericalColumnWriter {
    pub fn numerical_type(&self) -> NumericalType {
        self.compatible_numerical_types.to_numerical_type()
    }

    pub fn cardinality(&self, num_docs: RowId) -> Cardinality {
        self.column_writer.get_cardinality(num_docs)
    }

    pub fn record_numerical_value(
        &mut self,
        doc: RowId,
        value: NumericalValue,
        arena: &mut MemoryArena,
    ) {
        self.compatible_numerical_types.accept_value(value);
        self.column_writer.record(doc, value, arena);
    }

    pub(super) fn operation_iterator<'a>(
        self,
        arena: &MemoryArena,
        old_to_new_ids: Option<&[RowId]>,
        buffer: &'a mut Vec<u8>,
    ) -> impl Iterator<Item = ColumnOperation<NumericalValue>> + 'a + use<'a> {
        self.column_writer
            .operation_iterator(arena, old_to_new_ids, buffer)
    }
}

#[derive(Copy, Clone)]
pub(crate) struct StrOrBytesColumnWriter {
    pub(crate) dictionary_id: u32,
    pub(crate) column_writer: ColumnWriter,
    // If true, when facing a multivalued cardinality,
    // values associated to a given document will be sorted.
    //
    // This is useful for facets.
    //
    // If false, the order of appearance in the document will be
    // observed.
    pub(crate) sort_values_within_row: bool,
}

impl StrOrBytesColumnWriter {
    pub(crate) fn with_dictionary_id(dictionary_id: u32) -> StrOrBytesColumnWriter {
        StrOrBytesColumnWriter {
            dictionary_id,
            column_writer: Default::default(),
            sort_values_within_row: false,
        }
    }

    pub(crate) fn record_bytes(
        &mut self,
        doc: RowId,
        bytes: &[u8],
        dictionaries: &mut [DictionaryBuilder],
        arena: &mut MemoryArena,
    ) {
        let unordered_id =
            dictionaries[self.dictionary_id as usize].get_or_allocate_id(bytes, arena);
        self.column_writer.record(doc, unordered_id, arena);
    }

    pub(super) fn operation_iterator<'a>(
        &self,
        arena: &MemoryArena,
        old_to_new_ids: Option<&[RowId]>,
        byte_buffer: &'a mut Vec<u8>,
    ) -> impl Iterator<Item = ColumnOperation<UnorderedId>> + 'a + use<'a> {
        self.column_writer
            .operation_iterator(arena, old_to_new_ids, byte_buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_with_last_doc() {
        assert_eq!(delta_with_last_doc(None, 0u32), DocumentStep::Next);
        assert_eq!(delta_with_last_doc(None, 1u32), DocumentStep::Skipped);
        assert_eq!(delta_with_last_doc(None, 2u32), DocumentStep::Skipped);
        assert_eq!(delta_with_last_doc(Some(0u32), 0u32), DocumentStep::Same);
        assert_eq!(delta_with_last_doc(Some(1u32), 1u32), DocumentStep::Same);
        assert_eq!(delta_with_last_doc(Some(1u32), 2u32), DocumentStep::Next);
        assert_eq!(delta_with_last_doc(Some(1u32), 3u32), DocumentStep::Skipped);
        assert_eq!(delta_with_last_doc(Some(1u32), 4u32), DocumentStep::Skipped);
    }

    fn numerical_column_ops(
        docs: &[(RowId, Vec<NumericalValue>)],
        old_to_new_ids: Option<&[RowId]>,
    ) -> Vec<ColumnOperation<NumericalValue>> {
        let mut arena = MemoryArena::default();
        let mut column_writer = ColumnWriter::default();
        for (doc, values) in docs {
            for value in values {
                column_writer.record(*doc, *value, &mut arena);
            }
        }
        let mut buffer = Vec::new();
        column_writer
            .operation_iterator::<NumericalValue>(&arena, old_to_new_ids, &mut buffer)
            .collect()
    }

    /// The remapping performed before the document-group rewrite: one `(new_doc,
    /// op)` entry per operation, stable-sorted by new doc id.
    fn per_operation_remap<V: SymbolValue>(
        ops: &[ColumnOperation<V>],
        old_to_new_ids: &[RowId],
    ) -> Vec<ColumnOperation<V>> {
        let mut sorted_ops: Vec<(RowId, ColumnOperation<V>)> = Vec::new();
        let mut new_doc = 0u32;
        for op in ops {
            if let ColumnOperation::NewDoc(doc) = op {
                new_doc = old_to_new_ids[*doc as usize];
                sorted_ops.push((new_doc, ColumnOperation::NewDoc(new_doc)));
            } else {
                sorted_ops.push((new_doc, *op));
            }
        }
        sorted_ops.sort_by_key(|(new_doc_id, _)| *new_doc_id);
        sorted_ops.into_iter().map(|(_, op)| op).collect()
    }

    #[test]
    fn test_operation_iterator_remap_groups_documents() {
        // Doc 0 is multivalued, doc 1 single-valued, doc 2 absent, doc 4 single-valued.
        let docs: Vec<(RowId, Vec<NumericalValue>)> = vec![
            (0, vec![NumericalValue::I64(10), NumericalValue::I64(11)]),
            (1, vec![NumericalValue::I64(20)]),
            (4, vec![NumericalValue::I64(40)]),
        ];
        // Non-trivial permutation over 5 documents (old -> new).
        let old_to_new_ids: Vec<RowId> = vec![3, 0, 4, 2, 1];

        let remapped = numerical_column_ops(&docs, Some(&old_to_new_ids));
        assert_eq!(
            remapped,
            vec![
                ColumnOperation::NewDoc(0),
                ColumnOperation::Value(NumericalValue::I64(20)),
                ColumnOperation::NewDoc(1),
                ColumnOperation::Value(NumericalValue::I64(40)),
                ColumnOperation::NewDoc(3),
                ColumnOperation::Value(NumericalValue::I64(10)),
                ColumnOperation::Value(NumericalValue::I64(11)),
            ]
        );

        // Equivalent to the previous per-operation stable sort.
        let original = numerical_column_ops(&docs, None);
        assert_eq!(remapped, per_operation_remap(&original, &old_to_new_ids));
    }

    #[test]
    fn test_operation_iterator_remap_identity_and_empty() {
        let docs: Vec<(RowId, Vec<NumericalValue>)> = vec![(0, vec![NumericalValue::U64(1)])];
        // An identity mapping keeps the operation stream unchanged.
        assert_eq!(
            numerical_column_ops(&docs, Some(&[0])),
            numerical_column_ops(&docs, None)
        );
        // A column with no recorded value yields no operation.
        assert!(numerical_column_ops(&[], Some(&[0, 1])).is_empty());
    }

    #[track_caller]
    fn test_column_writer_coercion_iter_aux(
        values: impl Iterator<Item = NumericalValue>,
        expected_numerical_type: NumericalType,
    ) {
        let mut compatible_numerical_types = CompatibleNumericalTypes::default();
        for value in values {
            compatible_numerical_types.accept_value(value);
        }
        assert_eq!(
            compatible_numerical_types.to_numerical_type(),
            expected_numerical_type
        );
    }

    #[track_caller]
    fn test_column_writer_coercion_aux(
        values: &[NumericalValue],
        expected_numerical_type: NumericalType,
    ) {
        test_column_writer_coercion_iter_aux(values.iter().copied(), expected_numerical_type);
        test_column_writer_coercion_iter_aux(values.iter().rev().copied(), expected_numerical_type);
    }

    #[test]
    fn test_column_writer_coercion() {
        test_column_writer_coercion_aux(&[], NumericalType::I64);
        test_column_writer_coercion_aux(&[1i64.into()], NumericalType::I64);
        test_column_writer_coercion_aux(&[1u64.into()], NumericalType::I64);
        // We don't detect exact integer at the moment. We could!
        test_column_writer_coercion_aux(&[1f64.into()], NumericalType::F64);
        test_column_writer_coercion_aux(&[u64::MAX.into()], NumericalType::U64);
        test_column_writer_coercion_aux(&[(i64::MAX as u64).into()], NumericalType::U64);
        test_column_writer_coercion_aux(&[(1u64 << 63).into()], NumericalType::U64);
        test_column_writer_coercion_aux(&[1i64.into(), 1u64.into()], NumericalType::I64);
        test_column_writer_coercion_aux(&[u64::MAX.into(), (-1i64).into()], NumericalType::F64);
    }

    #[test]
    #[should_panic]
    fn test_compatible_numerical_types_static_incompatible_type() {
        let mut compatible_numerical_types =
            CompatibleNumericalTypes::StaticType(NumericalType::U64);
        compatible_numerical_types.accept_value(NumericalValue::I64(1i64));
    }

    #[test]
    fn test_compatible_numerical_types_static_different_type_forbidden() {
        let mut compatible_numerical_types =
            CompatibleNumericalTypes::StaticType(NumericalType::U64);
        compatible_numerical_types.accept_value(NumericalValue::U64(u64::MAX));
    }

    #[test]
    fn test_compatible_numerical_types_static() {
        for typ in [NumericalType::I64, NumericalType::I64, NumericalType::F64] {
            let compatible_numerical_types = CompatibleNumericalTypes::StaticType(typ);
            assert_eq!(compatible_numerical_types.to_numerical_type(), typ);
        }
    }
}
