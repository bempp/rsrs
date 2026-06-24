use std::time::{Duration, Instant};

use crate::{
    rsrs::{
        args::Symmetry,
        rsrs_factors::base_factors::{
            conjugate_array_in_place, ComposedFactorData, FactorData, LuSMat, RectArr, RegSMat,
            SquareArr,
        },
        sketch::SketchData,
    },
    utils::{
        data_ins_ext::extract_axis_into,
        linear_algebra::{
            add_diagonal, block_extraction_into, nullify_near_sketch, BlockExtractionMethod,
            NormalEquationScratch, NullMethod,
        },
        memory::{matrix_bytes, trace_memory_event, trace_memory_growth},
    },
};
use rand_distr::{Distribution, Standard, StandardNormal};
use rlst::{
    dense::{
        linalg::{interpolative_decomposition::MatrixIdNoSkel, lu::MatrixLu},
        tools::RandScalar,
    },
    prelude::*,
};
use serde::{Deserialize, Serialize};

type Real<T> = <T as rlst::RlstScalar>::Real;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum PivotMethod {
    DirectInversion,
    Lu(f64),       //TODO: Change to Item
    LuHybrid(f64), //TODO: Change to Item
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum NonSymmetricIdCombination {
    #[default]
    Sum,
    Concat,
}

#[derive(Debug, Clone)]
pub struct ExtractOptions<Item: RlstScalar> {
    pub block_extraction_method: BlockExtractionMethod,
    pub pivot_method: PivotMethod,
    pub tol_lstsq: Real<Item>,
}

#[derive(Debug, Clone)]
pub struct IdOptions<Item: RlstScalar> {
    pub null_method: NullMethod,
    pub qr_method: RankRevealingQrType<Real<Item>>,
    pub tol_null: Real<Item>,
    pub tol_id: Real<Item>,
    pub store_far: bool,
    pub nonsymmetric_id_combination: NonSymmetricIdCombination,
}

pub struct ExtractionScratch<Item: RlstScalar> {
    pub primary: DynamicArray<Item, 2>,
    pub secondary: DynamicArray<Item, 2>,
    pub tertiary: DynamicArray<Item, 2>,
    pub normal: NormalEquationScratch<Item>,
}

impl<Item: RlstScalar> ExtractionScratch<Item> {
    pub fn new() -> Self {
        Self {
            primary: empty_array(),
            secondary: empty_array(),
            tertiary: empty_array(),
            normal: NormalEquationScratch::new(),
        }
    }
}

impl<Item: RlstScalar> Default for ExtractionScratch<Item> {
    fn default() -> Self {
        Self::new()
    }
}

fn usable_nullspace_rows(subs_sample_dim: usize, near_field_len: usize) -> usize {
    subs_sample_dim.saturating_sub(near_field_len)
}

fn concat_nonsymmetric_id_sketches<Item: RlstScalar>(
    primary: &mut DynamicArray<Item, 2>,
    secondary: &DynamicArray<Item, 2>,
    rows_per_sketch: usize,
) {
    let primary_shape = primary.shape();
    let secondary_shape = secondary.shape();
    assert_eq!(
        primary_shape[1], secondary_shape[1],
        "Cannot concatenate ID sketches with different column counts"
    );

    let kept_primary_rows = primary_shape[0].min(rows_per_sketch);
    let kept_secondary_rows = secondary_shape[0].min(rows_per_sketch);
    let cols = primary_shape[1];
    let mut combined = rlst_dynamic_array2!(Item, [kept_primary_rows + kept_secondary_rows, cols]);

    if kept_primary_rows > 0 {
        combined
            .r_mut()
            .into_subview([0, 0], [kept_primary_rows, cols])
            .fill_from(primary.r().into_subview([0, 0], [kept_primary_rows, cols]));
    }
    if kept_secondary_rows > 0 {
        combined
            .r_mut()
            .into_subview([kept_primary_rows, 0], [kept_secondary_rows, cols])
            .fill_from(
                secondary
                    .r()
                    .into_subview([0, 0], [kept_secondary_rows, cols]),
            );
    }

    *primary = combined;
}

#[allow(clippy::too_many_arguments)]
fn null_sketch_near_field_into<
    Item: RlstScalar
        + MatrixId
        + MatrixIdNoSkel
        + MatrixInverse
        + MatrixPseudoInverse
        + RandScalar
        + MatrixLu
        + MatrixQr,
>(
    target_inds: &[usize],
    near_field_inds: &[usize],
    sketch: &DynamicArray<Item, 2>,
    test: &DynamicArray<Item, 2>,
    subs_sample_dim: usize,
    conjugate_data: bool,
    id_options: &IdOptions<Item>,
    sketch_t: &mut DynamicArray<Item, 2>,
    test_n: &mut DynamicArray<Item, 2>,
    normal_scratch: &mut NormalEquationScratch<Item>,
) where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    QrDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixQrDecomposition<Item = Item>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let dim = test.shape()[1];
    let sub_test = test.r().into_subview([0, 0], [subs_sample_dim, dim]);
    let sub_sketch = sketch.r().into_subview([0, 0], [subs_sample_dim, dim]);
    extract_axis_into(sketch_t, &sub_sketch, target_inds, 1, false);
    extract_axis_into(test_n, &sub_test, near_field_inds, 1, false);
    if conjugate_data {
        conjugate_array_in_place(sketch_t);
        conjugate_array_in_place(test_n);
    }
    trace_memory_growth(
        &format!(
            "null_sketch_near_field buffers (targets={}, near={}, samples={subs_sample_dim})",
            target_inds.len(),
            near_field_inds.len()
        ),
        Some(
            matrix_bytes::<Item>(subs_sample_dim, target_inds.len())
                + matrix_bytes::<Item>(subs_sample_dim, near_field_inds.len()),
        ),
    );
    nullify_near_sketch(test_n, sketch_t, id_options, normal_scratch);
}

#[allow(clippy::too_many_arguments)]
pub fn null_near_field_into<
    Item: RlstScalar
        + MatrixId
        + MatrixIdNoSkel
        + MatrixInverse
        + MatrixPseudoInverse
        + RandScalar
        + MatrixLu
        + MatrixQr,
>(
    target_inds: &[usize],
    near_field_inds: &[usize],
    y_data: &SketchData<Item>,
    z_data: &SketchData<Item>,
    subs_sample_dim: usize,
    symmetry: &Symmetry,
    _fixed_rank: bool,
    id_options: &IdOptions<Item>,
    far_field_sketch: &mut DynamicArray<Item, 2>,
    test_scratch: &mut DynamicArray<Item, 2>,
    aux_sketch: &mut DynamicArray<Item, 2>,
    normal_scratch: &mut NormalEquationScratch<Item>,
) where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    QrDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixQrDecomposition<Item = Item>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let complex_symmetric = symmetry.complex_symmetric_val::<Item>();

    if symmetry.symm_val() {
        null_sketch_near_field_into(
            target_inds,
            near_field_inds,
            &y_data.sketch,
            &y_data.test,
            subs_sample_dim,
            false,
            id_options,
            far_field_sketch,
            test_scratch,
            normal_scratch,
        );

        if complex_symmetric {
            null_sketch_near_field_into(
                target_inds,
                near_field_inds,
                &y_data.sketch,
                &y_data.test,
                subs_sample_dim,
                true,
                id_options,
                aux_sketch,
                test_scratch,
                normal_scratch,
            );
            far_field_sketch.sum_into(aux_sketch.r());
        }
    } else {
        null_sketch_near_field_into(
            target_inds,
            near_field_inds,
            &y_data.sketch,
            &y_data.test,
            subs_sample_dim,
            false,
            id_options,
            far_field_sketch,
            test_scratch,
            normal_scratch,
        );
        null_sketch_near_field_into(
            target_inds,
            near_field_inds,
            &z_data.sketch,
            &z_data.test,
            subs_sample_dim,
            false,
            id_options,
            aux_sketch,
            test_scratch,
            normal_scratch,
        );
        match id_options.nonsymmetric_id_combination {
            NonSymmetricIdCombination::Sum => {
                far_field_sketch.sum_into(aux_sketch.r());
            }
            NonSymmetricIdCombination::Concat => {
                concat_nonsymmetric_id_sketches(
                    far_field_sketch,
                    aux_sketch,
                    usable_nullspace_rows(subs_sample_dim, near_field_inds.len()),
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn null_near_field<
    Item: RlstScalar
        + MatrixId
        + MatrixIdNoSkel
        + MatrixInverse
        + MatrixPseudoInverse
        + RandScalar
        + MatrixLu
        + MatrixQr,
>(
    target_inds: &[usize],
    near_field_inds: &[usize],
    y_data: &SketchData<Item>,
    z_data: &SketchData<Item>,
    subs_sample_dim: usize,
    symmetry: &Symmetry,
    fixed_rank: bool,
    id_options: &IdOptions<Item>,
) -> DynamicArray<Item, 2>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    QrDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixQrDecomposition<Item = Item>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let mut scratch = ExtractionScratch::new();
    null_near_field_into(
        target_inds,
        near_field_inds,
        y_data,
        z_data,
        subs_sample_dim,
        symmetry,
        fixed_rank,
        id_options,
        &mut scratch.primary,
        &mut scratch.secondary,
        &mut scratch.tertiary,
        &mut scratch.normal,
    );
    scratch.primary
}

#[allow(clippy::too_many_arguments)]
pub fn near_box_extraction_into<Item: RlstScalar + MatrixPseudoInverse + MatrixLu>(
    ind_r: &[usize],
    near_field_inds: &[usize],
    sketch_data: &SketchData<Item>,
    subs_sample_dim: usize,
    _fixed_rank: bool,
    conjugate_data: bool,
    lu_options: &ExtractOptions<Item>,
    sample_r: &mut DynamicArray<Item, 2>,
    sample_n: &mut DynamicArray<Item, 2>,
    near_box: &mut DynamicArray<Item, 2>,
) -> (Duration, Duration)
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let dim = sketch_data.test.shape()[1];
    let test_subview = sketch_data
        .test
        .r()
        .into_subview([0, 0], [subs_sample_dim, dim]);
    let sketch_subview = sketch_data
        .sketch
        .r()
        .into_subview([0, 0], [subs_sample_dim, dim]);
    let start = Instant::now();
    extract_axis_into(sample_r, &sketch_subview, ind_r, 1, false);
    extract_axis_into(sample_n, &test_subview, near_field_inds, 1, false);
    if conjugate_data {
        conjugate_array_in_place(sample_r);
        conjugate_array_in_place(sample_n);
    }

    let lu_io_time = start.elapsed();
    let start = Instant::now();
    block_extraction_into(sample_n, sample_r, lu_options, near_box);
    trace_memory_growth(
        &format!(
            "near_box_extraction block (|r|={}, |near|={}, samples={subs_sample_dim})",
            ind_r.len(),
            near_field_inds.len()
        ),
        Some(
            matrix_bytes::<Item>(subs_sample_dim, ind_r.len())
                + matrix_bytes::<Item>(subs_sample_dim, near_field_inds.len())
                + matrix_bytes::<Item>(near_field_inds.len(), ind_r.len()),
        ),
    );
    let lu_b_ext_time = start.elapsed();
    (lu_io_time, lu_b_ext_time)
}

#[allow(clippy::too_many_arguments)]
pub fn near_box_extraction<Item: RlstScalar + MatrixPseudoInverse + MatrixLu>(
    ind_r: &[usize],
    near_field_inds: &[usize],
    sketch_data: &SketchData<Item>,
    subs_sample_dim: usize,
    fixed_rank: bool,
    conjugate_data: bool,
    lu_options: &ExtractOptions<Item>,
    r_numbering: &[usize],
    t_numbering: &[usize],
) -> (
    DynamicArray<Item, 2>,
    DynamicArray<Item, 2>,
    (Duration, Duration),
)
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let mut sample_r = empty_array();
    let mut sample_n = empty_array();
    let mut near_box = empty_array();
    let mut data_r = empty_array();
    let mut data_n = empty_array();
    let timings = near_box_extraction_into(
        ind_r,
        near_field_inds,
        sketch_data,
        subs_sample_dim,
        fixed_rank,
        conjugate_data,
        lu_options,
        &mut sample_r,
        &mut sample_n,
        &mut near_box,
    );
    extract_axis_into(&mut data_r, &near_box, r_numbering, 0, true);
    extract_axis_into(&mut data_n, &near_box, t_numbering, 0, true);
    (data_r, data_n, timings)
}

pub fn extract_lu_factor_from_blocks<Item: RlstScalar + MatrixInverse + MatrixLu>(
    pivot_block: &mut DynamicArray<Item, 2>,
    rect_block: &mut DynamicArray<Item, 2>,
    pivot_method: &PivotMethod,
) -> FactorData<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    let pivot_shape = pivot_block.shape();
    let rect_shape = rect_block.shape();
    let pivot_bytes = matrix_bytes::<Item>(pivot_shape[0], pivot_shape[1]);
    let rect_bytes = matrix_bytes::<Item>(rect_shape[0], rect_shape[1]);

    match pivot_method {
        PivotMethod::DirectInversion => {
            trace_memory_event(
                &format!(
                    "extract_lu_factor direct reuse pivot block (r={}, samples={})",
                    pivot_shape[0], pivot_shape[1]
                ),
                Some(pivot_bytes),
            );
            let mut arr = empty_array();
            std::mem::swap(&mut arr, pivot_block);

            trace_memory_event(
                &format!(
                    "extract_lu_factor direct clone pivot -> inv_arr (r={}, samples={})",
                    pivot_shape[0], pivot_shape[1]
                ),
                Some(pivot_bytes),
            );
            let mut y_r_inv = empty_array();
            y_r_inv.fill_from_resize(arr.r());
            y_r_inv.r_mut().into_inverse_alloc().unwrap();

            trace_memory_event(
                &format!(
                    "extract_lu_factor direct reuse rect block (t={}, samples={})",
                    rect_shape[1], rect_shape[0]
                ),
                Some(rect_bytes),
            );
            let mut rectg = empty_array();
            std::mem::swap(&mut rectg, rect_block);

            trace_memory_growth(
                &format!(
                    "extract_lu_factor direct inversion (r={}, t={})",
                    pivot_shape[0], rect_shape[1]
                ),
                Some(pivot_bytes * 2 + rect_bytes),
            );
            let sq = RegSMat {
                arr,
                inv_arr: y_r_inv,
            };
            let factor = ComposedFactorData {
                sq: SquareArr::Reg(sq),
                rectg: RectArr {
                    arr: Box::new(rectg),
                },
            };
            FactorData::Comp(factor)
        }
        PivotMethod::Lu(alpha) => {
            trace_memory_event(
                &format!(
                    "extract_lu_factor lu reuse pivot block (r={}, samples={})",
                    pivot_shape[0], pivot_shape[1]
                ),
                Some(pivot_bytes),
            );
            let mut lu_input = empty_array();
            std::mem::swap(&mut lu_input, pivot_block);
            add_diagonal(&mut lu_input, Item::real(*alpha));
            let lu: LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>> =
                <Item as MatrixLu>::into_lu_alloc(lu_input).unwrap();
            let square_factors = SquareLuFactors::from_lu(&lu).unwrap();
            trace_memory_growth(
                &format!("extract_lu_factor lu workspace (r={})", pivot_shape[0]),
                Some(pivot_bytes * 3 + rect_bytes),
            );
            let lu_arr = LuSMat { square_factors };
            trace_memory_event(
                &format!(
                    "extract_lu_factor lu reuse rect block (t={}, samples={})",
                    rect_shape[1], rect_shape[0]
                ),
                Some(rect_bytes),
            );
            let mut rectg = empty_array();
            std::mem::swap(&mut rectg, rect_block);
            let factor = ComposedFactorData {
                sq: SquareArr::Lu(lu_arr),
                rectg: RectArr {
                    arr: Box::new(rectg),
                },
            };
            FactorData::Comp(factor)
        }
        PivotMethod::LuHybrid(alpha) => {
            trace_memory_event(
                &format!(
                    "extract_lu_factor lu hybrid reuse pivot block (r={}, samples={})",
                    pivot_shape[0], pivot_shape[1]
                ),
                Some(pivot_bytes),
            );
            let mut lu_input = empty_array();
            std::mem::swap(&mut lu_input, pivot_block);
            add_diagonal(&mut lu_input, Item::real(*alpha));
            let lu: LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>> =
                <Item as MatrixLu>::into_lu_alloc(lu_input).unwrap();
            trace_memory_event(
                &format!(
                    "extract_lu_factor lu hybrid reuse rect block (t={}, samples={})",
                    rect_shape[1], rect_shape[0]
                ),
                Some(rect_bytes),
            );
            let mut rectg = empty_array();
            std::mem::swap(&mut rectg, rect_block);
            <LuDecomposition<Item, _> as MatrixLuDecomposition>::solve_mat(
                &lu,
                TransMode::NoTrans,
                rectg.r_mut(),
            )
            .unwrap();
            trace_memory_growth(
                &format!(
                    "extract_lu_factor lu hybrid solve update block (r={})",
                    pivot_shape[0]
                ),
                Some(pivot_bytes + rect_bytes),
            );
            FactorData::Reg(RectArr {
                arr: Box::new(rectg),
            })
        }
    }
}

pub fn extract_lu_factor<Item: RlstScalar + MatrixInverse + MatrixLu>(
    data_r: &DynamicArray<Item, 2>,
    data_n: &DynamicArray<Item, 2>,
    pivot_method: &PivotMethod,
) -> FactorData<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    let data_r_shape = data_r.shape();
    let data_n_shape = data_n.shape();

    match pivot_method {
        PivotMethod::DirectInversion => {
            let data_r_trans_bytes = matrix_bytes::<Item>(data_r_shape[1], data_r_shape[0]);
            let data_n_trans_bytes = matrix_bytes::<Item>(data_n_shape[1], data_n_shape[0]);

            trace_memory_event(
                &format!(
                    "extract_lu_factor direct transpose data_r -> y_r_inv (r={}, samples={})",
                    data_r_shape[0], data_r_shape[1]
                ),
                Some(data_r_trans_bytes),
            );
            let mut y_r_inv = empty_array();
            y_r_inv.r_mut().fill_from_resize(data_r.r().transpose());
            y_r_inv.r_mut().into_inverse_alloc().unwrap();

            trace_memory_event(
                &format!(
                    "extract_lu_factor direct transpose data_n -> rectg (t={}, samples={})",
                    data_n_shape[0], data_n_shape[1]
                ),
                Some(data_n_trans_bytes),
            );
            let mut rectg = empty_array();
            rectg.fill_from_resize(data_n.r().transpose());

            trace_memory_event(
                &format!(
                    "extract_lu_factor direct transpose data_r -> arr (r={}, samples={})",
                    data_r_shape[0], data_r_shape[1]
                ),
                Some(data_r_trans_bytes),
            );
            let mut arr = empty_array();
            arr.r_mut().fill_from_resize(data_r.r().transpose());
            trace_memory_growth(
                &format!(
                    "extract_lu_factor direct inversion (r={}, t={})",
                    data_r_shape[0], data_n_shape[0]
                ),
                Some(
                    matrix_bytes::<Item>(data_r_shape[1], data_r_shape[0]) * 2
                        + matrix_bytes::<Item>(data_n_shape[1], data_n_shape[0]),
                ),
            );
            let sq = RegSMat {
                arr,
                inv_arr: y_r_inv,
            };
            let factor = ComposedFactorData {
                sq: SquareArr::Reg(sq),
                rectg: RectArr {
                    arr: Box::new(rectg),
                },
            };
            FactorData::Comp(factor)
        }
        PivotMethod::Lu(alpha) => {
            let shape = data_r.shape();
            let data_r_trans_bytes = matrix_bytes::<Item>(shape[1], shape[0]);
            let data_n_trans_bytes = matrix_bytes::<Item>(data_n_shape[1], data_n_shape[0]);

            trace_memory_event(
                &format!(
                    "extract_lu_factor lu transpose data_r -> lu_input (r={}, samples={})",
                    shape[0], shape[1]
                ),
                Some(data_r_trans_bytes),
            );
            let mut data_r_trans = empty_array();
            data_r_trans.fill_from_resize(data_r.r().transpose());
            add_diagonal(&mut data_r_trans, Item::real(*alpha));
            let lu: LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>> =
                <Item as MatrixLu>::into_lu_alloc(data_r_trans).unwrap();
            let square_factors = SquareLuFactors::from_lu(&lu).unwrap();
            trace_memory_growth(
                &format!("extract_lu_factor lu workspace (r={})", shape[0]),
                Some(
                    matrix_bytes::<Item>(shape[0], shape[1]) * 3
                        + matrix_bytes::<Item>(data_n_shape[1], data_n_shape[0]),
                ),
            );
            let lu_arr = LuSMat { square_factors };
            let sq = SquareArr::Lu(lu_arr);
            trace_memory_event(
                &format!(
                    "extract_lu_factor lu transpose data_n -> rectg (t={}, samples={})",
                    data_n_shape[0], data_n_shape[1]
                ),
                Some(data_n_trans_bytes),
            );
            let mut rectg = empty_array();
            rectg.fill_from_resize(data_n.r().transpose());
            let factor = ComposedFactorData {
                sq,
                rectg: RectArr {
                    arr: Box::new(rectg),
                },
            };
            FactorData::Comp(factor)
        }
        PivotMethod::LuHybrid(alpha) => {
            let shape = data_r.shape();
            let data_r_trans_bytes = matrix_bytes::<Item>(shape[1], shape[0]);
            let data_n_trans_bytes = matrix_bytes::<Item>(data_n_shape[1], data_n_shape[0]);

            trace_memory_event(
                &format!(
                    "extract_lu_factor lu hybrid transpose data_r -> lu_input (r={}, samples={})",
                    shape[0], shape[1]
                ),
                Some(data_r_trans_bytes),
            );
            let mut data_r_trans = empty_array();
            data_r_trans.fill_from_resize(data_r.r().transpose());
            add_diagonal(&mut data_r_trans, Item::real(*alpha));
            let lu: LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>> =
                <Item as MatrixLu>::into_lu_alloc(data_r_trans).unwrap();
            trace_memory_event(
                &format!(
                    "extract_lu_factor lu hybrid transpose data_n -> rectg (t={}, samples={})",
                    data_n_shape[0], data_n_shape[1]
                ),
                Some(data_n_trans_bytes),
            );
            let mut rectg = empty_array();
            rectg.fill_from_resize(data_n.r().transpose());
            <LuDecomposition<Item, _> as MatrixLuDecomposition>::solve_mat(
                &lu,
                TransMode::NoTrans,
                rectg.r_mut(),
            )
            .unwrap();
            trace_memory_growth(
                &format!(
                    "extract_lu_factor lu hybrid solve update block (r={})",
                    shape[0]
                ),
                Some(data_r_trans_bytes + data_n_trans_bytes),
            );
            FactorData::Reg(RectArr {
                arr: Box::new(rectg),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concat_mode_stacks_sketches_by_rows() {
        let mut primary = rlst_dynamic_array2!(f64, [3, 2]);
        primary.r_mut()[[0, 0]] = 1.0;
        primary.r_mut()[[0, 1]] = 2.0;
        primary.r_mut()[[1, 0]] = 3.0;
        primary.r_mut()[[1, 1]] = 4.0;
        primary.r_mut()[[2, 0]] = 99.0;
        primary.r_mut()[[2, 1]] = 99.0;

        let mut secondary = rlst_dynamic_array2!(f64, [3, 2]);
        secondary.r_mut()[[0, 0]] = 5.0;
        secondary.r_mut()[[0, 1]] = 6.0;
        secondary.r_mut()[[1, 0]] = 7.0;
        secondary.r_mut()[[1, 1]] = 8.0;
        secondary.r_mut()[[2, 0]] = 88.0;
        secondary.r_mut()[[2, 1]] = 88.0;

        concat_nonsymmetric_id_sketches(&mut primary, &secondary, 2);

        assert_eq!(primary.shape(), [4, 2]);
        assert_eq!(primary.r()[[0, 0]], 1.0);
        assert_eq!(primary.r()[[0, 1]], 2.0);
        assert_eq!(primary.r()[[1, 0]], 3.0);
        assert_eq!(primary.r()[[1, 1]], 4.0);
        assert_eq!(primary.r()[[2, 0]], 5.0);
        assert_eq!(primary.r()[[2, 1]], 6.0);
        assert_eq!(primary.r()[[3, 0]], 7.0);
        assert_eq!(primary.r()[[3, 1]], 8.0);
    }
}
