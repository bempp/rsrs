use std::time::{Duration, Instant};

use crate::{
    rsrs::sketch::SketchData,
    utils::{
        data_ins_ext::{ExtInsType, Extraction, MatrixExtraction},
        linear_algebra::{
            block_extraction, nullify_near_sketch, BlockExtractionMethod, NullMethod,
        },
    },
};
use rand_distr::{Distribution, Standard, StandardNormal};
use rlst::{
    dense::{linalg::lu::MatrixLu, tools::RandScalar},
    prelude::*,
};
use serde::{Deserialize, Serialize};

type Real<T> = <T as rlst::RlstScalar>::Real;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum PivotMethod {
    DirectInversion,
    Lu(f64), //TODO: Change to Item
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
}

fn null_sketch_near_field<
    Item: RlstScalar + MatrixId + MatrixInverse + MatrixPseudoInverse + RandScalar + MatrixLu + MatrixQr,
>(
    target_inds: &[usize],
    near_field_inds: &[usize],
    sketch: &DynamicArray<Item, 2>,
    test: &DynamicArray<Item, 2>,
    subs_sample_dim: usize,
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
    let dim = test.shape()[1];
    let sub_test = test.r().into_subview([0, 0], [subs_sample_dim, dim]);
    let sub_sketch = sketch.r().into_subview([0, 0], [subs_sample_dim, dim]);
    let mut sketch_t = <Extraction<Item> as MatrixExtraction>::new(
        &sub_sketch,
        ExtInsType::Axis(target_inds.to_vec(), 1, false),
    )
    .unwrap()
    .ext;
    let test_n = <Extraction<Item> as MatrixExtraction>::new(
        &sub_test,
        ExtInsType::Axis(near_field_inds.to_vec(), 1, false),
    )
    .unwrap()
    .ext;
    nullify_near_sketch(&test_n, &mut sketch_t, &id_options);
    sketch_t
}

pub fn null_near_field<
    Item: RlstScalar + MatrixId + MatrixInverse + MatrixPseudoInverse + RandScalar + MatrixLu + MatrixQr,
>(
    target_inds: &[usize],
    near_field_inds: &[usize],
    y_data: &SketchData<Item>,
    z_data: &SketchData<Item>,
    subs_sample_dim: usize,
    symmetric: bool,
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
    let far_field_sketch = if symmetric {
        null_sketch_near_field(
            target_inds,
            near_field_inds,
            &y_data.sketch,
            &y_data.test,
            subs_sample_dim,
            id_options,
        )
    } else {
        let null_y_sketch = null_sketch_near_field(
            target_inds,
            near_field_inds,
            &y_data.sketch,
            &y_data.test,
            subs_sample_dim,
            id_options,
        );
        let null_z_sketch = null_sketch_near_field(
            target_inds,
            near_field_inds,
            &z_data.sketch,
            &z_data.test,
            subs_sample_dim,
            id_options,
        );
        let mut sketch_sum = empty_array();
        sketch_sum.fill_from_resize(null_y_sketch.r() + null_z_sketch.r());
        sketch_sum
    };

    far_field_sketch
}

pub fn near_box_extraction<Item: RlstScalar + MatrixPseudoInverse + MatrixLu>(
    ind_r: &[usize],
    near_field_inds: &[usize],
    sketch_data: &SketchData<Item>,
    subs_sample_dim: usize,
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
    let sketch_r: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
        &sketch_subview,
        ExtInsType::Axis(ind_r.to_vec(), 1, false),
    )
    .unwrap()
    .ext;
    let mut test_n: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
        &test_subview,
        ExtInsType::Axis(near_field_inds.to_vec(), 1, false),
    )
    .unwrap()
    .ext;

    let mut lu_io_time = start.elapsed();
    let start = Instant::now();
    let near_box = block_extraction(&mut test_n, &sketch_r, lu_options);
    let lu_b_ext_time = start.elapsed();
    let start = Instant::now();
    let data_r = <Extraction<Item> as MatrixExtraction>::new(
        &near_box,
        ExtInsType::Axis(r_numbering.to_vec(), 0, false),
    )
    .unwrap()
    .ext;
    let data_n = <Extraction<Item> as MatrixExtraction>::new(
        &near_box,
        ExtInsType::Axis(t_numbering.to_vec(), 0, false),
    )
    .unwrap()
    .ext;
    let lu_small_io_time = start.elapsed();
    lu_io_time += lu_small_io_time;
    (data_r, data_n, (lu_io_time, lu_b_ext_time))
}
