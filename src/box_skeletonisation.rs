pub use rlst::prelude::*;
pub use rlst::dense::array::empty_array;
use crate::{sketch::BoxesData, utils_linear_algebra::MatrixExt};
use rlst::dense::linalg::interpolative_decomposition::Accuracy;
use crate::elementary_matrix::{OpType, ElementaryOperations, ElementaryMatrix};
use crate::sketch::SketchUpdates;
use crate::utils_linear_algebra::LeastSquares;
//crate::utils_linear_algebra::pinv_right;

type ArrayImpl<Item> = BaseArray<Item, VectorContainer<Item>, 2>;
pub struct BoxFeatures<Item: RlstScalar> 
{
    id_tol: Item,
    null_tol: Item,
    target_inds: DynamicArray<usize, 1>,
    near_field_inds: DynamicArray<usize, 1>,
    ind_r: DynamicArray<usize, 1>,
    ind_s: DynamicArray<usize, 1>,
}

pub trait SkelBox{
    type Item: RlstScalar;
    type ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
        + Stride<2>
        + RawAccessMut<Item = Self::Item>
        + Shape<2>;
    fn new(null_tol: Self::Item, target_inds: DynamicArray<usize, 1>, near_field_inds: DynamicArray<usize, 1>, id_tol: f64)->Self;//TODO: ext to f32
    fn null_sketch_near_field(&self, arr: &mut Array<Self::Item, Self::ArrayImpl, 2>, sketch: &mut Array<Self::Item, Self::ArrayImpl, 2>, test: &mut Array<Self::Item, Self::ArrayImpl, 2>);
    fn null_near_field(&mut self, far_field_sketch: &mut Array<Self::Item, Self::ArrayImpl, 2>, y_data: &mut BoxesData<Self::Item>, z_data: &mut BoxesData<Self::Item>);  
    fn get_id_factor(&mut self, dim: usize, far_field_sketch: Array<Self::Item, Self::ArrayImpl, 2>)-> Option<ElementaryMatrix<f64>>;
    fn get_lu_factors(&mut self, dim: usize, y_data: &mut BoxesData<Self::Item>, z_data: &mut BoxesData<Self::Item>)-> (ElementaryMatrix<f64>, ElementaryMatrix<f64>); 
    fn near_box_extraction(&self, sketch_data: &mut BoxesData<Self::Item>, data_r: &mut DynamicArray<f64, 2>, data_n: &mut DynamicArray<f64, 2>);
    fn decouple_box(&mut self, dim: usize, y_data: &mut BoxesData<Self::Item>, z_data: &mut BoxesData<Self::Item>)->Rank<Self::Item>; 
}

pub struct Factor<Item: RlstScalar>
{
    left: ElementaryMatrix<Item>,
    right: ElementaryMatrix<Item>
}

pub enum Rank<T:RlstScalar> {
    Low(Factor<T>, Factor<T>),
    Full,
}


macro_rules! impl_skel_box{
    ($scalar:ty) => {
        impl SkelBox for BoxFeatures<$scalar> {
            type Item = $scalar;
            type ArrayImpl = ArrayImpl<$scalar>;

            fn new(null_tol: Self::Item, target_inds: DynamicArray<usize, 1>, near_field_inds: DynamicArray<usize, 1>, id_tol: f64)->Self{
                let ind_r = empty_array();
                let ind_s = empty_array();
                Self{id_tol, null_tol, target_inds, near_field_inds, ind_r, ind_s}
            }

            fn null_sketch_near_field(&self, arr: &mut Array<Self::Item, Self::ArrayImpl, 2>, sketch: &mut Array<Self::Item, Self::ArrayImpl, 2>, test: &mut Array<Self::Item, Self::ArrayImpl, 2>){
                let mut sub_test = <Self::Item as MatrixExt>::into_ext_alloc(test.view_mut(), self.near_field_inds.data(), 0).unwrap().ext;
                let mut sub_sketch = <Self::Item as MatrixExt>::into_ext_alloc(sketch.view_mut(), self.target_inds.data(), 0).unwrap().ext;
                let null_near_field = sub_test.view_mut().into_null_alloc(self.null_tol).unwrap();
                arr.view_mut().simple_mult_into_resize(sub_sketch.view_mut(), null_near_field.null_space_arr.view());
            }

            fn null_near_field(&mut self, far_field_sketch: &mut Array<Self::Item, Self::ArrayImpl, 2>, y_data: &mut BoxesData<Self::Item>, z_data: &mut BoxesData<Self::Item>){
                let mut null_y_sketch : DynamicArray<Self::Item, 2> = empty_array();
                let mut null_z_sketch : DynamicArray<Self::Item, 2> = empty_array();
                Self::null_sketch_near_field(&self, &mut null_y_sketch, & mut y_data.sketch, & mut y_data.test);
                Self::null_sketch_near_field(&self, &mut null_z_sketch, & mut z_data.sketch, & mut z_data.test);
                far_field_sketch.fill_from_resize(null_y_sketch.view() + null_z_sketch.view());
            }

            fn get_id_factor(&mut self, dim: usize, far_field_sketch: Array<Self::Item, Self::ArrayImpl, 2>)-> Option<ElementaryMatrix<f64>>{
                let max_rank: usize = *far_field_sketch.shape().iter().min().unwrap();
                let id_sketch: IdDecomposition<f64> = far_field_sketch.into_id_alloc(Accuracy::Tol(self.id_tol)).unwrap();
                let k: usize = id_sketch.rank;
                let target_inds_len = self.target_inds.shape()[0];

                if id_sketch.rank < max_rank{
                    self.ind_r.fill_from_resize(self.target_inds.view().into_subview([k], [target_inds_len-k]));
                    self.ind_s.fill_from_resize(self.target_inds.view().into_subview([0], [k]));
                    let id_el_matrix: ElementaryMatrix<f64> = <ElementaryMatrix<f64> as ElementaryOperations>::new(dim, self.ind_r.view().data().to_vec(), self.ind_s.view().data().to_vec(), OpType::Row(id_sketch.id_mat)).unwrap();
                    Some(id_el_matrix)
                }
                else{
                    None
                }
            }

            fn near_box_extraction(&self, sketch_data: &mut BoxesData<Self::Item>, data_r: &mut DynamicArray<f64, 2>, data_n: &mut DynamicArray<f64, 2>){
                let mut sketch_r = empty_array();
                sketch_r.fill_from_resize(<Self::Item as MatrixExt>::into_ext_alloc(sketch_data.sketch.view_mut(), self.ind_r.data(), 0).unwrap().ext);
                let mut test_n = empty_array();
                test_n.fill_from_resize(<Self::Item as MatrixExt>::into_ext_alloc(sketch_data.test.view_mut(), self.near_field_inds.data(), 0).unwrap().ext);
                let mut near_box = <Self::Item as LeastSquares>::solve_right(&sketch_r, &test_n);
                let mut r_numbering = Vec::new();
                let mut t_numbering = Vec::new();
                for &elem in self.ind_r.data().to_vec().iter(){
                    r_numbering.push(self.near_field_inds.data().to_vec().iter().position(|&y| y == elem).unwrap());
                }
                for &elem in self.target_inds.data().to_vec().iter(){
                    t_numbering.push(self.near_field_inds.data().to_vec().iter().position(|&y| y == elem).unwrap());
                }
                data_r.fill_from_resize(<Self::Item as MatrixExt>::into_ext_alloc(near_box.view_mut(), &r_numbering, 1).unwrap().ext);
                data_n.fill_from_resize(<Self::Item as MatrixExt>::into_ext_alloc(near_box.view_mut(), &t_numbering, 1).unwrap().ext);
            }

            fn get_lu_factors(&mut self, dim: usize, y_data: &mut BoxesData<Self::Item>, z_data: &mut BoxesData<Self::Item>)-> (ElementaryMatrix<f64>, ElementaryMatrix<f64>){
                let mut y_r: DynamicArray<f64, 2> = empty_array();
                let mut y_n: DynamicArray<f64, 2> = empty_array();
                let mut z_r: DynamicArray<f64, 2> = empty_array();
                let mut z_n: DynamicArray<f64, 2> = empty_array();

                self.near_box_extraction(y_data, &mut y_r, &mut y_n);
                self.near_box_extraction(z_data, &mut z_r, &mut z_n);
                y_r.view_mut().into_inverse_alloc().unwrap();
                z_r.view_mut().into_inverse_alloc().unwrap();

                let l_fact: DynamicArray<f64, 2> = empty_array().simple_mult_into_resize(y_n.view(), y_r.view());
                let u_fact: DynamicArray<f64, 2> = empty_array().simple_mult_into_resize(z_n.view(), z_r.view());
                let l_el_matrix: ElementaryMatrix<f64> = <ElementaryMatrix<f64> as ElementaryOperations>::new(dim, self.target_inds.data().to_vec(), self.ind_r.data().to_vec(), OpType::Row(l_fact)).unwrap();
                let u_el_matrix: ElementaryMatrix<f64> = <ElementaryMatrix<f64> as ElementaryOperations>::new(dim, self.target_inds.data().to_vec(), self.ind_r.data().to_vec(), OpType::Row(u_fact)).unwrap();
                (l_el_matrix, u_el_matrix)
            }

            fn decouple_box(&mut self, dim: usize, y_data: &mut BoxesData<Self::Item>, z_data: &mut BoxesData<Self::Item>)->Rank<Self::Item>{
                let mut far_field_sketch = empty_array();
                Self::null_near_field(self, & mut far_field_sketch, y_data, z_data);
                match Self::get_id_factor(self, dim, far_field_sketch){
                    Some(e_fact)=>{
                        let f_fact = e_fact.get_conj_transpose().unwrap();
                        if self.ind_r.len() > 0{
                            y_data.update_sketch(&e_fact, &f_fact, false);
                            z_data.update_sketch(&f_fact, &e_fact, true);
                            //y_data.permute_indices();
                            let (l_fact, u_fact) = self.get_lu_factors(dim, y_data, z_data);
                            y_data.update_sketch(&l_fact, &u_fact, false);//TODO: change to LU
                            z_data.update_sketch(&u_fact, &l_fact, true);
                            let fact_id = Factor{left:e_fact, right:f_fact};
                            let fact_lu = Factor{left:l_fact, right:u_fact};
                            Rank::Low(fact_id, fact_lu)
                        }
                        else{
                            Rank::Full
                        }
                    },
                    None=>{
                        Rank::Full
                    }
                }
            }

        }
    }
}

impl_skel_box!(f64);

