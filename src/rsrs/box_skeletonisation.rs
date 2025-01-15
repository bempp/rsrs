use rlst::dense::{linalg::interpolative_decomposition::Accuracy, tools::RandScalar};
use rand_distr::{Distribution, Standard, StandardNormal};
use std::time::{Duration, Instant};
pub use rlst::prelude::*;
use crate::{rsrs::sketch::{SketchOps, BoxesData}, utils::{elementary_matrix::{ElementaryMatrix, ElementaryOperations, OpType}, linear_algebra::{solve_right, ExtInsType, Extraction, MatrixExtraction}}};

type ArrayImpl<Item> = BaseArray<Item, VectorContainer<Item>, 2>;


pub struct Tols<T:RlstScalar> {
    pub id: <T as RlstScalar>::Real,
    pub null: <T as RlstScalar>::Real,
    pub lstq: <T as RlstScalar>::Real,
}
pub struct BoxFeatures
{
    pub target_inds: Vec<usize>,
    pub near_field_inds: Vec<usize>,
    pub ind_r: Vec<usize>,
    pub ind_s: Vec<usize>,
    pub ind_t: Vec<usize>
}

pub trait SkelBox<T: RlstScalar>{
    type Item: RlstScalar;
    type ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
        + Stride<2>
        + RawAccessMut<Item = Self::Item>
        + Shape<2>;
    fn new(target_inds: Vec<usize>, near_field_inds: Vec<usize>)->Self;//TODO: ext to f32
    fn null_sketch_near_field(&self, arr: &mut Array<Self::Item, Self::ArrayImpl, 2>, sketch: &mut Array<Self::Item, Self::ArrayImpl, 2>, test: &mut Array<Self::Item, Self::ArrayImpl, 2>, tol_null: <Self::Item as RlstScalar>::Real);
    fn null_near_field(&mut self, far_field_sketch: &mut Array<Self::Item, Self::ArrayImpl, 2>, y_data: &mut BoxesData<Self::Item>, z_data: &mut BoxesData<Self::Item>, tol_null: <Self::Item as RlstScalar>::Real);  
    fn get_id_factor(&mut self, dim: usize, far_field_sketch: Array<Self::Item, Self::ArrayImpl, 2>, tol_id: <Self::Item as RlstScalar>::Real)-> Option<(ElementaryMatrix<Self::Item>, ElementaryMatrix<Self::Item>, Vec<usize>)>;
    fn get_lu_factors(&mut self, dim: usize, y_data: &mut BoxesData<Self::Item>, z_data: &mut BoxesData<Self::Item>, tol_lstq: <Self::Item as RlstScalar>::Real)-> (ElementaryMatrix<Self::Item>, ElementaryMatrix<Self::Item>); 
    fn near_box_extraction(&self, sketch_data: &mut BoxesData<Self::Item>, tol_lstq: <Self::Item as RlstScalar>::Real, r_numbering: &Vec<usize>, t_numbering: &Vec<usize>)->(DynamicArray<Self::Item, 2>, DynamicArray<Self::Item, 2>, (Duration, Duration));
    fn decouple(&mut self, dim: usize, y_data: &mut BoxesData<Self::Item>, z_data: &mut BoxesData<Self::Item>, tols: &Tols<Self::Item>)->Rank<Self::Item>;//, level_indexing: &mut TreeData<'_, C>)->Rank<Self::Item>; 
}

pub struct Factor<T: RlstScalar>
{
    pub left: ElementaryMatrix<T>,
    pub right: ElementaryMatrix<T>,
}

pub struct DecoupledBox<T: RlstScalar>{
    pub fact_id: Factor<T>,
    pub fact_lu: Factor<T>,
    pub perm: Vec<usize>

}

pub enum Rank<T:RlstScalar> {
    Low(DecoupledBox<T>),
    Full,
}

impl <T: RlstScalar +
    MatrixId + MatrixNull + 
    MatrixInverse + MatrixPseudoInverse + 
    RandScalar + mpi::datatype::Equivalence> SkelBox<T> for BoxFeatures 
    where StandardNormal: Distribution<T::Real>,
    Standard: Distribution<T::Real>,
{
    type Item = T;
    type ArrayImpl = ArrayImpl<T>;

    fn new(target_inds: Vec<usize>, near_field_inds: Vec<usize>)->Self{
        let ind_r: Vec<usize> = Vec::new();
        let ind_s: Vec<usize> = Vec::new();
        let ind_t: Vec<usize> = Vec::new();
        Self{target_inds, near_field_inds, ind_r, ind_s, ind_t}
    }

    fn null_sketch_near_field(&self, arr: &mut Array<Self::Item, Self::ArrayImpl, 2>, sketch: &mut Array<Self::Item, Self::ArrayImpl, 2>, test: &mut Array<Self::Item, Self::ArrayImpl, 2>, tol_null: <Self::Item as RlstScalar>::Real){
        let null_dim = test.shape()[1]- self.near_field_inds.len();
        let mut sub_test: DynamicArray<Self::Item, 2> = <Extraction<Self::Item> as MatrixExtraction>::new(test, ExtInsType::Axis(self.near_field_inds.clone(), 0, false)).unwrap().ext;
        let mut sub_sketch: DynamicArray<Self::Item, 2> = <Extraction<Self::Item> as MatrixExtraction>::new(sketch, ExtInsType::Axis(self.target_inds.clone(), 0, false)).unwrap().ext;
        let null_near_field: NullSpace<Self::Item> = sub_test.view_mut().into_null_alloc(tol_null).unwrap();
        let shape = null_near_field.null_space_arr.shape(); 
        arr.view_mut().simple_mult_into_resize(sub_sketch.view_mut(), null_near_field.null_space_arr.into_subview([0, shape[1]-null_dim], [shape[0], null_dim]));
    }

    fn null_near_field(&mut self, far_field_sketch: &mut Array<Self::Item, Self::ArrayImpl, 2>, y_data: &mut BoxesData<Self::Item>, z_data: &mut BoxesData<Self::Item>, tol_null: <Self::Item as RlstScalar>::Real){
        let mut null_y_sketch : DynamicArray<Self::Item, 2> = empty_array();
        let mut null_z_sketch : DynamicArray<Self::Item, 2> = empty_array();
        Self::null_sketch_near_field(self, &mut null_y_sketch, & mut y_data.sketch, & mut y_data.test, tol_null);
        Self::null_sketch_near_field(self, &mut null_z_sketch, & mut z_data.sketch, & mut z_data.test, tol_null);
        far_field_sketch.fill_from_resize(null_y_sketch.view() + null_z_sketch.view());// See if AXPY can be applied here
    }

    fn get_id_factor(&mut self, dim: usize, far_field_sketch: Array<Self::Item, Self::ArrayImpl, 2>, tol_id: <Self::Item as RlstScalar>::Real)-> Option<(ElementaryMatrix<Self::Item>, ElementaryMatrix<Self::Item>, Vec<usize>)>{
        let max_rank: usize = *far_field_sketch.shape().iter().min().unwrap();
        let id_sketch: IdDecomposition<Self::Item> = far_field_sketch.into_id_alloc(Accuracy::Tol(tol_id)).unwrap();
        let k: usize = id_sketch.rank;
        println!("Rank of box: {}. Max rank: {}", k, max_rank);

        if id_sketch.rank < max_rank{
            let mut aux_indices: Vec<usize> = self.target_inds.clone();
            for (id, &elem) in id_sketch.perm.iter().enumerate(){
                *self.target_inds.get_mut(id).unwrap() = *aux_indices.get_mut(elem).unwrap();
                *self.near_field_inds.get_mut(id).unwrap() = *aux_indices.get_mut(elem).unwrap();
            }
            self.ind_r.append(&mut self.target_inds[k..].to_vec());
            self.ind_s.append(&mut self.target_inds[0..k].to_vec());
            let mut id_mat_copy = empty_array();
            id_mat_copy.fill_from_resize(id_sketch.id_mat.view());
            let e_factor: ElementaryMatrix<Self::Item> = <ElementaryMatrix<Self::Item> as ElementaryOperations>::new(dim, self.ind_r.clone(), self.ind_s.clone(), OpType::Row(id_sketch.id_mat), false).unwrap();
            let f_factor: ElementaryMatrix<Self::Item> = <ElementaryMatrix<Self::Item> as ElementaryOperations>::new(dim, self.ind_r.clone(), self.ind_s.clone(), OpType::Row(id_mat_copy), true).unwrap();
            Some((e_factor, f_factor, id_sketch.perm))
        }
        else{
            None
        }
    }

    fn near_box_extraction(&self, sketch_data: &mut BoxesData<Self::Item>, tol_lstq: <Self::Item as RlstScalar>::Real, r_numbering: &Vec<usize>, t_numbering: &Vec<usize>)->(DynamicArray<Self::Item, 2>, DynamicArray<Self::Item, 2>, (Duration, Duration)){
        let start = Instant::now();
        let sketch_r: DynamicArray<Self::Item, 2> = <Extraction<Self::Item> as MatrixExtraction>::new(&mut sketch_data.sketch, ExtInsType::Axis(self.ind_r.clone(), 0, false)).unwrap().ext;
        let test_n: DynamicArray<Self::Item, 2> = <Extraction<Self::Item> as MatrixExtraction>::new(&mut sketch_data.test, ExtInsType::Axis(self.near_field_inds.clone(), 0, false)).unwrap().ext;
        let mut lu_ext_time = start.elapsed();
        let start = Instant::now();
        let mut near_box: DynamicArray<Self::Item, 2> = solve_right(&sketch_r, &test_n, tol_lstq);
        let lu_solving_time = start.elapsed();
        let data_r: DynamicArray<Self::Item, 2>;
        let data_n: DynamicArray<Self::Item, 2>; 
        let start = Instant::now();
        if !sketch_data.trans{
            data_r = <Extraction<Self::Item> as MatrixExtraction>::new(&mut near_box, ExtInsType::Axis(r_numbering.to_vec(), 1, false)).unwrap().ext;
            data_n = <Extraction<Self::Item> as MatrixExtraction>::new(&mut near_box, ExtInsType::Axis(t_numbering.to_vec(), 1, false)).unwrap().ext;
        }else{
            data_r = <Extraction<Self::Item> as MatrixExtraction>::new(&mut near_box, ExtInsType::Axis(r_numbering.to_vec(), 1, true)).unwrap().ext;
            data_n = <Extraction<Self::Item> as MatrixExtraction>::new(&mut near_box, ExtInsType::Axis(t_numbering.to_vec(), 1, true)).unwrap().ext;
        }
        let lu_small_ext_time = start.elapsed();
        lu_ext_time += lu_small_ext_time;
        (data_r, data_n, (lu_ext_time, lu_solving_time))

    }

    fn get_lu_factors(&mut self, dim: usize, y_data: &mut BoxesData<Self::Item>, z_data: &mut BoxesData<Self::Item>, tol_lstq: <Self::Item as RlstScalar>::Real)-> (ElementaryMatrix<Self::Item>, ElementaryMatrix<Self::Item>){
        let mut r_numbering: Vec<usize> = Vec::new();
        let mut t_numbering: Vec<usize> = Vec::new();

        for &elem in self.ind_r.iter(){
            r_numbering.push(self.near_field_inds.iter().position(|&y| y == elem).unwrap());
        }
        for (pos, &elem) in self.near_field_inds.iter().enumerate(){
            if !self.ind_r.contains(&elem){
                t_numbering.push(pos);
                self.ind_t.push(elem);
            }
        }

        let (mut y_r,  y_n, (y_lu_ext_time, y_lu_solving_time)) = self.near_box_extraction(y_data, tol_lstq, &r_numbering, &t_numbering);
        let (mut z_r, z_n, (z_lu_ext_time, z_lu_solving_time)) = self.near_box_extraction(z_data, tol_lstq, &r_numbering, &t_numbering);
        y_r.view_mut().into_inverse_alloc().unwrap();
        z_r.view_mut().into_inverse_alloc().unwrap();

        let lu_ext_time = y_lu_ext_time + z_lu_ext_time;
        let lu_solving_time = y_lu_solving_time + z_lu_solving_time;
        println!("LU sketch/test extraction in {} ms", lu_ext_time.as_millis());
        println!("LU Solve in {} ms", lu_solving_time.as_millis());
        
        let l_fact: DynamicArray<Self::Item, 2> = empty_array().simple_mult_into_resize(z_n.view(), z_r.view());
        let u_fact: DynamicArray<Self::Item, 2> = empty_array().simple_mult_into_resize(y_r.view(), y_n.view());
        let l_el_matrix: ElementaryMatrix<Self::Item> = <ElementaryMatrix<Self::Item> as ElementaryOperations>::new(dim, self.ind_t.clone(), self.ind_r.clone(), OpType::Row(l_fact), false).unwrap();
        let u_el_matrix: ElementaryMatrix<Self::Item> = <ElementaryMatrix<Self::Item> as ElementaryOperations>::new(dim, self.ind_r.clone(), self.ind_t.clone(), OpType::Row(u_fact), false).unwrap();
        (l_el_matrix, u_el_matrix)
    }

    fn decouple(&mut self, dim: usize, y_data: &mut BoxesData<Self::Item>, z_data: &mut BoxesData<Self::Item>, tols: &Tols<Self::Item>)->Rank<Self::Item>{//, level_indexing: &mut TreeData<'_, C>)->Rank<Self::Item>{
        let mut far_field_sketch: DynamicArray<Self::Item, 2> = empty_array();
        let start: Instant = Instant::now();
        Self::null_near_field(self, & mut far_field_sketch, y_data, z_data, tols.null);
        let duration: Duration = start.elapsed();
        println!("Nullification in {} ms", duration.as_millis());

        let start: Instant = Instant::now();
        let id_factor: Option<(ElementaryMatrix<T>, ElementaryMatrix<T>, Vec<usize>)> = Self::get_id_factor(self, dim, far_field_sketch, tols.id);
        let duration: Duration = start.elapsed();
        println!("ID in {} ms", duration.as_millis());

        match id_factor{
            Some((e_fact, f_fact, perm))=>{
                if !self.ind_r.is_empty(){
                    let start: Instant = Instant::now();
                    y_data.update_sketch(&e_fact, &f_fact, false);
                    z_data.update_sketch(&f_fact, &e_fact, true);
                    let duration: Duration = start.elapsed();
                    println!("Update from ID factors in {} ms", duration.as_millis());

                    let start: Instant = Instant::now();
                    let (l_fact, u_fact) = self.get_lu_factors(dim, y_data, z_data, tols.lstq);
                    let duration: Duration = start.elapsed();
                    println!("LU in {} ms", duration.as_millis());

                    let start: Instant = Instant::now();
                    y_data.update_sketch(&l_fact, &u_fact, false);
                    z_data.update_sketch(&u_fact, &l_fact, true);
                    let duration: Duration = start.elapsed();
                    println!("Update from LU in {} ms", duration.as_millis());
                    let fact_id: Factor<T> = Factor{left:e_fact, right:f_fact};
                    let fact_lu: Factor<T> = Factor{left:l_fact, right:u_fact};
                    let res: DecoupledBox<T> = DecoupledBox{fact_id, fact_lu, perm};
                    Rank::Low(res)
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
 


