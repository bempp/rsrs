use rlst::{dense::linalg::interpolative_decomposition::Accuracy, empty_array, Array, DynamicArray, IdDecomposition, MatrixId, MatrixInverse, MatrixPseudoInverse, MultIntoResize, RawAccessMut, RlstResult, RlstScalar, Shape, UnsafeRandomAccessByRef, UnsafeRandomAccessByValue, UnsafeRandomAccessMut};
use crate::utils::{data_ins_ext::{matrix_insertion, solve_left, solve_right, ExtInsType, Extraction, MatrixExtraction}, elementary_matrix::{col_ops, col_perm, row_ops, row_perm}};
use super::{rsrs_cycle::RsrsOptions, sketch::BoxesData};
use std::time::{Duration, Instant};
use num::One;

pub struct FactorOptions {
    /// Inverse operation
    pub inv: bool,
    /// Transpose operation
    pub trans: bool

}

pub enum OpInfo<T:RlstScalar> {
    DecFact(DynamicArray<T, 2>, DynamicArray<T, 2>, Vec<usize>, Vec<usize>),
    DiagBlocks(Vec<DynamicArray<T, 2>>),
    Perm(Vec<usize>, Vec<usize>),
}

pub enum DecFactorOpType{
    Left,
    Right
}


pub enum FactorType{
    F,
    S
}

pub struct IdFactor<T:RlstScalar>{
    data: DynamicArray<T, 2>,
    pub perm: Vec<usize>,
    pub ind_r: Vec<usize>,//row_indices
    pub ind_s: Vec<usize>,//col_indices
}

pub trait IdFactorOperations: Sized {
    /// Item type
    type Item: RlstScalar;

    fn new(target_inds: &mut Vec<usize>, near_field_inds: &mut Vec<usize>, target_arr: DynamicArray<Self::Item, 2>, tol_id: <Self::Item as RlstScalar>::Real, options: &RsrsOptions)-> Option<Self>;

    fn mul<ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
    + Shape<2>
    + RawAccessMut<Item = Self::Item>
    + UnsafeRandomAccessMut<2, Item = Self::Item>
    + UnsafeRandomAccessByRef<2, Item = Self::Item>>(&self, target_arr: &mut Array<Self::Item, ArrayImplMut, 2>, options: &FactorOptions, factor_type: FactorType, operation_type: DecFactorOpType);
}

impl <T:RlstScalar + MatrixInverse + MatrixId>IdFactorOperations for IdFactor<T>
{
    type Item = T;

    fn new(target_inds: &mut Vec<usize>, near_field_inds: &mut Vec<usize>, target_arr: DynamicArray<Self::Item, 2>, tol_id: <Self::Item as RlstScalar>::Real, options: &RsrsOptions)-> Option<Self>{
        let max_rank: usize = *target_arr.shape().iter().min().unwrap();
        let id_sketch: IdDecomposition<Self::Item> = target_arr.into_id_alloc(Accuracy::Tol(tol_id)).unwrap();
        let k: usize = id_sketch.rank;
        let mut ind_r = Vec::new();
        let mut ind_s = Vec::new();

        if !options.silent{
            println!("Rank of box: {}. Max rank: {}", k, max_rank);
        }

        if id_sketch.rank < max_rank{
            let mut aux_indices: Vec<usize> = target_inds.clone();
            for (id, &elem) in id_sketch.perm.iter().enumerate(){
                *target_inds.get_mut(id).unwrap() = *aux_indices.get_mut(elem).unwrap();
                *near_field_inds.get_mut(id).unwrap() = *aux_indices.get_mut(elem).unwrap();
            }
            ind_r.append(&mut target_inds[k..].to_vec());
            ind_s.append(&mut target_inds[0..k].to_vec());

            Some(Self{data: id_sketch.id_mat, perm: id_sketch.perm, ind_r, ind_s})
        }
        else{
            None
        }
    }

    fn mul<ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
    + Shape<2>
    + RawAccessMut<Item = Self::Item>
    + UnsafeRandomAccessMut<2, Item = Self::Item>
    + UnsafeRandomAccessByRef<2, Item = Self::Item>>(&self, target_arr: &mut Array<Self::Item, ArrayImplMut, 2>, options: &FactorOptions, factor_type: FactorType, operation_type: DecFactorOpType){
        let mut beta: Self::Item = <Self::Item as One>::one();
        let mut trans = options.trans;
        if options.inv 
        {
            beta = -<Self::Item as One>::one();
        }

        match factor_type{
            FactorType::F => {
            },
            FactorType::S => {
                trans = !trans;
            },
        }

        match operation_type {
            DecFactorOpType::Left => {row_ops(self.ind_s.clone(), self.ind_r.clone(), &self.data, target_arr, beta, trans);},
            DecFactorOpType::Right => {col_ops(self.ind_s.clone(), self.ind_r.clone(), &self.data, target_arr, beta, trans);},
        }
    }
  
}

pub struct LuFactor<T:RlstScalar>{
    l_arr: DynamicArray<T, 2>,
    u_arr: DynamicArray<T, 2>,
    hermitian: bool,
    ind_r: Vec<usize>, //cols
    ind_t: Vec<usize> //rows
}

fn near_box_extraction<Item: RlstScalar + MatrixPseudoInverse>(ind_r: &[usize], near_field_inds: &[usize], sketch_data: &mut BoxesData<Item>, tol_lstq: <Item as RlstScalar>::Real, r_numbering: &Vec<usize>, t_numbering: &Vec<usize>)->(DynamicArray<Item, 2>, DynamicArray<Item, 2>, (Duration, Duration)){
    let start = Instant::now();
    let sketch_r: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(&mut sketch_data.sketch, ExtInsType::Axis(ind_r.to_vec(), 0, false)).unwrap().ext;
    let test_n: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(&mut sketch_data.test, ExtInsType::Axis(near_field_inds.to_vec(), 0, false)).unwrap().ext;
    let mut lu_io_time = start.elapsed();
    let start = Instant::now();
    let mut near_box: DynamicArray<Item, 2> = solve_right(&sketch_r, &test_n, tol_lstq);
    let lu_b_ext_time = start.elapsed();
    let data_r: DynamicArray<Item, 2>;
    let data_n: DynamicArray<Item, 2>; 
    let start = Instant::now();
    if !sketch_data.trans{
        data_r = <Extraction<Item> as MatrixExtraction>::new(&mut near_box, ExtInsType::Axis(r_numbering.to_vec(), 1, false)).unwrap().ext;
        data_n = <Extraction<Item> as MatrixExtraction>::new(&mut near_box, ExtInsType::Axis(t_numbering.to_vec(), 1, false)).unwrap().ext;
    }else{
        data_r = <Extraction<Item> as MatrixExtraction>::new(&mut near_box, ExtInsType::Axis(r_numbering.to_vec(), 1, true)).unwrap().ext;
        data_n = <Extraction<Item> as MatrixExtraction>::new(&mut near_box, ExtInsType::Axis(t_numbering.to_vec(), 1, true)).unwrap().ext;
    }
    let lu_small_io_time = start.elapsed();
    lu_io_time += lu_small_io_time;
    (data_r, data_n, (lu_io_time, lu_b_ext_time))
}

pub struct LuTimes{
    io: u128,
    extraction: u128,
    assembly: u128
}
pub trait LuFactorOperations: Sized {
    type Item: RlstScalar;
    fn new(ind_r: &[usize], near_field_inds: &[usize], y_data: &mut BoxesData<Self::Item>, z_data: &mut BoxesData<Self::Item>, tol_lstq: <Self::Item as RlstScalar>::Real, options: &RsrsOptions)-> (Self, LuTimes);
    fn mul<ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
    + Shape<2>
    + RawAccessMut<Item = Self::Item>
    + UnsafeRandomAccessMut<2, Item = Self::Item>
    + UnsafeRandomAccessByRef<2, Item = Self::Item>>(&self, target_arr: &mut Array<Self::Item, ArrayImplMut, 2>, options: &FactorOptions, factor_type: FactorType, operation_type: DecFactorOpType);
}

impl <T:RlstScalar + MatrixInverse + MatrixPseudoInverse>LuFactorOperations for LuFactor<T>
{
    type Item = T;

    fn new(ind_r: &[usize], near_field_inds: &[usize], y_data: &mut BoxesData<Self::Item>, z_data: &mut BoxesData<Self::Item>, tol_lstq: <Self::Item as RlstScalar>::Real, options: &RsrsOptions)->(Self, LuTimes){
        let mut r_numbering: Vec<usize> = Vec::new();
        let mut t_numbering: Vec<usize> = Vec::new();
        let mut ind_t = Vec::new();

        for &elem in ind_r.iter(){
            r_numbering.push(near_field_inds.iter().position(|&y| y == elem).unwrap());
        }
        for (pos, &elem) in near_field_inds.iter().enumerate(){
            if !ind_r.contains(&elem){
                t_numbering.push(pos);
                ind_t.push(elem);
            }
        }

        let (mut y_r,  y_n, (y_lu_io_time, y_lu_b_ext_time)) = near_box_extraction(ind_r, near_field_inds, y_data, tol_lstq, &r_numbering, &t_numbering);
        
        let start = Instant::now();
        y_r.view_mut().into_inverse_alloc().unwrap();
        let u_arr = empty_array().simple_mult_into_resize(y_r.view(), y_n.view());
        let u_assembly = start.elapsed();

        let mut l_arr: DynamicArray<Self::Item, 2> = empty_array();

        let lu_io_time;
        let lu_b_ext_time;
        let lu_assembly_time;

        if !options.hermitian{
            let (mut z_r, z_n, (z_lu_io_time, z_lu_b_ext_time)) = near_box_extraction(ind_r, near_field_inds, z_data, tol_lstq, &r_numbering, &t_numbering);
            let mut aux: DynamicArray<Self::Item, 2> = empty_array();

            let start = Instant::now();
            z_r.view_mut().into_inverse_alloc().unwrap();
            aux.view_mut().simple_mult_into_resize(z_n.view(), z_r.view());
            l_arr.view_mut().fill_from_resize(aux.view().conj());
            let l_assembly = start.elapsed();

            lu_io_time = y_lu_io_time + z_lu_io_time;
            lu_b_ext_time = y_lu_b_ext_time + z_lu_b_ext_time;
            lu_assembly_time = u_assembly + l_assembly;
        }
        else{
            lu_io_time = y_lu_io_time;
            lu_b_ext_time = y_lu_b_ext_time;
            lu_assembly_time = u_assembly;
        }

        if !options.silent{
            println!("LU io in {} ms", lu_io_time.as_millis());
            println!("LU block extraction in {} ms", lu_b_ext_time.as_millis());
        }
        
        let lu_times = LuTimes{io: lu_io_time.as_millis(), extraction: lu_b_ext_time.as_millis(), assembly: lu_assembly_time.as_millis()};

        (Self{l_arr, u_arr, hermitian: options.hermitian, ind_r: ind_r.to_vec(), ind_t}, lu_times)
        
    }

    fn mul<ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
    + Shape<2>
    + RawAccessMut<Item = Self::Item>
    + UnsafeRandomAccessMut<2, Item = Self::Item>
    + UnsafeRandomAccessByRef<2, Item = Self::Item>>(&self, target_arr: &mut Array<Self::Item, ArrayImplMut, 2>, options: &FactorOptions, factor_type: FactorType, operation_type: DecFactorOpType){
        let mut beta: Self::Item = <Self::Item as One>::one();
        let mut trans = options.trans;
        if options.inv 
        {
            beta = -<Self::Item as One>::one();
        }

        if self.hermitian{
            match factor_type{
                FactorType::F => {
                    trans = !trans;
                },
                FactorType::S => {
                },
            }
    
            match operation_type {
                DecFactorOpType::Left => {row_ops(self.ind_t.clone(), self.ind_r.clone(), &self.u_arr, target_arr, beta, trans);},
                DecFactorOpType::Right => {col_ops(self.ind_t.clone(), self.ind_r.clone(), &self.u_arr, target_arr, beta, trans);},
            }
        }
        else{
            match factor_type{
                FactorType::F => {
                    match operation_type {
                        DecFactorOpType::Left => {row_ops(self.ind_r.clone(), self.ind_t.clone(), &self.l_arr, target_arr, beta, options.trans);},
                        DecFactorOpType::Right => {col_ops( self.ind_r.clone(), self.ind_t.clone(), &self.l_arr, target_arr, beta, options.trans);},
                    }
                },
                FactorType::S => {
                    match operation_type {
                        DecFactorOpType::Left => {row_ops(self.ind_t.clone(), self.ind_r.clone(), &self.u_arr, target_arr, beta, options.trans);},
                        DecFactorOpType::Right => {col_ops( self.ind_t.clone(), self.ind_r.clone(), &self.u_arr, target_arr, beta, options.trans);},
                    }
                },
            }
        }
    }
  
}

pub struct PermFactor{
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>
}

pub trait PermOperations: Sized {

    fn new(row_indices: Vec<usize>, col_indices: Vec<usize>) -> RlstResult<Self>;

    fn left_mul<T: RlstScalar, ArrayImplMut: UnsafeRandomAccessByValue<2, Item = T>
    + Shape<2>
    + RawAccessMut<Item = T>
    + UnsafeRandomAccessMut<2, Item = T>
    + UnsafeRandomAccessByRef<2, Item = T>>(&self, right_arr: &mut Array<T, ArrayImplMut, 2>, options: &FactorOptions);

    fn right_mul<T: RlstScalar, ArrayImplMut: UnsafeRandomAccessByValue<2, Item = T>
    + Shape<2>
    + RawAccessMut<Item = T>
    + UnsafeRandomAccessMut<2, Item = T>
    + UnsafeRandomAccessByRef<2, Item = T>>(&self, left_arr: &mut Array<T, ArrayImplMut, 2>, options: &FactorOptions);
}

impl PermOperations for PermFactor {
    fn new(row_indices: Vec<usize>, col_indices: Vec<usize>) -> RlstResult<Self> {
        Ok(Self{row_indices, col_indices})
    }

    fn left_mul<T: RlstScalar, ArrayImplMut: UnsafeRandomAccessByValue<2, Item = T>
    + Shape<2>
    + RawAccessMut<Item = T>
    + UnsafeRandomAccessMut<2, Item = T>
    + UnsafeRandomAccessByRef<2, Item = T>>(&self, right_arr: &mut Array<T, ArrayImplMut, 2>, options: &FactorOptions) {
        assert_eq!(self.row_indices.len(), self.col_indices.len());
        let mut trans = options.trans;
        if options.inv{
            trans = !trans;

        }
        row_perm(self.col_indices.clone(), self.row_indices.clone(), right_arr, trans);
    }

    fn right_mul<T: RlstScalar, ArrayImplMut: UnsafeRandomAccessByValue<2, Item = T>
    + Shape<2>
    + RawAccessMut<Item = T>
    + UnsafeRandomAccessMut<2, Item = T>
    + UnsafeRandomAccessByRef<2, Item = T>>(&self, left_arr: &mut Array<T, ArrayImplMut, 2>, options: &FactorOptions) {
        assert_eq!(self.row_indices.len(), self.col_indices.len());
        let mut trans = options.trans;
        if options.inv{
            trans = !trans;

        }
        col_perm(self.col_indices.clone(), self.row_indices.clone(),left_arr, trans);
    }
}

pub struct DiagBoxFactor<T:RlstScalar>{
    pub diag_boxes: Vec<DynamicArray<T, 2>>
}

pub trait DiagBoxOperations: Sized {
    /// Item type
    type Item: RlstScalar;
    fn new(diag_boxes: Vec<DynamicArray<Self::Item, 2>>) -> RlstResult<Self>;

    fn left_mul<ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
    + Shape<2>
    + RawAccessMut<Item = Self::Item>
    + UnsafeRandomAccessMut<2, Item = Self::Item>
    + UnsafeRandomAccessByRef<2, Item = Self::Item>>(&self, right_arr: &mut Array<Self::Item, ArrayImplMut, 2>, options: &FactorOptions);

    fn right_mul<ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
    + Shape<2>
    + RawAccessMut<Item = Self::Item>
    + UnsafeRandomAccessMut<2, Item = Self::Item>
    + UnsafeRandomAccessByRef<2, Item = Self::Item>>(&self, right_arr: &mut Array<Self::Item, ArrayImplMut, 2>, options: &FactorOptions);
}

impl <T:RlstScalar + MatrixInverse>DiagBoxOperations for DiagBoxFactor<T>
{
    type Item = T;

    fn new(diag_boxes: Vec<DynamicArray<Self::Item, 2>>) -> RlstResult<Self> {
        
        Ok(Self{diag_boxes})
    }


    fn left_mul<ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
    + Shape<2>
    + RawAccessMut<Item = Self::Item>
    + UnsafeRandomAccessMut<2, Item = Self::Item>
    + UnsafeRandomAccessByRef<2, Item = Self::Item>>(&self, right_arr: &mut Array<Self::Item, ArrayImplMut, 2>, options: &FactorOptions){
        let count = 0;
        if options.inv{
            for diag_box in &self.diag_boxes{
                let mut inv_diag_box = empty_array();
                inv_diag_box.fill_from_resize(diag_box.view());
                inv_diag_box.view_mut().into_inverse_alloc().unwrap();
                let len_box = diag_box.shape()[0];
                let inds: Vec<usize> = (count..len_box + count).collect();
                let target_rows: DynamicArray<T, 2> = <Extraction<T> as MatrixExtraction>::new(right_arr, ExtInsType::Axis(inds.clone(), 0, false)).unwrap().ext;
                let mut new_target_rows = empty_array();
                new_target_rows.view_mut().simple_mult_into_resize(inv_diag_box.view(), target_rows.view());//TODO: Allow conj transpose
                matrix_insertion(right_arr, &mut new_target_rows, ExtInsType::Axis(inds.clone(), 0, false));
            }
        }
        else{
            for diag_box in &self.diag_boxes{
                let len_box = diag_box.shape()[0];
                let inds: Vec<usize> = (count..len_box + count).collect();
                let target_rows: DynamicArray<T, 2> = <Extraction<T> as MatrixExtraction>::new(right_arr, ExtInsType::Axis(inds.clone(), 0, false)).unwrap().ext;
                let mut new_target_rows = empty_array();
                new_target_rows.view_mut().simple_mult_into_resize(diag_box.view(), target_rows.view());//TODO: Allow conj transpose
                matrix_insertion(right_arr, &mut new_target_rows, ExtInsType::Axis(inds.clone(), 0, false));
            }

        }

    }

    fn right_mul<ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
    + Shape<2>
    + RawAccessMut<Item = Self::Item>
    + UnsafeRandomAccessMut<2, Item = Self::Item>
    + UnsafeRandomAccessByRef<2, Item = Self::Item>>(&self, left_arr: &mut Array<Self::Item, ArrayImplMut, 2>, options: &FactorOptions){

        let count = 0;

        if options.inv{
            for diag_box in &self.diag_boxes{
                let mut inv_diag_box = empty_array();
                inv_diag_box.fill_from_resize(diag_box.view());
                inv_diag_box.view_mut().into_inverse_alloc().unwrap();
                let len_box = diag_box.shape()[0];
                let inds: Vec<usize> = (count..len_box + count).collect();
                let target_rows: DynamicArray<T, 2> = <Extraction<T> as MatrixExtraction>::new(left_arr, ExtInsType::Axis(inds.clone(), 1, false)).unwrap().ext;
                let mut new_target_rows = empty_array();
                new_target_rows.view_mut().simple_mult_into_resize(target_rows.view(), inv_diag_box.view());//TODO: Allow conj transpose
                matrix_insertion(left_arr, &mut new_target_rows, ExtInsType::Axis(inds.clone(), 1, false));
            }
        }
        else{
            for diag_box in &self.diag_boxes{
                let len_box = diag_box.shape()[0];
                let inds: Vec<usize> = (count..len_box + count).collect();
                let target_rows: DynamicArray<T, 2> = <Extraction<T> as MatrixExtraction>::new(left_arr, ExtInsType::Axis(inds.clone(), 1, false)).unwrap().ext;
                let mut new_target_rows = empty_array();
                new_target_rows.view_mut().simple_mult_into_resize(target_rows.view(), diag_box.view());//TODO: Allow conj transpose
                matrix_insertion(left_arr, &mut new_target_rows, ExtInsType::Axis(inds.clone(), 1, false));
            }
        }
    }
  
}

pub enum DecFactorType<Item: RlstScalar>{
    Id(IdFactor<Item>),
    Lu(LuFactor<Item>)
}

pub struct DecFactors<Item:RlstScalar>{
    pub id_factor: IdFactor<Item>,
    pub lu_factor: Option<LuFactor<Item>>,
    pub near_field_inds: Vec<usize>
}

pub struct RsrsFactors<Item: RlstScalar> 
{
    pub dec_factors: Vec<DecFactors<Item>>,
    pub perm_factor: PermFactor,
    pub diag_box_factor: DiagBoxFactor<Item>
}

type Errors<T> = (<T as RlstScalar>::Real, <T as RlstScalar>::Real, <T as RlstScalar>::Real, <T as RlstScalar>::Real);
type VecErrors<T> = (Vec<Vec<<T as RlstScalar>::Real>>, Vec<Vec<<T as RlstScalar>::Real>>);

pub trait RsrsFactorsOps: Sized {
    type Item: RlstScalar;
    fn new() -> Self;

    fn operator_app_mul(&self, target_arr: &mut DynamicArray<Self::Item, 2>);

    fn operator_app_inv_mul(&self, target_arr: &mut DynamicArray<Self::Item, 2>);

    fn box_dec<ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
    + Shape<2>
    + RawAccessMut<Item = Self::Item>
    + UnsafeRandomAccessMut<2, Item = Self::Item>
    + UnsafeRandomAccessByRef<2, Item = Self::Item>>(&self, target_arr: &mut Array<Self::Item, ArrayImplMut, 2>, dec_factors: &DecFactors<Self::Item>, factor_options: &FactorOptions);

    fn boxes_diag_mul(&self, target_arr: &mut DynamicArray<Self::Item, 2>, inv: bool);

    fn set_diagonal_boxes(&self, target_arr: &mut DynamicArray<Self::Item, 2>);

    fn get_diag_errors(&self, arr: &mut DynamicArray<Self::Item, 2>)->Vec<<Self::Item as RlstScalar>::Real>;

    fn box_errors(&self, factor_num: usize, arr: &mut DynamicArray<Self::Item, 2>, near_field_inds: &[usize], silent: bool)->Errors<Self::Item>;

    fn get_boxes_errors(&self, arr: &mut DynamicArray<Self::Item, 2>, silent: bool)->VecErrors<<Self::Item as RlstScalar>::Real>;
}

fn get_far_indices(n: usize, near_indices: Vec<usize>)->Vec<usize>{
    let mut domain: Vec<usize> = (0..n).collect();
    domain.retain(|x| !near_indices.contains(x));
    domain
}

impl <T:RlstScalar + MatrixInverse + MatrixId + MatrixPseudoInverse>RsrsFactorsOps for RsrsFactors<T>
{
    type Item = T;

    fn new() -> Self {
        let dec_factors = Vec::new();
        let row_indices = Vec::new();
        let col_indices = Vec::new();
        let perm_factor = PermFactor{row_indices, col_indices};
        let diag_box_factor = DiagBoxFactor { diag_boxes: Vec::new()};
        Self{dec_factors, perm_factor, diag_box_factor}
    }

    fn operator_app_mul(&self, target_arr: &mut DynamicArray<Self::Item, 2>) {
        let factor_options = FactorOptions{ inv: false, trans: false };
        for dec_factor in self.dec_factors.iter(){
            self.box_dec(target_arr, dec_factor, &factor_options);
        }
    }
    
    fn operator_app_inv_mul(&self, target_arr: &mut DynamicArray<Self::Item, 2>) {
        let factor_options = FactorOptions{ inv: true, trans: false };
        for dec_factor in self.dec_factors.iter(){
            self.box_dec(target_arr, dec_factor, &factor_options);
        }
    }

    fn box_dec<ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
    + Shape<2>
    + RawAccessMut<Item = Self::Item>
    + UnsafeRandomAccessMut<2, Item = Self::Item>
    + UnsafeRandomAccessByRef<2, Item = Self::Item>>(&self, target_arr: &mut Array<Self::Item, ArrayImplMut, 2>, dec_factors: &DecFactors<Self::Item>, factor_options: &FactorOptions){
        dec_factors.id_factor.mul(target_arr, factor_options, FactorType::F, DecFactorOpType::Left);
        dec_factors.id_factor.mul(target_arr, factor_options, FactorType::S, DecFactorOpType::Right);
        if let Some(lu_factor) = &dec_factors.lu_factor {
            lu_factor.mul(target_arr, factor_options, FactorType::F, DecFactorOpType::Left);
            lu_factor.mul(target_arr, factor_options, FactorType::S, DecFactorOpType::Right);
        }
    }

    fn boxes_diag_mul(&self, target_arr: &mut DynamicArray<Self::Item, 2>, inv: bool)
    {
        let mut count = 0;
        for diag_box in self.diag_box_factor.diag_boxes.iter(){
            let len_box = diag_box.shape()[0];
            let inds: Vec<usize> = (count..len_box + count).collect();
            let target_rows: DynamicArray<Self::Item, 2> = <Extraction<Self::Item> as MatrixExtraction>::new(target_arr, ExtInsType::Axis(inds.clone(), 0, false)).unwrap().ext;

            if inv{
                matrix_insertion(target_arr, &mut solve_left(diag_box, &target_rows, num::Zero::zero()), ExtInsType::Axis(inds.clone(), 0, false));
            }
            else{
                let mut res = empty_array().simple_mult_into_resize(diag_box.view(), target_rows);
                matrix_insertion(target_arr, &mut res, ExtInsType::Axis(inds.clone(), 0, false));
            }
            count += len_box;
        }

    }

    fn set_diagonal_boxes(&self, target_arr: &mut DynamicArray<Self::Item, 2>){

        self.perm_factor.left_mul(target_arr, &FactorOptions{ inv: false, trans: false});
        self.perm_factor.right_mul(target_arr, &FactorOptions{ inv: false, trans: true});

    }

    fn get_diag_errors(&self, arr: &mut DynamicArray<Self::Item, 2>)->Vec<<Self::Item as RlstScalar>::Real>
    {
        let mut count = 0;
        let mut diag_ae = Vec::new();

        self.set_diagonal_boxes(arr);

        for diag_box in self.diag_box_factor.diag_boxes.iter(){
            let len_box = diag_box.shape()[0];
            let inds: Vec<usize> = (count..len_box + count).collect();
            let exact_diag_box: DynamicArray<Self::Item, 2> = <Extraction<Self::Item> as MatrixExtraction>::new(arr, ExtInsType::Cross(inds.clone(), inds.clone())).unwrap().ext;
            let mut res: DynamicArray<Self::Item, 2> = empty_array();
            let row_ext: DynamicArray<Self::Item, 2> = <Extraction<Self::Item> as MatrixExtraction>::new(arr, ExtInsType::Axis(inds.clone(), 0, false)).unwrap().ext;
            matrix_insertion(arr, &mut solve_left(diag_box, &row_ext, num::Zero::zero()), ExtInsType::Axis(inds.clone(), 0, false));
            res.fill_from_resize(exact_diag_box - diag_box.view());
            diag_ae.push(res.norm_1());
            count+=len_box;
        }
        diag_ae
    }

    fn box_errors(&self, factor_num: usize, arr: &mut DynamicArray<Self::Item, 2>, near_field_inds: &[usize], silent: bool)->Errors<Self::Item>{

        let ind_r = &self.dec_factors[factor_num].id_factor.ind_r;
        let ind_t = &self.dec_factors[factor_num].lu_factor.as_ref().unwrap().ind_t;
        let ind_s = &self.dec_factors[factor_num].id_factor.ind_s;
        let far_indices = get_far_indices(arr.shape()[0], near_field_inds.to_vec());

        if !silent{
            println!("Relevant indices sizes: residual {}, sketch {}, near field {}, far field {}", ind_r.len(), ind_s.len(), near_field_inds.len(), far_indices.len());
        }

        let arr_rf = <Extraction<Self::Item> as MatrixExtraction>::new(arr, ExtInsType::Cross(ind_r.clone(), far_indices.clone())).unwrap().ext;
        let arr_fr = <Extraction<Self::Item> as MatrixExtraction>::new(arr, ExtInsType::Cross(far_indices.clone(), ind_r.clone())).unwrap().ext;
        let arr_rt = <Extraction<Self::Item> as MatrixExtraction>::new(arr, ExtInsType::Cross(ind_r.clone(), ind_t.clone())).unwrap().ext;
        let arr_tr = <Extraction<Self::Item> as MatrixExtraction>::new(arr, ExtInsType::Cross(ind_t.clone(), ind_r.clone())).unwrap().ext;

        let arr_rf = arr_rf.norm_1();
        let arr_fr = arr_fr.norm_1();
        let arr_rt = arr_rt.norm_1();
        let arr_tr = arr_tr.norm_1();

        (arr_rf, arr_fr, arr_rt, arr_tr)
    }

    fn get_boxes_errors(&self, target_arr: &mut DynamicArray<Self::Item, 2>, silent: bool)->VecErrors<<Self::Item as RlstScalar>::Real>
    {
        let mut rel_errs: Vec<Vec<<T as RlstScalar>::Real>> = Vec::new();
        let mut abs_errs: Vec<Vec<<T as RlstScalar>::Real>> = Vec::new();
        rel_errs.resize(4, Vec::new());
        abs_errs.resize(4, Vec::new());

        let factor_options = FactorOptions{ inv: true, trans: false };

        for (box_num, dec_factors) in self.dec_factors.iter().enumerate(){
            let (arr_rf, arr_fr, arr_rt, arr_tr) = self.box_errors(box_num, target_arr, &dec_factors.near_field_inds, silent);            
            self.box_dec(target_arr, dec_factors, &factor_options);
            let (arr_rf_ae, arr_fr_ae, arr_rt_ae, arr_tr_ae) = self.box_errors(box_num, target_arr, &dec_factors.near_field_inds, silent);

            rel_errs[0].push(arr_rf_ae/arr_rf);
            rel_errs[1].push(arr_fr_ae/arr_fr);
            rel_errs[2].push(arr_rt_ae/arr_rt);
            rel_errs[3].push(arr_tr_ae/arr_tr);

            abs_errs[0].push(arr_rf_ae);
            abs_errs[1].push(arr_fr_ae);
            abs_errs[2].push(arr_rt_ae);
            abs_errs[3].push(arr_tr_ae);

            if !silent{
                println!("Absolute errors: {}, {}, {}, {}", arr_rf_ae, arr_fr_ae, arr_rt_ae, arr_tr_ae);
                println!("Relative errors: {}, {}, {}, {}\n\n", arr_rf_ae/arr_rf, arr_fr_ae/arr_fr, arr_rt_ae/arr_rt, arr_tr_ae/arr_tr);
            }
        }

        (rel_errs, abs_errs)
    }
}