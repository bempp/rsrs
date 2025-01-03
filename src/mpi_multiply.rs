//use core::slice::SlicePattern;

//use mpi::traits::{CommunicatorCollectives, Root};
use rlst::dense::gemm::Gemm;
use rlst::dense::traits::{RawAccess, RawAccessMut, Shape, Stride};
use rlst::dense::types::RlstScalar;
use rlst::dense::types::TransMode;
use mpi::datatype::DynBufferMut;
use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence, FromRaw, Group, Root, UncommittedDatatype};
use rlst::rlst_dynamic_array2;


 
pub fn matrix_multiply<
    Item: RlstScalar + Gemm + mpi::datatype::BufferMut,
    MatA: RawAccess<Item = Item> + Shape<2> + Stride<2>,
    MatB: RawAccess<Item = Item> + Shape<2> + Stride<2>,
    MatC: RawAccessMut<Item = Item> + Shape<2> + Stride<2>,
    C: CommunicatorCollectives
>
(
    transa: TransMode,
    transb: TransMode,
    alpha: Item,
    mat_a: &MatA,
    mat_b: &MatB,
    beta: Item,
    mat_c: &mut MatC,
    comm: &C
) where [Item]: mpi::datatype::Buffer, [Item]: mpi::datatype::BufferMut, std::vec::Vec<Item>: mpi::datatype::BufferMut{
    let m = mat_c.shape()[0];
    let n = mat_c.shape()[1];

    let a_shape = match transa {
        TransMode::NoTrans => mat_a.shape(),
        TransMode::ConjNoTrans => mat_a.shape(),
        TransMode::Trans => [mat_a.shape()[1], mat_a.shape()[0]],
        TransMode::ConjTrans => [mat_a.shape()[1], mat_a.shape()[0]],
    };

    let b_shape = match transb {
        TransMode::NoTrans => mat_b.shape(),
        TransMode::ConjNoTrans => mat_b.shape(),
        TransMode::Trans => [mat_b.shape()[1], mat_b.shape()[0]],
        TransMode::ConjTrans => [mat_b.shape()[1], mat_b.shape()[0]],
    };

    assert_eq!(m, a_shape[0], "Wrong dimension. {} != {}", m, a_shape[0]);
    assert_eq!(n, b_shape[1], "Wrong dimension. {} != {}", n, b_shape[1]);
    assert_eq!(
        a_shape[1], b_shape[0],
        "Wrong dimension. {} != {}",
        a_shape[1], b_shape[0]
    );


    let size: i32 = comm.size();
    let rank: i32 = comm.rank();

    let mut num_cols: usize = (mat_b.shape()[1] as f64 /size as f64) as usize;

    if rank == size - 1{
        num_cols  = mat_b.shape()[1] - (size - 1) as usize *num_cols;
    }
    

    let chunk_size: usize = num_cols*mat_b.shape()[0];

    //println!("rank, num_cols, chunk size, {}, {},  {}, {:?}", rank, num_cols, chunk_size, mat_c.shape());

    let mut chunk: Vec<Item>  = vec![<Item as num::Zero>::zero(); chunk_size];

    //println!("{:?}", chunk);
    
    if rank == 0 {
        //println!("rank0, {}", chunk_size);
        comm.process_at_rank(0).scatter_into_root(mat_b.data(), chunk.as_mut_slice());
    } else {
        //println!("other, {}", chunk_size);
        comm.process_at_rank(0).scatter_into(chunk.as_mut_slice());
    }

    //println!("scattered");
    let mut buffer = mat_a.data().to_vec();
    comm.process_at_rank(0).broadcast_into(&mut buffer);

    //println!("broadcasted");
    let mut partial_res: Vec<Item>  = vec![<Item as num::Zero>::zero(); chunk_size];
    
    let [rsa, csa] = mat_a.stride();
    let [rsb, csb] = mat_b.stride();
    let [rsc, csc] = mat_c.stride();

    let m = mat_b.shape()[0];
    let n = num_cols;

    <Item as Gemm>::gemm(
        transa,
        transb,
        m,
        n,
        a_shape[1],
        alpha,
        &buffer,
        rsa,
        csa,
        &chunk,
        rsb,
        csb,
        beta,
        &mut partial_res,
        rsc,
        csc,
    );


    /*<Item as Gemm>::gemm(
        transa,
        transb,
        m,
        n,
        a_shape[1],
        alpha,
        mat_a.data(),
        rsa,
        csa,
        mat_b.data(),
        rsb,
        csb,
        beta,
        mat_c.data_mut(),
        rsc,
        csc,
    );*/

    if comm.rank() == 0 {
        let mut target_buffer = vec![0f64; mat_c.data().len()];
        comm.process_at_rank(0).gather_into_root(&partial_res[..], &mut target_buffer[..]);
        //println!("{}, {:?}", comm.rank(), target_buffer);
    } else {
        comm.process_at_rank(0).gather_into(&partial_res[..]);
        //println!("{}, {:?}", comm.rank(), partial_res);
    }

    

}

/* 

fn mpi_testing<T:RlstScalar + RandScalar, C: CommunicatorCollectives> (num_samples: usize, arr: &Array<T, ArrayImpl<T>, 2>, sketch: &mut Array<T, ArrayImpl<T>, 2>, test: &mut Array<T, ArrayImpl<T>, 2>, trans: bool, comm: &C)
where StandardNormal: Distribution<T::Real>, Standard: Distribution<T::Real>, Vec<T>: mpi::datatype::BufferMut
{
    let test_shape: [usize; 2] = [arr.shape()[1], num_samples];
    let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
    test.resize_in_place(test_shape);
    test.fill_from_standard_normal(&mut rng); 
 
    let size: i32 = comm.size();
    let rank: i32 = comm.rank();
    
    comm.barrier();
    let mut chunk_size: usize = (test_shape[1] as f64 /size as f64) as usize;
    let mut start = chunk_size*(rank as usize);

    println!("{}, {}", rank, size);
    if rank == size - 1{
        start = (size - 1) as usize *chunk_size;
        chunk_size  = test_shape[1] - (size - 1) as usize *chunk_size;
    }

    if !trans{
        sketch.view_mut().into_subview([0, start], [test_shape[0], chunk_size]).simple_mult_into(arr.view(), test.view().into_subview([0, start], [test_shape[0], chunk_size]));
    }else{
        let mut aux: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2> = empty_array();
        aux.fill_from_resize(arr.view().conj().transpose());
        sketch.view_mut().into_subview([0, start], [test_shape[0], chunk_size]).simple_mult_into(aux.view(), test.view().into_subview([0, start], [test_shape[0], chunk_size]));
    }

}
*/