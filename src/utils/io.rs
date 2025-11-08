use hdf5::File;
use num_complex::Complex;
pub use rlst::prelude::*;
//use std::any::TypeId;
use std::path::Path;

pub fn resize_rows<
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Stride<2> + RawAccessMut<Item = Item> + Shape<2>,
>(
    arr: &Array<Item, ArrayImpl, 2>,
    new_shape: [usize; 2],
) -> DynamicArray<Item, 2> {
    let mut new_arr = rlst_dynamic_array2!(Item, new_shape);
    new_arr
        .r_mut()
        .into_subview([0, 0], arr.shape())
        .fill_from(arr.r());

    new_arr
}

pub trait IOData<T: RlstScalar> {
    type Item: RlstScalar;
    fn load(path: &str) -> hdf5::Result<Vec<Self::Item>>;
    //fn save(data: &[Self::Item], shape: [usize; 2], path: &str) -> hdf5::Result<()>;
    fn append<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = T> + Stride<2> + RawAccessMut<Item = T> + Shape<2>,
    >(
        data: &Array<T, ArrayImpl, 2>,
        path: &str,
    ) -> hdf5::Result<()>;
}

macro_rules! implement_io_data_real {
    ($scalar:ty) => {
        impl IOData<$scalar> for $scalar {
            type Item = $scalar;
            fn load(path: &str) -> hdf5::Result<Vec<Self::Item>> {
                let file = File::open(path)?;
                let ds = file.dataset("real")?;
                let array = ds.read()?;
                let data: Vec<$scalar> = array.to_vec();
                Ok(data)
            }

            fn append<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + RawAccessMut<Item = $scalar>
                    + Shape<2>,
            >(
                extra_arr: &Array<$scalar, ArrayImpl, 2>,
                path: &str,
            ) -> hdf5::Result<()> {
                let shape = extra_arr.shape();
                let data = extra_arr.data();
                println!(
                    "[append_real_array] incoming block shape = {:?}, flat len = {}",
                    shape,
                    data.len()
                );

                if Path::new(path).exists() {
                    println!("[append_real_array] appending to existing file: {path}");

                    let file = File::open_rw(path)?;
                    let ds = file.dataset("real")?;
                    let old = ds.read_raw()?;
                    drop(ds);

                    // Read old shape attribute
                    let ncols = shape[1];
                    let old_rows = old.len() / ncols;

                    println!(
                        "[append_real_array] old shape = [{}, {}], old flat len = {}",
                        old_rows,
                        ncols,
                        old.len()
                    );

                    // Build old array
                    let mut arr = rlst_dynamic_array2!($scalar, [old_rows, ncols]);
                    arr.data_mut()
                        .iter_mut()
                        .zip(old.iter())
                        .for_each(|(dst, src)| *dst = *src);

                    // Resize
                    let new_rows = old_rows + shape[0];
                    let new_shape = [new_rows, ncols];
                    arr = resize_rows(&arr, new_shape);
                    println!(
                        "[append_real_array] resized to {:?}, flat len now {}",
                        new_shape,
                        arr.data().len()
                    );

                    // Append new block (column-major order)
                    arr.r_mut()
                        .into_subview([old_rows, 0], extra_arr.shape())
                        .fill_from(extra_arr.r());

                    // Write back to file
                    let raw_data = arr.data();
                    println!(
                        "[append_real_array] writing back dataset, total flat len = {}",
                        raw_data.len()
                    );

                    file.unlink("real")?;
                    file.new_dataset::<$scalar>()
                        .shape((raw_data.len(),))
                        .create("real")?
                        .write(raw_data)?;
                    file.unlink("shape")?;

                    println!(
                        "[append_real_array] updated shape attr = {:?}, done.\n",
                        new_shape
                    );
                } else {
                    let file = File::create(path)?;
                    file.new_dataset::<$scalar>()
                        .shape((data.len(),))
                        .create("real")?
                        .write(data)?;
                }
                Ok(())
            }
        }
    };
}

macro_rules! implement_io_data_complex {
    ($scalar:ty) => {
        impl IOData<Complex<$scalar>> for Complex<$scalar> {
            type Item = Complex<$scalar>;
            fn load(path: &str) -> hdf5::Result<Vec<Self::Item>> {
                let file = File::open(path)?;
                let re_array = file.dataset("real")?.read()?;
                let im_array = file.dataset("imag")?.read()?;
                let re: Vec<$scalar> = re_array.to_vec();
                let im: Vec<$scalar> = im_array.to_vec();

                if re.len() != im.len() {
                    return Err(hdf5::Error::Internal("mismatched real/imag lengths".into()));
                }

                let data = re
                    .into_iter()
                    .zip(im.into_iter())
                    .map(|(r, i)| Complex::new(r, i))
                    .collect();

                Ok(data)
            }

            fn append<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = Complex<$scalar>>
                    + Stride<2>
                    + RawAccessMut<Item = Complex<$scalar>>
                    + Shape<2>,
            >(
                extra_arr: &Array<Complex<$scalar>, ArrayImpl, 2>,
                path: &str,
            ) -> hdf5::Result<()> {
                let shape = extra_arr.shape();
                let data = extra_arr.data();
                println!(
                    "[append_complex_array] incoming block shape = {:?}, flat len = {}",
                    shape,
                    data.len()
                );

                if Path::new(path).exists() {
                    println!("[append_complex_array] appending to existing file: {path}");

                    let file = File::open_rw(path)?;
                    let ds_re = file.dataset("real")?;
                    let ds_im = file.dataset("imag")?;
                    let old_re = ds_re.read_raw()?;
                    let old_im = ds_im.read_raw()?;
                    drop(ds_re);
                    drop(ds_im);

                    // Compute old shape from ncols
                    let ncols = shape[1];
                    let old_rows = old_re.len() / ncols;
                    println!(
                        "[append_complex_array] old shape = [{}, {}], old flat len = {}",
                        old_rows,
                        ncols,
                        old_re.len()
                    );

                    // Build old real/imag arrays
                    let mut re_data = rlst_dynamic_array2!($scalar, [old_rows, ncols]);
                    let mut im_data = rlst_dynamic_array2!($scalar, [old_rows, ncols]);
                    re_data
                        .data_mut()
                        .iter_mut()
                        .zip(old_re.iter())
                        .for_each(|(dst, src)| *dst = *src);
                    im_data
                        .data_mut()
                        .iter_mut()
                        .zip(old_im.iter())
                        .for_each(|(dst, src)| *dst = *src);

                    // Resize
                    let new_rows = old_rows + shape[0];
                    let new_shape = [new_rows, ncols];
                    re_data = resize_rows(&re_data, new_shape);
                    im_data = resize_rows(&im_data, new_shape);
                    println!(
                        "[append_complex_array] resized to {:?}, flat len now {}",
                        new_shape,
                        re_data.data().len()
                    );

                    // Prepare blocks for incoming real/imag parts (same shape as extra_arr)
                    let mut re_block = rlst_dynamic_array2!($scalar, [shape[0], ncols]);
                    let mut im_block = rlst_dynamic_array2!($scalar, [shape[0], ncols]);
                    re_block
                        .data_mut()
                        .iter_mut()
                        .zip(data.iter())
                        .for_each(|(dst, src)| *dst = src.re().clone());
                    im_block
                        .data_mut()
                        .iter_mut()
                        .zip(data.iter())
                        .for_each(|(dst, src)| *dst = src.im().clone());

                    // Insert the incoming blocks at row offset old_rows (column-major safe)
                    re_data
                        .r_mut()
                        .into_subview([old_rows, 0], [shape[0], ncols])
                        .fill_from(re_block.r());
                    im_data
                        .r_mut()
                        .into_subview([old_rows, 0], [shape[0], ncols])
                        .fill_from(im_block.r());

                    // Write back to file (recreate file datasets)
                    println!(
                        "[append_complex_array] writing back dataset, total flat len = {}",
                        re_data.data().len()
                    );

                    file.unlink("real")?;
                    file.unlink("imag")?;

                    let re = re_data.data().to_vec();
                    let im = im_data.data().to_vec();

                    file.new_dataset::<$scalar>()
                        .shape((re.len(),))
                        .create("real")?
                        .write(&re)?;
                    file.new_dataset::<$scalar>()
                        .shape((im.len(),))
                        .create("imag")?
                        .write(&im)?;
                    file.new_attr::<[u64; 2]>()
                        .create("shape")?
                        .write(&new_shape.map(|x| x as u64))?;

                    println!(
                        "[append_complex_array] updated shape attr = {:?}, done.\n",
                        new_shape
                    );
                } else {
                    println!("[append_complex_array] creating new file: {path}");
                    let file = File::create(path)?;
                    let re: Vec<$scalar> = data.iter().map(|c| c.re().clone()).collect();
                    let im: Vec<$scalar> = data.iter().map(|c| c.im().clone()).collect();

                    file.new_dataset::<$scalar>()
                        .shape((re.len(),))
                        .create("real")?
                        .write(&re)?;
                    file.new_dataset::<$scalar>()
                        .shape((im.len(),))
                        .create("imag")?
                        .write(&im)?;
                    file.new_attr::<[u64; 2]>()
                        .create("shape")?
                        .write(&(shape.map(|x| x as u64)))?;
                }

                Ok(())
            }
        }
    };
}

implement_io_data_real!(f64);
implement_io_data_real!(f32);
implement_io_data_complex!(f64);
implement_io_data_complex!(f32);
