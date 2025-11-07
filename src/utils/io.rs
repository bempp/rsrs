use hdf5::{File, H5Type};
use num_complex::Complex;
pub use rlst::prelude::*;
use std::any::TypeId;
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
// ==========================
// Internal helper functions
// ==========================

fn save_real_array<T>(data: &[T], shape: [usize; 2], path: &str) -> hdf5::Result<()>
where
    T: H5Type + RlstScalar,
{
    let file = File::create(path)?;
    file.new_dataset::<T>()
        .shape((data.len(),))
        .create("real")?
        .write(data)?;
    file.new_attr::<[u64; 2]>()
        .create("shape")?
        .write(&(shape.map(|x| x as u64)))?;
    Ok(())
}

fn save_complex_array<T>(data: &[Complex<T>], shape: [usize; 2], path: &str) -> hdf5::Result<()>
where
    T: H5Type + Clone + 'static,
{
    let file = File::create(path)?;
    let re: Vec<T> = data.iter().map(|c| c.re.clone()).collect();
    let im: Vec<T> = data.iter().map(|c| c.im.clone()).collect();

    file.new_dataset::<T>()
        .shape((data.len(),))
        .create("real")?
        .write(&re)?;
    file.new_dataset::<T>()
        .shape((data.len(),))
        .create("imag")?
        .write(&im)?;
    file.new_attr::<[u64; 2]>()
        .create("shape")?
        .write(&(shape.map(|x| x as u64)))?;
    Ok(())
}

fn append_real_array<T>(data: &[T], shape: [usize; 2], path: &str) -> hdf5::Result<()>
where
    T: H5Type + RlstScalar,
{
    println!(
        "[append_real_array] incoming block shape = {:?}, flat len = {}",
        shape,
        data.len()
    );

    if Path::new(path).exists() {
        println!("[append_real_array] appending to existing file: {path}");

        let file = File::open_rw(path)?;
        let ds = file.dataset("real")?;
        let old: Vec<T> = ds.read_raw::<T>()?;
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
        let mut arr = rlst_dynamic_array2!(T, [old_rows, ncols]);
        arr.data_mut()
            .iter_mut()
            .zip(old.iter())
            .for_each(|(dst, src)| *dst = src.clone());

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
        let row_offset = old_rows;
        for (k, val) in data.iter().enumerate() {
            let row = k % shape[0];
            let col = k / shape[0];
            let dst_row = row_offset + row;
            *arr.r_mut().get_mut([dst_row, col]).unwrap() = val.clone();
        }

        // Write back to file
        let raw_data = arr.data();
        println!(
            "[append_real_array] writing back dataset, total flat len = {}",
            raw_data.len()
        );

        file.unlink("real")?;
        file.new_dataset::<T>()
            .shape((raw_data.len(),))
            .create("real")?
            .write(raw_data)?;
        file.unlink("shape")?;
        file.new_attr::<[u64; 2]>()
            .create("shape")?
            .write(&new_shape.map(|x| x as u64))?;

        println!(
            "[append_real_array] updated shape attr = {:?}, done.\n",
            new_shape
        );
    } else {
        save_real_array(data, shape, path)?;
    }
    Ok(())
}

fn append_complex_array<T>(
    data: &[num::Complex<T>],
    shape: [usize; 2],
    path: &str,
) -> hdf5::Result<()>
where
    T: H5Type + RlstScalar,
{
    if Path::new(path).exists() {
        let file = File::open_rw(path)?;

        // Read old data
        let ds_re = file.dataset("real")?;
        let ds_im = file.dataset("imag")?;
        let old_re: Vec<T> = ds_re.read_raw::<T>()?;
        let old_im: Vec<T> = ds_im.read_raw::<T>()?;
        drop(ds_re);
        drop(ds_im);

        // Read shape
        //let old_shape_attr = file.attr("shape")?.read_scalar::<[usize; 2]>()?;
        //let old_rows = old_shape_attr[0];
        //let ncols = old_shape_attr[1];

        let ncols = shape[1];
        let old_rows = old_re.len() / ncols;

        // Build old real and imag arrays
        let mut re_data = rlst_dynamic_array2!(T, [old_rows, ncols]);
        let mut im_data = rlst_dynamic_array2!(T, [old_rows, ncols]);
        for (i, val) in old_re.iter().enumerate() {
            re_data.data_mut()[i] = val.clone();
        }
        for (i, val) in old_im.iter().enumerate() {
            im_data.data_mut()[i] = val.clone();
        }

        // Compute new shape and resize
        let new_rows = old_rows + shape[0];
        let new_shape = [new_rows, ncols];
        re_data = resize_rows(&re_data, new_shape);
        im_data = resize_rows(&im_data, new_shape);

        // Append new complex data column-major (RLST default)
        let row_offset = old_rows;
        for (k, c) in data.iter().enumerate() {
            let row = k % shape[0];
            let col = k / shape[0];
            let dst_row = row_offset + row;
            *re_data.r_mut().get_mut([dst_row, col]).unwrap() = c.re.clone();
            *im_data.r_mut().get_mut([dst_row, col]).unwrap() = c.im.clone();
        }

        // Write back to file
        let raw_re = re_data.data();
        let raw_im = im_data.data();

        file.unlink("real")?;
        file.unlink("imag")?;
        file.new_dataset::<T>()
            .shape((raw_re.len(),))
            .create("real")?
            .write(raw_re)?;
        file.new_dataset::<T>()
            .shape((raw_im.len(),))
            .create("imag")?
            .write(raw_im)?;
        file.unlink("shape")?;
        file.new_attr::<[u64; 2]>()
            .create("shape")?
            .write(&new_shape.map(|x| x as u64))?;
    } else {
        save_complex_array(data, shape, path)?;
    }
    Ok(())
}

fn load_real_array<T>(path: &str) -> hdf5::Result<Vec<T>>
where
    T: H5Type + Clone + 'static,
{
    let file = File::open(path)?;
    let ds = file.dataset("real")?;
    let array = ds.read()?;
    let data: Vec<T> = array.to_vec();
    Ok(data)
}

fn load_complex_array(path: &str) -> hdf5::Result<Vec<Complex<f64>>> {
    let file = File::open(path)?;
    let re_array = file.dataset("real")?.read()?;
    let im_array = file.dataset("imag")?.read()?;
    let re: Vec<f64> = re_array.to_vec();
    let im: Vec<f64> = im_array.to_vec();

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

// ===========================================
// Public generic save/append/load entrypoints
// ===========================================

pub fn save_array<T: RlstScalar>(data: &[T], shape: [usize; 2], path: &str) -> hdf5::Result<()> {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let d = unsafe { std::mem::transmute::<&[T], &[f64]>(data) };
        save_real_array(d, shape, path)
    } else if TypeId::of::<T>() == TypeId::of::<Complex<f64>>() {
        let d = unsafe { std::mem::transmute::<&[T], &[Complex<f64>]>(data) };
        save_complex_array(d, shape, path)
    } else {
        Err(hdf5::Error::Internal("unsupported scalar type".into()))
    }
}

pub fn append_array<T: RlstScalar>(data: &[T], shape: [usize; 2], path: &str) -> hdf5::Result<()> {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let d = unsafe { std::mem::transmute::<&[T], &[f64]>(data) };
        append_real_array(d, shape, path)
    } else if TypeId::of::<T>() == TypeId::of::<Complex<f64>>() {
        let d = unsafe { std::mem::transmute::<&[T], &[Complex<f64>]>(data) };
        append_complex_array(d, shape, path)
    } else {
        Err(hdf5::Error::Internal("unsupported scalar type".into()))
    }
}

pub fn load_array<T: RlstScalar>(path: &str) -> hdf5::Result<Vec<T>> {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let data = load_real_array::<f64>(path)?;
        let d = unsafe { std::mem::transmute::<Vec<f64>, Vec<T>>(data) };
        Ok(d)
    } else if TypeId::of::<T>() == TypeId::of::<Complex<f64>>() {
        let data = load_complex_array(path)?;
        let d = unsafe { std::mem::transmute::<Vec<Complex<f64>>, Vec<T>>(data) };
        Ok(d)
    } else {
        Err(hdf5::Error::Internal("unsupported scalar type".into()))
    }
}

// ============================
// IOData trait implementation
// ============================

pub trait IOData: Sized {
    type Item: RlstScalar;
    fn load(path: &str) -> hdf5::Result<Vec<Self::Item>>;
    fn save(data: &[Self::Item], shape: [usize; 2], path: &str) -> hdf5::Result<()>;
    fn append(data: &[Self::Item], shape: [usize; 2], path: &str) -> hdf5::Result<()>;
}

impl<T: RlstScalar> IOData for T {
    type Item = Self;

    fn load(path: &str) -> hdf5::Result<Vec<Self::Item>> {
        load_array::<T>(path)
    }

    fn save(data: &[Self::Item], shape: [usize; 2], path: &str) -> hdf5::Result<()> {
        save_array::<T>(data, shape, path)
    }

    fn append(data: &[Self::Item], shape: [usize; 2], path: &str) -> hdf5::Result<()> {
        append_array::<T>(data, shape, path)
    }
}
