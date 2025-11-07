use hdf5::{File, H5Type};
use num_complex::Complex;
pub use rlst::prelude::*;
use std::any::TypeId;
use std::path::Path;

// ==========================
// Internal helper functions
// ==========================

fn save_real_array<T>(data: &[T], shape: [usize; 2], path: &str) -> hdf5::Result<()>
where
    T: H5Type + Clone + 'static,
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
    T: H5Type + Clone + 'static,
{
    if Path::new(path).exists() {
        let file = File::open_rw(path)?;
        let ds = file.dataset("real")?;
        let mut old: Vec<T> = ds.read_raw::<T>()?;
        drop(ds);

        old.extend_from_slice(data);

        let old_shape_attr = file.attr("shape")?.read_scalar::<[u64; 2]>()?;
        let old_rows = old_shape_attr[0] as usize;
        let ncols = old_shape_attr[1] as usize;
        let new_rows = old_rows + shape[0];
        let new_shape = [new_rows as u64, ncols as u64];

        file.unlink("real")?;
        file.new_dataset::<T>()
            .shape((old.len(),))
            .create("real")?
            .write(&old)?;
        file.unlink("shape")?;
        file.new_attr::<[u64; 2]>()
            .create("shape")?
            .write(&new_shape)?;
    } else {
        save_real_array(data, shape, path)?;
    }
    Ok(())
}

fn append_complex_array<T>(data: &[Complex<T>], shape: [usize; 2], path: &str) -> hdf5::Result<()>
where
    T: H5Type + Clone + 'static,
{
    if Path::new(path).exists() {
        let file = File::open_rw(path)?;
        let ds_re = file.dataset("real")?;
        let ds_im = file.dataset("imag")?;
        let mut re: Vec<T> = ds_re.read_raw::<T>()?;
        let mut im: Vec<T> = ds_im.read_raw::<T>()?;
        drop(ds_re);
        drop(ds_im);

        re.extend(data.iter().map(|c| c.re.clone()));
        im.extend(data.iter().map(|c| c.im.clone()));

        let old_shape_attr = file.attr("shape")?.read_scalar::<[u64; 2]>()?;
        let old_rows = old_shape_attr[0] as usize;
        let ncols = old_shape_attr[1] as usize;
        let new_rows = old_rows + shape[0];
        let new_shape = [new_rows as u64, ncols as u64];

        file.unlink("real")?;
        file.unlink("imag")?;
        file.new_dataset::<T>()
            .shape((re.len(),))
            .create("real")?
            .write(&re)?;
        file.new_dataset::<T>()
            .shape((im.len(),))
            .create("imag")?
            .write(&im)?;
        file.unlink("shape")?;
        file.new_attr::<[u64; 2]>()
            .create("shape")?
            .write(&new_shape)?;
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
