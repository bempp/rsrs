use hdf5::{File, H5Type};
pub use rlst::prelude::*;
use std::any::TypeId;

fn save_complex_array(data: &[num::Complex<f64>], path: &str) -> hdf5::Result<()> {
    let file = File::create(path)?;

    // Split into real and imaginary parts
    let re: Vec<f64> = data.iter().map(|c| c.re).collect();
    let im: Vec<f64> = data.iter().map(|c| c.im).collect();

    // Save as two separate datasets
    file.new_dataset::<f64>().create("real")?.write(&re)?;
    file.new_dataset::<f64>().create("imag")?.write(&im)?;

    Ok(())
}

fn save_real_array<T>(data: &[T], path: &str) -> hdf5::Result<()>
where
    T: H5Type + Clone + 'static,
{
    let file = File::create(path)?;
    let ds = file.new_dataset::<T>().create("real")?;
    ds.write(data)?;
    Ok(())
}

pub fn save_array<T: RlstScalar>(data: &[T], path: &str) -> hdf5::Result<()> {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let d = unsafe { std::mem::transmute::<&[T], &[f64]>(data) };
        save_real_array(d, path)
    } else if TypeId::of::<T>() == TypeId::of::<num_complex::Complex<f64>>() {
        let d = unsafe { std::mem::transmute::<&[T], &[num_complex::Complex<f64>]>(data) };
        save_complex_array(d, path)
    } else {
        Err(hdf5::Error::Internal("unsupported scalar type".into()))
    }
}

fn load_real_array<T>(path: &str) -> hdf5::Result<Vec<T>>
where
    T: hdf5::H5Type + Clone + 'static,
{
    let file = File::open(path)?;
    let ds = file.dataset("real")?;
    let array = ds.read()?;
    let data: Vec<T> = array.to_vec();
    Ok(data)
}

fn load_complex_array(path: &str) -> hdf5::Result<Vec<num::Complex<f64>>> {
    let file = File::open(path)?;
    let re_array = file.dataset("real")?.read()?;
    let re: Vec<f64> = re_array.to_vec();
    let im_array = file.dataset("imag")?.read()?;
    let im: Vec<f64> = im_array.to_vec();
    if re.len() != im.len() {
        return Err(hdf5::Error::Internal("mismatched real/imag lengths".into()));
    }
    let data = re
        .into_iter()
        .zip(im.into_iter())
        .map(|(r, i)| num::Complex::new(r, i))
        .collect();
    Ok(data)
}

pub fn load_array<T: RlstScalar>(path: &str) -> hdf5::Result<Vec<T>> {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let data = load_real_array::<f64>(path)?;
        // SAFETY: T = f64
        let d = unsafe { std::mem::transmute::<Vec<f64>, Vec<T>>(data) };
        Ok(d)
    } else if TypeId::of::<T>() == TypeId::of::<num_complex::Complex<f64>>() {
        let data = load_complex_array(path)?;
        // SAFETY: T = num_complex::Complex<f64>
        let d = unsafe { std::mem::transmute::<Vec<num::Complex<f64>>, Vec<T>>(data) };
        Ok(d)
    } else {
        Err(hdf5::Error::Internal("unsupported scalar type".into()))
    }
}

pub fn append_real_array<T>(data: &[T], path: &str) -> hdf5::Result<()>
where
    T: H5Type + Clone + 'static,
{
    if std::path::Path::new(path).exists() {
        let file = File::open_rw(path)?;
        let mut old_data: Vec<T> = file.dataset("real")?.read()?.to_vec();
        old_data.extend_from_slice(data);
        file.unlink("real")?;

        file.new_dataset::<T>()
            .shape(old_data.len())
            .create("real")?
            .write(&old_data)?;
    } else {
        let file = File::create(path)?;
        file.new_dataset::<T>()
            .shape(data.len())
            .create("real")?
            .write(data)?;
    }
    Ok(())
}

pub fn append_complex_array<T>(data: &[num::Complex<T>], path: &str) -> hdf5::Result<()>
where
    T: H5Type + Clone + 'static,
{
    if std::path::Path::new(path).exists() {
        let file = File::open_rw(path)?;

        let mut re: Vec<T> = file.dataset("real")?.read()?.to_vec();
        let mut im: Vec<T> = file.dataset("imag")?.read()?.to_vec();

        re.extend(data.iter().map(|c| c.re.clone()));
        im.extend(data.iter().map(|c| c.im.clone()));

        file.unlink("real")?;
        file.unlink("imag")?;

        file.new_dataset::<T>()
            .shape(re.len())
            .create("real")?
            .write(&re)?;

        file.new_dataset::<T>()
            .shape(im.len())
            .create("imag")?
            .write(&im)?;
    } else {
        let file = File::create(path)?;
        let re: Vec<T> = data.iter().map(|c| c.re.clone()).collect();
        let im: Vec<T> = data.iter().map(|c| c.im.clone()).collect();

        file.new_dataset::<T>()
            .shape(data.len())
            .create("real")?
            .write(&re)?;

        file.new_dataset::<T>()
            .shape(data.len())
            .create("imag")?
            .write(&im)?;
    }
    Ok(())
}

pub trait IOData: Sized {
    /// Item type
    type Item: RlstScalar;
    fn load(path: &str) -> hdf5::Result<Vec<Self::Item>>;
    fn save(data: &[Self::Item], path: &str) -> hdf5::Result<()>;
    fn append(data: &[Self::Item], path: &str) -> hdf5::Result<()>;
}

impl<T: RlstScalar> IOData for T {
    type Item = Self;

    fn load(path: &str) -> hdf5::Result<Vec<Self::Item>> {
        if TypeId::of::<Self::Item>() == TypeId::of::<f64>() {
            let data = load_real_array::<f64>(path)?;
            // SAFETY: T = f64
            let d = unsafe { std::mem::transmute::<Vec<f64>, Vec<Self::Item>>(data) };
            Ok(d)
        } else if TypeId::of::<Self::Item>() == TypeId::of::<num_complex::Complex<f64>>() {
            let data = load_complex_array(path)?;
            // SAFETY: T = num_complex::Complex<f64>
            let d = unsafe { std::mem::transmute::<Vec<num::Complex<f64>>, Vec<T>>(data) };
            Ok(d)
        } else {
            Err(hdf5::Error::Internal("unsupported scalar type".into()))
        }
    }

    fn save(data: &[Self::Item], path: &str) -> hdf5::Result<()> {
        if TypeId::of::<Self::Item>() == TypeId::of::<f64>() {
            let d = unsafe { std::mem::transmute::<&[Self::Item], &[f64]>(data) };
            save_real_array(d, path)
        } else if TypeId::of::<Self::Item>() == TypeId::of::<num_complex::Complex<f64>>() {
            let d =
                unsafe { std::mem::transmute::<&[Self::Item], &[num_complex::Complex<f64>]>(data) };
            save_complex_array(d, path)
        } else {
            Err(hdf5::Error::Internal("unsupported scalar type".into()))
        }
    }

    fn append(data: &[Self::Item], path: &str) -> hdf5::Result<()> {
        if TypeId::of::<Self::Item>() == TypeId::of::<f64>() {
            let d = unsafe { std::mem::transmute::<&[Self::Item], &[f64]>(data) };
            append_real_array(d, path)
        } else if TypeId::of::<Self::Item>() == TypeId::of::<num_complex::Complex<f64>>() {
            let d =
                unsafe { std::mem::transmute::<&[Self::Item], &[num_complex::Complex<f64>]>(data) };
            append_complex_array(d, path)
        } else {
            Err(hdf5::Error::Internal("unsupported scalar type".into()))
        }
    }
}
