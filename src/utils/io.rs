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
// ==========================
// Internal helper functions
// ==========================

/*fn save_real_array<T>(data: &[T], shape: [usize; 2], path: &str) -> hdf5::Result<()>
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

fn append_real_array<
    T,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = <T as rlst::RlstScalar>::Real>
        + Stride<2>
        + RawAccessMut<Item = <T as rlst::RlstScalar>::Real>
        + Shape<2>,
>(
    extra_arr: &Array<<T as rlst::RlstScalar>::Real, ArrayImpl, 2>,
    path: &str,
) -> hdf5::Result<()>
where
    T: RlstScalar,
    <T as rlst::RlstScalar>::Real: H5Type,
{
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
        let old: Vec<<T as rlst::RlstScalar>::Real> =
            ds.read_raw::<<T as rlst::RlstScalar>::Real>()?;
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
        let mut arr = rlst_dynamic_array2!(<T as rlst::RlstScalar>::Real, [old_rows, ncols]);
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
        file.new_dataset::<<T as rlst::RlstScalar>::Real>()
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
        let file = File::create(path)?;
        file.new_dataset::<<T as rlst::RlstScalar>::Real>()
            .shape((data.len(),))
            .create("real")?
            .write(data)?;
        file.new_attr::<[u64; 2]>()
            .create("shape")?
            .write(&(shape.map(|x| x as u64)))?;
    }
    Ok(())
}

fn append_complex_array<
    T,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = T> + Stride<2> + RawAccessMut<Item = T> + Shape<2>,
>(
    extra_arr: &Array<T, ArrayImpl, 2>,
    path: &str,
) -> hdf5::Result<()>
where
    T: RlstScalar,
    <T as rlst::RlstScalar>::Real: H5Type + RlstScalar,
{
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
        let old_re: Vec<<T as rlst::RlstScalar>::Real> =
            ds_re.read_raw::<<T as rlst::RlstScalar>::Real>()?;
        let old_im: Vec<<T as rlst::RlstScalar>::Real> =
            ds_im.read_raw::<<T as rlst::RlstScalar>::Real>()?;
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
        let mut re_data = rlst_dynamic_array2!(<T as rlst::RlstScalar>::Real, [old_rows, ncols]);
        let mut im_data = rlst_dynamic_array2!(<T as rlst::RlstScalar>::Real, [old_rows, ncols]);
        re_data
            .data_mut()
            .iter_mut()
            .zip(old_re.iter())
            .for_each(|(dst, src)| *dst = src.clone());
        im_data
            .data_mut()
            .iter_mut()
            .zip(old_im.iter())
            .for_each(|(dst, src)| *dst = src.clone());

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
        let mut re_block = rlst_dynamic_array2!(<T as rlst::RlstScalar>::Real, [shape[0], ncols]);
        let mut im_block = rlst_dynamic_array2!(<T as rlst::RlstScalar>::Real, [shape[0], ncols]);
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

        let file = File::create(path)?;
        let re = re_data.data().to_vec();
        let im = im_data.data().to_vec();

        file.new_dataset::<<T as rlst::RlstScalar>::Real>()
            .shape((re.len(),))
            .create("real")?
            .write(&re)?;
        file.new_dataset::<<T as rlst::RlstScalar>::Real>()
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
        let re: Vec<<T as rlst::RlstScalar>::Real> = data.iter().map(|c| c.re().clone()).collect();
        let im: Vec<<T as rlst::RlstScalar>::Real> = data.iter().map(|c| c.im().clone()).collect();

        file.new_dataset::<<T as rlst::RlstScalar>::Real>()
            .shape((re.len(),))
            .create("real")?
            .write(&re)?;
        file.new_dataset::<<T as rlst::RlstScalar>::Real>()
            .shape((im.len(),))
            .create("imag")?
            .write(&im)?;
        file.new_attr::<[u64; 2]>()
            .create("shape")?
            .write(&(shape.map(|x| x as u64)))?;
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
}*/

// ===========================================
// Public generic append/load entrypoints
// ===========================================
/*
pub fn append_array<
    T: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = T> + Stride<2> + RawAccessMut<Item = T> + Shape<2>,
>(
    data: &Array<T, ArrayImpl, 2>,
    path: &str,
) -> hdf5::Result<()>
where
    <T as rlst::RlstScalar>::Real: H5Type,
{
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        //let d = unsafe { std::mem::transmute::<&[T], &[f64]>(data) };
        append_real_array(data, path)
    } else if TypeId::of::<T>() == TypeId::of::<Complex<f64>>() {
        //let d = unsafe { std::mem::transmute::<&[T], &[Complex<f64>]>(data) };
        append_complex_array(data, path)
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
}*/

// ============================
// IOData trait implementation
// ============================

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

/*impl<T: RlstScalar> IOData for T
where
    <T as rlst::RlstScalar>::Real: H5Type,
{
    type Item = Self;

    fn load(path: &str) -> hdf5::Result<Vec<Self::Item>> {
        load_array::<T>(path)
    }

    fn append<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + RawAccessMut<Item = Self::Item>
            + Shape<2>,
    >(
        data: &Array<Self::Item, ArrayImpl, 2>,
        path: &str,
    ) -> hdf5::Result<()> {
        append_array(data, path)
    }
}*/

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

                    let file = File::create(path)?;
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
