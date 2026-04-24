use hdf5::File;
use num_complex::Complex;
pub use rlst::prelude::*;
use std::path::{Path, PathBuf};

// ----------------------
// Chunking / blocking parameters
// ----------------------
// BLOCK_COLS: number of columns (across N) stored per PART FILE.
const BLOCK_COLS: usize = 4096;
// CHUNK_ROWS: HDF5 internal chunk length ~ CHUNK_ROWS * block_width elements.
const CHUNK_ROWS: usize = 256;

pub const DEFAULT_SAMPLING_DIR: &str = "sampling";

pub fn preferred_sampling_dir(configured_sampling_dir: Option<&str>) -> PathBuf {
    configured_sampling_dir
        .filter(|dir| !dir.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_SAMPLING_DIR))
}

fn candidate_sampling_dirs(configured_sampling_dir: Option<&Path>) -> Vec<PathBuf> {
    let preferred = configured_sampling_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_SAMPLING_DIR));
    let legacy = PathBuf::from(DEFAULT_SAMPLING_DIR);

    if preferred == legacy {
        vec![preferred]
    } else {
        vec![preferred, legacy]
    }
}

fn ensure_sampling_dir(sampling_dir: &Path) -> hdf5::Result<()> {
    std::fs::create_dir_all(sampling_dir).map_err(|e| {
        hdf5::Error::Internal(format!(
            "failed to create '{}' directory: {e}",
            sampling_dir.display()
        ))
    })
}

// ----------------------
// Helpers
// ----------------------
fn nblocks(ncols: usize) -> usize {
    ncols.div_ceil(BLOCK_COLS)
}

fn block_width(ncols: usize, b: usize) -> usize {
    let col0 = b * BLOCK_COLS;
    (ncols - col0).min(BLOCK_COLS)
}

fn chunk_len(total_len: usize, wk: usize) -> usize {
    total_len.min((CHUNK_ROWS * wk).max(1)).max(1)
}

fn append_rows_column_major<T: Copy + Default>(
    old_flat: &[T],
    extra_flat: &[T],
    m_old: usize,
    m_add: usize,
    wk: usize,
) -> Vec<T> {
    let m_new = m_old + m_add;
    let mut merged = vec![T::default(); m_new * wk];

    for j in 0..wk {
        let old_src = &old_flat[j * m_old..(j + 1) * m_old];
        let extra_src = &extra_flat[j * m_add..(j + 1) * m_add];
        let dst_col = &mut merged[j * m_new..(j + 1) * m_new];
        let (dst_old, dst_extra) = dst_col.split_at_mut(m_old);
        dst_old.copy_from_slice(old_src);
        dst_extra.copy_from_slice(extra_src);
    }

    merged
}

fn gather_block_column_major<
    T: RlstBase,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = T> + Shape<2>,
>(
    arr: &Array<T, ArrayImpl, 2>,
    col0: usize,
    wk: usize,
) -> Vec<T> {
    let shape = arr.shape();
    let rows = shape[0];
    let mut flat = Vec::with_capacity(rows * wk);

    for j in 0..wk {
        let col = col0 + j;
        for i in 0..rows {
            flat.push(arr.get_value([i, col]).unwrap());
        }
    }

    flat
}

/// Accepts base names like:
///   "y_sketch_file"
///   "y_sketch_file.h5"
///   "y_sketch_file.00000.h5"
/// and returns the canonical base:
///   "y_sketch_file"
fn canonical_base(input: &str) -> String {
    let s = input.to_string();

    // Strip suffix ".00000.h5" if present
    if s.len() >= 9 && s.ends_with(".h5") {
        let tail = &s[s.len() - 9..]; // ".00000.h5"
        if tail.as_bytes()[0] == b'.'
            && tail.as_bytes()[6] == b'.'
            && &tail[7..] == "h5"
            && tail[1..6].chars().all(|c| c.is_ascii_digit())
        {
            return s[..s.len() - 9].to_string();
        }
    }

    // Strip plain ".h5"
    if let Some(stripped) = s.strip_suffix(".h5") {
        return stripped.to_string();
    }

    s
}

/// Part file naming: "{dir}/{base}.00000.h5"
fn part_path(sampling_dir: &Path, base: &str, b: usize) -> PathBuf {
    let b0 = canonical_base(base);
    sampling_dir.join(format!("{b0}.{:05}.h5", b))
}

/// Discover all part files in the given directory that match "{base}.00000.h5", sort by index.
fn find_part_files_in_dir(sampling_dir: &Path, base: &str) -> hdf5::Result<Vec<(usize, PathBuf)>> {
    let stem = canonical_base(base);

    if !sampling_dir.exists() {
        return Ok(Vec::new());
    }

    let mut parts: Vec<(usize, PathBuf)> = Vec::new();

    let rd = std::fs::read_dir(sampling_dir).map_err(|e| {
        hdf5::Error::Internal(format!(
            "read_dir failed for '{}': {e}",
            sampling_dir.display()
        ))
    })?;

    for entry in rd {
        let entry =
            entry.map_err(|e| hdf5::Error::Internal(format!("read_dir entry error: {e}")))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let fname = match path.file_name() {
            Some(x) => x.to_string_lossy(),
            None => continue,
        };

        let prefix = format!("{stem}.");
        if !fname.starts_with(&prefix) || !fname.ends_with(".h5") {
            continue;
        }

        let middle = &fname[prefix.len()..fname.len() - 3];
        if middle.len() != 5 || !middle.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }

        let idx: usize = middle.parse().unwrap();
        parts.push((idx, path));
    }

    parts.sort_by_key(|(i, _)| *i);

    for (expected, (idx, _)) in parts.iter().enumerate() {
        if *idx != expected {
            return Err(hdf5::Error::Internal(format!(
                "missing part file index {} (found {}) in {}",
                expected,
                idx,
                sampling_dir.display()
            )));
        }
    }

    Ok(parts)
}

type LocatedPartFiles = (PathBuf, Vec<(usize, PathBuf)>);

fn find_part_files(
    base: &str,
    sampling_dir: Option<&Path>,
) -> hdf5::Result<Option<LocatedPartFiles>> {
    for dir in candidate_sampling_dirs(sampling_dir) {
        let parts = find_part_files_in_dir(&dir, base)?;
        if !parts.is_empty() {
            return Ok(Some((dir, parts)));
        }
    }
    Ok(None)
}

pub fn resolve_sampling_dir(
    configured_sampling_dir: Option<&str>,
    bases: &[&str],
) -> hdf5::Result<Option<PathBuf>> {
    for dir in candidate_sampling_dirs(configured_sampling_dir.map(Path::new)) {
        let mut all_present = true;

        for base in bases {
            let parts = find_part_files_in_dir(&dir, base)?;
            if parts.is_empty() {
                all_present = false;
                break;
            }
        }

        if all_present {
            return Ok(Some(dir));
        }
    }

    Ok(None)
}

// Robust shape attr IO (avoids ndarray conversion issues)
fn write_shape(file: &File, shape: [usize; 2]) -> hdf5::Result<()> {
    let sh: [u64; 2] = [shape[0] as u64, shape[1] as u64];
    if file.attr("shape").is_ok() {
        file.attr("shape")?.write_raw(&sh)?;
    } else {
        file.new_attr::<u64>()
            .shape([2])
            .create("shape")?
            .write_raw(&sh)?;
    }
    Ok(())
}

fn read_shape(file: &File) -> hdf5::Result<[usize; 2]> {
    let sh: Vec<u64> = file.attr("shape")?.read_raw()?;
    if sh.len() != 2 {
        return Err(hdf5::Error::Internal(
            "shape attr must have length 2".into(),
        ));
    }
    Ok([sh[0] as usize, sh[1] as usize])
}

// Your existing helper (unchanged)
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

// ----------------------
// Trait
// ----------------------
pub trait IOData<T: RlstScalar> {
    type Item: RlstScalar;

    fn load(path: &str) -> hdf5::Result<Vec<Self::Item>> {
        Self::load_in_dir(path, None)
    }

    fn load_in_dir(path: &str, sampling_dir: Option<&Path>) -> hdf5::Result<Vec<Self::Item>>;

    fn load_into_in_dir(
        target: &mut DynamicArray<Self::Item, 2>,
        dim: usize,
        path: &str,
        sampling_dir: Option<&Path>,
    ) -> hdf5::Result<()>;

    fn append<ArrayImpl: UnsafeRandomAccessByValue<2, Item = T> + Shape<2>>(
        data: &Array<T, ArrayImpl, 2>,
        path: &str,
    ) -> hdf5::Result<()> {
        Self::append_in_dir(data, path, None)
    }

    fn append_in_dir<ArrayImpl: UnsafeRandomAccessByValue<2, Item = T> + Shape<2>>(
        data: &Array<T, ArrayImpl, 2>,
        path: &str,
        sampling_dir: Option<&Path>,
    ) -> hdf5::Result<()>;
}

// ----------------------
// Real implementation
// ----------------------
macro_rules! implement_io_data_real {
    ($scalar:ty) => {
        impl IOData<$scalar> for $scalar {
            type Item = $scalar;

            fn load_in_dir(
                path: &str,
                sampling_dir: Option<&Path>,
            ) -> hdf5::Result<Vec<Self::Item>> {
                if Path::new(path).exists() {
                    let file = File::open(path)?;
                    if file.dataset("real").is_ok() {
                        let ds = file.dataset("real")?;
                        let array = ds.read()?;
                        return Ok(array.to_vec());
                    }
                }

                let Some((_dir, parts)) = find_part_files(path, sampling_dir)? else {
                    return Err(hdf5::Error::Internal(format!(
                        "no '{}' (old format) and no multipart parts found for base '{}'",
                        path,
                        canonical_base(path)
                    )));
                };

                let file0 = File::open(&parts[0].1)?;
                let [m, ncols] = read_shape(&file0)?;

                let mut out = vec![<$scalar as Default>::default(); m * ncols];

                for (b, p) in parts.iter() {
                    let wk = block_width(ncols, *b);
                    let col0 = (*b) * BLOCK_COLS;

                    let fb = File::open(p)?;
                    let [m2, n2] = read_shape(&fb)?;
                    if m2 != m || n2 != ncols {
                        return Err(hdf5::Error::Internal(format!(
                            "shape mismatch in part '{}': got [{m2},{n2}] expected [{m},{ncols}]",
                            p.display()
                        )));
                    }

                    let ds = fb.dataset("real")?;
                    let flat: Vec<$scalar> = ds.read_raw().map_err(|e| {
                        hdf5::Error::Internal(format!(
                            "failed reading '{}::real' as {}: {e}",
                            p.display(),
                            stringify!($scalar)
                        ))
                    })?;

                    if flat.len() != m * wk {
                        return Err(hdf5::Error::Internal(format!(
                            "length mismatch in '{}::real': got {}, expected {}",
                            p.display(),
                            flat.len(),
                            m * wk
                        )));
                    }

                    for j in 0..wk {
                        let gcol = col0 + j;
                        let src = &flat[j * m..(j + 1) * m];
                        let dst = &mut out[gcol * m..(gcol + 1) * m];
                        dst.copy_from_slice(src);
                    }
                }

                Ok(out)
            }

            fn load_into_in_dir(
                target: &mut DynamicArray<Self::Item, 2>,
                dim: usize,
                path: &str,
                sampling_dir: Option<&Path>,
            ) -> hdf5::Result<()> {
                if Path::new(path).exists() {
                    let file = File::open(path)?;
                    if file.dataset("real").is_ok() {
                        let ds = file.dataset("real")?;
                        let flat: Vec<$scalar> = ds.read_raw().map_err(|e| {
                            hdf5::Error::Internal(format!(
                                "failed reading '{}::real' as {}: {e}",
                                path,
                                stringify!($scalar)
                            ))
                        })?;
                        if dim == 0 {
                            target.resize_in_place([0, 0]);
                            return Ok(());
                        }
                        if flat.len() % dim != 0 {
                            return Err(hdf5::Error::Internal(format!(
                                "length mismatch in '{}::real': got {}, not divisible by dim {}",
                                path,
                                flat.len(),
                                dim
                            )));
                        }
                        let rows = flat.len() / dim;
                        target.resize_in_place([rows, dim]);
                        target.data_mut().copy_from_slice(&flat);
                        return Ok(());
                    }
                }

                let Some((_dir, parts)) = find_part_files(path, sampling_dir)? else {
                    return Err(hdf5::Error::Internal(format!(
                        "no '{}' (old format) and no multipart parts found for base '{}'",
                        path,
                        canonical_base(path)
                    )));
                };

                let file0 = File::open(&parts[0].1)?;
                let [m, ncols] = read_shape(&file0)?;
                if dim != ncols {
                    return Err(hdf5::Error::Internal(format!(
                        "column mismatch for base '{}': stored {}, expected {}",
                        canonical_base(path),
                        ncols,
                        dim
                    )));
                }

                target.resize_in_place([m, ncols]);
                let out = target.data_mut();

                for (b, p) in parts.iter() {
                    let wk = block_width(ncols, *b);
                    let col0 = (*b) * BLOCK_COLS;

                    let fb = File::open(p)?;
                    let [m2, n2] = read_shape(&fb)?;
                    if m2 != m || n2 != ncols {
                        return Err(hdf5::Error::Internal(format!(
                            "shape mismatch in part '{}': got [{m2},{n2}] expected [{m},{ncols}]",
                            p.display()
                        )));
                    }

                    let ds = fb.dataset("real")?;
                    let flat: Vec<$scalar> = ds.read_raw().map_err(|e| {
                        hdf5::Error::Internal(format!(
                            "failed reading '{}::real' as {}: {e}",
                            p.display(),
                            stringify!($scalar)
                        ))
                    })?;

                    if flat.len() != m * wk {
                        return Err(hdf5::Error::Internal(format!(
                            "length mismatch in '{}::real': got {}, expected {}",
                            p.display(),
                            flat.len(),
                            m * wk
                        )));
                    }

                    for j in 0..wk {
                        let gcol = col0 + j;
                        let src = &flat[j * m..(j + 1) * m];
                        let dst = &mut out[gcol * m..(gcol + 1) * m];
                        dst.copy_from_slice(src);
                    }
                }

                Ok(())
            }

            fn append_in_dir<ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar> + Shape<2>>(
                extra_arr: &Array<$scalar, ArrayImpl, 2>,
                path: &str,
                sampling_dir: Option<&Path>,
            ) -> hdf5::Result<()> {
                let sampling_dir = sampling_dir.unwrap_or_else(|| Path::new(DEFAULT_SAMPLING_DIR));
                ensure_sampling_dir(sampling_dir)?;

                let shape = extra_arr.shape();
                let m_add = shape[0];
                let ncols = shape[1];
                if m_add == 0 {
                    return Ok(());
                }

                let existing_parts = find_part_files_in_dir(sampling_dir, path)?;
                let m_old = if existing_parts.is_empty() {
                    0
                } else {
                    if existing_parts.len() != nblocks(ncols) {
                        return Err(hdf5::Error::Internal(format!(
                            "part count mismatch for base '{}': found {}, expected {}",
                            canonical_base(path),
                            existing_parts.len(),
                            nblocks(ncols)
                        )));
                    }

                    let file0 = File::open(&existing_parts[0].1)?;
                    let [stored_rows, stored_ncols] = read_shape(&file0)?;
                    if stored_ncols != ncols {
                        return Err(hdf5::Error::Internal(format!(
                            "column mismatch for base '{}': stored {}, incoming {}",
                            canonical_base(path),
                            stored_ncols,
                            ncols
                        )));
                    }
                    stored_rows
                };

                let total_rows = m_old + m_add;

                for b in 0..nblocks(ncols) {
                    let wk = block_width(ncols, b);
                    let col0 = b * BLOCK_COLS;
                    let extra_block = gather_block_column_major(extra_arr, col0, wk);

                    let merged = if m_old == 0 {
                        extra_block
                    } else {
                        let p = &existing_parts[b].1;
                        let fb = File::open(p)?;
                        let flat: Vec<$scalar> = fb.dataset("real")?.read_raw().map_err(|e| {
                            hdf5::Error::Internal(format!(
                                "failed reading '{}::real' as {}: {e}",
                                p.display(),
                                stringify!($scalar)
                            ))
                        })?;

                        if flat.len() != m_old * wk {
                            return Err(hdf5::Error::Internal(format!(
                                "length mismatch in '{}::real': got {}, expected {}",
                                p.display(),
                                flat.len(),
                                m_old * wk
                            )));
                        }

                        append_rows_column_major(&flat, &extra_block, m_old, m_add, wk)
                    };

                    let p = part_path(sampling_dir, path, b);
                    let file = File::create(&p)?;
                    write_shape(&file, [total_rows, ncols])?;

                    file.new_dataset::<$scalar>()
                        .shape((merged.len(),))
                        .chunk((chunk_len(merged.len(), wk),))
                        .create("real")?
                        .write(&merged)?;
                }

                Ok(())
            }
        }
    };
}

// ----------------------
// Complex implementation
// ----------------------
macro_rules! implement_io_data_complex {
    ($scalar:ty) => {
        impl IOData<Complex<$scalar>> for Complex<$scalar> {
            type Item = Complex<$scalar>;

            fn load_in_dir(
                path: &str,
                sampling_dir: Option<&Path>,
            ) -> hdf5::Result<Vec<Self::Item>> {
                if Path::new(path).exists() {
                    let file = File::open(path)?;
                    if file.dataset("real").is_ok() && file.dataset("imag").is_ok() {
                        let re_array = file.dataset("real")?.read()?;
                        let im_array = file.dataset("imag")?.read()?;
                        let re: Vec<$scalar> = re_array.to_vec();
                        let im: Vec<$scalar> = im_array.to_vec();

                        if re.len() != im.len() {
                            return Err(hdf5::Error::Internal(
                                "mismatched real/imag lengths".into(),
                            ));
                        }

                        return Ok(re
                            .into_iter()
                            .zip(im.into_iter())
                            .map(|(r, i)| Complex::new(r, i))
                            .collect());
                    }
                }

                let Some((_dir, parts)) = find_part_files(path, sampling_dir)? else {
                    return Err(hdf5::Error::Internal(format!(
                        "no '{}' (old format) and no multipart parts found for base '{}'",
                        path,
                        canonical_base(path)
                    )));
                };

                let file0 = File::open(&parts[0].1)?;
                let [m, ncols] = read_shape(&file0)?;

                let mut re_full = vec![<$scalar as Default>::default(); m * ncols];
                let mut im_full = vec![<$scalar as Default>::default(); m * ncols];

                for (b, p) in parts.iter() {
                    let wk = block_width(ncols, *b);
                    let col0 = (*b) * BLOCK_COLS;

                    let fb = File::open(p)?;
                    let [m2, n2] = read_shape(&fb)?;
                    if m2 != m || n2 != ncols {
                        return Err(hdf5::Error::Internal(format!(
                            "shape mismatch in part '{}': got [{m2},{n2}] expected [{m},{ncols}]",
                            p.display()
                        )));
                    }

                    let re_blk: Vec<$scalar> = fb.dataset("real")?.read_raw().map_err(|e| {
                        hdf5::Error::Internal(format!(
                            "failed reading '{}::real' as {}: {e}",
                            p.display(),
                            stringify!($scalar)
                        ))
                    })?;
                    let im_blk: Vec<$scalar> = fb.dataset("imag")?.read_raw().map_err(|e| {
                        hdf5::Error::Internal(format!(
                            "failed reading '{}::imag' as {}: {e}",
                            p.display(),
                            stringify!($scalar)
                        ))
                    })?;

                    if re_blk.len() != m * wk || im_blk.len() != m * wk {
                        return Err(hdf5::Error::Internal(format!(
                            "length mismatch in '{}': re={}, im={}, expected={}",
                            p.display(),
                            re_blk.len(),
                            im_blk.len(),
                            m * wk
                        )));
                    }

                    for j in 0..wk {
                        let gcol = col0 + j;
                        re_full[gcol * m..(gcol + 1) * m]
                            .copy_from_slice(&re_blk[j * m..(j + 1) * m]);
                        im_full[gcol * m..(gcol + 1) * m]
                            .copy_from_slice(&im_blk[j * m..(j + 1) * m]);
                    }
                }

                Ok(re_full
                    .into_iter()
                    .zip(im_full.into_iter())
                    .map(|(r, i)| Complex::new(r, i))
                    .collect())
            }

            fn load_into_in_dir(
                target: &mut DynamicArray<Self::Item, 2>,
                dim: usize,
                path: &str,
                sampling_dir: Option<&Path>,
            ) -> hdf5::Result<()> {
                if Path::new(path).exists() {
                    let file = File::open(path)?;
                    if file.dataset("real").is_ok() && file.dataset("imag").is_ok() {
                        let re: Vec<$scalar> = file.dataset("real")?.read_raw().map_err(|e| {
                            hdf5::Error::Internal(format!(
                                "failed reading '{}::real' as {}: {e}",
                                path,
                                stringify!($scalar)
                            ))
                        })?;
                        let im: Vec<$scalar> = file.dataset("imag")?.read_raw().map_err(|e| {
                            hdf5::Error::Internal(format!(
                                "failed reading '{}::imag' as {}: {e}",
                                path,
                                stringify!($scalar)
                            ))
                        })?;
                        if re.len() != im.len() {
                            return Err(hdf5::Error::Internal(
                                "mismatched real/imag lengths".into(),
                            ));
                        }
                        if dim == 0 {
                            target.resize_in_place([0, 0]);
                            return Ok(());
                        }
                        if re.len() % dim != 0 {
                            return Err(hdf5::Error::Internal(format!(
                                "length mismatch in '{}': got {}, not divisible by dim {}",
                                path,
                                re.len(),
                                dim
                            )));
                        }
                        let rows = re.len() / dim;
                        target.resize_in_place([rows, dim]);
                        for (dst, (r, i)) in
                            target.data_mut().iter_mut().zip(re.into_iter().zip(im))
                        {
                            *dst = Complex::new(r, i);
                        }
                        return Ok(());
                    }
                }

                let Some((_dir, parts)) = find_part_files(path, sampling_dir)? else {
                    return Err(hdf5::Error::Internal(format!(
                        "no '{}' (old format) and no multipart parts found for base '{}'",
                        path,
                        canonical_base(path)
                    )));
                };

                let file0 = File::open(&parts[0].1)?;
                let [m, ncols] = read_shape(&file0)?;
                if dim != ncols {
                    return Err(hdf5::Error::Internal(format!(
                        "column mismatch for base '{}': stored {}, expected {}",
                        canonical_base(path),
                        ncols,
                        dim
                    )));
                }

                target.resize_in_place([m, ncols]);
                let out = target.data_mut();

                for (b, p) in parts.iter() {
                    let wk = block_width(ncols, *b);
                    let col0 = (*b) * BLOCK_COLS;

                    let fb = File::open(p)?;
                    let [m2, n2] = read_shape(&fb)?;
                    if m2 != m || n2 != ncols {
                        return Err(hdf5::Error::Internal(format!(
                            "shape mismatch in part '{}': got [{m2},{n2}] expected [{m},{ncols}]",
                            p.display()
                        )));
                    }

                    let re_blk: Vec<$scalar> = fb.dataset("real")?.read_raw().map_err(|e| {
                        hdf5::Error::Internal(format!(
                            "failed reading '{}::real' as {}: {e}",
                            p.display(),
                            stringify!($scalar)
                        ))
                    })?;
                    let im_blk: Vec<$scalar> = fb.dataset("imag")?.read_raw().map_err(|e| {
                        hdf5::Error::Internal(format!(
                            "failed reading '{}::imag' as {}: {e}",
                            p.display(),
                            stringify!($scalar)
                        ))
                    })?;

                    if re_blk.len() != m * wk || im_blk.len() != m * wk {
                        return Err(hdf5::Error::Internal(format!(
                            "length mismatch in '{}': real {}, imag {}, expected {}",
                            p.display(),
                            re_blk.len(),
                            im_blk.len(),
                            m * wk
                        )));
                    }

                    for j in 0..wk {
                        let gcol = col0 + j;
                        let re_src = &re_blk[j * m..(j + 1) * m];
                        let im_src = &im_blk[j * m..(j + 1) * m];
                        let dst = &mut out[gcol * m..(gcol + 1) * m];
                        for (dst_val, (r, i)) in
                            dst.iter_mut().zip(re_src.iter().zip(im_src.iter()))
                        {
                            *dst_val = Complex::new(*r, *i);
                        }
                    }
                }

                Ok(())
            }

            fn append_in_dir<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = Complex<$scalar>> + Shape<2>,
            >(
                extra_arr: &Array<Complex<$scalar>, ArrayImpl, 2>,
                path: &str,
                sampling_dir: Option<&Path>,
            ) -> hdf5::Result<()> {
                let sampling_dir = sampling_dir.unwrap_or_else(|| Path::new(DEFAULT_SAMPLING_DIR));
                ensure_sampling_dir(sampling_dir)?;

                let shape = extra_arr.shape();
                let m_add = shape[0];
                let ncols = shape[1];
                if m_add == 0 {
                    return Ok(());
                }

                let existing_parts = find_part_files_in_dir(sampling_dir, path)?;
                let m_old = if existing_parts.is_empty() {
                    0
                } else {
                    if existing_parts.len() != nblocks(ncols) {
                        return Err(hdf5::Error::Internal(format!(
                            "part count mismatch for base '{}': found {}, expected {}",
                            canonical_base(path),
                            existing_parts.len(),
                            nblocks(ncols)
                        )));
                    }

                    let file0 = File::open(&existing_parts[0].1)?;
                    let [stored_rows, stored_ncols] = read_shape(&file0)?;
                    if stored_ncols != ncols {
                        return Err(hdf5::Error::Internal(format!(
                            "column mismatch for base '{}': stored {}, incoming {}",
                            canonical_base(path),
                            stored_ncols,
                            ncols
                        )));
                    }
                    stored_rows
                };

                let total_rows = m_old + m_add;

                for b in 0..nblocks(ncols) {
                    let wk = block_width(ncols, b);
                    let col0 = b * BLOCK_COLS;
                    let extra_block = gather_block_column_major(extra_arr, col0, wk);

                    let mut extra_re: Vec<$scalar> = Vec::with_capacity(extra_block.len());
                    let mut extra_im: Vec<$scalar> = Vec::with_capacity(extra_block.len());
                    for z in extra_block.iter() {
                        extra_re.push(z.re);
                        extra_im.push(z.im);
                    }

                    let (merged_re, merged_im) = if m_old == 0 {
                        (extra_re, extra_im)
                    } else {
                        let p = &existing_parts[b].1;
                        let fb = File::open(p)?;
                        let old_re: Vec<$scalar> = fb.dataset("real")?.read_raw().map_err(|e| {
                            hdf5::Error::Internal(format!(
                                "failed reading '{}::real' as {}: {e}",
                                p.display(),
                                stringify!($scalar)
                            ))
                        })?;
                        let old_im: Vec<$scalar> = fb.dataset("imag")?.read_raw().map_err(|e| {
                            hdf5::Error::Internal(format!(
                                "failed reading '{}::imag' as {}: {e}",
                                p.display(),
                                stringify!($scalar)
                            ))
                        })?;

                        if old_re.len() != m_old * wk || old_im.len() != m_old * wk {
                            return Err(hdf5::Error::Internal(format!(
                                "length mismatch in '{}': re={}, im={}, expected={}",
                                p.display(),
                                old_re.len(),
                                old_im.len(),
                                m_old * wk
                            )));
                        }

                        (
                            append_rows_column_major(&old_re, &extra_re, m_old, m_add, wk),
                            append_rows_column_major(&old_im, &extra_im, m_old, m_add, wk),
                        )
                    };

                    let p = part_path(sampling_dir, path, b);
                    let file = File::create(&p)?;
                    write_shape(&file, [total_rows, ncols])?;

                    file.new_dataset::<$scalar>()
                        .shape((merged_re.len(),))
                        .chunk((chunk_len(merged_re.len(), wk),))
                        .create("real")?
                        .write(&merged_re)?;
                    file.new_dataset::<$scalar>()
                        .shape((merged_im.len(),))
                        .chunk((chunk_len(merged_im.len(), wk),))
                        .create("imag")?
                        .write(&merged_im)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn unique_temp_dir(tag: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        dir.push(format!("rsrs_{tag}_{stamp}_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn array_from_rows<Item: RlstScalar + Copy>(rows: &[&[Item]]) -> DynamicArray<Item, 2> {
        let m = rows.len();
        let n = rows.first().map_or(0, |row| row.len());
        let mut arr = rlst_dynamic_array2!(Item, [m, n]);
        let mut view = arr.r_mut();

        for i in 0..m {
            for j in 0..n {
                view[[i, j]] = rows[i][j];
            }
        }

        arr
    }

    fn flatten_column_major<T: Copy>(rows: &[&[T]]) -> Vec<T> {
        let m = rows.len();
        let n = rows.first().map_or(0, |row| row.len());
        let mut flat = Vec::with_capacity(m * n);

        for j in 0..n {
            for row in rows.iter().take(m) {
                flat.push(row[j]);
            }
        }

        flat
    }

    #[test]
    fn real_append_preserves_column_major_blocks() {
        let dir = unique_temp_dir("io_real_append");

        let initial = array_from_rows::<f64>(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        <f64 as IOData<f64>>::append_in_dir(&initial, "y_test_file", Some(dir.as_path())).unwrap();

        let extra = array_from_rows::<f64>(&[&[7.0, 8.0, 9.0]]);
        <f64 as IOData<f64>>::append_in_dir(&extra, "y_test_file", Some(dir.as_path())).unwrap();

        let loaded = <f64 as IOData<f64>>::load_in_dir("y_test_file", Some(dir.as_path())).unwrap();
        assert_eq!(
            loaded,
            flatten_column_major(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0],])
        );

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn complex_append_preserves_column_major_blocks() {
        let dir = unique_temp_dir("io_complex_append");

        let initial = array_from_rows::<Complex<f64>>(&[
            &[Complex::new(1.0, 1.5), Complex::new(2.0, 2.5)],
            &[Complex::new(3.0, 3.5), Complex::new(4.0, 4.5)],
        ]);
        <Complex<f64> as IOData<Complex<f64>>>::append_in_dir(
            &initial,
            "z_sketch_file",
            Some(dir.as_path()),
        )
        .unwrap();

        let extra =
            array_from_rows::<Complex<f64>>(&[&[Complex::new(5.0, 5.5), Complex::new(6.0, 6.5)]]);
        <Complex<f64> as IOData<Complex<f64>>>::append_in_dir(
            &extra,
            "z_sketch_file",
            Some(dir.as_path()),
        )
        .unwrap();

        let loaded = <Complex<f64> as IOData<Complex<f64>>>::load_in_dir(
            "z_sketch_file",
            Some(dir.as_path()),
        )
        .unwrap();
        assert_eq!(
            loaded,
            flatten_column_major(&[
                &[Complex::new(1.0, 1.5), Complex::new(2.0, 2.5)],
                &[Complex::new(3.0, 3.5), Complex::new(4.0, 4.5)],
                &[Complex::new(5.0, 5.5), Complex::new(6.0, 6.5)],
            ])
        );

        fs::remove_dir_all(dir).unwrap();
    }
}
