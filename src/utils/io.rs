use hdf5::File;
use num_complex::Complex;
pub use rlst::prelude::*;
use std::path::Path;

// ----------------------
// Chunking / blocking parameters
// ----------------------
// BLOCK_COLS: number of columns (across N) stored per PART FILE.
const BLOCK_COLS: usize = 4096;
// CHUNK_ROWS: HDF5 internal chunk length ~ CHUNK_ROWS * block_width elements.
const CHUNK_ROWS: usize = 256;

const SAMPLING_DIR: &str = "sampling";

fn ensure_sampling_dir() -> hdf5::Result<()> {
    std::fs::create_dir_all(SAMPLING_DIR).map_err(|e| {
        hdf5::Error::Internal(format!("failed to create '{SAMPLING_DIR}' directory: {e}").into())
    })
}

// ----------------------
// Helpers
// ----------------------
fn nblocks(ncols: usize) -> usize {
    (ncols + BLOCK_COLS - 1) / BLOCK_COLS
}

fn block_width(ncols: usize, b: usize) -> usize {
    let col0 = b * BLOCK_COLS;
    (ncols - col0).min(BLOCK_COLS)
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
                                      // tail layout: '.' + 5 digits + ".h5"
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

/// Part file naming: "{base}.00000.h5"
fn part_path(base: &str, b: usize) -> String {
    let b0 = canonical_base(base);
    format!("{SAMPLING_DIR}/{b0}.{:05}.h5", b)
}

/// Best-effort cleanup: remove existing part files base.00000.h5, base.00001.h5, ...
fn remove_existing_parts(base: &str) -> std::io::Result<()> {
    let b0 = canonical_base(base);
    for b in 0.. {
        let p = format!("{SAMPLING_DIR}/{b0}.{:05}.h5", b);
        if !Path::new(&p).exists() {
            break;
        }
        std::fs::remove_file(&p)?;
    }
    Ok(())
}

/// Discover all part files in the current directory that match "{base}.00000.h5", sort by index.
/// Since you said you don't pass a directory in `path`, we always scan ".".
fn find_part_files(base: &str) -> hdf5::Result<Vec<(usize, String)>> {
    let stem = canonical_base(base);

    // If sampling/ doesn't exist, just return "no parts" (caller will ignore)
    if !Path::new(SAMPLING_DIR).exists() {
        return Ok(Vec::new());
    }

    let mut parts: Vec<(usize, String)> = Vec::new();

    let rd = std::fs::read_dir(SAMPLING_DIR).map_err(|e| {
        hdf5::Error::Internal(format!("read_dir failed for '{SAMPLING_DIR}': {e}").into())
    })?;

    for entry in rd {
        let entry = entry
            .map_err(|e| hdf5::Error::Internal(format!("read_dir entry error: {e}").into()))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let fname = match path.file_name() {
            Some(x) => x.to_string_lossy(),
            None => continue,
        };

        // Match exactly: "{stem}.00000.h5"
        let prefix = format!("{stem}.");
        if !fname.starts_with(&prefix) || !fname.ends_with(".h5") {
            continue;
        }

        let middle = &fname[prefix.len()..fname.len() - 3]; // strip prefix and ".h5"
        if middle.len() != 5 || !middle.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }

        let idx: usize = middle.parse().unwrap();
        parts.push((idx, format!("{SAMPLING_DIR}/{}", fname)));
    }

    parts.sort_by_key(|(i, _)| *i);

    // Require contiguous indices 0..K-1
    for (expected, (idx, _)) in parts.iter().enumerate() {
        if *idx != expected {
            return Err(hdf5::Error::Internal(
                format!(
                    "missing part file index {} (found {}) in {SAMPLING_DIR}",
                    expected, idx
                )
                .into(),
            ));
        }
    }

    Ok(parts)
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
// Trait (unchanged)
// ----------------------
pub trait IOData<T: RlstScalar> {
    type Item: RlstScalar;
    fn load(path: &str) -> hdf5::Result<Vec<Self::Item>>;
    fn append<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = T> + Stride<2> + RawAccessMut<Item = T> + Shape<2>,
    >(
        data: &Array<T, ArrayImpl, 2>,
        path: &str,
    ) -> hdf5::Result<()>;
}

// ----------------------
// Real implementation
// ----------------------
macro_rules! implement_io_data_real {
    ($scalar:ty) => {
        impl IOData<$scalar> for $scalar {
            type Item = $scalar;

            fn load(path: &str) -> hdf5::Result<Vec<Self::Item>> {
                // Backward compatibility: old single file layout (dataset "real")
                if Path::new(path).exists() {
                    let file = File::open(path)?;
                    if file.dataset("real").is_ok() {
                        let ds = file.dataset("real")?;
                        let array = ds.read()?;
                        return Ok(array.to_vec());
                    }
                }

                // New multipart layout: discover parts by scanning current directory
                let parts = find_part_files(path)?;
                if parts.is_empty() {
                    return Err(hdf5::Error::Internal(format!(
                        "no '{}' (old format) and no multipart parts found for base '{}'",
                        path,
                        canonical_base(path)
                    )
                    .into()));
                }

                // Read global shape [m, ncols] from part 0
                let file0 = File::open(&parts[0].1)?;
                let [m, ncols] = read_shape(&file0)?;

                // Allocate full output flat column-major (m x ncols)
                let mut out = vec![<$scalar as Default>::default(); m * ncols];

                // For each part file in order, read dataset "real" (flat column-major for that block)
                for (b, p) in parts.iter() {
                    let wk = block_width(ncols, *b);
                    let col0 = (*b) * BLOCK_COLS;

                    let fb = File::open(p)?;
                    // Sanity: ensure all parts agree on shape
                    let [m2, n2] = read_shape(&fb)?;
                    if m2 != m || n2 != ncols {
                        return Err(hdf5::Error::Internal(format!(
                            "shape mismatch in part '{p}': got [{m2},{n2}] expected [{m},{ncols}]"
                        )
                        .into()));
                    }

                    let ds = fb.dataset("real")?;
                    let flat: Vec<$scalar> = ds.read_raw().map_err(|e| {
                        hdf5::Error::Internal(format!(
                            "failed reading '{p}::real' as {}: {e}",
                            stringify!($scalar)
                        )
                        .into())
                    })?;

                    if flat.len() != m * wk {
                        return Err(hdf5::Error::Internal(format!(
                            "length mismatch in '{p}::real': got {}, expected {}",
                            flat.len(),
                            m * wk
                        )
                        .into()));
                    }

                    // Scatter into full column-major buffer
                    for j in 0..wk {
                        let gcol = col0 + j;
                        let src = &flat[j * m..(j + 1) * m];
                        let dst = &mut out[gcol * m..(gcol + 1) * m];
                        dst.copy_from_slice(src);
                    }
                }

                Ok(out)
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
                ensure_sampling_dir()?;

                let shape = extra_arr.shape(); // [m, ncols]  (num_samples x N)
                let m = shape[0];
                let ncols = shape[1];
                let data = extra_arr.data();

                println!(
                    "[save_real_multipart] base='{}' (from '{}'), incoming shape = {:?}, flat len = {}",
                    canonical_base(path),
                    path,
                    shape,
                    data.len()
                );

                // Save/load one: overwrite by removing old single file and all part files.
                if Path::new(path).exists() {
                    std::fs::remove_file(path).map_err(|e| {
                        hdf5::Error::Internal(format!("failed to remove existing file '{path}': {e}").into())
                    })?;
                }
                remove_existing_parts(path).map_err(|e| {
                    hdf5::Error::Internal(format!("failed to remove existing part files: {e}").into())
                })?;

                // Create one part file per block, each containing dataset "real" + global shape attr.
                for b in 0..nblocks(ncols) {
                    let wk = block_width(ncols, b);
                    let col0 = b * BLOCK_COLS;

                    // Column-major contiguous slice for this block:
                    // block = data[col0*m .. (col0+wk)*m]
                    let start = col0 * m;
                    let end = (col0 + wk) * m;
                    let block = &data[start..end];

                    let p = part_path(path, b);
                    let file = File::create(&p)?;
                    write_shape(&file, [m, ncols])?;

                    let chunk_len = (CHUNK_ROWS * wk).max(1);

                    file.new_dataset::<$scalar>()
                        .shape((block.len(),))
                        .chunk((chunk_len,))
                        .create("real")?
                        .write(block)?;

                    println!(
                        "[save_real_multipart] wrote part {} -> '{}', cols [{}..{}), len={}",
                        b,
                        p,
                        col0,
                        col0 + wk,
                        block.len()
                    );
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

            fn load(path: &str) -> hdf5::Result<Vec<Self::Item>> {
                // Backward compatibility: old single file layout "real"/"imag"
                if Path::new(path).exists() {
                    let file = File::open(path)?;
                    if file.dataset("real").is_ok() && file.dataset("imag").is_ok() {
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
                        return Ok(data);
                    }
                }

                // New multipart layout: discover parts by scanning current directory
                let parts = find_part_files(path)?;
                if parts.is_empty() {
                    return Err(hdf5::Error::Internal(format!(
                        "no '{}' (old format) and no multipart parts found for base '{}'",
                        path,
                        canonical_base(path)
                    )
                    .into()));
                }

                // Global shape from part 0
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
                            "shape mismatch in part '{p}': got [{m2},{n2}] expected [{m},{ncols}]"
                        )
                        .into()));
                    }

                    let ds_re = fb.dataset("real")?;
                    let ds_im = fb.dataset("imag")?;
                    let re_blk: Vec<$scalar> = ds_re.read_raw().map_err(|e| {
                        hdf5::Error::Internal(format!(
                            "failed reading '{p}::real' as {}: {e}",
                            stringify!($scalar)
                        )
                        .into())
                    })?;
                    let im_blk: Vec<$scalar> = ds_im.read_raw().map_err(|e| {
                        hdf5::Error::Internal(format!(
                            "failed reading '{p}::imag' as {}: {e}",
                            stringify!($scalar)
                        )
                        .into())
                    })?;

                    if re_blk.len() != m * wk || im_blk.len() != m * wk {
                        return Err(hdf5::Error::Internal(format!(
                            "length mismatch in '{p}': re={}, im={}, expected={}",
                            re_blk.len(),
                            im_blk.len(),
                            m * wk
                        )
                        .into()));
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

            fn append<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = Complex<$scalar>>
                    + Stride<2>
                    + RawAccessMut<Item = Complex<$scalar>>
                    + Shape<2>,
            >(
                extra_arr: &Array<Complex<$scalar>, ArrayImpl, 2>,
                path: &str,
            ) -> hdf5::Result<()> {
                ensure_sampling_dir()?;

                let shape = extra_arr.shape(); // [m, ncols]
                let m = shape[0];
                let ncols = shape[1];
                let data = extra_arr.data();

                println!(
                    "[save_complex_multipart] base='{}' (from '{}'), incoming shape = {:?}, flat len = {}",
                    canonical_base(path),
                    path,
                    shape,
                    data.len()
                );

                // overwrite by removing old single file + part files
                if Path::new(path).exists() {
                    std::fs::remove_file(path).map_err(|e| {
                        hdf5::Error::Internal(format!("failed to remove existing file '{path}': {e}").into())
                    })?;
                }
                remove_existing_parts(path).map_err(|e| {
                    hdf5::Error::Internal(format!("failed to remove existing part files: {e}").into())
                })?;

                for b in 0..nblocks(ncols) {
                    let wk = block_width(ncols, b);
                    let col0 = b * BLOCK_COLS;

                    let start = col0 * m;
                    let end = (col0 + wk) * m;
                    let block = &data[start..end]; // &[Complex<$scalar>]

                    let mut re: Vec<$scalar> = Vec::with_capacity(block.len());
                    let mut im: Vec<$scalar> = Vec::with_capacity(block.len());
                    for z in block.iter() {
                        re.push(z.re);
                        im.push(z.im);
                    }

                    let p = part_path(path, b);
                    let file = File::create(&p)?;
                    write_shape(&file, [m, ncols])?;

                    let chunk_len = (CHUNK_ROWS * wk).max(1);

                    file.new_dataset::<$scalar>()
                        .shape((re.len(),))
                        .chunk((chunk_len,))
                        .create("real")?
                        .write(&re)?;
                    file.new_dataset::<$scalar>()
                        .shape((im.len(),))
                        .chunk((chunk_len,))
                        .create("imag")?
                        .write(&im)?;

                    println!(
                        "[save_complex_multipart] wrote part {} -> '{}', cols [{}..{}), len={}",
                        b,
                        p,
                        col0,
                        col0 + wk,
                        re.len()
                    );
                }

                Ok(())
            }
        }
    };
}

// Instantiate for f64/f32 and complex variants
implement_io_data_real!(f64);
implement_io_data_real!(f32);
implement_io_data_complex!(f64);
implement_io_data_complex!(f32);
