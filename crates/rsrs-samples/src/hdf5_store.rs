//! Streaming reader for the versioned sketch-only HDF5 cache.

use crate::{
    digest_complex, digest_real, digest_recipe, generate_complex, generate_real, MetadataError,
    SampleCacheMetadata, ScalarKind,
};
use hdf5::{types::VarLenUnicode, File};
use ndarray::s;
use num_complex::Complex64;
use std::{fmt, path::Path};

const CACHE_KIND: &str = "rsrs_sketch_cache";

#[derive(Debug)]
pub enum CacheError {
    Hdf5(hdf5::Error),
    Json(serde_json::Error),
    Metadata(MetadataError),
    Invalid(String),
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hdf5(error) => error.fmt(f),
            Self::Json(error) => error.fmt(f),
            Self::Metadata(error) => error.fmt(f),
            Self::Invalid(message) => message.fmt(f),
        }
    }
}

impl std::error::Error for CacheError {}

impl From<hdf5::Error> for CacheError {
    fn from(value: hdf5::Error) -> Self {
        Self::Hdf5(value)
    }
}

impl From<serde_json::Error> for CacheError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<MetadataError> for CacheError {
    fn from(value: MetadataError) -> Self {
        Self::Metadata(value)
    }
}

pub enum SketchBlock {
    Real {
        y_test: Vec<f64>,
        y_sketch: Vec<f64>,
        z_test: Option<Vec<f64>>,
        z_sketch: Option<Vec<f64>>,
        rows: usize,
        columns: usize,
    },
    Complex {
        y_test: Vec<Complex64>,
        y_sketch: Vec<Complex64>,
        z_test: Option<Vec<Complex64>>,
        z_sketch: Option<Vec<Complex64>>,
        rows: usize,
        columns: usize,
    },
}

pub struct Hdf5SketchCache {
    file: File,
    metadata: SampleCacheMetadata,
    committed_rows: usize,
}

impl Hdf5SketchCache {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, CacheError> {
        let file = File::open(path)?;
        let kind: VarLenUnicode = file.attr("cache_kind")?.read_scalar()?;
        if kind.as_str() != CACHE_KIND {
            return Err(CacheError::Invalid(format!(
                "unexpected cache kind {kind:?}"
            )));
        }
        let metadata_json: VarLenUnicode = file.attr("metadata_json")?.read_scalar()?;
        let metadata: SampleCacheMetadata = serde_json::from_str(metadata_json.as_str())?;
        metadata.validate()?;
        let committed_rows = usize::try_from(file.attr("committed_rows")?.read_scalar::<u64>()?)
            .map_err(|_| CacheError::Invalid("committed row count overflows usize".into()))?;
        if committed_rows > metadata.y_test.rows {
            return Err(CacheError::Invalid(
                "committed rows exceed the test recipe".into(),
            ));
        }
        let families = if metadata.z_test.is_some() {
            &["y", "z"][..]
        } else {
            &["y"][..]
        };
        for family in families {
            let shape = file.dataset(&format!("sketches/{family}/real"))?.shape();
            if shape.len() != 2
                || shape[0] < committed_rows
                || shape[1] != metadata.operator_dimension
            {
                return Err(CacheError::Invalid(format!(
                    "invalid {family} sketch shape {shape:?}"
                )));
            }
        }
        Ok(Self {
            file,
            metadata,
            committed_rows,
        })
    }

    pub fn metadata(&self) -> &SampleCacheMetadata {
        &self.metadata
    }

    pub fn committed_rows(&self) -> usize {
        self.committed_rows
    }

    pub fn read_block(&self, first: usize, rows: usize) -> Result<SketchBlock, CacheError> {
        let last = first
            .checked_add(rows)
            .ok_or_else(|| CacheError::Invalid("sample row range overflow".into()))?;
        if last > self.committed_rows {
            return Err(CacheError::Invalid(
                "requested rows exceed the committed cache".into(),
            ));
        }
        let columns = self.metadata.operator_dimension;
        match self.metadata.scalar_kind {
            ScalarKind::Real64 => Ok(SketchBlock::Real {
                y_test: generate_real(&self.metadata.y_test.generator, first, rows, columns)
                    .map_err(|error| CacheError::Invalid(error.to_string()))?,
                y_sketch: self.read_real("y", first, last)?,
                z_test: self
                    .metadata
                    .z_test
                    .as_ref()
                    .map(|recipe| generate_real(&recipe.generator, first, rows, columns))
                    .transpose()
                    .map_err(|error| CacheError::Invalid(error.to_string()))?,
                z_sketch: self
                    .metadata
                    .z_test
                    .as_ref()
                    .map(|_| self.read_real("z", first, last))
                    .transpose()?,
                rows,
                columns,
            }),
            ScalarKind::Complex64 => Ok(SketchBlock::Complex {
                y_test: generate_complex(&self.metadata.y_test.generator, first, rows, columns)
                    .map_err(|error| CacheError::Invalid(error.to_string()))?,
                y_sketch: self.read_complex("y", first, last)?,
                z_test: self
                    .metadata
                    .z_test
                    .as_ref()
                    .map(|recipe| generate_complex(&recipe.generator, first, rows, columns))
                    .transpose()
                    .map_err(|error| CacheError::Invalid(error.to_string()))?,
                z_sketch: self
                    .metadata
                    .z_test
                    .as_ref()
                    .map(|_| self.read_complex("z", first, last))
                    .transpose()?,
                rows,
                columns,
            }),
        }
    }

    pub fn validate_manifest(&self, index: usize) -> Result<(), CacheError> {
        let first_rows = self.file.dataset("manifests/first_row")?;
        if index >= first_rows.size() {
            return Err(CacheError::Invalid("manifest index out of range".into()));
        }
        let first = first_rows.read_slice_1d::<u64, _>(s![index..index + 1])?[0] as usize;
        let rows = self
            .file
            .dataset("manifests/rows")?
            .read_slice_1d::<u64, _>(s![index..index + 1])?[0] as usize;
        let expected = self
            .file
            .dataset("manifests/sha256")?
            .read_slice_2d::<u8, _>(s![index, .., ..])?;
        let block = self.read_block(first, rows)?;
        let actual = match block {
            SketchBlock::Real {
                y_sketch,
                z_sketch,
                columns,
                ..
            } => [
                digest_recipe(&self.metadata.y_test, first, rows),
                digest_real(first, rows, columns, &y_sketch),
                self.metadata
                    .z_test
                    .as_ref()
                    .map(|recipe| digest_recipe(recipe, first, rows))
                    .unwrap_or([0; 32]),
                z_sketch
                    .as_ref()
                    .map(|values| digest_real(first, rows, columns, values))
                    .unwrap_or([0; 32]),
            ],
            SketchBlock::Complex {
                y_sketch,
                z_sketch,
                columns,
                ..
            } => [
                digest_recipe(&self.metadata.y_test, first, rows),
                digest_complex(first, rows, columns, &y_sketch),
                self.metadata
                    .z_test
                    .as_ref()
                    .map(|recipe| digest_recipe(recipe, first, rows))
                    .unwrap_or([0; 32]),
                z_sketch
                    .as_ref()
                    .map(|values| digest_complex(first, rows, columns, values))
                    .unwrap_or([0; 32]),
            ],
        };
        let expected = expected.as_slice().ok_or_else(|| {
            CacheError::Invalid("manifest digest dataset is not contiguous".into())
        })?;
        if expected != actual.concat() {
            return Err(CacheError::Invalid(format!(
                "sample digest mismatch in manifest {index}"
            )));
        }
        Ok(())
    }

    pub fn validate_quick(&self) -> Result<(), CacheError> {
        let count = self.file.dataset("manifests/first_row")?.size();
        if count != 0 {
            self.validate_manifest(0)?;
            if count > 1 {
                self.validate_manifest(count - 1)?;
            }
        }
        Ok(())
    }

    fn read_real(&self, family: &str, first: usize, last: usize) -> Result<Vec<f64>, CacheError> {
        Ok(self
            .file
            .dataset(&format!("sketches/{family}/real"))?
            .read_slice_2d::<f64, _>(s![first..last, ..])?
            .into_raw_vec())
    }

    fn read_complex(
        &self,
        family: &str,
        first: usize,
        last: usize,
    ) -> Result<Vec<Complex64>, CacheError> {
        let real = self.read_real(family, first, last)?;
        let imag = self
            .file
            .dataset(&format!("sketches/{family}/imag"))?
            .read_slice_2d::<f64, _>(s![first..last, ..])?
            .into_raw_vec();
        if real.len() != imag.len() {
            return Err(CacheError::Invalid(
                "real and imaginary sketch shapes differ".into(),
            ));
        }
        Ok(real
            .into_iter()
            .zip(imag)
            .map(|(re, im)| Complex64::new(re, im))
            .collect())
    }
}
