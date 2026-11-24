//! Portable RSRS test-matrix generation and sample metadata.
//!
//! The generator is counter based: any row range can be regenerated without
//! producing the preceding rows. Python calls this implementation through the
//! RSRS extension, ensuring that Rust and Python use identical test values.

use num_complex::Complex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[cfg(feature = "hdf5-store")]
pub mod hdf5_store;

/// Current on-disk sample schema.
pub const FORMAT_VERSION: u32 = 2;
pub const LEGACY_FORMAT_VERSION: u32 = 1;
/// Current deterministic generator algorithm.
pub const GENERATOR_VERSION: u32 = 1;

/// Identifies the statistically independent RSRS test family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestFamily {
    /// Tests used to form `Y = A Omega_y`.
    Y,
    /// Tests used to form the transpose or adjoint sketch `Z`.
    Z,
}

impl TestFamily {
    fn stream(self) -> u64 {
        match self {
            Self::Y => 0x595F_5445_5354_5F31,
            Self::Z => 0x5A5F_5445_5354_5F31,
        }
    }
}

/// Operation used to form the second sketch family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZMode {
    /// `Z = A^T Omega_z`.
    Transpose,
    /// `Z = A^* Omega_z`.
    Adjoint,
}

/// Scalar representation used by the operator and its sketches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarKind {
    Real64,
    Complex64,
}

/// Recipe for one regenerable test matrix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestMatrixRecipe {
    pub rows: usize,
    pub columns: usize,
    pub generator: GeneratorSpec,
}

/// Immutable metadata stored with every sketch cache.
///
/// Unknown descriptive fields belong in `annotations`; fields that affect
/// numerical interpretation are explicit and versioned here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SampleCacheMetadata {
    pub format_version: u32,
    pub scalar_kind: ScalarKind,
    pub operator_dimension: usize,
    pub y_test: TestMatrixRecipe,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub z_test: Option<TestMatrixRecipe>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub z_mode: Option<ZMode>,
    /// Digest of the canonical operator configuration and geometry identity.
    pub operator_fingerprint: [u8; 32],
    #[serde(default)]
    pub annotations: BTreeMap<String, String>,
}

impl SampleCacheMetadata {
    pub fn validate(&self) -> Result<(), MetadataError> {
        if self.format_version != LEGACY_FORMAT_VERSION && self.format_version != FORMAT_VERSION {
            return Err(MetadataError::UnsupportedFormat(self.format_version));
        }
        if self.y_test.columns != self.operator_dimension
            || self
                .z_test
                .as_ref()
                .is_some_and(|recipe| recipe.columns != self.operator_dimension)
        {
            return Err(MetadataError::DimensionMismatch);
        }
        if self
            .z_test
            .as_ref()
            .is_some_and(|recipe| self.y_test.rows != recipe.rows)
        {
            return Err(MetadataError::RowCountMismatch);
        }
        if self.y_test.generator.family != TestFamily::Y
            || self
                .z_test
                .as_ref()
                .is_some_and(|recipe| recipe.generator.family != TestFamily::Z)
        {
            return Err(MetadataError::FamilyMismatch);
        }
        if self.z_test.is_some() != self.z_mode.is_some() {
            return Err(MetadataError::IncompleteZMetadata);
        }
        if self.format_version == LEGACY_FORMAT_VERSION && self.z_test.is_none() {
            return Err(MetadataError::LegacyFormatRequiresZ);
        }
        let expected_distribution = match self.scalar_kind {
            ScalarKind::Real64 => TestDistribution::RealStandardNormal,
            ScalarKind::Complex64 => TestDistribution::CircularComplexStandardNormal,
        };
        if self.y_test.generator.distribution != expected_distribution
            || self
                .z_test
                .as_ref()
                .is_some_and(|recipe| recipe.generator.distribution != expected_distribution)
        {
            return Err(MetadataError::DistributionMismatch);
        }
        self.y_test
            .generator
            .validate()
            .map_err(MetadataError::Generator)?;
        if let Some(z_test) = &self.z_test {
            z_test
                .generator
                .validate()
                .map_err(MetadataError::Generator)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetadataError {
    UnsupportedFormat(u32),
    DimensionMismatch,
    RowCountMismatch,
    FamilyMismatch,
    DistributionMismatch,
    IncompleteZMetadata,
    LegacyFormatRequiresZ,
    Generator(GeneratorError),
}

impl std::fmt::Display for MetadataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedFormat(version) => write!(f, "unsupported sample format {version}"),
            Self::DimensionMismatch => write!(f, "test columns do not match operator dimension"),
            Self::RowCountMismatch => write!(f, "Y and Z test row counts differ"),
            Self::FamilyMismatch => write!(f, "test generator family does not match Y/Z role"),
            Self::DistributionMismatch => {
                write!(f, "test distribution does not match the scalar kind")
            }
            Self::IncompleteZMetadata => {
                write!(
                    f,
                    "Z recipe and Z mode must either both be present or both be absent"
                )
            }
            Self::LegacyFormatRequiresZ => write!(f, "format version 1 requires a Z family"),
            Self::Generator(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for MetadataError {}

/// Integrity record for one append transaction. A writer first appends sketch
/// data, then this record, and finally advances the committed-row counter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchManifest {
    pub first_row: usize,
    pub rows: usize,
    pub y_test_digest: [u8; 32],
    pub y_sketch_digest: [u8; 32],
    pub z_test_digest: [u8; 32],
    pub z_sketch_digest: [u8; 32],
}

/// Scalar and distribution used by a reproducible test matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestDistribution {
    /// Independent real `N(0, 1)` entries.
    RealStandardNormal,
    /// Circular complex entries `(x + i y) / sqrt(2)`, with independent
    /// `x, y ~ N(0, 1)`, hence `E[|z|^2] = 1` and `E[z^2] = 0`.
    CircularComplexStandardNormal,
}

/// Complete recipe required to regenerate a test matrix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratorSpec {
    /// Generator algorithm version.
    pub version: u32,
    /// User-selected root seed.
    pub seed: u64,
    /// Matrix family, providing an independent random stream.
    pub family: TestFamily,
    /// Entry distribution and normalization.
    pub distribution: TestDistribution,
}

impl GeneratorSpec {
    /// Construct a specification using the current generator version.
    pub fn new(seed: u64, family: TestFamily, distribution: TestDistribution) -> Self {
        Self {
            version: GENERATOR_VERSION,
            seed,
            family,
            distribution,
        }
    }

    fn validate(&self) -> Result<(), GeneratorError> {
        if self.version != GENERATOR_VERSION {
            return Err(GeneratorError::UnsupportedVersion(self.version));
        }
        Ok(())
    }
}

/// Generator failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeneratorError {
    /// The metadata requests an unknown generator algorithm.
    UnsupportedVersion(u32),
    /// A matrix dimension or global scalar index overflowed `u64`.
    IndexOverflow,
    /// The requested scalar generator does not match the metadata.
    DistributionMismatch {
        expected: TestDistribution,
        actual: TestDistribution,
    },
}

impl std::fmt::Display for GeneratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported RSRS test generator version {version}")
            }
            Self::IndexOverflow => write!(f, "RSRS test matrix index overflow"),
            Self::DistributionMismatch { expected, actual } => write!(
                f,
                "RSRS test distribution mismatch: expected {expected:?}, got {actual:?}"
            ),
        }
    }
}

impl std::error::Error for GeneratorError {}

#[inline]
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[inline]
fn word(spec: &GeneratorSpec, counter: u64) -> u64 {
    splitmix64(spec.seed ^ spec.family.stream() ^ counter.wrapping_mul(0xD134_2543_DE82_EF95))
}

#[inline]
fn open_unit(word: u64) -> f64 {
    // The midpoint of one of the 2^53 equal bins in (0, 1), avoiding both
    // logarithm singularities in Box-Muller.
    (((word >> 11) as f64) + 0.5) * (1.0 / ((1_u64 << 53) as f64))
}

#[inline]
fn normal_pair(spec: &GeneratorSpec, pair: u64) -> (f64, f64) {
    let first = pair.wrapping_mul(2);
    let u1 = open_unit(word(spec, first));
    let u2 = open_unit(word(spec, first.wrapping_add(1)));
    let radius = (-2.0 * u1.ln()).sqrt();
    let angle = std::f64::consts::TAU * u2;
    (radius * angle.cos(), radius * angle.sin())
}

#[inline]
fn normal_at(spec: &GeneratorSpec, scalar_index: u64) -> f64 {
    let (first, second) = normal_pair(spec, scalar_index / 2);
    if scalar_index % 2 == 0 {
        first
    } else {
        second
    }
}

fn element_offset(row_offset: usize, rows: usize, dimension: usize) -> Result<u64, GeneratorError> {
    let end_row = row_offset
        .checked_add(rows)
        .ok_or(GeneratorError::IndexOverflow)?;
    let end = end_row
        .checked_mul(dimension)
        .ok_or(GeneratorError::IndexOverflow)?;
    u64::try_from(end).map_err(|_| GeneratorError::IndexOverflow)?;
    let start = row_offset
        .checked_mul(dimension)
        .ok_or(GeneratorError::IndexOverflow)?;
    u64::try_from(start).map_err(|_| GeneratorError::IndexOverflow)
}

/// Generate a row-major block of real Gaussian test rows.
pub fn generate_real(
    spec: &GeneratorSpec,
    row_offset: usize,
    rows: usize,
    dimension: usize,
) -> Result<Vec<f64>, GeneratorError> {
    spec.validate()?;
    if spec.distribution != TestDistribution::RealStandardNormal {
        return Err(GeneratorError::DistributionMismatch {
            expected: TestDistribution::RealStandardNormal,
            actual: spec.distribution,
        });
    }
    let start = element_offset(row_offset, rows, dimension)?;
    let count = rows
        .checked_mul(dimension)
        .ok_or(GeneratorError::IndexOverflow)?;
    Ok((0..count)
        .map(|index| normal_at(spec, start + index as u64))
        .collect())
}

/// Generate a row-major block of circular complex Gaussian test rows.
pub fn generate_complex(
    spec: &GeneratorSpec,
    row_offset: usize,
    rows: usize,
    dimension: usize,
) -> Result<Vec<Complex<f64>>, GeneratorError> {
    spec.validate()?;
    if spec.distribution != TestDistribution::CircularComplexStandardNormal {
        return Err(GeneratorError::DistributionMismatch {
            expected: TestDistribution::CircularComplexStandardNormal,
            actual: spec.distribution,
        });
    }
    let element_start = element_offset(row_offset, rows, dimension)?;
    let count = rows
        .checked_mul(dimension)
        .ok_or(GeneratorError::IndexOverflow)?;
    if count != 0 {
        let final_element = element_start
            .checked_add((count - 1) as u64)
            .ok_or(GeneratorError::IndexOverflow)?;
        final_element
            .checked_mul(2)
            .and_then(|index| index.checked_add(1))
            .ok_or(GeneratorError::IndexOverflow)?;
    }
    let scale = std::f64::consts::FRAC_1_SQRT_2;
    Ok((0..count)
        .map(|index| {
            let element = element_start
                .checked_add(index as u64)
                .expect("element range was checked before generation");
            let real_index = element
                .checked_mul(2)
                .expect("complex scalar range was checked before generation");
            Complex::new(
                normal_at(spec, real_index) * scale,
                normal_at(spec, real_index + 1) * scale,
            )
        })
        .collect())
}

/// Stable digest of a real test block, including its shape and row offset.
pub fn digest_real(row_offset: usize, rows: usize, dimension: usize, values: &[f64]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(&(row_offset as u64).to_le_bytes());
    digest.update(&(rows as u64).to_le_bytes());
    digest.update(&(dimension as u64).to_le_bytes());
    for value in values {
        digest.update(&value.to_bits().to_le_bytes());
    }
    digest.finalize().into()
}

/// Stable digest of a complex test block, including its shape and row offset.
pub fn digest_complex(
    row_offset: usize,
    rows: usize,
    dimension: usize,
    values: &[Complex<f64>],
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(&(row_offset as u64).to_le_bytes());
    digest.update(&(rows as u64).to_le_bytes());
    digest.update(&(dimension as u64).to_le_bytes());
    for value in values {
        digest.update(&value.re.to_bits().to_le_bytes());
        digest.update(&value.im.to_bits().to_le_bytes());
    }
    digest.finalize().into()
}

/// Stable digest of a generator recipe and row interval. Unlike hashing the
/// generated floating-point values, this remains identical across libm/CPU
/// implementations while still detecting a changed seed, stream, shape, or
/// generator version.
pub fn digest_recipe(recipe: &TestMatrixRecipe, row_offset: usize, rows: usize) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"rsrs-test-recipe-v1\0");
    digest.update(recipe.generator.version.to_le_bytes());
    digest.update(recipe.generator.seed.to_le_bytes());
    digest.update([match recipe.generator.family {
        TestFamily::Y => 0,
        TestFamily::Z => 1,
    }]);
    digest.update([match recipe.generator.distribution {
        TestDistribution::RealStandardNormal => 0,
        TestDistribution::CircularComplexStandardNormal => 1,
    }]);
    digest.update((row_offset as u64).to_le_bytes());
    digest.update((rows as u64).to_le_bytes());
    digest.update((recipe.columns as u64).to_le_bytes());
    digest.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_generation_is_chunk_invariant() {
        let spec = GeneratorSpec::new(12345, TestFamily::Y, TestDistribution::RealStandardNormal);
        let all = generate_real(&spec, 0, 13, 17).unwrap();
        let mut chunked = generate_real(&spec, 0, 5, 17).unwrap();
        chunked.extend(generate_real(&spec, 5, 8, 17).unwrap());
        assert_eq!(all, chunked);
    }

    #[test]
    fn complex_generation_is_chunk_invariant_and_streams_are_distinct() {
        let y = GeneratorSpec::new(
            9,
            TestFamily::Y,
            TestDistribution::CircularComplexStandardNormal,
        );
        let z = GeneratorSpec::new(
            9,
            TestFamily::Z,
            TestDistribution::CircularComplexStandardNormal,
        );
        let all = generate_complex(&y, 0, 11, 19).unwrap();
        let mut chunked = generate_complex(&y, 0, 7, 19).unwrap();
        chunked.extend(generate_complex(&y, 7, 4, 19).unwrap());
        assert_eq!(all, chunked);
        assert_ne!(all, generate_complex(&z, 0, 11, 19).unwrap());
    }

    #[test]
    fn circular_complex_normalization_is_correct() {
        let spec = GeneratorSpec::new(
            20261123,
            TestFamily::Y,
            TestDistribution::CircularComplexStandardNormal,
        );
        let values = generate_complex(&spec, 0, 1000, 100).unwrap();
        let count = values.len() as f64;
        let mean = values.iter().copied().sum::<Complex<f64>>() / count;
        let variance = values.iter().map(|value| value.norm_sqr()).sum::<f64>() / count;
        let pseudo_variance = values
            .iter()
            .map(|value| value * value)
            .sum::<Complex<f64>>()
            / count;
        assert!(mean.norm() < 1.0e-2, "mean={mean}");
        assert!((variance - 1.0).abs() < 1.0e-2, "variance={variance}");
        assert!(
            pseudo_variance.norm() < 1.5e-2,
            "pseudo_variance={pseudo_variance}"
        );
    }

    #[test]
    fn unsupported_version_is_rejected() {
        let mut spec = GeneratorSpec::new(0, TestFamily::Y, TestDistribution::RealStandardNormal);
        spec.version += 1;
        assert_eq!(
            generate_real(&spec, 0, 1, 1),
            Err(GeneratorError::UnsupportedVersion(spec.version))
        );
    }

    fn example_metadata() -> SampleCacheMetadata {
        let distribution = TestDistribution::CircularComplexStandardNormal;
        SampleCacheMetadata {
            format_version: FORMAT_VERSION,
            scalar_kind: ScalarKind::Complex64,
            operator_dimension: 101,
            y_test: TestMatrixRecipe {
                rows: 30,
                columns: 101,
                generator: GeneratorSpec::new(11, TestFamily::Y, distribution),
            },
            z_test: Some(TestMatrixRecipe {
                rows: 30,
                columns: 101,
                generator: GeneratorSpec::new(11, TestFamily::Z, distribution),
            }),
            z_mode: Some(ZMode::Transpose),
            operator_fingerprint: [7; 32],
            annotations: BTreeMap::from([("kernel".into(), "helmholtz".into())]),
        }
    }

    #[test]
    fn metadata_round_trips_and_validates() {
        let metadata = example_metadata();
        metadata.validate().unwrap();
        let encoded = serde_json::to_vec(&metadata).unwrap();
        let decoded: SampleCacheMetadata = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(metadata, decoded);
    }

    #[test]
    fn metadata_rejects_semantic_mismatches() {
        let mut metadata = example_metadata();
        metadata.z_test.as_mut().unwrap().generator.family = TestFamily::Y;
        assert_eq!(metadata.validate(), Err(MetadataError::FamilyMismatch));

        let mut metadata = example_metadata();
        metadata.z_test.as_mut().unwrap().rows -= 1;
        assert_eq!(metadata.validate(), Err(MetadataError::RowCountMismatch));

        let mut metadata = example_metadata();
        metadata.scalar_kind = ScalarKind::Real64;
        assert_eq!(
            metadata.validate(),
            Err(MetadataError::DistributionMismatch)
        );
    }

    #[test]
    fn version_two_y_only_metadata_round_trips() {
        let mut metadata = example_metadata();
        metadata.z_test = None;
        metadata.z_mode = None;
        metadata.validate().unwrap();
        let encoded = serde_json::to_vec(&metadata).unwrap();
        let decoded: SampleCacheMetadata = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, metadata);
    }

    #[test]
    fn version_one_remains_readable_but_requires_z() {
        let mut metadata = example_metadata();
        metadata.format_version = LEGACY_FORMAT_VERSION;
        metadata.validate().unwrap();
        metadata.z_test = None;
        metadata.z_mode = None;
        assert_eq!(
            metadata.validate(),
            Err(MetadataError::LegacyFormatRequiresZ)
        );
    }

    fn hex(bytes: [u8; 32]) -> String {
        bytes.iter().map(|value| format!("{value:02x}")).collect()
    }

    #[test]
    fn digests_match_the_cross_language_test_vectors() {
        assert_eq!(
            hex(digest_real(7, 2, 2, &[1.0, 2.0, 3.0, 4.0])),
            "0c96d9ac15d0dac7342ba06c8c07a53b42f9ad0910569b051c67eb657bedf3ca"
        );
        let values = [
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        ];
        assert_eq!(
            hex(digest_complex(7, 2, 2, &values)),
            "e02d00e003cf5a1ae10f32bd395280ba23a6c9d3fb15bb56f8f0192fefd111ec"
        );
        let recipe = TestMatrixRecipe {
            rows: 30,
            columns: 2,
            generator: GeneratorSpec::new(
                19,
                TestFamily::Y,
                TestDistribution::CircularComplexStandardNormal,
            ),
        };
        assert_eq!(
            hex(digest_recipe(&recipe, 7, 2)),
            "3dd7704f1faff3070705689a302b611c716349ebbbbc788edd3fe6a51be61bf3"
        );
    }
}
