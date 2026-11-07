//! Sample acquisition contracts shared by native and Python-driven RSRS runs.

use std::{error::Error, fmt};

/// Identifies one RSRS test/sketch family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFamily {
    Y,
    Z,
}
/// Additional rows requested by a paused precomputed-sample factorisation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleRequest {
    pub y_rows: usize,
    pub z_rows: usize,
    pub total_required: usize,
    pub level: usize,
    pub limiting_box: Option<usize>,
}

impl SampleRequest {
    pub fn is_empty(&self) -> bool {
        self.y_rows == 0 && self.z_rows == 0
    }
}

/// Structured error returned instead of sampling through a Python callback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsufficientSamples {
    pub request: SampleRequest,
    pub y_rows_available: usize,
    pub z_rows_available: Option<usize>,
}

impl fmt::Display for InsufficientSamples {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "RSRS needs {} additional y rows and {} additional z rows at level {} ({} total required; {} y and {:?} z available)",
            self.request.y_rows,
            self.request.z_rows,
            self.request.level,
            self.request.total_required,
            self.y_rows_available,
            self.z_rows_available,
        )
    }
}

impl Error for InsufficientSamples {}

/// Validate current availability and construct a resumable sample request.
pub fn require_sample_rows(
    target: usize,
    y_rows: usize,
    z_rows: Option<usize>,
    level: usize,
    limiting_box: Option<usize>,
) -> Result<(), InsufficientSamples> {
    let y_missing = target.saturating_sub(y_rows);
    let z_missing = z_rows.map_or(0, |rows| target.saturating_sub(rows));
    if y_missing == 0 && z_missing == 0 {
        return Ok(());
    }
    Err(InsufficientSamples {
        request: SampleRequest {
            y_rows: y_missing,
            z_rows: z_missing,
            total_required: target,
            level,
            limiting_box,
        },
        y_rows_available: y_rows,
        z_rows_available: z_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_each_missing_family_independently() {
        let error = require_sample_rows(100, 80, Some(95), 7, Some(12)).unwrap_err();
        assert_eq!(error.request.y_rows, 20);
        assert_eq!(error.request.z_rows, 5);
        assert_eq!(error.request.level, 7);
        assert_eq!(error.request.limiting_box, Some(12));
    }

    #[test]
    fn symmetric_availability_does_not_request_z_rows() {
        assert!(require_sample_rows(100, 100, None, 5, None).is_ok());
    }
}
