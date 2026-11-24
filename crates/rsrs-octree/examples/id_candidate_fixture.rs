//! Apply modern RLST RRQR/ID to an exact row-major candidate fixture.

extern crate blas_src;
extern crate lapack_src;
#[cfg(target_os = "linux")]
extern crate openblas_src;

use rlst::dense::array::DynArray;
use rsrs_core::interpolative_decomposition::{compute_candidate_id, IdMethod, IdRankSelection};

fn main() {
    let path = std::env::args().nth(1).expect("candidate fixture path");
    let text = std::fs::read_to_string(path).unwrap();
    let mut fields = text.split_whitespace();
    let rows: usize = fields.next().unwrap().parse().unwrap();
    let columns: usize = fields.next().unwrap().parse().unwrap();
    let mut candidate = DynArray::<f64, 2>::from_shape([rows, columns]);
    for row in 0..rows {
        for column in 0..columns {
            let re: f64 = fields.next().unwrap().parse().unwrap();
            let im: f64 = fields.next().unwrap().parse().unwrap();
            assert_eq!(im, 0.0);
            candidate[[row, column]] = re;
        }
    }
    let id = compute_candidate_id(
        &candidate,
        IdRankSelection::Fixed(16),
        IdMethod::Rrqr,
        false,
    )
    .unwrap();
    let mut permutation_hash = 0xcbf2_9ce4_8422_2325u64;
    for &index in &id.perm {
        permutation_hash = (permutation_hash ^ index as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    let [id_rows, id_columns] = id.id_mat.shape();
    let mut sum = 0.0;
    let mut weighted = 0.0;
    for row in 0..id_rows {
        for column in 0..id_columns {
            let value = id.id_mat[[row, column]];
            sum += value;
            weighted += value * (1 + row * id_columns + column) as f64;
        }
    }
    println!(
        "rank={} permutation_hash={permutation_hash:016x} interpolation_sum={sum:.16e} interpolation_weighted={weighted:.16e}",
        id.rank
    );
}
