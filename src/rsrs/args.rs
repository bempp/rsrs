use crate::{
    rsrs::{
        rsrs_factors::{
            null_and_extract::{ExtractOptions, IdOptions, PivotMethod},
            rsrs_operator::FactType,
        },
        sketch::Shift,
    },
    utils::linear_algebra::{BlockExtractionMethod, NullMethod},
};
use rlst::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::Write;

type Real<T> = <T as rlst::RlstScalar>::Real;

#[derive(Debug, Clone, Deserialize)]
pub enum RankPicking {
    Min,
    DoubleMin,
    Max,
    Avg,
    Mid,
    Tol,
}

#[derive(Debug, Clone)]
pub struct SketchingOptions {
    pub oversampling: usize,
    pub oversampling_diag_blocks: usize,
    pub initial_num_samples: usize,
    pub min_num_samples: usize,
    pub shift: Shift,
    pub save_samples: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum Symmetry {
    NoSymm,
    Symmetric,
    Hermitian,
}

impl Symmetry {
    /// Returns true if the factor is transposed and no it it isn't.
    /// Conjugations of the factor are not implemented for simplicity.
    pub fn symm_val(&self) -> bool {
        match self {
            Symmetry::NoSymm => false,
            Symmetry::Symmetric => true,
            Symmetry::Hermitian => true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RsrsOptions<Item: RlstScalar> {
    pub fact_type: FactType,
    pub sketching: SketchingOptions,
    pub id_options: IdOptions<Item>,
    pub lu_options: ExtractOptions<Item>,
    pub extract_db_options: ExtractOptions<Item>,
    pub min_rank: usize,
    pub symmetry: Symmetry,
    pub min_level: usize,
    pub rank_picking: RankPicking,
    pub num_threads: usize,
    pub flush_factors: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(bound = "Real<Item>: Deserialize<'de>")]
pub struct RsrsArgs<Item: RlstScalar> {
    oversampling: usize,
    oversampling_diag_blocks: usize,
    min_num_samples: usize,
    initial_num_samples: usize,
    shift: Shift,
    null_method: NullMethod,
    qr_method: RankRevealingQrType<Real<Item>>,
    near_block_extraction_method: BlockExtractionMethod,
    diag_block_extraction_method: BlockExtractionMethod,
    lu_pivot_method: PivotMethod,
    diag_pivot_method: PivotMethod,
    tol_null: Real<Item>,
    tol_id: Real<Item>,
    tol_ext_near: Real<Item>,
    tol_diag_ext: Real<Item>,
    min_rank: usize,
    min_level: usize,
    symmetry: Symmetry,
    rank_picking: RankPicking,
    fact_type: FactType,
    save_samples: bool,
    num_threads: usize,
    flush_factors: bool,
    store_far: bool,
}

impl<Item> RsrsArgs<Item>
where
    Item: RlstScalar,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        oversampling: usize,
        oversampling_diag_blocks: usize,
        min_num_samples: usize,
        initial_num_samples: usize,
        shift: Shift,
        null_method: NullMethod,
        qr_method: RankRevealingQrType<Real<Item>>,
        near_block_extraction_method: BlockExtractionMethod,
        diag_block_extraction_method: BlockExtractionMethod,
        lu_pivot_method: PivotMethod,
        diag_pivot_method: PivotMethod,
        tol_null: Real<Item>,
        tol_id: Real<Item>,
        tol_ext_near: Real<Item>,
        tol_diag_ext: Real<Item>,
        min_rank: usize,
        min_level: usize,
        symmetry: Symmetry,
        rank_picking: RankPicking,
        fact_type: FactType,
        save_samples: bool,
        num_threads: usize,
        flush_factors: bool,
        store_far: bool,
    ) -> Self {
        Self {
            oversampling,
            oversampling_diag_blocks,
            min_num_samples,
            initial_num_samples,
            shift,
            null_method,
            qr_method,
            near_block_extraction_method,
            diag_block_extraction_method,
            lu_pivot_method,
            diag_pivot_method,
            tol_null,
            tol_id,
            tol_ext_near,
            tol_diag_ext,
            min_rank,
            min_level,
            symmetry,
            rank_picking,
            fact_type,
            save_samples,
            num_threads,
            flush_factors,
            store_far,
        }
    }
}

impl<Item: RlstScalar + std::fmt::Display> RsrsOptions<Item> {
    pub fn new(args: Option<RsrsArgs<Item>>) -> Self {
        let args = match args {
            Some(input) => input,
            None => RsrsArgs::new(
                8,
                16,
                0,
                420,
                Shift::False,
                NullMethod::Projection,
                RankRevealingQrType::RRQR,
                BlockExtractionMethod::LuLstSq,
                BlockExtractionMethod::LuLstSq,
                PivotMethod::Lu(1e-10),
                PivotMethod::Lu(0.0),
                Item::real(1e-10),
                Item::real(1e-2),
                Item::real(1e-10),
                Item::real(1e-10),
                4,
                1,
                Symmetry::NoSymm,
                RankPicking::Min,
                FactType::Joint,
                false,
                num_cpus::get(),
                false,
                true,
            ),
        };

        let min_rank = if args.tol_id > num::One::one() {
            let k = num::ToPrimitive::to_usize(&args.tol_id).unwrap();
            println!("For tolerances > 1, ID will use this as a fixed rank instead. This fixed rank is: {k}");

            if k <= args.min_rank {
                k
            } else {
                args.min_rank
            }
        } else {
            args.min_rank
        };

        Self {
            sketching: SketchingOptions {
                oversampling: args.oversampling,
                oversampling_diag_blocks: args.oversampling_diag_blocks,
                min_num_samples: args.min_num_samples,
                initial_num_samples: args.initial_num_samples,
                shift: args.shift,
                save_samples: args.save_samples,
            },
            id_options: IdOptions {
                null_method: args.null_method,
                qr_method: args.qr_method,
                tol_null: args.tol_null,
                tol_id: args.tol_id,
                store_far: args.store_far,
            },
            lu_options: ExtractOptions {
                block_extraction_method: args.near_block_extraction_method,
                pivot_method: args.lu_pivot_method,
                tol_lstsq: args.tol_ext_near,
            },
            extract_db_options: ExtractOptions {
                block_extraction_method: args.diag_block_extraction_method,
                pivot_method: args.diag_pivot_method,
                tol_lstsq: args.tol_diag_ext,
            },
            fact_type: args.fact_type,
            min_rank,
            min_level: args.min_level,
            symmetry: args.symmetry,
            rank_picking: args.rank_picking,
            num_threads: args.num_threads,
            flush_factors: args.flush_factors,
        }
    }

    pub fn to_identifier(&self) -> String {
        let mut id = String::from("rsrs");

        write!(
            &mut id,
            "_null_{:?}_toln_{:e}",
            self.id_options.null_method, self.id_options.tol_null,
        )
        .unwrap();

        match self.sketching.shift {
            Shift::True(alpha) => write!(
                &mut id,
                "_os_{os}_osdiag_{osdiag}_initsam_{init}_shiftd_{alpha:.e}",
                os = self.sketching.oversampling,
                osdiag = self.sketching.oversampling_diag_blocks,
                init = self.sketching.initial_num_samples,
                alpha = alpha
            )
            .unwrap(),
            Shift::False => write!(
                &mut id,
                "_os_{os}_osdiag_{osdiag}_initsam_{init}",
                os = self.sketching.oversampling,
                osdiag = self.sketching.oversampling_diag_blocks,
                init = self.sketching.initial_num_samples
            )
            .unwrap(),
        };

        match self.id_options.qr_method{
            RankRevealingQrType::RRQR => write!(
            &mut id,
            "_mrnk_{}_mlvl_{}_{:?}_rpick_{:?}_next_{:?}_tolextn_{:e}_db_ext_{:?}_tol_lstsq_{:e}_rrqr",
            self.min_rank,
            self.min_level,
            self.symmetry,
            self.rank_picking,
            self.lu_options.block_extraction_method,
            self.lu_options.tol_lstsq,
            self.extract_db_options.block_extraction_method,
            self.extract_db_options.tol_lstsq
        )
        .unwrap(),
            RankRevealingQrType::SRRQR(f) => write!(
            &mut id,
            "_mrnk_{}_mlvl_{}_{:?}_rpick_{:?}_next_{:?}_tolextn_{:e}_db_ext_{:?}_tol_lstsq_{:e}_srrqr_{:e}",
            self.min_rank,
            self.min_level,
            self.symmetry,
            self.rank_picking,
            self.lu_options.block_extraction_method,
            self.lu_options.tol_lstsq,
            self.extract_db_options.block_extraction_method,
            self.extract_db_options.tol_lstsq,
            f
        )
        .unwrap(),
        };

        id
    }
}
