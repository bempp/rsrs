"""Versioned configuration shared by Python, rsrs_config, and Rust."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ThreadOptions:
    factorization_threads: int = 1
    apply_threads: int = 1

    def __post_init__(self) -> None:
        if self.factorization_threads < 1 or self.apply_threads < 1:
            raise ValueError("thread counts must be positive")


@dataclass(frozen=True)
class TreeOptions:
    max_tree_depth: int = 6
    leaf_size: int | None = None

    def __post_init__(self) -> None:
        if self.max_tree_depth < 1:
            raise ValueError("max_tree_depth must be positive")
        if self.leaf_size is not None and self.leaf_size < 1:
            raise ValueError("leaf_size must be positive when provided")


@dataclass(frozen=True)
class RsrsOptions:
    schema_version: int = SCHEMA_VERSION
    oversampling: int = 8
    oversampling_diag_blocks: int = 16
    min_num_samples: int = 0
    initial_num_samples: int = 0
    fixed_rank_sampling_mode: str = "PerLevel"
    nonsymmetric_id_combination: str = "Concat"
    run_seed: int | None = None
    shift: Any = field(default_factory=lambda: {"type": "False"})
    null_method: str = "Projection"
    qr_method: Any = "RRQR"
    near_block_extraction_method: str = "LuLstSq"
    diag_block_extraction_method: str = "LuLstSq"
    lu_pivot_method: Any = field(default_factory=lambda: {"type": "LuHybrid", "value": 0})
    diag_pivot_method: Any = field(default_factory=lambda: {"type": "LuHybrid", "value": 0})
    tol_null: float = 1e-16
    tol_id: float = 1e-2
    tol_ext_near: float = 1e-16
    tol_diag_ext: float = 1e-16
    min_rank: int = 1
    min_level: int = 1
    symmetry: str = "NoSymm"
    rank_picking: str = "Min"
    fact_type: str = "Joint"
    save_samples: bool = False
    load_samples: bool = True
    sample_storage_dir: str | None = None
    flush_factors: bool = False
    store_far: bool = False
    tree: TreeOptions = field(default_factory=TreeOptions)
    threads: ThreadOptions = field(default_factory=ThreadOptions)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        scenario: Mapping[str, Any] | None = None,
    ) -> "RsrsOptions":
        data = dict(values)
        schema_version = int(data.pop("schema_version", SCHEMA_VERSION))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported RSRS schema version {schema_version}")

        legacy_threads = int(data.pop("num_threads", 1))
        thread_data = dict(data.pop("threads", {}))
        factorization_threads = int(
            data.pop(
                "factorization_threads",
                thread_data.pop("factorization_threads", legacy_threads),
            )
        )
        apply_threads = int(
            data.pop("apply_threads", thread_data.pop("apply_threads", legacy_threads))
        )
        if thread_data:
            raise ValueError(f"unknown thread options: {sorted(thread_data)}")

        scenario_data = dict(scenario or {})
        tree_data = dict(data.pop("tree", {}))
        max_tree_depth = int(
            data.pop(
                "max_tree_depth",
                tree_data.pop("max_tree_depth", scenario_data.get("max_tree_depth", 6)),
            )
        )
        leaf_size = data.pop(
            "leaf_size",
            tree_data.pop("leaf_size", scenario_data.get("leaf_size")),
        )
        if tree_data:
            raise ValueError(f"unknown tree options: {sorted(tree_data)}")

        data.pop("symmetric", None)
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown RSRS options: {unknown}")

        return cls(
            schema_version=schema_version,
            tree=TreeOptions(max_tree_depth=max_tree_depth, leaf_size=leaf_size),
            threads=ThreadOptions(factorization_threads, apply_threads),
            **data,
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

