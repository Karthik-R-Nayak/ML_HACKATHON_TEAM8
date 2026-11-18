from dataclasses import dataclass, field


@dataclass(frozen=True)
class CandidateConfig:
    radius_m: float = 100.0      # candidate search radius
    max_candidates: int = 30     # keep the K nearest arcs within radius
    cell_m: float = 100.0        # spatial-index grid cell size


@dataclass(frozen=True)
class FeatureConfig:
    # 'central': use previous+next fix (offline inference). 'backward': previous fix only (causal).
    heading_mode: str = "central"
    min_move_m: float = 3.0      # below this displacement GPS heading is undefined (NaN)


@dataclass(frozen=True)
class AlignConfig:
    search_radius_m: float = 150.0   # route arcs farther than this are not label states
    sigma_m: float = 20.0            # emission noise scale
    back_tol_m: float = 30.0         # allowed backwards progress on the same arc
    tol_abs_m: float = 30.0          # free slack on |progress - gps displacement|
    tol_rel: float = 0.3
    lam_per_m: float = 0.02          # transition penalty per metre beyond slack
    skip_cap_m: float = 500.0        # cap so long gaps/skipped arcs stay feasible


@dataclass(frozen=True)
class IOConfig:
    node_id_base: int = 0            # node id of first row in .nodes
    arc_id_base: int = 0             # arc id of first row in .arcs (route ids refer to this)
    time_unit: str = "auto"          # 'auto' | 's' | 'ms'


@dataclass(frozen=True)
class PipelineConfig:
    io: IOConfig = field(default_factory=IOConfig)
    cand: CandidateConfig = field(default_factory=CandidateConfig)
    feat: FeatureConfig = field(default_factory=FeatureConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
