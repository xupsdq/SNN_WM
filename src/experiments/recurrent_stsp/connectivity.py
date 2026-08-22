"""Deterministic sparse connectivity generation for the Tiddia network."""

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .config import TiddiaNetworkConfig


PathLike = Union[str, Path]


@dataclass(frozen=True)
class ConnectionBlockRecord:
    """Persisted description of one upstream-equivalent ``Connect`` call."""

    name: str
    plastic: bool
    source_start: int
    source_count: int
    target_start: int
    target_count: int
    indegree: int
    edge_count: int
    weight_mode: str
    facilitated: bool


@dataclass(frozen=True)
class _ConnectionBlock:
    name: str
    plastic: bool
    source_start: int
    source_count: int
    target_start: int
    target_count: int
    indegree: int
    weight_mode: str
    facilitated: bool = False

    @property
    def edge_count(self) -> int:
        return self.target_count * self.indegree

    def record(self) -> ConnectionBlockRecord:
        return ConnectionBlockRecord(
            name=self.name,
            plastic=self.plastic,
            source_start=self.source_start,
            source_count=self.source_count,
            target_start=self.target_start,
            target_count=self.target_count,
            indegree=self.indegree,
            edge_count=self.edge_count,
            weight_mode=self.weight_mode,
            facilitated=self.facilitated,
        )


@dataclass
class StaticCsrEdges:
    """Source-major CSR storage for static recurrent connections."""

    row_ptr: torch.Tensor
    targets: torch.Tensor
    weights: torch.Tensor
    delay_steps: torch.Tensor

    @property
    def num_edges(self) -> int:
        return int(self.targets.numel())

    def to(
        self,
        device: Union[str, torch.device],
        *,
        float_dtype: Optional[torch.dtype] = None,
    ) -> "StaticCsrEdges":
        weight_dtype = self.weights.dtype if float_dtype is None else float_dtype
        return StaticCsrEdges(
            row_ptr=self.row_ptr.to(device=device),
            targets=self.targets.to(device=device),
            weights=self.weights.to(device=device, dtype=weight_dtype),
            delay_steps=self.delay_steps.to(device=device),
        )


@dataclass
class PlasticCsrEdges(StaticCsrEdges):
    """CSR connections plus immutable initial and time-constant STSP data."""

    initial_u: torch.Tensor
    initial_x: torch.Tensor
    tau_rec_ms: torch.Tensor
    tau_fac_ms: torch.Tensor

    def to(
        self,
        device: Union[str, torch.device],
        *,
        float_dtype: Optional[torch.dtype] = None,
    ) -> "PlasticCsrEdges":
        dtype = self.weights.dtype if float_dtype is None else float_dtype
        return PlasticCsrEdges(
            row_ptr=self.row_ptr.to(device=device),
            targets=self.targets.to(device=device),
            weights=self.weights.to(device=device, dtype=dtype),
            delay_steps=self.delay_steps.to(device=device),
            initial_u=self.initial_u.to(device=device, dtype=dtype),
            initial_x=self.initial_x.to(device=device, dtype=dtype),
            tau_rec_ms=self.tau_rec_ms.to(device=device, dtype=dtype),
            tau_fac_ms=self.tau_fac_ms.to(device=device, dtype=dtype),
        )


@dataclass
class SparseRecurrentConnectivity:
    """Complete recurrent graph, split into plastic and static CSR edges."""

    config: TiddiaNetworkConfig
    plastic: PlasticCsrEdges
    static: StaticCsrEdges
    blocks: Tuple[ConnectionBlockRecord, ...]

    SCHEMA_VERSION = 1

    @property
    def num_edges(self) -> int:
        return self.plastic.num_edges + self.static.num_edges

    @property
    def storage_bytes(self) -> int:
        tensors = (
            self.plastic.row_ptr,
            self.plastic.targets,
            self.plastic.weights,
            self.plastic.delay_steps,
            self.plastic.initial_u,
            self.plastic.initial_x,
            self.plastic.tau_rec_ms,
            self.plastic.tau_fac_ms,
            self.static.row_ptr,
            self.static.targets,
            self.static.weights,
            self.static.delay_steps,
        )
        return sum(t.numel() * t.element_size() for t in tensors)

    def validate(self) -> None:
        _validate_csr(self.static, self.config, plastic=False)
        _validate_csr(self.plastic, self.config, plastic=True)
        expected = sum(block.edge_count for block in self.blocks)
        if self.num_edges != expected:
            raise ValueError(
                "Connectivity has {} edges but block manifest declares {}.".format(
                    self.num_edges, expected
                )
            )

    def to(
        self,
        device: Union[str, torch.device],
        *,
        float_dtype: Optional[torch.dtype] = None,
    ) -> "SparseRecurrentConnectivity":
        return SparseRecurrentConnectivity(
            config=self.config,
            plastic=self.plastic.to(device, float_dtype=float_dtype),
            static=self.static.to(device, float_dtype=float_dtype),
            blocks=self.blocks,
        )

    def save(self, path: PathLike) -> Path:
        """Atomically persist the reusable graph artifact on CPU."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "config": asdict(self.config),
            "blocks": [asdict(block) for block in self.blocks],
            "plastic": _edge_payload(self.plastic),
            "static": _edge_payload(self.static),
        }
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=destination.name + ".",
                suffix=".tmp",
                dir=str(destination.parent),
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
            torch.save(payload, temporary_name)
            os.replace(temporary_name, destination)
        finally:
            if temporary_name is not None and os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return destination

    @classmethod
    def load(cls, path: PathLike) -> "SparseRecurrentConnectivity":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("Unsupported recurrent connectivity schema version.")
        config = TiddiaNetworkConfig(**payload["config"])
        plastic_data = payload["plastic"]
        static_data = payload["static"]
        graph = cls(
            config=config,
            plastic=PlasticCsrEdges(**plastic_data),
            static=StaticCsrEdges(**static_data),
            blocks=tuple(ConnectionBlockRecord(**item) for item in payload["blocks"]),
        )
        graph.validate()
        return graph


def _edge_payload(edges: StaticCsrEdges) -> Dict[str, torch.Tensor]:
    payload = {
        "row_ptr": edges.row_ptr.detach().cpu(),
        "targets": edges.targets.detach().cpu(),
        "weights": edges.weights.detach().cpu(),
        "delay_steps": edges.delay_steps.detach().cpu(),
    }
    if isinstance(edges, PlasticCsrEdges):
        payload.update(
            {
                "initial_u": edges.initial_u.detach().cpu(),
                "initial_x": edges.initial_x.detach().cpu(),
                "tau_rec_ms": edges.tau_rec_ms.detach().cpu(),
                "tau_fac_ms": edges.tau_fac_ms.detach().cpu(),
            }
        )
    return payload


def _validate_csr(
    edges: StaticCsrEdges,
    config: TiddiaNetworkConfig,
    *,
    plastic: bool,
) -> None:
    if edges.row_ptr.dtype != torch.int64:
        raise TypeError("CSR row_ptr must use int64.")
    if edges.targets.dtype != torch.int32:
        raise TypeError("CSR targets must use int32 compact storage.")
    if edges.delay_steps.dtype != torch.int16:
        raise TypeError("CSR delay_steps must use int16 compact storage.")
    if edges.row_ptr.shape != (config.n_neurons + 1,):
        raise ValueError("CSR row_ptr has the wrong shape.")
    if int(edges.row_ptr[0].item()) != 0 or int(edges.row_ptr[-1].item()) != edges.num_edges:
        raise ValueError("CSR row_ptr endpoints do not match edge storage.")
    if bool((edges.row_ptr[1:] < edges.row_ptr[:-1]).any().item()):
        raise ValueError("CSR row_ptr must be nondecreasing.")
    for name in ("weights", "delay_steps"):
        if getattr(edges, name).shape != (edges.num_edges,):
            raise ValueError("{} has the wrong edge dimension.".format(name))
    if edges.num_edges:
        if int(edges.targets.min().item()) < 0:
            raise ValueError("Connection targets must be non-negative.")
        if int(edges.targets.max().item()) >= config.n_neurons:
            raise ValueError("Connection target exceeds the neuron count.")
        if int(edges.delay_steps.min().item()) < config.min_delay_steps:
            raise ValueError("Connection delay is below the configured minimum.")
        if int(edges.delay_steps.max().item()) > config.max_delay_steps:
            raise ValueError("Connection delay exceeds the configured maximum.")
        if not bool(torch.isfinite(edges.weights).all().item()):
            raise ValueError("Connection weights must be finite.")
    if plastic:
        assert isinstance(edges, PlasticCsrEdges)
        if int(edges.row_ptr[config.n_exc].item()) != edges.num_edges:
            raise ValueError("Plastic edges must originate only from excitatory neurons.")
        for name in ("initial_u", "initial_x", "tau_rec_ms", "tau_fac_ms"):
            tensor = getattr(edges, name)
            if tensor.shape != (edges.num_edges,):
                raise ValueError("{} has the wrong edge dimension.".format(name))
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError("{} must contain finite values.".format(name))
        if bool(((edges.initial_u < 0) | (edges.initial_u > 1)).any().item()):
            raise ValueError("initial_u must lie in [0, 1].")
        if bool(((edges.initial_x < 0) | (edges.initial_x > 1)).any().item()):
            raise ValueError("initial_x must lie in [0, 1].")
        if bool((edges.tau_rec_ms <= 0).any().item()):
            raise ValueError("tau_rec_ms must be positive.")
        if bool((edges.tau_fac_ms <= 0).any().item()):
            raise ValueError("tau_fac_ms must be positive.")


def psp_to_psc_weight(
    psp_mv: float,
    tau_m_ms: float,
    *,
    capacitance_pf: float = 250.0,
    tau_syn_ms: float = 2.0,
) -> float:
    """Convert a target PSP amplitude to the upstream PSC weight in pA."""

    ratio = tau_m_ms / tau_syn_ms
    conversion = (
        capacitance_pf ** -1
        * tau_m_ms
        * tau_syn_ms
        / (tau_syn_ms - tau_m_ms)
        * (
            ratio ** (-tau_m_ms / (tau_m_ms - tau_syn_ms))
            - ratio ** (-tau_syn_ms / (tau_m_ms - tau_syn_ms))
        )
    ) ** -1
    return conversion * psp_mv


def _append_split_blocks(
    blocks: List[_ConnectionBlock],
    *,
    name: str,
    source_start: int,
    source_count: int,
    target_start: int,
    target_count: int,
    base_indegree: float,
    weight_mode: str,
    facilitated_fraction: float,
) -> None:
    facilitated_indegree = int(base_indegree * facilitated_fraction)
    nonfacilitated_indegree = int(base_indegree * (1.0 - facilitated_fraction))
    if facilitated_indegree:
        blocks.append(
            _ConnectionBlock(
                name=name + "_facilitated",
                plastic=True,
                source_start=source_start,
                source_count=source_count,
                target_start=target_start,
                target_count=target_count,
                indegree=facilitated_indegree,
                weight_mode=weight_mode,
                facilitated=True,
            )
        )
    if nonfacilitated_indegree:
        blocks.append(
            _ConnectionBlock(
                name=name + "_nonfacilitated",
                plastic=True,
                source_start=source_start,
                source_count=source_count,
                target_start=target_start,
                target_count=target_count,
                indegree=nonfacilitated_indegree,
                weight_mode=weight_mode,
                facilitated=False,
            )
        )


def connection_blocks(config: TiddiaNetworkConfig) -> Tuple[ConnectionBlockRecord, ...]:
    """Return the exact block manifest without allocating any edges."""

    return tuple(block.record() for block in _build_blocks(config))


def _build_blocks(config: TiddiaNetworkConfig) -> List[_ConnectionBlock]:
    blocks: List[_ConnectionBlock] = []
    selective_size = config.selective_population_size
    nonselective_start = config.n_memories * selective_size
    nonselective_size = config.nonselective_population_size
    inhibitory_start = config.n_exc
    c = config.connection_probability
    f = config.coding_fraction
    facilitated = config.facilitated_fraction

    for target_memory in range(config.n_memories):
        target_start = target_memory * selective_size
        for source_memory in range(config.n_memories):
            source_start = source_memory * selective_size
            mode = "jp_lognormal" if target_memory == source_memory else "jb_lognormal"
            _append_split_blocks(
                blocks,
                name="sel{}_to_sel{}".format(source_memory, target_memory),
                source_start=source_start,
                source_count=selective_size,
                target_start=target_start,
                target_count=selective_size,
                base_indegree=f * c * config.n_exc,
                weight_mode=mode,
                facilitated_fraction=facilitated,
            )
        baseline_nonselective = (
            (1.0 - config.gamma_0)
            * c
            * (1.0 - f * config.n_memories)
            * config.n_exc
        )
        potentiated_nonselective = (
            config.gamma_0
            * c
            * (1.0 - f * config.n_memories)
            * config.n_exc
        )
        _append_split_blocks(
            blocks,
            name="nonselective_baseline_to_sel{}".format(target_memory),
            source_start=nonselective_start,
            source_count=nonselective_size,
            target_start=target_start,
            target_count=selective_size,
            base_indegree=baseline_nonselective,
            weight_mode="jb_lognormal",
            facilitated_fraction=facilitated,
        )
        _append_split_blocks(
            blocks,
            name="nonselective_potentiated_to_sel{}".format(target_memory),
            source_start=nonselective_start,
            source_count=nonselective_size,
            target_start=target_start,
            target_count=selective_size,
            base_indegree=potentiated_nonselective,
            weight_mode="jp_normal",
            facilitated_fraction=facilitated,
        )
        blocks.append(
            _ConnectionBlock(
                name="inh_to_sel{}".format(target_memory),
                plastic=False,
                source_start=inhibitory_start,
                source_count=config.n_inh,
                target_start=target_start,
                target_count=selective_size,
                indegree=int(c * config.n_inh),
                weight_mode="i_to_e",
            )
        )

    for source_memory in range(config.n_memories):
        _append_split_blocks(
            blocks,
            name="sel{}_to_nonselective".format(source_memory),
            source_start=source_memory * selective_size,
            source_count=selective_size,
            target_start=nonselective_start,
            target_count=nonselective_size,
            base_indegree=f * c * config.n_exc,
            weight_mode="jb_lognormal",
            facilitated_fraction=facilitated,
        )
    baseline_nonselective = (
        (1.0 - config.gamma_0)
        * c
        * (1.0 - f * config.n_memories)
        * config.n_exc
    )
    potentiated_nonselective = (
        config.gamma_0
        * c
        * (1.0 - f * config.n_memories)
        * config.n_exc
    )
    _append_split_blocks(
        blocks,
        name="nonselective_baseline_to_nonselective",
        source_start=nonselective_start,
        source_count=nonselective_size,
        target_start=nonselective_start,
        target_count=nonselective_size,
        base_indegree=baseline_nonselective,
        weight_mode="jb_lognormal",
        facilitated_fraction=facilitated,
    )
    _append_split_blocks(
        blocks,
        name="nonselective_potentiated_to_nonselective",
        source_start=nonselective_start,
        source_count=nonselective_size,
        target_start=nonselective_start,
        target_count=nonselective_size,
        base_indegree=potentiated_nonselective,
        weight_mode="jp_lognormal",
        facilitated_fraction=facilitated,
    )
    blocks.append(
        _ConnectionBlock(
            name="inh_to_nonselective",
            plastic=False,
            source_start=inhibitory_start,
            source_count=config.n_inh,
            target_start=nonselective_start,
            target_count=nonselective_size,
            indegree=int(c * config.n_inh),
            weight_mode="i_to_e",
        )
    )

    for source_memory in range(config.n_memories):
        blocks.append(
            _ConnectionBlock(
                name="sel{}_to_inh".format(source_memory),
                plastic=False,
                source_start=source_memory * selective_size,
                source_count=selective_size,
                target_start=inhibitory_start,
                target_count=config.n_inh,
                indegree=int(f * c * config.n_exc),
                weight_mode="e_to_i_selective",
            )
        )
    blocks.append(
        _ConnectionBlock(
            name="nonselective_to_inh",
            plastic=False,
            source_start=nonselective_start,
            source_count=nonselective_size,
            target_start=inhibitory_start,
            target_count=config.n_inh,
            indegree=int(c * (1.0 - f * config.n_memories) * config.n_exc),
            weight_mode="e_to_i_nonselective",
        )
    )
    blocks.append(
        _ConnectionBlock(
            name="inh_to_inh",
            plastic=False,
            source_start=inhibitory_start,
            source_count=config.n_inh,
            target_start=inhibitory_start,
            target_count=config.n_inh,
            indegree=int(c * config.n_inh),
            weight_mode="i_to_i",
        )
    )
    return blocks


def expected_edge_counts(config: TiddiaNetworkConfig) -> Dict[str, int]:
    blocks = _build_blocks(config)
    plastic = sum(block.edge_count for block in blocks if block.plastic)
    static = sum(block.edge_count for block in blocks if not block.plastic)
    return {"plastic": plastic, "static": static, "total": plastic + static}


def _sample_sources(
    block: _ConnectionBlock,
    config: TiddiaNetworkConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if block.indegree <= 0:
        return np.empty(0, dtype=np.int32)
    overlap_start = max(block.source_start, block.target_start)
    overlap_stop = min(
        block.source_start + block.source_count,
        block.target_start + block.target_count,
    )
    populations_overlap = overlap_start < overlap_stop
    available = block.source_count - (1 if populations_overlap and not config.allow_autapses else 0)
    if not config.allow_multapses and block.indegree > available:
        raise ValueError("Block {} requests more unique sources than available.".format(block.name))

    sampled = np.empty(block.edge_count, dtype=np.int32)
    write_offset = 0
    target_stop = block.target_start + block.target_count
    for chunk_start in range(
        block.target_start, target_stop, config.sampling_target_chunk
    ):
        chunk_stop = min(chunk_start + config.sampling_target_chunk, target_stop)
        target_ids = np.arange(chunk_start, chunk_stop, dtype=np.int32)
        rows = target_ids.size
        if config.allow_multapses:
            selected = rng.integers(
                0, block.source_count, size=(rows, block.indegree), dtype=np.int32
            )
            if populations_overlap and not config.allow_autapses:
                absolute = selected + block.source_start
                invalid = absolute == target_ids[:, None]
                while bool(invalid.any()):
                    selected[invalid] = rng.integers(
                        0, block.source_count, size=int(invalid.sum()), dtype=np.int32
                    )
                    absolute = selected + block.source_start
                    invalid = absolute == target_ids[:, None]
        else:
            keys = rng.random((rows, block.source_count), dtype=np.float32)
            if populations_overlap and not config.allow_autapses:
                local = target_ids - block.source_start
                valid_rows = (local >= 0) & (local < block.source_count)
                keys[np.nonzero(valid_rows)[0], local[valid_rows]] = np.inf
            selected = np.argpartition(
                keys, kth=block.indegree - 1, axis=1
            )[:, : block.indegree].astype(np.int32, copy=False)
        flat = selected.reshape(-1) + block.source_start
        sampled[write_offset : write_offset + flat.size] = flat
        write_offset += flat.size
    return sampled


def _lognormal_parameters(mean: float, std: float) -> Tuple[float, float]:
    sigma = math.sqrt(math.log((std / mean) ** 2 + 1.0))
    mu = math.log(mean) - 0.5 * sigma * sigma
    return mu, sigma


def _weight_mean(block: _ConnectionBlock, config: TiddiaNetworkConfig) -> float:
    common = {
        "capacitance_pf": config.capacitance_pf,
        "tau_syn_ms": config.tau_syn_ms,
    }
    if block.weight_mode.startswith("jb"):
        return psp_to_psc_weight(config.j_baseline_mv, config.tau_m_exc_ms, **common)
    if block.weight_mode.startswith("jp"):
        return psp_to_psc_weight(config.j_potentiated_mv, config.tau_m_exc_ms, **common)
    if block.weight_mode == "i_to_e":
        return psp_to_psc_weight(-config.j_i_to_e_mv, config.tau_m_inh_ms, **common)
    if block.weight_mode == "i_to_i":
        return psp_to_psc_weight(-config.j_i_to_i_mv, config.tau_m_inh_ms, **common)
    if block.weight_mode == "e_to_i_selective":
        return psp_to_psc_weight(config.j_e_to_i_mv, config.tau_m_inh_ms, **common)
    if block.weight_mode == "e_to_i_nonselective":
        return psp_to_psc_weight(config.j_e_to_i_mv, config.tau_m_exc_ms, **common)
    raise ValueError("Unknown weight mode {}.".format(block.weight_mode))


def _sample_weights(
    block: _ConnectionBlock,
    config: TiddiaNetworkConfig,
    rng: np.random.Generator,
) -> Union[np.float32, np.ndarray]:
    mean = _weight_mean(block, config)
    if block.weight_mode.startswith("jb"):
        std_mv = config.j_baseline_lognormal_std_mv
        tau_m = config.tau_m_exc_ms
    elif block.weight_mode.startswith("jp"):
        std_mv = config.j_potentiated_lognormal_std_mv
        tau_m = config.tau_m_exc_ms
    else:
        std_mv = None
        tau_m = config.tau_m_exc_ms
    if std_mv is None:
        return np.float32(mean)
    std = abs(
        psp_to_psc_weight(
            std_mv,
            tau_m,
            capacitance_pf=config.capacitance_pf,
            tau_syn_ms=config.tau_syn_ms,
        )
    )
    if block.weight_mode == "jp_normal":
        return rng.normal(mean, std, size=block.edge_count).astype(np.float32)
    mu, sigma = _lognormal_parameters(abs(mean), std)
    values = rng.lognormal(mu, sigma, size=block.edge_count).astype(np.float32)
    return -values if mean < 0 else values


def _sample_truncated_normal(
    rng: np.random.Generator,
    *,
    mean: float,
    std: float,
    lower: float,
    upper: float,
    size: int,
) -> np.ndarray:
    from scipy.stats import truncnorm

    lower_standard = (lower - mean) / std
    upper_standard = (upper - mean) / std
    return truncnorm.rvs(
        lower_standard,
        upper_standard,
        loc=mean,
        scale=std,
        size=size,
        random_state=rng,
    ).astype(np.float32)


def _sample_delays(
    block: _ConnectionBlock,
    config: TiddiaNetworkConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    raw = rng.uniform(config.delay_min_ms, config.delay_max_ms, block.edge_count)
    steps = np.floor(raw / config.dt_ms + 0.5).astype(np.int16)
    return np.clip(steps, config.min_delay_steps, config.max_delay_steps)


def _block_attributes(
    block: _ConnectionBlock,
    config: TiddiaNetworkConfig,
    rng: np.random.Generator,
) -> Dict[str, Union[np.float32, np.ndarray]]:
    attributes: Dict[str, Union[np.float32, np.ndarray]] = {
        "weights": _sample_weights(block, config, rng),
        "delay_steps": _sample_delays(block, config, rng),
    }
    if not block.plastic:
        return attributes
    if config.stsp_initial_u_truncnorm_std is None:
        initial_u: Union[np.float32, np.ndarray] = np.float32(config.stsp_initial_u)
    else:
        initial_u = _sample_truncated_normal(
            rng,
            mean=config.stsp_initial_u,
            std=config.stsp_initial_u_truncnorm_std,
            lower=0.0,
            upper=1.0,
            size=block.edge_count,
        )
    if config.stsp_initial_x_uniform:
        initial_x: Union[np.float32, np.ndarray] = rng.random(
            block.edge_count, dtype=np.float32
        )
    else:
        initial_x = np.float32(config.stsp_initial_x)

    if block.facilitated:
        if config.stsp_tau_rec_truncnorm_std_ms is None:
            tau_rec: Union[np.float32, np.ndarray] = np.float32(
                config.stsp_tau_rec_ms
            )
        else:
            tau_rec = _sample_truncated_normal(
                rng,
                mean=config.stsp_tau_rec_ms,
                std=config.stsp_tau_rec_truncnorm_std_ms,
                lower=1.0,
                upper=6_000.0,
                size=block.edge_count,
            )
        if config.stsp_tau_fac_truncnorm_std_ms is None:
            tau_fac: Union[np.float32, np.ndarray] = np.float32(
                config.stsp_tau_fac_ms
            )
        else:
            tau_fac = _sample_truncated_normal(
                rng,
                mean=config.stsp_tau_fac_ms,
                std=config.stsp_tau_fac_truncnorm_std_ms,
                lower=1.0,
                upper=6_000.0,
                size=block.edge_count,
            )
    elif config.stsp_tau_rec_truncnorm_std_ms is None:
        tau_rec = np.float32(config.nonfacilitating_tau_ms)
        tau_fac = np.float32(config.nonfacilitating_tau_ms)
    else:
        tau_rec = rng.normal(
            config.nonfacilitating_tau_ms,
            config.nonfacilitating_tau_std_ms,
            block.edge_count,
        ).astype(np.float32)
        tau_fac = rng.normal(
            config.nonfacilitating_tau_ms,
            config.nonfacilitating_tau_std_ms,
            block.edge_count,
        ).astype(np.float32)
    attributes.update(
        {
            "initial_u": initial_u,
            "initial_x": initial_x,
            "tau_rec_ms": tau_rec,
            "tau_fac_ms": tau_fac,
        }
    )
    return attributes


def _allocate_edge_arrays(
    edge_count: int,
    *,
    plastic: bool,
) -> Dict[str, np.ndarray]:
    arrays = {
        "targets": np.empty(edge_count, dtype=np.int32),
        "weights": np.empty(edge_count, dtype=np.float32),
        "delay_steps": np.empty(edge_count, dtype=np.int16),
    }
    if plastic:
        arrays.update(
            {
                "initial_u": np.empty(edge_count, dtype=np.float32),
                "initial_x": np.empty(edge_count, dtype=np.float32),
                "tau_rec_ms": np.empty(edge_count, dtype=np.float32),
                "tau_fac_ms": np.empty(edge_count, dtype=np.float32),
            }
        )
    return arrays


def _assemble_csr(
    sampled_blocks: Sequence[Tuple[int, _ConnectionBlock, np.ndarray]],
    config: TiddiaNetworkConfig,
    *,
    plastic: bool,
) -> Union[StaticCsrEdges, PlasticCsrEdges]:
    selected = [item for item in sampled_blocks if item[1].plastic == plastic]
    source_counts = np.zeros(config.n_neurons, dtype=np.int64)
    for _, _, sources in selected:
        source_counts += np.bincount(sources, minlength=config.n_neurons)
    row_ptr = np.empty(config.n_neurons + 1, dtype=np.int64)
    row_ptr[0] = 0
    np.cumsum(source_counts, out=row_ptr[1:])
    arrays = _allocate_edge_arrays(int(row_ptr[-1]), plastic=plastic)
    cursor = np.zeros(config.n_neurons, dtype=np.int64)

    for block_index, block, sources in selected:
        target_ids = np.repeat(
            np.arange(
                block.target_start,
                block.target_start + block.target_count,
                dtype=np.int32,
            ),
            block.indegree,
        )
        order = np.argsort(sources, kind="stable")
        sorted_sources = sources[order]
        counts = np.bincount(sorted_sources, minlength=config.n_neurons)
        nonzero = counts > 0
        local_starts = np.cumsum(counts) - counts
        within_source = np.arange(block.edge_count, dtype=np.int64) - np.repeat(
            local_starts[nonzero], counts[nonzero]
        )
        destination = (
            row_ptr[sorted_sources]
            + cursor[sorted_sources]
            + within_source
        )
        arrays["targets"][destination] = target_ids[order]

        rng = np.random.default_rng(
            np.random.SeedSequence([config.seed, block_index, 1])
        )
        attributes = _block_attributes(block, config, rng)
        for name, values in attributes.items():
            if np.ndim(values) == 0:
                arrays[name][destination] = values
            else:
                arrays[name][destination] = values[order]
        cursor += counts

    if not np.array_equal(cursor, source_counts):
        raise RuntimeError("CSR assembly did not consume every sampled edge.")
    common = {
        "row_ptr": torch.from_numpy(row_ptr),
        "targets": torch.from_numpy(arrays["targets"]),
        "weights": torch.from_numpy(arrays["weights"]),
        "delay_steps": torch.from_numpy(arrays["delay_steps"]),
    }
    if not plastic:
        return StaticCsrEdges(**common)
    return PlasticCsrEdges(
        **common,
        initial_u=torch.from_numpy(arrays["initial_u"]),
        initial_x=torch.from_numpy(arrays["initial_x"]),
        tau_rec_ms=torch.from_numpy(arrays["tau_rec_ms"]),
        tau_fac_ms=torch.from_numpy(arrays["tau_fac_ms"]),
    )


def generate_sparse_connectivity(
    config: Optional[TiddiaNetworkConfig] = None,
) -> SparseRecurrentConnectivity:
    """Generate the complete source-major recurrent graph on CPU."""

    resolved = TiddiaNetworkConfig() if config is None else config
    blocks = _build_blocks(resolved)
    sampled: List[Tuple[int, _ConnectionBlock, np.ndarray]] = []
    for block_index, block in enumerate(blocks):
        rng = np.random.default_rng(
            np.random.SeedSequence([resolved.seed, block_index, 0])
        )
        sampled.append((block_index, block, _sample_sources(block, resolved, rng)))

    plastic = _assemble_csr(sampled, resolved, plastic=True)
    static = _assemble_csr(sampled, resolved, plastic=False)
    assert isinstance(plastic, PlasticCsrEdges)
    assert isinstance(static, StaticCsrEdges)
    graph = SparseRecurrentConnectivity(
        config=resolved,
        plastic=plastic,
        static=static,
        blocks=tuple(block.record() for block in blocks),
    )
    graph.validate()
    return graph
