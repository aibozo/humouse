"""PyTorch dataset utilities built atop event loaders and gesture segmentation."""
from __future__ import annotations

import hashlib
import logging
import math
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from . import (
    load_attentive_cursor,
    load_balabit,
    load_bogazici,
    load_local_mouse,
    load_sapimouse,
    segment_event_stream,
)
from .common import NormalisedEvent
from .segmenter import GestureSequence
from features import (
    compute_features as compute_neuromotor_features,
    sigma_lognormal_features,
    sigma_lognormal_features_from_sequence,
)
from train.replay_buffer import ReplayBuffer

LoaderFn = Callable[..., Iterable]

GEOMETRY_FEATURE_DIM = 5
NUM_DIRECTION_BUCKETS = 8

_DATASET_LOADERS: dict[str, LoaderFn] = {
    "balabit": load_balabit,
    "bogazici": load_bogazici,
    "local_mouse": load_local_mouse,
    "sapimouse": load_sapimouse,
    "attentive_cursor": load_attentive_cursor,
}

_STATS_VERSION = 1


def _estimate_sampling_rate(
    events: Sequence[NormalisedEvent],
    *,
    max_samples: int = 50000,
) -> Optional[float]:
    dt_values: list[float] = []
    prev: Optional[NormalisedEvent] = None
    for event in events:
        if prev is not None:
            if (
                event.user_id == prev.user_id
                and event.session_id == prev.session_id
                and (event.split or "") == (prev.split or "")
            ):
                dt_ms = max(float(event.timestamp_ms - prev.timestamp_ms), 0.0)
                if dt_ms > 0.0:
                    dt_values.append(dt_ms / 1000.0)
                    if len(dt_values) >= max_samples:
                        break
        prev = event

    if not dt_values:
        return None

    median_dt = float(np.median(dt_values))
    if median_dt <= 0.0:
        return None
    return 1.0 / median_dt


def _direction_bucket(dx: float, dy: float, *, epsilon: float = 1e-6) -> Optional[int]:
    magnitude = math.hypot(dx, dy)
    if magnitude <= epsilon:
        return None
    angle = math.atan2(dy, dx)
    bucket = int(((angle + math.pi) / (2 * math.pi)) * NUM_DIRECTION_BUCKETS) % NUM_DIRECTION_BUCKETS
    return bucket


def _bucket_center_angle(bucket: int) -> float:
    width = 2 * math.pi / NUM_DIRECTION_BUCKETS
    return -math.pi + width * (bucket + 0.5)


@dataclass
class GestureDatasetConfig:
    dataset_id: str
    sequence_length: int = 64
    max_gestures: Optional[int] = 2000
    sampling_rate: Optional[float | str] = None
    min_events: int = 5
    split: Optional[str] = "train"
    user_filter: Optional[list[str]] = None
    use_generated_negatives: bool = True
    negative_ratio: float = 1.0
    cache_enabled: bool = True
    cache_dir: str = "data/processed"
    replay_path: Optional[str] = None
    replay_sample_ratio: float = 1.0
    normalize_sequences: bool = True
    normalize_features: bool = True
    feature_mode: str = "neuromotor"
    canonicalize_path: bool = False
    canonicalize_duration: bool = False
    include_goal_geometry: bool = False
    use_click_boundaries: bool = False
    click_button: Optional[str] = "left"
    direction_buckets: Optional[list[int]] = None
    rotate_to_buckets: bool = False
    min_path_length: float = 0.0
    min_path_length: float = 0.0
    feature_reservoir_size: Optional[int] = None


def _goal_geometry_features(
    sequence: np.ndarray,
    mask: np.ndarray,
    dataset_id: str,
    metadata: dict,
    fallback_distance: float,
) -> np.ndarray:
    GEOMETRY_FEATURE_DIM = 5
    valid_len = int(mask.sum()) if mask.size else sequence.shape[0]
    if valid_len <= 0:
        return np.zeros(5, dtype=np.float32)

    seq_slice = sequence[:valid_len]
    dx = seq_slice[:, 0]
    dy = seq_slice[:, 1]
    total_dx = float(np.sum(dx))
    total_dy = float(np.sum(dy))
    distance = float(np.hypot(total_dx, total_dy))
    if distance <= 1e-6:
        cos_theta = 1.0
        sin_theta = 0.0
        distance = float(max(distance, fallback_distance))
    else:
        angle = float(np.arctan2(total_dy, total_dx))
        cos_theta = float(np.cos(angle))
        sin_theta = float(np.sin(angle))
    distance = float(np.log1p(max(distance, 0.0)))

    extent = None
    metadata = metadata or {}
    for key in ("target_width", "target_diameter", "target_radius", "target_size", "target_height"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            extent = float(value)
            break
        except (TypeError, ValueError):
            continue
    if extent is None or extent <= 0.0:
        extent = float(fallback_distance)
    extent = float(np.log1p(max(extent, 0.0)))

    style_field = metadata.get("input_device") or metadata.get("device") or metadata.get("pointer_type") or metadata.get("cursor_style")
    if style_field is not None:
        style_str = str(style_field).lower()
        if any(token in style_str for token in ("touchpad", "trackpad")):
            style_value = 1.0
        elif any(token in style_str for token in ("pen", "tablet")):
            style_value = 0.5
        else:
            style_value = 0.0
    else:
        default_styles = {
            "balabit": 0.0,
            "bogazici": 0.25,
            "sapimouse": 0.5,
            "attentive_cursor": 0.75,
        }
        style_value = float(default_styles.get(dataset_id, 0.0))

    return np.array([cos_theta, sin_theta, distance, extent, style_value], dtype=np.float32)


def _canonicalize_sequence(
    sequence: np.ndarray,
    *,
    unit_path: bool,
    unit_time: bool,
) -> np.ndarray:
    seq = sequence.astype(np.float32, copy=True)
    if unit_path:
        dx = seq[:, 0]
        dy = seq[:, 1]
        path = float(np.sum(np.sqrt(dx**2 + dy**2)))
        if path > 1e-8:
            seq[:, 0] = dx / path
            seq[:, 1] = dy / path
    if unit_time:
        dt = seq[:, 2]
        total_time = float(np.sum(dt))
        if total_time > 1e-8:
            seq[:, 2] = dt / total_time
    return seq


class GestureDataset(Dataset):
    """In-memory dataset of gesture sequences + neuromotor features."""

    def __init__(self, config: GestureDatasetConfig):
        if config.dataset_id not in _DATASET_LOADERS:
            raise ValueError(f"Unsupported dataset_id: {config.dataset_id}")
        self.config = config
        self._geometry_dim: int = GEOMETRY_FEATURE_DIM if config.include_goal_geometry else 0
        self._positive_sequences: List[np.ndarray] = []
        self._positive_features: List[np.ndarray] = []
        self._positive_features_tensor: Optional[torch.Tensor] = None
        self._feature_reservoir_tensor: Optional[torch.Tensor] = None
        self._conditioning_features_tensor: Optional[torch.Tensor] = None
        self._positive_feature_cache: Dict[str, int] = {}
        self._shared_positive_sequences: Optional[torch.Tensor] = None
        self._shared_positive_features: Optional[torch.Tensor] = None
        self._sequence_mean: Optional[torch.Tensor] = None
        self._sequence_std: Optional[torch.Tensor] = None
        self._feature_mean: Optional[torch.Tensor] = None
        self._feature_std: Optional[torch.Tensor] = None
        self._effective_sampling_rate: Optional[float] = None
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._build_samples()

    def _extract_features_from_gesture(self, gesture: GestureSequence) -> np.ndarray:
        if self.config.feature_mode == "neuromotor":
            return compute_neuromotor_features(gesture)
        if self.config.feature_mode == "sigma_lognormal":
            return sigma_lognormal_features(gesture)
        raise ValueError(f"Unsupported feature_mode: {self.config.feature_mode}")

    def _build_samples(self) -> None:
        if self._load_from_cache():
            logger = logging.getLogger(__name__)
            logger.debug(
                "Loaded cached gesture dataset for %s (seq_len=%d)",
                self.config.dataset_id,
                self.config.sequence_length,
            )
        else:
            loader = _DATASET_LOADERS[self.config.dataset_id]
            split = self.config.split or "train"
            if self.config.dataset_id == "attentive_cursor":
                raw_events = list(loader())
            elif self.config.dataset_id in {"bogazici", "balabit"}:
                raw_events = list(loader(split))
            else:
                raw_events = list(loader())

            if not raw_events:
                self._positive_sequences = []
                self._positive_features = []
                self._save_to_cache()
                self._compute_statistics()
                return

            sampling_rate_value: Optional[float]
            sr_cfg = self.config.sampling_rate
            if isinstance(sr_cfg, str):
                if sr_cfg.lower() == "auto":
                    sampling_rate_value = _estimate_sampling_rate(raw_events)
                else:
                    try:
                        sampling_rate_value = float(sr_cfg)
                    except ValueError:
                        sampling_rate_value = None
            elif sr_cfg is None:
                sampling_rate_value = _estimate_sampling_rate(raw_events)
            else:
                sampling_rate_value = float(sr_cfg)

            if sampling_rate_value is not None and sampling_rate_value <= 0.0:
                sampling_rate_value = None

            self._effective_sampling_rate = sampling_rate_value

            count = 0
            positive_sequences: List[np.ndarray] = []
            positive_features: List[np.ndarray] = []
            allowed_users = set(self.config.user_filter) if self.config.user_filter else None
            target_buckets = set(self.config.direction_buckets or [])
            for gesture in segment_event_stream(
                raw_events,
                target_len=self.config.sequence_length,
                sampling_rate=self._effective_sampling_rate,
                min_events=self.config.min_events,
                use_click_boundaries=self.config.use_click_boundaries,
                click_button=self.config.click_button,
            ):
                if self.config.split and gesture.split and gesture.split != self.config.split:
                    continue
                if allowed_users and gesture.user_id not in allowed_users:
                    continue
                base_sequence = gesture.sequence.astype(np.float32)
                total_dx = float(np.sum(base_sequence[:, 0]))
                total_dy = float(np.sum(base_sequence[:, 1]))
                bucket = _direction_bucket(total_dx, total_dy)
                raw_variants: list[tuple[np.ndarray, float, dict]] = []

                if target_buckets:
                    if bucket is not None and bucket in target_buckets:
                        base_metadata = dict(gesture.metadata or {})
                        base_metadata.setdefault("start_xy", (0.0, 0.0))
                        base_metadata["end_xy"] = (total_dx, total_dy)
                        base_path_length = float(np.sum(np.linalg.norm(base_sequence[:, :2], axis=1)))
                        raw_variants.append((base_sequence, base_path_length, base_metadata))
                    elif (
                        self.config.rotate_to_buckets
                        and bucket is not None
                        and math.hypot(total_dx, total_dy) > 1e-6
                    ):
                        base_angle = math.atan2(total_dy, total_dx)
                        for target in target_buckets:
                            if target == bucket:
                                continue
                            angle_delta = _bucket_center_angle(target) - base_angle
                            cos_a = math.cos(angle_delta)
                            sin_a = math.sin(angle_delta)
                            rotated = base_sequence.copy()
                            dx = rotated[:, 0]
                            dy = rotated[:, 1]
                            rotated[:, 0] = dx * cos_a - dy * sin_a
                            rotated[:, 1] = dx * sin_a + dy * cos_a
                            metadata = dict(gesture.metadata or {})
                            metadata.setdefault("start_xy", (0.0, 0.0))
                            metadata["end_xy"] = (
                                float(np.sum(rotated[:, 0])),
                                float(np.sum(rotated[:, 1])),
                            )
                            rotated_path_length = float(np.sum(np.linalg.norm(rotated[:, :2], axis=1)))
                            raw_variants.append((rotated, rotated_path_length, metadata))
                    else:
                        continue
                else:
                    metadata = dict(gesture.metadata or {})
                    metadata.setdefault("start_xy", (0.0, 0.0))
                    metadata["end_xy"] = (total_dx, total_dy)
                    base_path_length = float(np.sum(np.linalg.norm(base_sequence[:, :2], axis=1)))
                    raw_variants.append((base_sequence, base_path_length, metadata))

                for raw_seq, raw_path_length, metadata in raw_variants:
                    if self.config.min_path_length > 0.0 and raw_path_length < self.config.min_path_length:
                        continue

                    seq_np = raw_seq.copy()
                    canon_gesture = replace(
                        gesture,
                        sequence=seq_np,
                        path_length=raw_path_length,
                        metadata=metadata,
                    )
                    if self.config.canonicalize_path or self.config.canonicalize_duration:
                        seq_np = _canonicalize_sequence(
                            seq_np,
                            unit_path=self.config.canonicalize_path,
                            unit_time=self.config.canonicalize_duration,
                        )
                        canon_gesture = replace(
                            canon_gesture,
                            sequence=seq_np,
                            duration=1.0 if self.config.canonicalize_duration else canon_gesture.duration,
                            path_length=1.0 if self.config.canonicalize_path else canon_gesture.path_length,
                        )

                    feature_vec = self._extract_features_from_gesture(canon_gesture)
                    if self.config.include_goal_geometry:
                        geometry_vec = _goal_geometry_features(
                            seq_np,
                            gesture.mask,
                            gesture.dataset_id,
                            canon_gesture.metadata,
                            canon_gesture.path_length,
                        )
                        if self._geometry_dim == 0:
                            self._geometry_dim = geometry_vec.shape[-1]
                        feature_vec = np.concatenate([feature_vec, geometry_vec], axis=0)

                    positive_sequences.append(seq_np)
                    positive_features.append(feature_vec)
                    count += 1
                    if self.config.max_gestures and count >= self.config.max_gestures:
                        break
                if self.config.max_gestures and count >= self.config.max_gestures:
                    break

            self._positive_sequences = positive_sequences
            self._positive_features = positive_features
            self._populate_feature_cache()
            self._save_to_cache()
        self._init_shared_positive_tensors()
        if not self._load_statistics_metadata():
            self._compute_statistics()
            self._save_statistics_metadata()

        feature_dim = None
        for seq_np, feat_np in zip(self._positive_sequences, self._positive_features):
            sequence = torch.from_numpy(seq_np).contiguous()
            features = torch.from_numpy(feat_np).contiguous()
            feature_dim = features.shape[0]
            if self.config.normalize_sequences:
                sequence = self._apply_sequence_normalization(sequence)
            if self.config.normalize_features:
                features = self._apply_feature_normalization(features)
            self.samples.append((sequence, features, torch.tensor(1.0)))

        self._add_replay_negatives(feature_dim)

        if self.config.use_generated_negatives and feature_dim is not None:
            neg_count = int(len(self._positive_sequences) * self.config.negative_ratio)
            base_len = self._positive_sequences[0].shape[0] if self._positive_sequences else self.config.sequence_length
            for _ in range(max(1, neg_count)):
                noise = torch.randn(base_len, 3) * 0.1
                noise[:, 2] = noise[:, 2].abs() + 1e-2
                noise_features = torch.zeros(feature_dim)
                if self.config.normalize_sequences:
                    noise = self._apply_sequence_normalization(noise)
                if self.config.normalize_features:
                    noise_features = self._apply_feature_normalization(noise_features)
                noise = noise.contiguous()
                noise_features = noise_features.contiguous()
                self.samples.append((noise, noise_features, torch.tensor(0.0)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[index]

    def _cache_path(self) -> Optional[Path]:
        if not self.config.cache_enabled:
            return None
        cache_dir = Path(self.config.cache_dir) / self.config.dataset_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        max_g = self.config.max_gestures if self.config.max_gestures is not None else "all"
        split_suffix = f"_{self.config.split}" if self.config.split else ""
        user_suffix = ""
        if self.config.user_filter:
            encoded = ",".join(sorted(self.config.user_filter))
            digest = hashlib.md5(encoded.encode("utf-8")).hexdigest()[:8]
            user_suffix = f"_users{digest}"
        canon_suffix = ""
        if self.config.canonicalize_path:
            canon_suffix += "_canonpath"
        if self.config.canonicalize_duration:
            canon_suffix += "_canontime"
        sr_cfg = self.config.sampling_rate
        if isinstance(sr_cfg, str):
            sr_suffix = f"_sr-{sr_cfg.lower()}"
        elif sr_cfg:
            sr_value = f"{float(sr_cfg):g}".replace(".", "p")
            sr_suffix = f"_sr{sr_value}"
        else:
            sr_suffix = "_sr-native"
        click_suffix = "_click" if self.config.use_click_boundaries else ""
        direction_suffix = ""
        if self.config.direction_buckets:
            bucket_str = "-".join(str(int(b)) for b in sorted(self.config.direction_buckets))
            direction_suffix = f"_dir{bucket_str}"
            if self.config.rotate_to_buckets:
                direction_suffix += "rot"
        minpath_suffix = "" if self.config.min_path_length <= 0 else f"_minpath{int(self.config.min_path_length)}"
        geometry_suffix = "_geom" if self.config.include_goal_geometry else ""
        feature_suffix = f"_feat-{self.config.feature_mode}"
        filename = (
            f"gestures_seq{self.config.sequence_length}_max{max_g}"
            f"{split_suffix}{user_suffix}{canon_suffix}{sr_suffix}{click_suffix}{direction_suffix}{minpath_suffix}{geometry_suffix}{feature_suffix}.npz"
        )
        return cache_dir / filename

    def _load_from_cache(self) -> bool:
        cache_path = self._cache_path()
        if cache_path is None or not cache_path.exists():
            self._positive_sequences = []
            self._positive_features = []
            return False

        with np.load(cache_path, allow_pickle=False) as data:
            sequences = data["sequences"]
            features = data["features"]
        self._positive_sequences = [seq.astype(np.float32) for seq in sequences]
        self._positive_features = [feat.astype(np.float32) for feat in features]
        self._positive_features_tensor = None
        self._populate_feature_cache()
        return True

    def _save_to_cache(self) -> None:
        cache_path = self._cache_path()
        if cache_path is None:
            return
        sequences = np.array(self._positive_sequences, dtype=np.float32)
        features = np.array(self._positive_features, dtype=np.float32)
        np.savez_compressed(cache_path, sequences=sequences, features=features)
        # Ensure positive feature tensors are re-derived through the normalisation
        # pipeline the next time they are requested. Storing the raw tensor here
        # would bypass normalisation when get_positive_features_tensor() is first
        # invoked, leading to conditioning vectors with incorrect scale.
        self._positive_features_tensor = None

    def _cache_signature(self) -> Optional[str]:
        cache_path = self._cache_path()
        if cache_path is None:
            return None
        return cache_path.stem

    def _metadata_paths(self) -> tuple[Optional[Path], Optional[Path]]:
        cache_path = self._cache_path()
        if cache_path is None:
            return None, None
        base = Path(str(cache_path))
        meta_path = Path(f"{base}.meta.json")
        reservoir_path = Path(f"{base}.reservoir.pt")
        return meta_path, reservoir_path

    def metadata_paths(self) -> tuple[Optional[Path], Optional[Path]]:
        """Return paths to cached metadata (JSON) and reservoir tensor."""
        return self._metadata_paths()

    def _load_statistics_metadata(self) -> bool:
        meta_path, reservoir_path = self._metadata_paths()
        if meta_path is None or not meta_path.exists():
            return False
        try:
            raw = meta_path.read_text()
            meta = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            return False

        if meta.get("version") != _STATS_VERSION:
            return False

        signature = self._cache_signature()
        if signature is not None and meta.get("signature") != signature:
            return False

        config_meta = meta.get("config", {})
        if bool(config_meta.get("normalize_sequences", self.config.normalize_sequences)) != self.config.normalize_sequences:
            return False
        if bool(config_meta.get("normalize_features", self.config.normalize_features)) != self.config.normalize_features:
            return False
        if config_meta.get("feature_mode", self.config.feature_mode) != self.config.feature_mode:
            return False

        requested_reservoir = int(self.config.feature_reservoir_size or 0)
        stored_reservoir = int(meta.get("feature_reservoir_size", 0))
        if requested_reservoir != stored_reservoir:
            return False

        try:
            seq_mean = torch.tensor(meta["sequence_mean"], dtype=torch.float32)
            seq_std = torch.tensor(meta["sequence_std"], dtype=torch.float32)
            feat_mean = torch.tensor(meta["feature_mean"], dtype=torch.float32)
            feat_std = torch.tensor(meta["feature_std"], dtype=torch.float32)
        except (KeyError, TypeError, ValueError):
            return False

        self._sequence_mean = seq_mean
        self._sequence_std = torch.clamp(seq_std, min=1e-6)
        self._feature_mean = feat_mean
        self._feature_std = torch.clamp(feat_std, min=1e-6)
        self._positive_features_tensor = None

        self._feature_reservoir_tensor = None
        self._conditioning_features_tensor = None
        if stored_reservoir > 0:
            if reservoir_path is None or not reservoir_path.exists():
                return False
            try:
                reservoir_tensor = torch.load(reservoir_path)
            except (OSError, RuntimeError):
                return False
            if not isinstance(reservoir_tensor, torch.Tensor):
                return False
            if reservoir_tensor.dtype != torch.float32:
                reservoir_tensor = reservoir_tensor.float()
            self._feature_reservoir_tensor = reservoir_tensor.contiguous()
            self._conditioning_features_tensor = self._feature_reservoir_tensor
        return True

    def _save_statistics_metadata(self) -> None:
        meta_path, reservoir_path = self._metadata_paths()
        if meta_path is None:
            return
        meta = {
            "version": _STATS_VERSION,
            "signature": self._cache_signature(),
            "sequence_mean": self._sequence_mean.tolist() if self._sequence_mean is not None else [],
            "sequence_std": self._sequence_std.tolist() if self._sequence_std is not None else [],
            "feature_mean": self._feature_mean.tolist() if self._feature_mean is not None else [],
            "feature_std": self._feature_std.tolist() if self._feature_std is not None else [],
            "feature_reservoir_size": int(self._feature_reservoir_tensor.size(0)) if self._feature_reservoir_tensor is not None else 0,
            "config": {
                "normalize_sequences": self.config.normalize_sequences,
                "normalize_features": self.config.normalize_features,
                "feature_mode": self.config.feature_mode,
                "feature_reservoir_size": int(self.config.feature_reservoir_size or 0),
            },
        }
        try:
            meta_path.write_text(json.dumps(meta, indent=2))
        except OSError:
            return
        if self._feature_reservoir_tensor is not None and reservoir_path is not None:
            try:
                torch.save(self._feature_reservoir_tensor.cpu(), reservoir_path)
            except OSError:
                pass
        elif reservoir_path is not None and reservoir_path.exists():
            try:
                reservoir_path.unlink()
            except OSError:
                pass

    def _populate_feature_cache(self) -> None:
        self._positive_feature_cache = {}
        if not self._positive_sequences or not self._positive_features:
            return
        for idx, seq_np in enumerate(self._positive_sequences):
            key = self._sequence_hash(seq_np)
            self._positive_feature_cache[key] = idx

    def _init_shared_positive_tensors(self) -> None:
        if self._positive_sequences:
            seq_stack = np.stack(self._positive_sequences, axis=0).astype(np.float32, copy=False)
            seq_tensor = torch.from_numpy(seq_stack).contiguous()
            try:
                seq_tensor.share_memory_()
            except RuntimeError:
                pass
            self._shared_positive_sequences = seq_tensor
        else:
            self._shared_positive_sequences = None

        if self._positive_features:
            feat_stack = np.stack(self._positive_features, axis=0).astype(np.float32, copy=False)
            feat_tensor = torch.from_numpy(feat_stack).contiguous()
            try:
                feat_tensor.share_memory_()
            except RuntimeError:
                pass
            self._shared_positive_features = feat_tensor
        else:
            self._shared_positive_features = None

    def _sequence_hash(self, sequence: np.ndarray) -> str:
        arr = np.ascontiguousarray(sequence.astype(np.float32, copy=False))
        return hashlib.sha1(arr.tobytes()).hexdigest()

    def get_cached_feature_vectors(self, sequences: torch.Tensor) -> List[Optional[torch.Tensor]]:
        if not self._positive_feature_cache:
            return [None] * sequences.size(0)
        if sequences.device.type != "cpu":
            sequences = sequences.cpu()
        np_sequences = sequences.detach().numpy()
        cached: List[Optional[torch.Tensor]] = []
        for seq in np_sequences:
            key = self._sequence_hash(seq)
            idx = self._positive_feature_cache.get(key)
            if idx is None:
                cached.append(None)
                continue
            if self._shared_positive_features is not None:
                cached.append(self._shared_positive_features[idx])
            else:
                cached.append(torch.from_numpy(self._positive_features[idx]))
        return cached

    def _add_replay_negatives(self, feature_dim: Optional[int]) -> None:
        if self.config.replay_path is None:
            return
        if not self._positive_sequences:
            return
        buffer = ReplayBuffer.load(self.config.replay_path)
        if len(buffer) == 0:
            return

        sample_count = int(len(self._positive_sequences) * self.config.replay_sample_ratio)
        sample_count = max(1, sample_count)
        sequences, features = buffer.sample(sample_count)
        if feature_dim is None and features.numel() > 0:
            feature_dim = features.shape[-1]
        if feature_dim is not None and features.shape[-1] != feature_dim:
            features = features[:, :feature_dim]
        for seq, feat in zip(sequences, features):
            seq_t = seq.float()
            feat_t = feat.float()
            if self.config.normalize_sequences:
                seq_t = self._apply_sequence_normalization(seq_t)
            if self.config.normalize_features and feature_dim is not None:
                feat_t = self._apply_feature_normalization(feat_t)
            seq_t = seq_t.contiguous()
            feat_t = feat_t.contiguous()
            self.samples.append((seq_t, feat_t, torch.tensor(0.0)))

    def _get_full_positive_features_tensor(self) -> torch.Tensor:
        if self._positive_features_tensor is None:
            if not self._positive_features:
                self._positive_features_tensor = torch.empty((0, 0), dtype=torch.float32)
            else:
                features = np.array(self._positive_features, dtype=np.float32)
                tensor = torch.from_numpy(features)
                if self.config.normalize_features:
                    tensor = self._apply_feature_normalization(tensor)
                self._positive_features_tensor = tensor.contiguous()
        return self._positive_features_tensor

    def get_positive_features_tensor(
        self,
        device: Optional[torch.device] = None,
        *,
        use_full: bool = False,
    ) -> torch.Tensor:
        if use_full:
            tensor = self._get_full_positive_features_tensor()
        else:
            tensor = self._conditioning_features_tensor
            if tensor is None:
                tensor = self._get_full_positive_features_tensor()
        if device is not None:
            tensor = tensor.to(device)
        return tensor.clone()

    # ------------------------------------------------------------------
    # Normalisation helpers & statistics
    # ------------------------------------------------------------------

    def _compute_statistics(self) -> None:
        seq_sum = np.zeros(3, dtype=np.float64)
        seq_sq_sum = np.zeros(3, dtype=np.float64)
        seq_count = 0
        for seq_np in self._positive_sequences:
            seq64 = seq_np.astype(np.float64, copy=False)
            seq_sum += seq64.sum(axis=0)
            seq_sq_sum += (seq64**2).sum(axis=0)
            seq_count += seq_np.shape[0]

        if seq_count == 0:
            seq_mean = np.zeros(3, dtype=np.float32)
            seq_std = np.ones(3, dtype=np.float32)
        else:
            mean64 = seq_sum / seq_count
            var64 = np.maximum(seq_sq_sum / seq_count - mean64**2, 1e-6)
            seq_mean = mean64.astype(np.float32)
            seq_std = np.sqrt(var64).astype(np.float32)

        self._sequence_mean = torch.from_numpy(seq_mean)
        self._sequence_std = torch.from_numpy(seq_std)

        feat_count = len(self._positive_features)
        if feat_count == 0:
            feat_mean = np.zeros(1, dtype=np.float32)
            feat_std = np.ones(1, dtype=np.float32)
        else:
            feat_dim = self._positive_features[0].shape[0]
            feat_sum = np.zeros(feat_dim, dtype=np.float64)
            feat_sq_sum = np.zeros(feat_dim, dtype=np.float64)
            for feat_np in self._positive_features:
                feat64 = feat_np.astype(np.float64, copy=False)
                feat_sum += feat64
                feat_sq_sum += feat64**2
            mean64 = feat_sum / feat_count
            var64 = np.maximum(feat_sq_sum / feat_count - mean64**2, 1e-6)
            feat_mean = mean64.astype(np.float32)
            feat_std = np.sqrt(var64).astype(np.float32)

        self._feature_mean = torch.from_numpy(feat_mean)
        self._feature_std = torch.from_numpy(feat_std)
        self._positive_features_tensor = None

        reservoir_size = int(self.config.feature_reservoir_size or 0)
        self._feature_reservoir_tensor = None
        self._conditioning_features_tensor = None
        if reservoir_size > 0 and feat_count > 0:
            max_size = min(reservoir_size, feat_count)
            reservoir: list[np.ndarray] = []
            rng = random.Random(1337)
            for idx, feat_np in enumerate(self._positive_features):
                feat_vec = feat_np.astype(np.float32, copy=False)
                if idx < max_size:
                    reservoir.append(feat_vec.copy())
                else:
                    replace_idx = rng.randint(0, idx)
                    if replace_idx < max_size:
                        reservoir[replace_idx] = feat_vec.copy()
            if reservoir:
                reservoir_array = np.stack(reservoir, axis=0).astype(np.float32)
                reservoir_tensor = torch.from_numpy(reservoir_array)
                if self.config.normalize_features:
                    reservoir_tensor = self._apply_feature_normalization(reservoir_tensor)
                self._feature_reservoir_tensor = reservoir_tensor.contiguous()
                self._conditioning_features_tensor = self._feature_reservoir_tensor

    def _apply_sequence_normalization(self, sequence: torch.Tensor) -> torch.Tensor:
        if self._sequence_mean is None or self._sequence_std is None:
            return sequence
        mean = self._sequence_mean.to(dtype=sequence.dtype, device=sequence.device)
        std = self._sequence_std.to(dtype=sequence.dtype, device=sequence.device)
        return (sequence - mean) / std

    def _apply_feature_normalization(self, features: torch.Tensor) -> torch.Tensor:
        if self._feature_mean is None or self._feature_std is None:
            return features
        mean = self._feature_mean.to(dtype=features.dtype, device=features.device)
        std = self._feature_std.to(dtype=features.dtype, device=features.device)
        return (features - mean) / std

    def get_sequence_stats(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self._sequence_mean if self._sequence_mean is not None else torch.zeros(3)
        std = self._sequence_std if self._sequence_std is not None else torch.ones(3)
        if device is not None:
            mean = mean.to(device)
            std = std.to(device)
        return mean.clone(), std.clone()

    def get_feature_stats(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._feature_mean is None or self._feature_std is None:
            feature_dim = self.samples[0][1].shape[0] if self.samples else 1
            mean = torch.zeros(feature_dim)
            std = torch.ones(feature_dim)
        else:
            mean = self._feature_mean
            std = self._feature_std
        if device is not None:
            mean = mean.to(device)
            std = std.to(device)
        return mean.clone(), std.clone()

    def normalize_sequences(self, sequences: torch.Tensor) -> torch.Tensor:
        if not self.config.normalize_sequences:
            return sequences
        mean, std = self.get_sequence_stats(device=sequences.device)
        mean = mean.to(dtype=sequences.dtype)
        std = std.to(dtype=sequences.dtype)
        while mean.dim() < sequences.dim():
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return (sequences - mean) / std

    def denormalize_sequences(self, sequences: torch.Tensor) -> torch.Tensor:
        if not self.config.normalize_sequences:
            return sequences
        mean, std = self.get_sequence_stats(device=sequences.device)
        mean = mean.to(dtype=sequences.dtype)
        std = std.to(dtype=sequences.dtype)
        while mean.dim() < sequences.dim():
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return sequences * std + mean

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        if not self.config.normalize_features:
            return features
        target_device = features.device if hasattr(features, "device") else None
        mean, std = self.get_feature_stats(device=target_device)
        mean = mean.to(dtype=features.dtype)
        std = std.to(dtype=features.dtype)
        if features.dim() == 1:
            return (features - mean) / std
        return (features - mean.unsqueeze(0)) / std.unsqueeze(0)

    def denormalize_features(self, features: torch.Tensor) -> torch.Tensor:
        if not self.config.normalize_features:
            return features
        target_device = features.device if hasattr(features, "device") else None
        mean, std = self.get_feature_stats(device=target_device)
        mean = mean.to(dtype=features.dtype)
        std = std.to(dtype=features.dtype)
        if features.dim() == 1:
            return features * std + mean
        return features * std.unsqueeze(0) + mean.unsqueeze(0)



def create_gesture_dataloader(
    config: GestureDatasetConfig,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    dataset = GestureDataset(config)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
