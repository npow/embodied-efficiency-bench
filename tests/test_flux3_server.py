"""FLUX 3 Action server: camera mapping, hardware entry and recipe. No GPU or flux_action needed."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from eeb.protocol.policy import Observation
from eeb.servers.flux3.policy import MODEL_CAMERAS, Flux3ActionPolicy
from embodied_bench.schema import HARDWARE_SKU_ALIASES, HARDWARE_SKU_SPECS

ROOT = Path(__file__).resolve().parents[1]


def _policy() -> Flux3ActionPolicy:
  return Flux3ActionPolicy.__new__(Flux3ActionPolicy)   # skips the model load


def _obs(names):
  return Observation(session_id="s", step=0, images={n: np.full((4, 4, 3), i, dtype=np.uint8) for i, n in enumerate(names)})


def test_droid_camera_names_map_to_wrist_left_right():
  mapped = _policy()._cameras(_obs(["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"]))
  assert [int(mapped[c][0, 0, 0]) for c in MODEL_CAMERAS] == [2, 0, 1]   # wrist, left (exterior 1), right (exterior 2)


def test_unknown_camera_names_fall_back_to_order():
  mapped = _policy()._cameras(_obs(["a", "b", "c"]))
  assert [int(mapped[c][0, 0, 0]) for c in MODEL_CAMERAS] == [0, 1, 2]


def test_two_cameras_are_an_error_nothing_is_repeated():
  with pytest.raises(ValueError, match="missing cameras"):
    _policy()._cameras(_obs(["agentview", "wrist"]))


def test_missing_wrist_camera_is_an_error():
  with pytest.raises(ValueError, match="missing cameras"):
    _policy()._cameras(_obs(["exterior_image_1_left", "exterior_image_2_left"]))


def test_rtx_pro_6000_is_a_hardware_sku_with_a_provider_price():
  sku = HARDWARE_SKU_SPECS["RTX_PRO_6000_96GB"]
  assert HARDWARE_SKU_ALIASES["NVIDIA_RTX_PRO_6000"] == "RTX_PRO_6000_96GB"
  assert sku["vram_mb"] == 98304 and sku["hourly_cost_usd"] > 0 and sku["capex_usd"] > 0


def test_recipe_matches_the_model_contract():
  recipe = yaml.safe_load((ROOT / "recipes" / "track1_vla_flux3_action_rtx_pro_6000.yaml").read_text())
  assert recipe["hardware_sku"] in HARDWARE_SKU_SPECS and recipe["native_action_chunk_size"] == 32 and recipe["num_cameras"] == 3


def test_a_96gb_part_fails_the_single_chip_envelope_and_passes_the_tier_its_recipe_names():
  from eeb.envelope.check import DeviceInfo, DeviceInventory, check_envelope

  inv = DeviceInventory(devices=[DeviceInfo(index=0, name="NVIDIA RTX PRO 6000 Blackwell Server Edition", vendor="NVIDIA", total_memory_mb=97894.0, source="nvidia-smi")])
  default = check_envelope(hardware_sku="RTX_PRO_6000_96GB", inventory=inv)
  assert not default.passed and "64 GB single-chip envelope" in default.reason         # unchanged behaviour without a perf cap
  tier = check_envelope(hardware_sku="RTX_PRO_6000_96GB", inventory=inv, perf_cap="CAP_DATACENTER_1000T")
  assert tier.passed and tier.envelope_limits["perf_cap"] == "CAP_DATACENTER_1000T" and tier.envelope_limits["max_memory_gb"] == 192.0
  assert check_envelope(hardware_sku="RTX_PRO_6000_96GB", inventory=inv, perf_cap="CAP_SINGLE_CHIP").passed is False
  recipe = yaml.safe_load((ROOT / "recipes" / "track1_vla_flux3_action_rtx_pro_6000.yaml").read_text())
  assert recipe["perf_cap"] == "CAP_DATACENTER_1000T" == HARDWARE_SKU_SPECS["RTX_PRO_6000_96GB"]["default_perf_cap"]


class _FakeTrace:
  """A decoded LeRobot episode with three cameras, as the DROID loader returns it."""

  def __init__(self, n=24):
    rng = np.random.default_rng(0)
    self.episode_id, self.instruction, self.cameras = "fake", "put the marker in the pot", ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"]
    still = rng.integers(0, 255, (1, 18, 32, 3)).astype(int)
    drift = np.linspace(0, 3, n)[:, None, None, None].astype(int)                  # frames change slowly, as a real episode does
    self.images = {c: np.clip(still + drift + 40 * i, 0, 255).astype(np.uint8) for i, c in enumerate(self.cameras)}   # distinct cameras
    self.proprio = rng.standard_normal((n, 8)).astype(np.float32)
    self.action = rng.standard_normal((n, 7)).astype(np.float32)

  def __len__(self):
    return len(self.proprio)


def test_three_camera_shard_keeps_each_real_camera_and_round_trips(tmp_path):
  from embodied_bench.dataset_loader import build_shard_from_real_lerobot_trace, load_episode_shard_npz, save_episode_shard_npz

  kw = dict(episode_id="e", source_dataset="DROID", subset_name="s", license_spdx="MIT", resolution_hw=32)
  shard = build_shard_from_real_lerobot_trace(_FakeTrace(), num_cameras=3, **kw)
  assert shard.rgb_frames.shape[:3] == (16, 3, 3) and shard.camera_names == tuple(_FakeTrace().cameras)
  assert not np.array_equal(shard.rgb_frames[:, 0], shard.rgb_frames[:, 1])                 # three different views, none repeated
  save_episode_shard_npz(shard, tmp_path / "e.npz")
  assert load_episode_shard_npz(tmp_path / "e.npz").camera_names == shard.camera_names
  legacy = build_shard_from_real_lerobot_trace(_FakeTrace(), **kw)
  assert legacy.rgb_frames.shape[1] == 2 and legacy.camera_names == ("agentview", "wrist")  # the default is unchanged
  save_episode_shard_npz(legacy, tmp_path / "l.npz")
  assert "camera_names" not in bytes(np.load(tmp_path / "l.npz")["meta_json"]).decode()      # legacy shards keep their metadata


def test_three_cameras_are_refused_when_the_source_has_fewer(tmp_path):
  from embodied_bench.dataset_loader import build_shard_from_real_lerobot_trace

  trace = _FakeTrace()
  trace.cameras, trace.images = trace.cameras[:2], {k: v for k, v in list(trace.images.items())[:2]}
  with pytest.raises(ValueError, match="needs three"):
    build_shard_from_real_lerobot_trace(trace, episode_id="e", source_dataset="DROID", subset_name="s", license_spdx="MIT", resolution_hw=32, num_cameras=3)
