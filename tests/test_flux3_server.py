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


def test_missing_camera_is_an_error():
  with pytest.raises(ValueError, match="missing cameras"):
    _policy()._cameras(_obs(["wrist_image_left", "exterior_image_1_left"]))


def test_rtx_pro_6000_is_a_hardware_sku_without_a_default_price():
  sku = HARDWARE_SKU_SPECS["RTX_PRO_6000_96GB"]
  assert HARDWARE_SKU_ALIASES["NVIDIA_RTX_PRO_6000"] == "RTX_PRO_6000_96GB"
  assert sku["vram_mb"] == 98304 and "hourly_cost_usd" not in sku


def test_recipe_matches_the_model_contract():
  recipe = yaml.safe_load((ROOT / "recipes" / "track1_vla_flux3_action_rtx_pro_6000.yaml").read_text())
  assert recipe["hardware_sku"] in HARDWARE_SKU_SPECS and recipe["native_action_chunk_size"] == 32 and recipe["num_cameras"] == 3
