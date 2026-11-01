"""The pi0.5 PyTorch server adapts to whichever lerobot release is installed. Fake lerobot modules stand in for the releases."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("lerobot_compat", Path(__file__).resolve().parents[1] / "eeb/servers/pi05/lerobot_compat.py")
compat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compat)

MODELING, VLA_UTILS = "lerobot.policies.pi05.modeling_pi05", "lerobot.policies.common.vla_utils"


@pytest.fixture
def modules(monkeypatch):
    """Install fake lerobot modules; returns a function to set their attributes."""
    for name in (MODELING, VLA_UTILS):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    def put(module, **attrs):
        for key, value in attrs.items():
            setattr(sys.modules[module], key, value)
    return put


def test_helpers_resolve_from_the_modeling_module_first_then_vla_utils(modules):
    modules(VLA_UTILS, clone_past_key_values=lambda c: "from-vla")
    assert compat.clone_past_key_values("x") == "from-vla"
    modules(MODELING, clone_past_key_values=lambda c: "from-modeling")
    assert compat.clone_past_key_values("x") == "from-modeling"


def test_clone_falls_back_to_a_deep_copy_when_the_release_has_no_helper(modules):
    cache = {"k": [1, 2]}
    clone = compat.clone_past_key_values(cache)
    assert clone == cache and clone is not cache and clone["k"] is not cache["k"]


def test_make_att_2d_masks_is_required(modules):
    with pytest.raises(ImportError):
        compat.make_att_2d_masks(1, 2)
    modules(VLA_UTILS, make_att_2d_masks=lambda pad, att: (pad, att))
    assert compat.make_att_2d_masks(1, 2) == (1, 2)


class Mask:
    def __init__(self, tag):
        self.tag = tag

    def to(self, dtype):
        return Mask(f"{self.tag}->{dtype}")


def test_mask_uses_the_model_method_with_dtype_when_it_takes_one(modules):
    model = types.SimpleNamespace(_prepare_attention_masks_4d=lambda att, dtype=None: f"method:{dtype}")
    assert compat.attention_mask_4d(model, "m", "bf16") == "method:bf16"


def test_mask_casts_when_the_release_method_takes_no_dtype(modules):
    model = types.SimpleNamespace(_prepare_attention_masks_4d=lambda att: Mask("float32"))
    assert compat.attention_mask_4d(model, "m", "bf16").tag == "float32->bf16"


def test_mask_uses_the_module_helper_when_the_model_has_no_method(modules):
    modules(VLA_UTILS, prepare_attention_masks_4d=lambda att, dtype=None: f"helper:{dtype}")
    assert compat.attention_mask_4d(types.SimpleNamespace(), "m", "bf16") == "helper:bf16"


def test_mask_fails_loudly_when_nothing_is_available(modules):
    with pytest.raises(ImportError):
        compat.attention_mask_4d(types.SimpleNamespace(), "m", "bf16")


class TorchDtype:
    def __str__(self):
        return "torch.bfloat16"


def test_config_dtype_matches_the_type_lerobot_declares():
    assert compat.config_dtype(types.SimpleNamespace(dtype="float32"), TorchDtype()) == "bfloat16"
    dtype = TorchDtype()
    assert compat.config_dtype(types.SimpleNamespace(dtype=object()), dtype) is dtype


class Base:
    @classmethod
    def from_pretrained(cls, checkpoint, **kwargs):
        return Pi05Cfg(checkpoint)


class Pi05Cfg(Base):
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    @classmethod
    def from_pretrained(cls, checkpoint, **kwargs):
        raise ValueError("The fields `type` are not valid for PI05Config")


def test_config_loader_falls_back_to_the_base_class(monkeypatch):
    configs = types.ModuleType("lerobot.configs")
    configs.PreTrainedConfig = Base
    monkeypatch.setitem(sys.modules, "lerobot.configs", configs)
    monkeypatch.delitem(sys.modules, "lerobot.configs.policies", raising=False)
    assert compat.load_config(Pi05Cfg, "ckpt").checkpoint == "ckpt"


def test_config_loader_returns_the_subclass_result_when_it_works():
    class Fine:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            return ("ok", checkpoint, kwargs)
    assert compat.load_config(Fine, "c", revision="r") == ("ok", "c", {"revision": "r"})
