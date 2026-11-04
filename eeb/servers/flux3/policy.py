"""FLUX 3 Action (black-forest-labs/flux-3-action-droid) as an EEB Track 1 policy.

The reference serves the released one-step FP8r package (`variants/sd-fp8r`) with the checkpoint's own settings: one denoising step,
BF16 activations, text encoder and video VAE as shipped. Every request is a B=1 call; calls serialize on one lock (single chip).

Observation contract (DROID): three cameras (wrist, left exterior, right exterior), 8-dim state, a language instruction.
Action contract: a (32, 8) chunk. Trace cameras are matched by name (`wrist`, `exterior_1`/`left`, `exterior_2`/`right`) and, when the
names are unknown, by order [wrist, left, right]. Frames that are not the model's 360x640 are resized (bilinear) inside the timed call.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np

from eeb.protocol.policy import ActionChunk, BasePolicy, Caps, Context, Frame, Observation

FLUX3_INSTALL_HINT = (
    "FLUX 3 Action needs the `flux-action` package (pip install 'flux-action[encoders] @ "
    "git+https://github.com/black-forest-labs/flux-action.git@<pinned commit>'), torch>=2.10 and a Hugging Face token with access to "
    "black-forest-labs/flux-3-action-droid and flux-3-action-base."
)

DEFAULT_REPO = "black-forest-labs/flux-3-action-droid"
DEFAULT_REVISION = "e3b6c915cdcdfb1d5db08e133b8ca3d29be2d7d4"
MODEL_HW = (360, 640)
MODEL_CAMERAS = ("images.wrist", "images.left", "images.right")
# trace camera name fragments -> model camera, first match wins
_CAMERA_HINTS = (("wrist", "images.wrist"), ("exterior_1", "images.left"), ("exterior_image_1", "images.left"), ("left", "images.left"),
                 ("exterior_2", "images.right"), ("exterior_image_2", "images.right"), ("right", "images.right"))
VARIANTS = {"sd-fp8r": "variants/sd-fp8r", "base": None}


class Flux3ActionPolicy(BasePolicy):
  """One-step FLUX 3 Action server policy.

  Args:
    device: torch device ("cuda").
    precision_tag: label reported in Caps (the package's own dtype is BF16).
    variant: "sd-fp8r" (released one-step FP8r package) or "base" (4-step BF16 root package).
    repo / revision: checkpoint, pinned.
    compile_dit: torch.compile the DiT in `prepare_for_serving` (off by default: the reference is eager).
    execute_horizon: actions executed per chunk by the loadgen (the model emits 32).
    control_hz: DROID control rate; sets Caps.dt_ms.
    seed: base seed; each call runs with inference_seed = seed + obs.step, so replays are deterministic.
    hf_offline: read the checkpoint from the local cache only.
  """

  def __init__(self, device: str = "cuda", precision_tag: str = "BF16", variant: str = "sd-fp8r", repo: str = DEFAULT_REPO,
               revision: str = DEFAULT_REVISION, compile_dit: bool = False, execute_horizon: int = 8, control_hz: float = 15.0,
               seed: int = 0, hf_offline: bool = False) -> None:
    if variant not in VARIANTS:
      raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {variant!r}.")
    try:
      import torch
      from huggingface_hub import snapshot_download
      from flux_action.inference.precision import prepare_for_serving
      from flux_action.policy import FluxActionPolicy
    except ImportError as exc:
      raise ImportError(f"{exc}. {FLUX3_INSTALL_HINT}") from exc
    self._torch = torch
    self.device = device
    self.precision_tag = precision_tag
    self.variant = variant
    self.execute_horizon = int(execute_horizon)
    self.dt_ms = 1000.0 / float(control_hz)
    self._seed = int(seed)
    self._lock = threading.Lock()
    subfolder = VARIANTS[variant]
    patterns = ["config.json", "config.native.json", "manifest.json", "model.safetensors"] if subfolder is None else [
        "config.json", f"{subfolder}/config.json", f"{subfolder}/manifest.json", f"{subfolder}/model.safetensors"]
    root = snapshot_download(repo, revision=revision, allow_patterns=patterns, local_files_only=bool(hf_offline))
    policy = FluxActionPolicy.from_pretrained(root, subfolder=subfolder, device="cpu")
    policy.config.validate_inference()
    prepare_for_serving(policy, device=device, compile_dit=bool(compile_dit), offload_text_encoder=False)
    self._policy = policy
    self._steps = int(policy.config.num_inference_steps)
    self._horizon, self._action_dim = 32, 8
    self._flux_version = _package_version("flux-action")
    self._warm_start()

  def _warm_start(self) -> None:
    """Two untimed calls so compilation, autotuning and allocator growth finish before the first measured request."""
    blank = {name: np.zeros((*MODEL_HW, 3), dtype=np.uint8) for name in MODEL_CAMERAS}
    for _ in range(2):
      self._infer(blank, np.zeros(8, dtype=np.float32), "warmup", self._seed)

  def capabilities(self) -> Caps:
    return Caps(track="VLA_TRACK", horizon=self._horizon, dt_ms=self.dt_ms, modalities=("rgb", "proprio", "language"),
                precision=self.precision_tag, model_id=f"flux-3-action-droid:{self.variant}",
                server_impl=f"flux-action-{self._flux_version}+torch-{self._torch.__version__}-{self.device}-steps{self._steps}",
                action_dim=self._action_dim, proprio_dim=8, cameras=MODEL_CAMERAS, image_height=MODEL_HW[0], image_width=MODEL_HW[1],
                execute_horizon=self.execute_horizon)

  def _cameras(self, obs: Observation) -> Dict[str, np.ndarray]:
    """Map the observation's cameras onto the model's three, by name then by order [wrist, left, right]."""
    mapped: Dict[str, np.ndarray] = {}
    for name, image in obs.images.items():
      lowered = name.lower()
      for hint, target in _CAMERA_HINTS:
        if hint in lowered and target not in mapped:
          mapped[target] = image
          break
    if len(mapped) < len(MODEL_CAMERAS):
      leftover = [img for name, img in obs.images.items() if all(img is not m for m in mapped.values())]
      for target in MODEL_CAMERAS:
        if target not in mapped and leftover:
          mapped[target] = leftover.pop(0)
    missing = [t for t in MODEL_CAMERAS if t not in mapped]
    if missing:
      raise ValueError(f"Observation is missing cameras for {missing} (got {sorted(obs.images)}).")
    return mapped

  def _infer(self, cameras: Dict[str, np.ndarray], state: np.ndarray, instruction: str, seed: int) -> np.ndarray:
    torch = self._torch
    batch: Dict[str, Any] = {}
    for name in MODEL_CAMERAS:
      arr = np.asarray(cameras[name], dtype=np.uint8)
      tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)[None])).to(self.device)
      if tuple(tensor.shape[-2:]) != MODEL_HW:
        resized = torch.nn.functional.interpolate(tensor.float(), size=MODEL_HW, mode="bilinear", align_corners=False)
        tensor = resized.round().clamp(0, 255).to(torch.uint8)
      batch[name] = tensor
    batch["state"] = torch.from_numpy(np.asarray(state, dtype=np.float32)[None].copy()).to(self.device)
    batch["task"] = [instruction]
    config = self._policy.config
    with self._lock, torch.inference_mode():
      original = config.inference_seed
      try:
        config.inference_seed = int(seed)
        out = self._policy.predict_action_chunk(batch)
      finally:
        config.inference_seed = original
      if self.device.startswith("cuda"):
        torch.cuda.synchronize()
    actions = out.detach().to("cpu").float().numpy()
    if actions.ndim == 3 and actions.shape[0] == 1:
      actions = actions[0]
    return actions

  def predict(self, obs: Observation) -> ActionChunk:
    state = np.asarray(obs.proprio, dtype=np.float32).reshape(-1)
    if state.shape != (8,):
      raise ValueError(f"FLUX 3 Action needs an 8-dim state, got shape {state.shape}.")
    actions = self._infer(self._cameras(obs), state, obs.instruction, self._seed + int(obs.step))
    self._horizon, self._action_dim = int(actions.shape[0]), int(actions.shape[1])
    return ActionChunk(actions=actions)

  def rollout(self, ctx: Context, actions: np.ndarray) -> Iterator[Frame]:
    raise NotImplementedError("FLUX 3 Action is a Track 1 VLA policy; it does not implement Track 2 rollout.")

  def close(self) -> None:
    self._policy = None


def _package_version(name: str) -> str:
  try:
    from importlib.metadata import version
    return version(name)
  except Exception:  # noqa: BLE001 - provenance only
    return "unknown"


__all__ = ["FLUX3_INSTALL_HINT", "Flux3ActionPolicy"]
