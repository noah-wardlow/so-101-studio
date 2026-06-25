#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL_ID = "allenai/MolmoAct2-SO100_101"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_TASK = "pick up the cube and place it into the target bin"
DEFAULT_NORM_TAG = "so100_so101_molmoact2"
DEFAULT_SEED = 1005


def env_default_float_list(name: str, fallback: list[float], expected_length: int) -> list[float]:
    value = os.environ.get(name)
    if value is None:
        return fallback
    try:
        result = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated floats, got {value!r}") from exc
    if len(result) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} values, got {len(result)}")
    return result


SO101_GRIPPER_MIN_RAD = np.deg2rad(-10.0)
SO101_GRIPPER_MAX_RAD = np.deg2rad(100.0)
SO101_POLICY_GRIPPER_STATE_MIN = 0.9435578
SO101_POLICY_GRIPPER_STATE_MAX = 44.1375560
SO101_POLICY_GRIPPER_ACTION_MIN = -0.3016557
SO101_POLICY_GRIPPER_ACTION_MAX = 44.7464934

# MolmoAct2-SO100_101 was trained on LeRobot SO-100/101 motor features:
# arm joints in calibrated servo degrees and gripper in the raw robot-scale
# range recorded in norm_stats.json. Those are not the same coordinate
# system as the geometric MuJoCo hinge radians.
#
# This affine bridge anchors the current MuJoCo visual home to the median
# state of the checkpoint's SO100/SO101 norm stats and applies the known
# sign flips needed to keep the physical arm convention from folding the
# simulated arm through itself.
SO101_SIM_HOME_RAD = np.asarray(
    [0.069314, -1.685636, 0.810382, 1.600638, -1.396152],
    dtype=np.float32,
)
SO101_POLICY_HOME_DEG = np.asarray(
    [3.0663757, 123.1648209, 124.3993006, 57.8860546, -11.0374367],
    dtype=np.float32,
)
SO101_SIM_TO_POLICY_SIGN = np.asarray(
    env_default_float_list("SO101_SIM_TO_POLICY_SIGN", [1.0, -1.0, 1.0, -1.0, 1.0], 5),
    dtype=np.float32,
)
SO101_SIM_TO_POLICY_OFFSET_DEG = SO101_POLICY_HOME_DEG - (
    SO101_SIM_TO_POLICY_SIGN * np.rad2deg(SO101_SIM_HOME_RAD)
)
SO101_SIM_ACTION_BIAS_RAD = np.asarray(
    env_default_float_list("SO101_SIM_ACTION_BIAS_RAD", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6),
    dtype=np.float32,
)
SO101_SIM_CTRL_MIN_RAD = np.asarray(
    [-1.91986, -1.74533, -1.69, -1.65806, -2.74385, SO101_GRIPPER_MIN_RAD],
    dtype=np.float32,
)
SO101_SIM_CTRL_MAX_RAD = np.asarray(
    [1.91986, 1.74533, 1.69, 1.65806, 2.84121, SO101_GRIPPER_MAX_RAD],
    dtype=np.float32,
)


def env_default(name: str, fallback: str) -> str:
    return os.environ.get(name, fallback)


def env_default_int(name: str, fallback: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return fallback
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def env_default_bool(name: str, fallback: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return fallback
    return value.lower() in {"1", "true", "yes", "on"}


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_dtype(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"auto", "bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {value!r}")


def decode_image(value: str | None) -> Image.Image | None:
    if not value:
        return None
    if "," in value and value.lstrip().startswith("data:"):
        value = value.split(",", 1)[1]
    raw = base64.b64decode(value)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def sim_state_to_policy_units(state: np.ndarray) -> np.ndarray:
    robot_state = np.asarray(state[:6], dtype=np.float32)
    policy_state = np.empty(6, dtype=np.float32)
    policy_state[:5] = (
        SO101_SIM_TO_POLICY_SIGN * np.rad2deg(robot_state[:5])
        + SO101_SIM_TO_POLICY_OFFSET_DEG
    )
    gripper = np.clip(robot_state[5], SO101_GRIPPER_MIN_RAD, SO101_GRIPPER_MAX_RAD)
    policy_state[5] = (
        (gripper - SO101_GRIPPER_MIN_RAD)
        / (SO101_GRIPPER_MAX_RAD - SO101_GRIPPER_MIN_RAD)
        * (SO101_POLICY_GRIPPER_STATE_MAX - SO101_POLICY_GRIPPER_STATE_MIN)
        + SO101_POLICY_GRIPPER_STATE_MIN
    )
    return policy_state


def policy_action_to_sim_radians(action: np.ndarray) -> np.ndarray:
    policy_action = np.asarray(action[:6], dtype=np.float32)
    sim_action = np.empty(6, dtype=np.float32)
    sim_action[:5] = np.deg2rad(
        SO101_SIM_TO_POLICY_SIGN * (policy_action[:5] - SO101_SIM_TO_POLICY_OFFSET_DEG)
    )
    gripper_policy = np.clip(
        policy_action[5],
        SO101_POLICY_GRIPPER_ACTION_MIN,
        SO101_POLICY_GRIPPER_ACTION_MAX,
    )
    sim_action[5] = SO101_GRIPPER_MIN_RAD + (
        (gripper_policy - SO101_POLICY_GRIPPER_ACTION_MIN)
        / (SO101_POLICY_GRIPPER_ACTION_MAX - SO101_POLICY_GRIPPER_ACTION_MIN)
    ) * (SO101_GRIPPER_MAX_RAD - SO101_GRIPPER_MIN_RAD)
    sim_action += SO101_SIM_ACTION_BIAS_RAD
    return np.clip(sim_action, SO101_SIM_CTRL_MIN_RAD, SO101_SIM_CTRL_MAX_RAD)


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def canonical_image_key(key: str) -> str:
    if key.startswith("observation.images."):
        return key.removeprefix("observation.images.")
    return key


def ordered_image_entries(images: dict[str, str]) -> list[tuple[str, Image.Image]]:
    preferred_keys = [
        "top",
        "side",
        "external",
        "wrist",
        "front",
        "image",
        "observation.images.top",
        "observation.images.side",
        "observation.images.external",
        "observation.images.wrist",
        "observation.images.front",
        "observation.images.image",
    ]
    result: list[tuple[str, Image.Image]] = []
    seen: set[str] = set()
    for key in preferred_keys + sorted(images):
        canonical_key = canonical_image_key(key)
        if canonical_key in seen:
            continue
        image = decode_image(images.get(key))
        if image is not None:
            result.append((canonical_key, image))
            seen.add(canonical_key)
    if len(result) == 1:
        key, image = result[0]
        result.append((f"{key}_copy", image.copy()))
    return result


class MolmoAct2Runtime:
    def __init__(
        self,
        model_id: str,
        device: torch.device,
        dtype: torch.dtype,
        norm_tag: str,
        num_steps: int,
        seed: int,
        enable_cuda_graph: bool,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.norm_tag = norm_tag
        self.num_steps = num_steps
        self.seed = seed
        self.enable_cuda_graph = enable_cuda_graph
        self._lock = threading.Lock()
        self._generator: torch.Generator | None = None
        self._episode_index = -1
        self._request_index = 0

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=dtype,
        ).to(device).eval()

    def _make_generator(self) -> torch.Generator:
        try:
            generator = torch.Generator(device=self.device)
        except RuntimeError:
            generator = torch.Generator()
        generator.manual_seed(int(self.seed))
        return generator

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(payload.get("state", []), dtype=np.float32)
        if state.shape[0] < 6:
            raise ValueError(f"Expected at least 6 state values, got shape {list(state.shape)}")

        image_entries = ordered_image_entries(payload.get("images") or {})
        if not image_entries:
            raise ValueError("MolmoAct2-SO100_101 requires at least one RGB image")
        images = [image for _, image in image_entries]

        task = str(payload.get("task") or DEFAULT_TASK)
        policy_state = sim_state_to_policy_units(state)
        started_at = time.perf_counter()

        with self._lock:
            if bool(payload.get("reset")) or self._generator is None:
                self._episode_index += 1
                self._request_index = 0
                self._generator = self._make_generator()
            request_index = self._request_index
            self._request_index += 1

            autocast_enabled = self.device.type == "cuda" and self.dtype in (torch.bfloat16, torch.float16)
            with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype, enabled=autocast_enabled):
                out = self.model.predict_action(
                    processor=self.processor,
                    images=images,
                    task=task,
                    state=policy_state,
                    norm_tag=self.norm_tag,
                    inference_action_mode="continuous",
                    enable_depth_reasoning=False,
                    num_steps=self.num_steps,
                    generator=self._generator,
                    normalize_language=True,
                    enable_cuda_graph=self.enable_cuda_graph,
                )

        action_policy_chunk = as_numpy(out.actions).astype(np.float32)
        if action_policy_chunk.ndim == 3:
            if action_policy_chunk.shape[0] != 1:
                raise ValueError(
                    f"Expected one MolmoAct2 action batch, got shape {list(action_policy_chunk.shape)}"
                )
            action_policy_chunk = action_policy_chunk[0]
        if action_policy_chunk.ndim == 1:
            action_policy_chunk = action_policy_chunk[None, :]
        action_policy_chunk = action_policy_chunk[:, :6]
        action_chunk = np.stack([
            policy_action_to_sim_radians(step)
            for step in action_policy_chunk
        ])

        return {
            "model_id": self.model_id,
            "dataset_id": "SO100_101 mixture",
            "device": str(self.device),
            "policy_space": "molmoact2-so100_101-absolute-joint",
            "state_dim": 6,
            "stats_source": f"{self.model_id}:norm_stats.json:{self.norm_tag}",
            "server_ms": (time.perf_counter() - started_at) * 1000.0,
            "action": action_chunk[0].astype(float).tolist(),
            "actions": action_chunk.astype(float).tolist(),
            "action_policy_units": action_policy_chunk[0].astype(float).tolist(),
            "image_keys": [key for key, _ in image_entries],
            "image_count": len(image_entries),
            "action_names": [
                "shoulder_pan.pos",
                "shoulder_lift.pos",
                "elbow_flex.pos",
                "wrist_flex.pos",
                "wrist_roll.pos",
                "gripper.pos",
            ],
            "calibration": {
                "state_units": "MuJoCo hinge radians mapped to SO100/101 calibrated motor features",
                "action_units": "SO100/101 calibrated motor features mapped to MuJoCo ctrl radians",
                "arm_sim_home_rad": SO101_SIM_HOME_RAD.astype(float).tolist(),
                "arm_policy_home_deg": SO101_POLICY_HOME_DEG.astype(float).tolist(),
                "sim_to_policy_sign": SO101_SIM_TO_POLICY_SIGN.astype(float).tolist(),
                "sim_to_policy_offset_deg": SO101_SIM_TO_POLICY_OFFSET_DEG.astype(float).tolist(),
                "sim_action_bias_rad": SO101_SIM_ACTION_BIAS_RAD.astype(float).tolist(),
                "gripper_policy_state_range": [
                    SO101_POLICY_GRIPPER_STATE_MIN,
                    SO101_POLICY_GRIPPER_STATE_MAX,
                ],
                "gripper_policy_action_range": [
                    SO101_POLICY_GRIPPER_ACTION_MIN,
                    SO101_POLICY_GRIPPER_ACTION_MAX,
                ],
                "norm_tag": self.norm_tag,
                "num_steps": self.num_steps,
                "seed": self.seed,
                "episode_index": self._episode_index,
                "request_index": request_index,
                "dtype": str(self.dtype).removeprefix("torch."),
                "enable_cuda_graph": self.enable_cuda_graph,
            },
        }


def make_handler(runtime: MolmoAct2Runtime):
    class Handler(BaseHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            super().end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self.end_headers()

        def do_GET(self):
            if self.path not in ("/", "/health"):
                self.send_error(404)
                return
            self.send_json({
                "ok": True,
                "model_id": runtime.model_id,
                "device": str(runtime.device),
                "dtype": str(runtime.dtype).removeprefix("torch."),
                "policy_space": "molmoact2-so100_101-absolute-joint",
                "state_dim": 6,
                "image_features": {
                    "images": "one or more RGB images; duplicated if only one is provided",
                },
                "norm_tag": runtime.norm_tag,
                "num_steps": runtime.num_steps,
                "seed": runtime.seed,
                "enable_cuda_graph": runtime.enable_cuda_graph,
            })

        def do_POST(self):
            if self.path != "/infer":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length))
                self.send_json(runtime.infer(payload))
            except Exception as exc:  # noqa: BLE001
                self.send_json({"ok": False, "error": str(exc)}, status=500)

        def send_json(self, payload: dict[str, Any], status: int = 200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any):
            print(fmt % args, file=sys.stderr)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=env_default("MOLMO_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--host", default=env_default("POLICY_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=env_default_int("POLICY_PORT", DEFAULT_PORT))
    parser.add_argument("--device", default=env_default("POLICY_DEVICE", "auto"))
    parser.add_argument("--dtype", default=env_default("MOLMO_DTYPE", "bfloat16"))
    parser.add_argument("--norm-tag", default=env_default("MOLMO_NORM_TAG", DEFAULT_NORM_TAG))
    parser.add_argument("--num-steps", type=int, default=env_default_int("MOLMO_NUM_STEPS", 10))
    parser.add_argument("--seed", type=int, default=env_default_int("MOLMO_SEED", DEFAULT_SEED))
    parser.add_argument(
        "--enable-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=env_default_bool("MOLMO_ENABLE_CUDA_GRAPH", False),
    )
    args = parser.parse_args()

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype)
    runtime = MolmoAct2Runtime(
        model_id=args.model_id,
        device=device,
        dtype=dtype,
        norm_tag=args.norm_tag,
        num_steps=args.num_steps,
        seed=args.seed,
        enable_cuda_graph=args.enable_cuda_graph,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    print(f"MolmoAct2 policy server listening on http://{args.host}:{args.port}", flush=True)
    print(f"model={args.model_id} device={device} dtype={dtype}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
