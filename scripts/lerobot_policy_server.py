#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import packaging.version  # noqa: F401 - LeRobot expects packaging.version to be imported.
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file as load_safetensors

from lerobot.datasets import LeRobotDatasetMetadata
import lerobot.policies.factory  # noqa: F401 - registers policy config choices.
from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.factory import get_policy_class
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference


DEFAULT_MODEL_ID = "davidlinjiahao/lerobot_so101_base_sim_pickplace"
DEFAULT_DATASET_ID = "davidlinjiahao/lerobot_batch_001"
DEFAULT_POLICY_SPACE = "sim-radians-act"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8776
SO101_ACTION_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
SO101_GRIPPER_MIN_RAD = np.deg2rad(-10.0)
SO101_GRIPPER_MAX_RAD = np.deg2rad(100.0)
SO101_POLICY_HOME_DEG = np.asarray(
    [1.9561, -98.7437, 98.9242, 74.8198, -51.4530],
    dtype=np.float32,
)
SO101_SIM_HOME_RAD = np.asarray(
    [0.03414, -1.7234, 1.72655, 1.30585, -0.89802],
    dtype=np.float32,
)
SO101_SIM_TO_POLICY_OFFSET_DEG = SO101_POLICY_HOME_DEG - np.rad2deg(SO101_SIM_HOME_RAD)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def decode_image(value: str | None, width: int, height: int) -> np.ndarray:
    if not value:
        return np.zeros((height, width, 3), dtype=np.uint8)

    if "," in value and value.lstrip().startswith("data:"):
        value = value.split(",", 1)[1]

    raw = base64.b64decode(value)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def sim_state_to_policy_units(state: np.ndarray) -> np.ndarray:
    policy_state = np.empty_like(state, dtype=np.float32)
    policy_state[:5] = np.rad2deg(state[:5]) + SO101_SIM_TO_POLICY_OFFSET_DEG
    gripper = np.clip(state[5], SO101_GRIPPER_MIN_RAD, SO101_GRIPPER_MAX_RAD)
    policy_state[5] = (gripper - SO101_GRIPPER_MIN_RAD) / (SO101_GRIPPER_MAX_RAD - SO101_GRIPPER_MIN_RAD) * 100.0
    return policy_state


def policy_action_to_sim_radians(action: np.ndarray) -> np.ndarray:
    sim_action = np.empty_like(action, dtype=np.float32)
    sim_action[:5] = np.deg2rad(action[:5] - SO101_SIM_TO_POLICY_OFFSET_DEG)
    gripper_percent = np.clip(action[5], 0.0, 100.0)
    sim_action[5] = SO101_GRIPPER_MIN_RAD + (gripper_percent / 100.0) * (
        SO101_GRIPPER_MAX_RAD - SO101_GRIPPER_MIN_RAD
    )
    return sim_action


def mujoco_degree_action_to_sim_radians(action: np.ndarray) -> np.ndarray:
    sim_action = np.empty_like(action, dtype=np.float32)
    sim_action[:] = np.deg2rad(action[:])
    sim_action[5] = np.clip(sim_action[5], SO101_GRIPPER_MIN_RAD, SO101_GRIPPER_MAX_RAD)
    return sim_action


def mujoco_degree_arm_percent_gripper_action_to_sim_radians(action: np.ndarray) -> np.ndarray:
    sim_action = np.empty_like(action, dtype=np.float32)
    sim_action[:5] = np.deg2rad(action[:5])
    gripper_percent = np.clip(action[5], 0.0, 100.0)
    sim_action[5] = SO101_GRIPPER_MIN_RAD + (gripper_percent / 100.0) * (
        SO101_GRIPPER_MAX_RAD - SO101_GRIPPER_MIN_RAD
    )
    return sim_action


def mujoco_radian_state_to_degree_state(state: np.ndarray) -> np.ndarray:
    return np.rad2deg(state).astype(np.float32)


def make_ordered_action(action: torch.Tensor, features: dict[str, Any], action_names: list[str]) -> np.ndarray:
    action_dict = make_robot_action(action, features)
    return np.asarray([action_dict[name] for name in action_names], dtype=np.float32)


def episode_stat_count(stat: dict[str, Any]) -> float:
    count = np.asarray(stat["count"], dtype=np.float64).reshape(-1)
    return float(count[0])


def load_v21_dataset_stats(dataset_id: str) -> dict[str, dict[str, np.ndarray]]:
    stats_path = hf_hub_download(dataset_id, "meta/episodes_stats.jsonl", repo_type="dataset")
    accumulators: dict[str, dict[str, Any]] = {}

    with open(stats_path, encoding="utf-8") as stats_file:
        for line in stats_file:
            if not line.strip():
                continue
            episode_stats = json.loads(line)["stats"]
            for key, stat in episode_stats.items():
                count = episode_stat_count(stat)
                if count <= 0:
                    continue

                minimum = np.asarray(stat["min"], dtype=np.float64)
                maximum = np.asarray(stat["max"], dtype=np.float64)
                mean = np.asarray(stat["mean"], dtype=np.float64)
                std = np.asarray(stat["std"], dtype=np.float64)
                second_moment = std**2 + mean**2

                accumulator = accumulators.get(key)
                if accumulator is None:
                    accumulators[key] = {
                        "count": count,
                        "min": minimum,
                        "max": maximum,
                        "mean_sum": mean * count,
                        "second_sum": second_moment * count,
                    }
                else:
                    accumulator["count"] += count
                    accumulator["min"] = np.minimum(accumulator["min"], minimum)
                    accumulator["max"] = np.maximum(accumulator["max"], maximum)
                    accumulator["mean_sum"] += mean * count
                    accumulator["second_sum"] += second_moment * count

    if not accumulators:
        raise RuntimeError(f"No episode stats found for dataset {dataset_id}")

    stats: dict[str, dict[str, np.ndarray]] = {}
    for key, accumulator in accumulators.items():
        count = float(accumulator["count"])
        mean = accumulator["mean_sum"] / count
        variance = np.maximum(accumulator["second_sum"] / count - mean**2, 1e-12)
        stats[key] = {
            "min": accumulator["min"].astype(np.float32),
            "max": accumulator["max"].astype(np.float32),
            "mean": mean.astype(np.float32),
            "std": np.sqrt(variance).astype(np.float32),
            "count": np.asarray([int(count)], dtype=np.int64),
        }

    return stats


def arrays_from_json_stats(raw_stats: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    stats: dict[str, dict[str, np.ndarray]] = {}
    for key, feature_stats in raw_stats.items():
        if not isinstance(feature_stats, dict):
            continue
        converted: dict[str, np.ndarray] = {}
        for stat_name in ("min", "max", "mean", "std", "count"):
            if stat_name in feature_stats:
                converted[stat_name] = np.asarray(feature_stats[stat_name], dtype=np.float32)
        if converted:
            stats[key] = converted
    return stats


def load_v30_dataset_metadata(dataset_id: str) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any] | None]:
    stats_path = hf_hub_download(dataset_id, "meta/stats.json", repo_type="dataset")
    with open(stats_path, encoding="utf-8") as stats_file:
        stats = arrays_from_json_stats(json.load(stats_file))

    features = None
    try:
        info_path = hf_hub_download(dataset_id, "meta/info.json", repo_type="dataset")
        with open(info_path, encoding="utf-8") as info_file:
            features = json.load(info_file).get("features")
    except Exception:  # noqa: BLE001
        features = None

    return stats, features


def load_dataset_metadata_or_stats(dataset_id: str) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any] | None, str]:
    try:
        metadata = LeRobotDatasetMetadata(dataset_id)
        return metadata.stats, metadata.features, "lerobot-metadata"
    except Exception as exc:  # noqa: BLE001
        print(
            f"Could not load LeRobotDatasetMetadata for {dataset_id}; "
            f"falling back to v2.1 episode stats: {exc}",
            file=sys.stderr,
        )
        try:
            stats, features = load_v30_dataset_metadata(dataset_id)
            return stats, features, "v3-meta-stats"
        except Exception as v30_exc:  # noqa: BLE001
            print(
                f"Could not load v3 meta/stats.json for {dataset_id}; "
                f"falling back to v2.1 episode stats: {v30_exc}",
                file=sys.stderr,
            )
            return load_v21_dataset_stats(dataset_id), None, "v2.1-episodes_stats"


def model_has_policy_processors(model_id: str) -> bool:
    try:
        hf_hub_download(model_id, "policy_preprocessor.json")
        hf_hub_download(model_id, "policy_postprocessor.json")
    except Exception:  # noqa: BLE001
        return False
    return True


def checkpoint_buffer_feature_name(buffer_name: str) -> str | None:
    if buffer_name == "observation_state":
        return "observation.state"
    if buffer_name.startswith("observation_images_"):
        return "observation.images." + buffer_name.removeprefix("observation_images_")
    if buffer_name == "action":
        return "action"
    return None


def override_mujoco_smolvla_features(
    cfg: PreTrainedConfig,
    model_id: str,
    dataset_id: str,
    enabled: bool,
) -> bool:
    if not enabled:
        return False

    hint = f"{model_id} {dataset_id}".lower()
    if cfg.type != "smolvla" or ("mujoco" not in hint and "so101-merged" not in hint):
        return False
    if "observation.images.image" not in cfg.input_features:
        return False

    cfg.input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }
    return True


def apply_checkpoint_normalization_stats(
    model_id: str,
    stats: dict[str, dict[str, np.ndarray]],
) -> bool:
    model_path = hf_hub_download(model_id, "model.safetensors")
    tensors = load_safetensors(model_path, device="cpu")
    updated = False

    for tensor_name, tensor in tensors.items():
        if ".buffer_" not in tensor_name:
            continue
        if not tensor_name.endswith(".mean") and not tensor_name.endswith(".std"):
            continue

        _, suffix = tensor_name.split(".buffer_", 1)
        buffer_name, stat_name = suffix.rsplit(".", 1)
        feature_name = checkpoint_buffer_feature_name(buffer_name)
        if feature_name is None:
            continue

        feature_stats = stats.setdefault(feature_name, {})
        feature_stats[stat_name] = tensor.detach().cpu().numpy().astype(np.float32)
        updated = True

    return updated


class LeRobotPolicyRuntime:
    def __init__(
        self,
        model_id: str,
        dataset_id: str,
        device: torch.device,
        policy_space: str,
        feature_override: bool,
    ):
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.device = device

        cfg = PreTrainedConfig.from_pretrained(model_id)
        cfg.device = str(device)
        if getattr(cfg, "compile_model", False) and (
            device.type == "mps" or os.environ.get("LEROBOT_DISABLE_TORCH_COMPILE") == "1"
        ):
            print(
                f"Disabling torch.compile for {model_id} on device {device}; "
                "compiled SmolVLA inference is not reliable on this backend.",
                file=sys.stderr,
            )
            cfg.compile_model = False
        self.feature_override = override_mujoco_smolvla_features(cfg, model_id, dataset_id, feature_override)
        self.uses_model_processors = model_has_policy_processors(model_id)
        try:
            self.dataset_stats, self.dataset_features, self.stats_source = load_dataset_metadata_or_stats(dataset_id)
        except Exception as exc:  # noqa: BLE001
            if not self.uses_model_processors:
                raise
            print(
                f"Could not load dataset stats for {dataset_id}; "
                f"using bundled model processors from {model_id}: {exc}",
                file=sys.stderr,
            )
            self.dataset_stats = {}
            self.dataset_features = None
            self.stats_source = "model-processors"
        if not self.uses_model_processors and apply_checkpoint_normalization_stats(model_id, self.dataset_stats):
            self.stats_source += "+checkpoint_mean_std"

        policy_cls = get_policy_class(cfg.type)
        self.policy = policy_cls.from_pretrained(model_id, config=cfg)
        self.policy.to(device)
        self.policy.eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            pretrained_path=model_id if self.uses_model_processors else None,
            dataset_stats=self.dataset_stats,
            preprocessor_overrides={"device_processor": {"device": str(device)}} if self.uses_model_processors else None,
            postprocessor_overrides={"device_processor": {"device": str(device)}} if self.uses_model_processors else None,
        )

        self.state_key = next(
            key for key, feature in self.policy.config.input_features.items() if feature.type == FeatureType.STATE
        )
        self.state_shape = tuple(self.policy.config.input_features[self.state_key].shape)
        if len(self.state_shape) != 1:
            raise ValueError(f"Expected 1D state feature, got {self.state_key} shape {self.state_shape}")
        self.state_dim = int(self.state_shape[0])
        self.image_features = {
            key: feature.shape for key, feature in self.policy.config.input_features.items()
            if feature.type == FeatureType.VISUAL
        }
        self.policy_space = self.resolve_policy_space(policy_space)
        self.uses_real_so101_units = self.policy_space == "smolvla-real-camera-calibrated"
        self.action_names = SO101_ACTION_NAMES if self.uses_real_so101_units else [
            f"action_{index}" for index in range(6)
        ]

    def resolve_policy_space(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if self.state_dim != 6:
            return "sim-radians-act"
        dataset_hint = self.dataset_id.lower()
        model_hint = self.model_id.lower()
        if "mujoco" in dataset_hint or "mujoco" in model_hint or "sim" in dataset_hint:
            action_stats = self.dataset_stats.get("action", {})
            action_max = np.asarray(action_stats.get("max", []), dtype=np.float32).reshape(-1)
            action_min = np.asarray(action_stats.get("min", []), dtype=np.float32).reshape(-1)
            if action_max.size >= 5 and action_min.size >= 5:
                joint_abs_max = float(np.max(np.abs(np.concatenate([action_min[:5], action_max[:5]]))))
                if joint_abs_max <= np.pi + 0.25:
                    return "sim-radians-act"
                if action_max.size >= 6 and action_min.size >= 6:
                    gripper_min = float(action_min[5])
                    gripper_max = float(action_max[5])
                    if gripper_min >= -1.0 and gripper_max >= 50.0:
                        return "mujoco-radians-state-degrees-action-gripper-percent"
            return "mujoco-radians-state-degrees-action-smolvla"
        return "smolvla-real-camera-calibrated"

    def reset(self) -> None:
        self.policy.reset()
        self.preprocess.reset()
        self.postprocess.reset()

    def calibration(self) -> dict[str, Any]:
        if self.uses_real_so101_units:
            return {
                "arm_policy_home_deg": SO101_POLICY_HOME_DEG.astype(float).tolist(),
                "arm_sim_home_rad": SO101_SIM_HOME_RAD.astype(float).tolist(),
                "sim_to_policy_offset_deg": SO101_SIM_TO_POLICY_OFFSET_DEG.astype(float).tolist(),
                "gripper_policy_units": "range_0_100",
            }
        calibration = {
            "state_units": "sim_radians",
            "gripper_closed_rad": float(SO101_GRIPPER_MIN_RAD),
            "gripper_open_rad": float(SO101_GRIPPER_MAX_RAD),
        }
        if self.policy_space == "mujoco-radians-state-degrees-action-smolvla":
            calibration["action_units"] = "joint_degrees"
        elif self.policy_space == "mujoco-radians-state-degrees-action-gripper-percent":
            calibration["action_units"] = "arm_joint_degrees_gripper_percent"
        elif self.policy_space == "mujoco-radians-state-degrees-state-action":
            calibration["state_units"] = "sim_radians_converted_to_joint_degrees"
            calibration["action_units"] = "joint_degrees_converted_to_sim_radians"
        else:
            calibration["action_units"] = "sim_radians"
        if self.feature_override:
            calibration["input_features_override"] = "mujoco_smolvla_image_state"
        return calibration

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("reset"):
            self.reset()

        state = np.asarray(payload.get("state", []), dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise ValueError(f"Expected state shape [{self.state_dim}], got {list(state.shape)}")
        if self.uses_real_so101_units:
            state = sim_state_to_policy_units(state)
        elif self.policy_space == "mujoco-radians-state-degrees-state-action":
            state = mujoco_radian_state_to_degree_state(state)

        images = payload.get("images") or {}
        observation: dict[str, np.ndarray] = {self.state_key: state}

        for key, shape in self.image_features.items():
            _, height, width = shape
            short_name = key.removeprefix("observation.images.")
            observation[key] = decode_image(images.get(key) or images.get(short_name), width, height)

        with torch.inference_mode():
            batch = prepare_observation_for_inference(
                observation,
                self.device,
                task=payload.get("task") or "pick and place the cube",
                robot_type=payload.get("robot_type") or "so100_follower",
            )
            batch = self.preprocess(batch)
            action_chunk = self.policy.predict_action_chunk(batch)
            action_chunk = self.postprocess(action_chunk).squeeze(0).detach().to("cpu")

            if self.uses_real_so101_units:
                if self.dataset_features is None:
                    raise RuntimeError("SmolVLA action ordering requires dataset metadata features")
                action_policy_chunk = np.stack([
                    make_ordered_action(step, self.dataset_features, self.action_names)
                    for step in action_chunk
                ])
                action_chunk = np.stack([
                    policy_action_to_sim_radians(step)
                    for step in action_policy_chunk
                ])
            else:
                action_policy_chunk = action_chunk.numpy().astype(np.float32)
                if self.policy_space == "mujoco-radians-state-degrees-action-smolvla":
                    action_chunk = np.stack([
                        mujoco_degree_action_to_sim_radians(step)
                        for step in action_policy_chunk
                    ])
                elif self.policy_space == "mujoco-radians-state-degrees-action-gripper-percent":
                    action_chunk = np.stack([
                        mujoco_degree_arm_percent_gripper_action_to_sim_radians(step)
                        for step in action_policy_chunk
                    ])
                elif self.policy_space == "mujoco-radians-state-degrees-state-action":
                    action_chunk = np.stack([
                        mujoco_degree_action_to_sim_radians(step)
                        for step in action_policy_chunk
                    ])
                else:
                    action_chunk = action_policy_chunk

            action_policy_units = action_policy_chunk[0].astype(float).tolist()
            action = action_chunk[0].astype(float).tolist()
            actions = action_chunk.astype(float).tolist()

        return {
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "device": str(self.device),
            "policy_space": self.policy_space,
            "state_dim": self.state_dim,
            "stats_source": self.stats_source,
            "action": action,
            "actions": actions,
            "action_policy_units": action_policy_units,
            "action_names": self.action_names,
            "calibration": self.calibration(),
        }


def make_handler(runtime: LeRobotPolicyRuntime):
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
                "dataset_id": runtime.dataset_id,
                "device": str(runtime.device),
                "state_key": runtime.state_key,
                "state_dim": runtime.state_dim,
                "image_features": runtime.image_features,
                "action_names": runtime.action_names,
                "policy_space": runtime.policy_space,
                "stats_source": runtime.stats_source,
                "uses_model_processors": runtime.uses_model_processors,
                "calibration": runtime.calibration(),
                "feature_override": runtime.feature_override,
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
    parser.add_argument("--model-id", default=env_default("MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--dataset-id", default=env_default("DATASET_ID", DEFAULT_DATASET_ID))
    parser.add_argument("--host", default=env_default("POLICY_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=env_default_int("POLICY_PORT", DEFAULT_PORT))
    parser.add_argument("--device", default=env_default("POLICY_DEVICE", "auto"))
    parser.add_argument(
        "--policy-space",
        choices=[
            "auto",
            "smolvla-real-camera-calibrated",
            "mujoco-radians-state-degrees-action-smolvla",
            "mujoco-radians-state-degrees-action-gripper-percent",
            "mujoco-radians-state-degrees-state-action",
            "sim-radians-smolvla",
            "sim-radians-act",
        ],
        default=env_default("POLICY_SPACE", DEFAULT_POLICY_SPACE),
    )
    parser.set_defaults(feature_override=True)
    parser.add_argument(
        "--feature-override",
        dest="feature_override",
        action="store_true",
        help="Use the single-image SmolVLA feature override when applicable.",
    )
    parser.add_argument(
        "--no-feature-override",
        dest="feature_override",
        action="store_false",
        help="Preserve the checkpoint-declared SmolVLA visual feature layout.",
    )
    args = parser.parse_args()

    device = choose_device(args.device)
    runtime = LeRobotPolicyRuntime(args.model_id, args.dataset_id, device, args.policy_space, args.feature_override)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    print(f"LeRobot policy server listening on http://{args.host}:{args.port}")
    print(f"model={args.model_id} device={device}")
    server.serve_forever()


if __name__ == "__main__":
    main()
