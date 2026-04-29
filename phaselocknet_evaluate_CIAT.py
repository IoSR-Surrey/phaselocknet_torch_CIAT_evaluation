import argparse
import inspect
import os
import re
from pathlib import Path

import numpy as np
import torch
import torchaudio

import phaselocknet_model
import util
import util_localization_psychophysics as ulp

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # fallback to current frame (works when executed via exec/pyrun)
    try:
        SCRIPT_DIR = Path(inspect.getfile(inspect.currentframe())).resolve().parent
    except Exception:
        SCRIPT_DIR = Path.cwd()

HARDCODED_MODEL_PARENT_DIR = os.path.join(
    SCRIPT_DIR,
    "models",
    "sound_localization",
    "simplified_IHC3000_delayed_integration",
)


def _resolve_model_dir(model_dir_override=None):
    """Resolve model directory robustly across CLI and MATLAB sessions."""
    if model_dir_override:
        print(
            f"[phaselocknet_evaluate_CIAT] trying explicit model dir: "
            f"{model_dir_override}"
        )
        candidate = Path(model_dir_override).expanduser().resolve()
        if candidate.exists():
            print(f"[phaselocknet_evaluate_CIAT] using model dir: {candidate}")
            return str(candidate)
        raise FileNotFoundError(f"Model directory not found: {candidate}")

    env_override = os.environ.get("PHASELOCKNET_MODEL_DIR", "")
    if env_override:
        print(
            "[phaselocknet_evaluate_CIAT] trying PHASELOCKNET_MODEL_DIR: "
            f"{env_override}"
        )
        candidate = Path(env_override).expanduser().resolve()
        if candidate.exists():
            print(f"[phaselocknet_evaluate_CIAT] using model dir: {candidate}")
            return str(candidate)
        raise FileNotFoundError(
            f"PHASELOCKNET_MODEL_DIR is set but path does not exist: {candidate}"
        )

    candidates = [
        Path(HARDCODED_MODEL_PARENT_DIR),
        Path.cwd()
        / "models"
        / "sound_localization"
        / "simplified_IHC3000_delayed_integration",
    ]
    for candidate in candidates:
        print(f"[phaselocknet_evaluate_CIAT] trying fallback model dir: {candidate}")
        if (candidate / "arch01" / "config.json").exists():
            print(f"[phaselocknet_evaluate_CIAT] using model dir: {candidate}")
            return str(candidate.resolve())

    raise FileNotFoundError(
        "Could not resolve model directory. Tried HARDCODED_MODEL_PARENT_DIR and "
        "CWD-based path. Pass `--dir-model`, set `PHASELOCKNET_MODEL_DIR`, "
        "or update HARDCODED_MODEL_PARENT_DIR."
    )


def _parse_arch_indices(arch_indices):
    """
    Parse explicit architecture indices (1..10) from MATLAB or CLI.
    Supports list/tuple/np.ndarray or comma-separated string.
    """
    if arch_indices is None:
        return [1]

    values = arch_indices
    if isinstance(values, str):
        values = [v.strip() for v in values.split(",") if v.strip()]
    elif isinstance(values, np.ndarray):
        values = values.reshape(-1).tolist()
    elif isinstance(values, (int, float, np.integer, np.floating)):
        values = [int(values)]
    elif not isinstance(values, (list, tuple)):
        raise ValueError(
            "arch_indices must be a comma-separated string, scalar, list, tuple, "
            f"or numpy array; got type {type(values)}."
        )

    parsed = []
    for value in values:
        idx = int(value)
        if idx < 1 or idx > 10:
            raise ValueError(f"Invalid architecture index {idx}; expected 1..10.")
        if idx not in parsed:
            parsed.append(idx)

    if not parsed:
        raise ValueError("arch_indices resolved to an empty set.")
    return parsed


def _resolve_model_dir_for_arch(base_dir_model, arch_index):
    """
    Resolve model directory for the requested architecture index.
    """
    base = Path(base_dir_model).resolve()
    arch_name = f"arch{int(arch_index):02d}"

    if re.search(r"arch\d+$", base.name):
        candidate = base.parent / arch_name
    else:
        candidate = base / arch_name

    required_files = ["config.json", "arch.json", "ckpt_BEST.pt"]
    missing = [fn for fn in required_files if not (candidate / fn).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {candidate}: {missing}. "
            "Provide a base dir containing archXX folders or a valid arch dir."
        )
    return str(candidate)


def _build_frontal_elevation0_prior(num_classes):
    """Prior: frontal hemifield azimuths and elevation == 0."""
    labels = np.arange(num_classes)
    azim_deg, elev_deg = ulp.label_to_azim_elev(labels)
    azim_deg = ulp.normalize_angle(azim_deg)
    prior = np.logical_and.reduce(
        [
            azim_deg >= -90,
            azim_deg <= 90,
            elev_deg == 0,
        ]
    ).astype(np.float32)
    if np.sum(prior) == 0:
        raise ValueError("Prior has zero support.")
    return prior, azim_deg


def _prepare_input(signal, signal_sr, model, target_sr):
    """
    Convert MATLAB signal to model input tensor.

    Supported input shapes:
    - (n_samples, n_channels)
    - (n_positions, n_samples, n_channels)
    """
    print(
        "[phaselocknet_evaluate_CIAT] preparing input "
        f"(signal_sr={signal_sr}, target_sr={target_sr})"
    )
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim == 2:
        # Single example -> add batch dimension.
        x = torch.from_numpy(x).unsqueeze(0)
    elif x.ndim == 3:
        # Batched examples from MATLAB.
        x = torch.from_numpy(x)
    else:
        raise ValueError(
            "Expected shape (n_samples, n_channels) or "
            f"(n_positions, n_samples, n_channels), got shape {x.shape}."
        )
    print(f"[phaselocknet_evaluate_CIAT] input numpy shape: {x.shape}")

    if signal_sr != target_sr:
        print(
            "[phaselocknet_evaluate_CIAT] resampling input from "
            f"{signal_sr} Hz to {target_sr} Hz"
        )
        resampler = torchaudio.transforms.Resample(
            orig_freq=int(signal_sr),
            new_freq=int(target_sr),
        )
        x = torch.stack(
            [resampler(x[..., channel]) for channel in range(x.shape[-1])],
            axis=-1,
        )

    # Match expected channel count if needed.
    expected_channels = model.input_shape[-1] if len(model.input_shape) > 2 else 1
    if x.shape[-1] != expected_channels:
        print(
            "[phaselocknet_evaluate_CIAT] channel count mismatch: "
            f"got {x.shape[-1]}, expected {expected_channels}"
        )
        if x.shape[-1] == 1 and expected_channels > 1:
            x = torch.cat([x for _ in range(expected_channels)], dim=-1)
        else:
            raise ValueError(
                f"Model expects {expected_channels} channels, got {x.shape[-1]}."
            )

    x = util.pad_or_trim_to_len(x, n=model.input_shape[1], dim=1)
    print("[phaselocknet_evaluate_CIAT] tensor after pad/trim: " f"{tuple(x.shape)}")
    if list(x.shape[1:]) != list(model.input_shape[1:]):
        raise ValueError(
            f"Prepared input shape {tuple(x.shape)} does not match model input "
            f"shape {tuple(model.input_shape)}."
        )
    return x


def _run_single_model_inference(logits):
    """
    Run single-model inference from prepared input and decode localization outputs.
    """
    probs = torch.nn.functional.softmax(logits, dim=1)
    probs_np = probs.detach().cpu().numpy()
    num_positions = int(probs_np.shape[0])
    print(
        "[phaselocknet_evaluate_CIAT] stage: posterior processing "
        f"(num_positions={num_positions}, num_classes={probs_np.shape[1]})"
    )

    prior, azim_deg_all = _build_frontal_elevation0_prior(num_classes=probs_np.shape[1])
    posterior = probs_np * prior

    pred_idx = ulp.probs_to_label(probs_np, prior=prior).astype(int)
    argmax_azimuth_deg = azim_deg_all[pred_idx].astype(float)

    posterior_sum = np.sum(posterior, axis=1, keepdims=True)
    if np.any(posterior_sum <= 0):
        raise ValueError("Prior-weighted posterior sums to zero.")
    posterior = posterior / posterior_sum
    expected_azimuth_deg = np.sum(posterior * azim_deg_all.reshape(1, -1), axis=1)
    return {
        "predicted_class_index": pred_idx,
        "argmax_azimuth_deg": argmax_azimuth_deg,
        "expected_azimuth_deg": expected_azimuth_deg,
    }


def estimate_angle_from_signal(
    binaural_signal,
    signal_sr=50000,
    dir_model=None,
    eval_batch_size=1,
    arch_indices=None,
):
    """
    Run PhaselockNet on one or many binaural signals.

    Returns a dict with:
    - predicted_class_index: shape (n_positions,)
    - argmax_azimuth_deg: shape (n_positions,)
    - expected_azimuth_deg: shape (n_positions,)
    """
    print("[phaselocknet_evaluate_CIAT] stage: resolve model directory")
    base_dir_model = _resolve_model_dir(model_dir_override=dir_model)
    arch_indices = _parse_arch_indices(arch_indices)
    print(f"[phaselocknet_evaluate_CIAT] selected architectures: {arch_indices}")

    print("[phaselocknet_evaluate_CIAT] stage: initialize model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[phaselocknet_evaluate_CIAT] device: {device}")
    print(
        "[phaselocknet_evaluate_CIAT] stage: forward inference "
        f"(eval_batch_size={eval_batch_size})"
    )
    result_per_arch = []
    for i_arch, arch_idx in enumerate(arch_indices, start=1):
        arch_dir_model = _resolve_model_dir_for_arch(base_dir_model, arch_idx)
        print(
            "[phaselocknet_evaluate_CIAT] stage: architecture "
            f"[{i_arch}/{len(arch_indices)}] arch{arch_idx:02d}"
        )
        model, config_model = phaselocknet_model.get_model(
            dir_model=arch_dir_model,
            fn_config="config.json",
            fn_arch="arch.json",
        )
        print("[phaselocknet_evaluate_CIAT] stage: load checkpoint ckpt_BEST.pt")
        util.load_model_checkpoint(
            model=model.perceptual_model,
            dir_model=arch_dir_model,
            step=None,
            fn_ckpt="ckpt_BEST.pt",
        )
        model.eval()
        model.to(device)

        print("[phaselocknet_evaluate_CIAT] stage: preprocess input signal")
        sr_model = int(config_model["kwargs_cochlea"]["sr_input"])
        x = _prepare_input(
            signal=binaural_signal,
            signal_sr=int(signal_sr),
            model=model,
            target_sr=sr_model,
        )
        num_positions = int(x.shape[0])
        logits_chunks = []
        with torch.no_grad():
            for i0 in range(0, num_positions, int(eval_batch_size)):
                i1 = min(i0 + int(eval_batch_size), num_positions)
                print(
                    "[phaselocknet_evaluate_CIAT] forward chunk "
                    f"{i0}:{i1} / {num_positions}"
                )
                x_chunk = x[i0:i1].to(device)
                logits_chunk_by_task = model(x_chunk)
                logits_chunks.append(logits_chunk_by_task)

        first_task = sorted(logits_chunks[0].keys())[0]
        logits = torch.cat(
            [chunk[first_task].detach().cpu() for chunk in logits_chunks],
            dim=0,
        )
        result_per_arch.append(_run_single_model_inference(logits))
        del model

    # Aggregate final estimates across models (precedence-style interpretation).
    pred_idx_mat = np.stack(
        [entry["predicted_class_index"] for entry in result_per_arch], axis=0
    )
    argmax_azim_mat = np.stack(
        [entry["argmax_azimuth_deg"] for entry in result_per_arch], axis=0
    )
    expected_azim_mat = np.stack(
        [entry["expected_azimuth_deg"] for entry in result_per_arch], axis=0
    )
    argmax_azimuth_deg_ensemble = np.mean(argmax_azim_mat, axis=0)
    expected_azimuth_deg_ensemble = np.mean(expected_azim_mat, axis=0)

    # Nearest frontal/elev=0 class to ensemble expected azimuth.
    prior, azim_deg_all = _build_frontal_elevation0_prior(num_classes=504)
    support_azim = azim_deg_all[prior.astype(bool)]
    support_labels = np.arange(azim_deg_all.shape[0])[prior.astype(bool)]
    diff = np.abs(
        expected_azimuth_deg_ensemble.reshape(-1, 1) - support_azim.reshape(1, -1)
    )
    nearest_idx = np.argmin(diff, axis=1)
    pred_idx_ensemble = support_labels[nearest_idx].astype(int)
    print("[phaselocknet_evaluate_CIAT] stage: completed inference")

    return {
        "arch_indices": np.array(arch_indices, dtype=int),
        "per_model": {
            "predicted_class_index": pred_idx_mat,
            "argmax_azimuth_deg": argmax_azim_mat,
            "expected_azimuth_deg": expected_azim_mat,
        },
        "predicted_class_index": pred_idx_ensemble,
        "argmax_azimuth_deg": argmax_azimuth_deg_ensemble,
        "expected_azimuth_deg": expected_azimuth_deg_ensemble,
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run PhaselockNet inference for one binaural signal."
    )
    parser.add_argument(
        "--signal-npy",
        type=str,
        default=None,
        help=(
            "Optional path to .npy array with shape (n_samples, n_channels) "
            "or (n_positions, n_samples, n_channels)."
        ),
    )
    parser.add_argument(
        "--signal-sr",
        type=int,
        default=50000,
        help="Input sample rate (Hz).",
    )
    parser.add_argument(
        "--dir-model",
        type=str,
        default=None,
        help=(
            "Optional explicit model directory containing config.json, "
            "arch.json, and ckpt_BEST.pt."
        ),
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1,
        help="Mini-batch size used during forward inference.",
    )
    parser.add_argument(
        "--arch-indices",
        type=str,
        default="1",
        help="Comma-separated architecture indices from 1..10, e.g. '1,2,3'.",
    )
    return parser.parse_args()


def main():
    """
    Supports two modes:
    1) MATLAB pyrunfile globals: binaural_signal (+ optional signal_sr)
    2) CLI argument: --signal-npy /path/to/signal.npy --signal-sr 50000
    """
    args = _parse_args()
    print("[phaselocknet_evaluate_CIAT] starting script")

    if "binaural_signal" in globals():
        print("[phaselocknet_evaluate_CIAT] input source: MATLAB globals")
        signal = globals()["binaural_signal"]
        signal_sr = int(globals().get("signal_sr", args.signal_sr))
        dir_model = globals().get("dir_model", args.dir_model)
        eval_batch_size = int(globals().get("eval_batch_size", args.eval_batch_size))
        arch_indices = globals().get("arch_indices", args.arch_indices)
    elif args.signal_npy is not None:
        print(
            "[phaselocknet_evaluate_CIAT] input source: --signal-npy "
            f"({args.signal_npy})"
        )
        signal = np.load(args.signal_npy)
        signal_sr = int(args.signal_sr)
        dir_model = args.dir_model
        eval_batch_size = int(args.eval_batch_size)
        arch_indices = args.arch_indices
    else:
        raise ValueError(
            "Provide input via MATLAB global `binaural_signal` or CLI `--signal-npy`."
        )

    result = estimate_angle_from_signal(
        binaural_signal=signal,
        signal_sr=signal_sr,
        dir_model=dir_model,
        eval_batch_size=eval_batch_size,
        arch_indices=arch_indices,
    )
    # Export one MATLAB-friendly container from a single script execution.
    result = {
        "arch_indices": result["arch_indices"].tolist(),
        "per_model": {
            "predicted_class_index": result["per_model"][
                "predicted_class_index"
            ].tolist(),
            "argmax_azimuth_deg": result["per_model"]["argmax_azimuth_deg"].tolist(),
            "expected_azimuth_deg": result["per_model"][
                "expected_azimuth_deg"
            ].tolist(),
        },
        "predicted_class_index": result["predicted_class_index"].tolist(),
        "argmax_azimuth_deg": result["argmax_azimuth_deg"].tolist(),
        "expected_azimuth_deg": result["expected_azimuth_deg"].tolist(),
    }
    print("[phaselocknet_evaluate_CIAT] exporting outputs to globals")
    globals()["result"] = result
    # Ensemble outputs (averaged across selected architectures).
    globals()["estimated_class_index"] = result["predicted_class_index"]
    globals()["estimated_angle"] = result["argmax_azimuth_deg"]
    globals()["estimated_angle_expected"] = result["expected_azimuth_deg"]
    # Per-model outputs with shape (n_models, n_positions).
    globals()["estimated_class_index_per_model"] = result["per_model"][
        "predicted_class_index"
    ]
    globals()["estimated_angle_per_model"] = result["per_model"]["argmax_azimuth_deg"]
    globals()["estimated_angle_expected_per_model"] = result["per_model"][
        "expected_azimuth_deg"
    ]
    globals()["used_arch_indices"] = result["arch_indices"]
    if len(result["predicted_class_index"]) == 1:
        print(
            f"estimated_class_index={result['predicted_class_index'][0]}",
            f"estimated_angle={result['argmax_azimuth_deg'][0]}",
            f"estimated_angle_expected={result['expected_azimuth_deg']}",
        )
    n_models = len(result["per_model"]["argmax_azimuth_deg"])
    n_positions = (
        len(result["per_model"]["argmax_azimuth_deg"][0]) if n_models > 0 else 0
    )
    print(
        "[phaselocknet_evaluate_CIAT] output shapes: "
        f"ensemble={len(result['argmax_azimuth_deg'])}, "
        f"per_model=({n_models}, {n_positions})"
    )


if __name__ == "__main__":
    main()
