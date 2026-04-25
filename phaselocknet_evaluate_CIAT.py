import argparse
import inspect
import os
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

HARDCODED_MODEL_DIR = os.path.join(
    SCRIPT_DIR,
    "models",
    "sound_localization",
    "simplified_IHC3000_delayed_integration",
    "arch01",
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
        Path(HARDCODED_MODEL_DIR),
        Path.cwd()
        / "models"
        / "sound_localization"
        / "simplified_IHC3000_delayed_integration"
        / "arch01",
    ]
    for candidate in candidates:
        print(f"[phaselocknet_evaluate_CIAT] trying fallback model dir: {candidate}")
        if (candidate / "config.json").exists():
            print(f"[phaselocknet_evaluate_CIAT] using model dir: {candidate}")
            return str(candidate.resolve())

    raise FileNotFoundError(
        "Could not resolve model directory. Tried HARDCODED_MODEL_DIR and "
        "CWD-based path. Pass `--dir-model`, set `PHASELOCKNET_MODEL_DIR`, "
        "or update HARDCODED_MODEL_DIR."
    )


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
    """Convert MATLAB signal (n_samples, n_channels) to model input tensor."""
    print(
        "[phaselocknet_evaluate_CIAT] preparing input "
        f"(signal_sr={signal_sr}, target_sr={target_sr})"
    )
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(
            f"Expected shape (n_samples, n_channels), got shape {x.shape}."
        )
    print(f"[phaselocknet_evaluate_CIAT] input numpy shape: {x.shape}")

    # Batch dimension -> (1, n_samples, n_channels)
    x = torch.from_numpy(x).unsqueeze(0)

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


def estimate_angle_from_signal(binaural_signal, signal_sr=50000, dir_model=None):
    """
    Run PhaselockNet on one binaural signal.

    Returns a dict with:
    - predicted_class_index: argmax class index under prior
    - argmax_azimuth_deg: azimuth mapped from predicted class
    - expected_azimuth_deg: posterior expectation in degrees under prior
    """
    print("[phaselocknet_evaluate_CIAT] stage: resolve model directory")
    dir_model = _resolve_model_dir(model_dir_override=dir_model)
    print("[phaselocknet_evaluate_CIAT] stage: initialize model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[phaselocknet_evaluate_CIAT] device: {device}")
    model, config_model = phaselocknet_model.get_model(
        dir_model=dir_model,
        fn_config="config.json",
        fn_arch="arch.json",
    )
    print("[phaselocknet_evaluate_CIAT] stage: load checkpoint ckpt_BEST.pt")
    util.load_model_checkpoint(
        model=model.perceptual_model,
        dir_model=dir_model,
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
    ).to(device)

    print("[phaselocknet_evaluate_CIAT] stage: forward inference")
    with torch.no_grad():
        logits_by_task = model(x)

    # Use the first task head when a multi-head dict is returned.
    first_task = sorted(logits_by_task.keys())[0]
    probs = torch.nn.functional.softmax(logits_by_task[first_task], dim=1)[0]
    probs_np = probs.detach().cpu().numpy()
    print(
        "[phaselocknet_evaluate_CIAT] stage: posterior processing "
        f"(num_classes={probs_np.shape[0]})"
    )

    prior, azim_deg_all = _build_frontal_elevation0_prior(num_classes=probs_np.shape[0])
    posterior = probs_np * prior

    pred_idx = int(ulp.probs_to_label(probs_np.reshape(1, -1), prior=prior)[0])
    argmax_azimuth_deg = float(azim_deg_all[pred_idx])

    posterior_sum = float(np.sum(posterior))
    if posterior_sum <= 0:
        raise ValueError("Prior-weighted posterior sums to zero.")
    posterior = posterior / posterior_sum
    expected_azimuth_deg = float(np.sum(posterior * azim_deg_all))
    print("[phaselocknet_evaluate_CIAT] stage: completed inference")

    return {
        "predicted_class_index": pred_idx,
        "argmax_azimuth_deg": argmax_azimuth_deg,
        "expected_azimuth_deg": expected_azimuth_deg,
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run PhaselockNet inference for one binaural signal."
    )
    parser.add_argument(
        "--signal-npy",
        type=str,
        default=None,
        help="Optional path to .npy array with shape (n_samples, n_channels).",
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
    elif args.signal_npy is not None:
        print(
            "[phaselocknet_evaluate_CIAT] input source: --signal-npy "
            f"({args.signal_npy})"
        )
        signal = np.load(args.signal_npy)
        signal_sr = int(args.signal_sr)
        dir_model = args.dir_model
    else:
        raise ValueError(
            "Provide input via MATLAB global `binaural_signal` or CLI `--signal-npy`."
        )

    result = estimate_angle_from_signal(
        binaural_signal=signal,
        signal_sr=signal_sr,
        dir_model=dir_model,
    )
    print("[phaselocknet_evaluate_CIAT] exporting outputs to globals")
    globals()["estimated_class_index"] = result["predicted_class_index"]
    globals()["estimated_angle"] = result["argmax_azimuth_deg"]
    globals()["estimated_angle_expected"] = result["expected_azimuth_deg"]
    print(
        "estimated_class_index={idx}, estimated_angle={argmax_deg}, "
        "estimated_angle_expected={expected_deg}".format(
            idx=result["predicted_class_index"],
            argmax_deg=result["argmax_azimuth_deg"],
            expected_deg=result["expected_azimuth_deg"],
        )
    )


if __name__ == "__main__":
    main()
