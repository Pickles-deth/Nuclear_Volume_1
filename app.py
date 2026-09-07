import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import re
import time
import colorsys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tifffile
import plotly.graph_objects as go

from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import marching_cubes


# ============================================================
# Streamlit
# ============================================================

st.set_page_config(
    page_title="3D Nuclear Volume Analyzer - ROI Fast",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 3D Nuclear Volume Analyzer - ROI Fast")
st.caption(
    "元画像のXY/Z解像度は変更しません。StarDist 3Dは核のseed検出に使い、"
    "最終的な体積はseedにつながる白いDAPI核全体の3D領域から計算します。"
)


# ============================================================
# Session state
# ============================================================

DEFAULTS = {
    "raw_stack": None,
    "file_signature": None,
    "candidates": None,
    "roi_results": None,
    "result_table": None,
    "prediction_done": False,
    "selected_nucleus": None,
    "excluded_ids": set(),
    "selected_roi_ids": None,
    "timings": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# General helpers
# ============================================================

def natural_key(name):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", str(name))
    ]


def normalize_for_display(img):
    img = np.asarray(img, dtype=np.float32)

    lo = float(np.percentile(img, 1.0))
    hi = float(np.percentile(img, 99.8))

    if hi <= lo:
        hi = lo + 1.0

    out = (img - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def object_color(object_id):
    hue = (int(object_id) * 0.618033988749895) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.85, 1.0)


def rgb255(rgb):
    return tuple(int(round(x * 255)) for x in rgb)


# ============================================================
# Loading
# ============================================================

def load_uploaded_stack(uploaded_files, channel):
    """
    Read 2D TIFF slices and stack as Z,Y,X.
    No resize / resampling.
    """
    uploaded_files = sorted(
        uploaded_files,
        key=lambda x: natural_key(x.name),
    )

    slices = []

    for f in uploaded_files:
        img = tifffile.imread(f)

        if img.ndim == 2:
            pass

        elif img.ndim == 3 and img.shape[-1] == 3:
            idx = {"R": 0, "G": 1, "B": 2}[channel]
            img = img[..., idx]

        elif img.ndim == 3 and img.shape[0] == 1:
            img = img[0]

        else:
            raise ValueError(
                f"{f.name}: 対応できないshapeです: {img.shape}"
            )

        slices.append(np.asarray(img))

    shapes = [x.shape for x in slices]

    if len(set(shapes)) != 1:
        raise ValueError(
            "すべてのZ画像でXYサイズを一致させてください。"
        )

    # Keep the original TIFF dtype in memory.
    # StarDist converts only the small ROI to float32 later.
    # This is important on Streamlit Community Cloud (limited RAM).
    stack = np.stack(slices, axis=0)
    if stack.dtype == np.float64:
        stack = stack.astype(np.float32)
    return stack


# ============================================================
# Candidate detection
# ============================================================

def detect_nucleus_candidates(
    stack,
    expected_count=5,
    xy_margin=96,
    z_margin=8,
    min_component_area=300,
    max_component_area_fraction=0.25,
):
    """
    Fast candidate detection on a 2D maximum-intensity projection.

    The candidate detector only determines ROIs.
    Final nucleus segmentation and volume are always obtained by StarDist3D.

    Returns one candidate ROI per detected component:
        z0,z1,y0,y1,x0,x1

    Coordinates follow Python slicing:
        stack[z0:z1, y0:y1, x0:x1]
    """
    zdim, h, w = stack.shape

    mip = np.max(stack, axis=0)
    norm = normalize_for_display(mip)

    sm = gaussian(
        norm,
        sigma=2.0,
        preserve_range=True,
    )

    try:
        otsu = float(threshold_otsu(sm))
    except Exception:
        otsu = 0.25

    # Slightly conservative threshold so nuclear edges remain included.
    thr = max(0.04, otsu * 0.80)
    mask = sm > thr

    # Clean small speckles and fill tiny holes.
    mask = ndi.binary_opening(
        mask,
        structure=np.ones((3, 3), dtype=bool),
    )
    mask = ndi.binary_closing(
        mask,
        structure=np.ones((5, 5), dtype=bool),
    )
    mask = ndi.binary_fill_holes(mask)

    labeled, n = ndi.label(mask)

    if n == 0:
        return [], mip, mask

    sizes = np.bincount(labeled.ravel())
    max_area = int(h * w * max_component_area_fraction)

    components = []

    for label_id in range(1, n + 1):
        area = int(sizes[label_id])

        if area < int(min_component_area):
            continue

        if area > max_area:
            # Usually background/merged illumination artifact.
            continue

        ys, xs = np.where(labeled == label_id)

        if len(xs) == 0:
            continue

        y_min = int(ys.min())
        y_max = int(ys.max())
        x_min = int(xs.min())
        x_max = int(xs.max())

        components.append(
            {
                "component_id": label_id,
                "area_px": area,
                "y_min": y_min,
                "y_max": y_max,
                "x_min": x_min,
                "x_max": x_max,
                "center_y": float(ys.mean()),
                "center_x": float(xs.mean()),
            }
        )

    # If thresholding fragmented a nucleus, keeping the largest few tends
    # to match the user's small-number-of-nuclei use case.
    components = sorted(
        components,
        key=lambda d: d["area_px"],
        reverse=True,
    )

    # Keep a few extras because StarDist can later reject false candidates.
    keep_n = max(int(expected_count) + 3, int(expected_count))
    components = components[:keep_n]

    candidates = []

    for i, comp in enumerate(components, start=1):
        y0 = max(0, comp["y_min"] - int(xy_margin))
        y1 = min(h, comp["y_max"] + int(xy_margin) + 1)
        x0 = max(0, comp["x_min"] - int(xy_margin))
        x1 = min(w, comp["x_max"] + int(xy_margin) + 1)

        # Determine useful Z range from intensity inside the XY candidate.
        # This does not alter the image; it only avoids empty Z planes.
        roi_xy = stack[:, y0:y1, x0:x1]

        # Robust high-intensity signal per z plane.
        z_signal = np.percentile(
            roi_xy.reshape(zdim, -1),
            99.5,
            axis=1,
        ).astype(np.float32)

        z_signal = ndi.gaussian_filter1d(
            z_signal,
            sigma=1.0,
        )

        baseline = float(np.percentile(z_signal, 20))
        peak = float(z_signal.max())

        if peak <= baseline:
            z0 = 0
            z1 = zdim
        else:
            z_thr = baseline + 0.20 * (peak - baseline)
            present = np.where(z_signal >= z_thr)[0]

            if len(present) == 0:
                z0 = 0
                z1 = zdim
            else:
                z0 = max(0, int(present.min()) - int(z_margin))
                z1 = min(zdim, int(present.max()) + int(z_margin) + 1)

                # Safety: do not make Z crop implausibly thin.
                if (z1 - z0) < 12:
                    mid = int(round((z0 + z1) / 2))
                    z0 = max(0, mid - 10)
                    z1 = min(zdim, mid + 11)

        candidates.append(
            {
                "candidate": i,
                "z0": z0,
                "z1": z1,
                "y0": y0,
                "y1": y1,
                "x0": x0,
                "x1": x1,
                "area_px": comp["area_px"],
                "center_y": comp["center_y"],
                "center_x": comp["center_x"],
            }
        )

    # Remove highly overlapping duplicate ROIs.
    filtered = []

    def intersection_over_smaller(a, b):
        iy0 = max(a["y0"], b["y0"])
        iy1 = min(a["y1"], b["y1"])
        ix0 = max(a["x0"], b["x0"])
        ix1 = min(a["x1"], b["x1"])

        if iy1 <= iy0 or ix1 <= ix0:
            return 0.0

        inter = (iy1 - iy0) * (ix1 - ix0)
        aa = (a["y1"] - a["y0"]) * (a["x1"] - a["x0"])
        bb = (b["y1"] - b["y0"]) * (b["x1"] - b["x0"])

        return inter / float(min(aa, bb))

    for cand in candidates:
        duplicate = False

        for kept in filtered:
            if intersection_over_smaller(cand, kept) > 0.85:
                duplicate = True
                break

        if not duplicate:
            filtered.append(cand)

    # Re-number.
    for i, c in enumerate(filtered, start=1):
        c["candidate"] = i

    return filtered, mip, mask


def make_candidate_figure(mip, candidates):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(normalize_for_display(mip), cmap="gray")

    for c in candidates:
        color = object_color(c["candidate"])

        rect = patches.Rectangle(
            (c["x0"], c["y0"]),
            c["x1"] - c["x0"],
            c["y1"] - c["y0"],
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            c["x0"] + 6,
            c["y0"] + 18,
            f"ROI {c['candidate']}",
            color="white",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                facecolor="black",
                alpha=0.65,
                edgecolor=color,
                pad=2,
            ),
        )

    ax.set_title("Detected nuclear candidate ROIs")
    ax.axis("off")
    fig.tight_layout()

    return fig


# ============================================================
# StarDist model - Windows 1314 fix
# ============================================================

@st.cache_resource(show_spinner=False)
def load_stardist_model():
    """
    Try standard pretrained loading first.

    Windows may raise WinError 1314 when CSBDeep tries to create a symbolic
    link. If so, load the extracted model directory directly.
    """
    from stardist.models import StarDist3D

    try:
        return StarDist3D.from_pretrained("3D_demo")

    except OSError as e:
        if getattr(e, "winerror", None) != 1314:
            raise

        model_root = (
            Path.home()
            / ".keras"
            / "models"
            / "StarDist3D"
        )

        candidates = []

        if model_root.exists():
            candidates.extend(
                model_root.rglob("config.json")
            )

        candidates = sorted(
            candidates,
            key=lambda p: (
                "3D_demo_extracted" not in str(p),
                "3D_demo" not in str(p),
                len(str(p)),
            ),
        )

        last_error = None

        for config_path in candidates:
            model_dir = config_path.parent

            try:
                return StarDist3D(
                    config=None,
                    name=model_dir.name,
                    basedir=model_dir.parent,
                )
            except Exception as inner:
                last_error = inner

        raise RuntimeError(
            "StarDist 3D_demoの展開済みモデルを直接読み込めませんでした。"
            "Windowsの開発者モードをONにするか、"
            "管理者権限で一度モデルを取得してください。"
        ) from (last_error or e)


def _is_memory_error(exc):
    msg = str(exc).lower()
    return (
        isinstance(exc, MemoryError)
        or "unable to allocate" in msg
        or "resourceexhausted" in msg
        or "oom" in msg
        or "out of memory" in msg
    )


def _tile_plans(tile_z, tile_y, tile_x):
    plans = [
        (int(tile_z), int(tile_y), int(tile_x)),
        (2, 1, 1),
        (2, 2, 2),
        (4, 1, 1),
        (4, 2, 2),
        (4, 3, 3),
        (8, 2, 2),
        (8, 3, 3),
    ]

    out = []
    seen = set()

    for p in plans:
        p = tuple(max(1, int(v)) for v in p)
        if p not in seen:
            out.append(p)
            seen.add(p)

    return out


def run_stardist_roi(
    stack_roi,
    prob_thresh,
    nms_thresh,
    tile_z,
    tile_y,
    tile_x,
):
    """
    Full-resolution StarDist3D on one ROI.

    - no resize / resampling
    - memory-safe tiling fallback
    - if StarDist returns 0 labels, retry with a slightly lower
      probability threshold
    """
    import gc
    from csbdeep.utils import normalize

    model = load_stardist_model()

    x = normalize(
        stack_roi,
        1,
        99.8,
        axis=(0, 1, 2),
    ).astype(np.float32, copy=False)

    requested_prob = float(prob_thresh)

    # Only lower the threshold when the previous attempt detected NOTHING.
    # This avoids unnecessarily huge NMS workloads.
    prob_plans = []
    for p in [
        requested_prob,
        min(requested_prob, 0.30),
        min(requested_prob, 0.25),
        min(requested_prob, 0.20),
        min(requested_prob, 0.15),
    ]:
        p = round(max(0.05, float(p)), 3)
        if p not in prob_plans:
            prob_plans.append(p)

    memory_errors = []
    zero_detection_attempts = []

    for p in prob_plans:
        for n_tiles in _tile_plans(
            tile_z,
            tile_y,
            tile_x,
        ):
            try:
                labels, details = model.predict_instances(
                    x,
                    axes="ZYX",
                    prob_thresh=float(p),
                    nms_thresh=float(nms_thresh),
                    n_tiles=n_tiles,
                    verbose=False,
                )

                labels = labels.astype(
                    np.int32,
                    copy=False,
                )

                detected = int(labels.max())

                if detected > 0:
                    return (
                        labels,
                        details,
                        n_tiles,
                        memory_errors,
                        float(p),
                        zero_detection_attempts,
                    )

                zero_detection_attempts.append(
                    {
                        "prob_thresh": float(p),
                        "n_tiles": tuple(n_tiles),
                    }
                )

                # Detection was 0, so no reason to retry the same
                # probability with more tiling. Tiling does not
                # improve recognition, only memory usage.
                break

            except Exception as exc:
                if not _is_memory_error(exc):
                    raise

                memory_errors.append(
                    {
                        "prob_thresh": float(p),
                        "n_tiles": tuple(n_tiles),
                        "error": str(exc),
                    }
                )
                gc.collect()
                continue

    # Return the final zero-label result instead of crashing.
    # This allows the UI to show exactly which ROI failed.
    empty = np.zeros(
        stack_roi.shape,
        dtype=np.int32,
    )

    return (
        empty,
        {},
        (
            int(tile_z),
            int(tile_y),
            int(tile_x),
        ),
        memory_errors,
        float(prob_plans[-1]),
        zero_detection_attempts,
    )


# ============================================================
# Pick the nucleus belonging to a candidate ROI
# ============================================================

def choose_best_object(
    labels,
    roi_shape,
    target_y=None,
    target_x=None,
):
    """
    Pick the StarDist object belonging to the candidate.

    Prefer:
    - object center close to the actual candidate center
    - reasonably large object
    """
    max_id = int(labels.max())

    if max_id <= 0:
        return None

    zdim, h, w = roi_shape

    if target_y is None:
        target_y = (h - 1) / 2.0
    if target_x is None:
        target_x = (w - 1) / 2.0

    counts = np.bincount(
        labels.ravel(),
        minlength=max_id + 1,
    )

    objects = ndi.find_objects(
        labels,
        max_label=max_id,
    )

    scored = []

    for object_id in range(1, max_id + 1):
        count = int(counts[object_id])

        if count == 0:
            continue

        sl = objects[object_id - 1]

        if sl is None:
            continue

        crop = labels[sl] == object_id
        cz, cy, cx = ndi.center_of_mass(crop)

        global_cy = float(cy + sl[1].start)
        global_cx = float(cx + sl[2].start)

        dist_xy = np.hypot(
            global_cy - float(target_y),
            global_cx - float(target_x),
        )

        diag = max(1.0, np.hypot(h, w))
        distance_score = dist_xy / diag
        size_reward = np.log1p(count) * 0.015

        score = distance_score - size_reward
        scored.append((score, -count, object_id))

    if not scored:
        return None

    scored.sort()
    return int(scored[0][2])


def object_touches_z_boundary(
    labels,
    object_id,
    guard_slices=1,
):
    if object_id is None:
        return False, False

    mask = labels == int(object_id)

    if not np.any(mask):
        return False, False

    guard_slices = max(1, int(guard_slices))

    return (
        bool(np.any(mask[:guard_slices])),
        bool(np.any(mask[-guard_slices:])),
    )


# ============================================================
# Expand StarDist seed to the whole bright DAPI nucleus
# ============================================================

def _largest_component_overlapping_seed(mask, seed_mask):
    """
    Keep the 3D connected bright component that overlaps the StarDist seed most.
    """
    structure = ndi.generate_binary_structure(3, 2)
    lab, n = ndi.label(mask, structure=structure)

    if n <= 0:
        return np.zeros_like(mask, dtype=bool)

    seed_ids = lab[seed_mask]
    seed_ids = seed_ids[seed_ids > 0]

    if seed_ids.size > 0:
        ids, counts = np.unique(seed_ids, return_counts=True)
        best_id = int(ids[np.argmax(counts)])
        return lab == best_id

    # Fallback: choose component nearest the seed center.
    seed_coords = np.argwhere(seed_mask)
    if seed_coords.size == 0:
        return np.zeros_like(mask, dtype=bool)

    target = seed_coords.mean(axis=0)
    objects = ndi.find_objects(lab)
    sizes = np.bincount(lab.ravel())

    best_id = None
    best_score = None

    for obj_id in range(1, n + 1):
        sl = objects[obj_id - 1]
        if sl is None or sizes[obj_id] == 0:
            continue

        cz = (sl[0].start + sl[0].stop - 1) / 2.0
        cy = (sl[1].start + sl[1].stop - 1) / 2.0
        cx = (sl[2].start + sl[2].stop - 1) / 2.0

        dist = (
            ((cz - target[0]) / max(1, mask.shape[0])) ** 2
            + ((cy - target[1]) / max(1, mask.shape[1])) ** 2
            + ((cx - target[2]) / max(1, mask.shape[2])) ** 2
        )

        # Mildly prefer a larger connected object.
        score = dist - np.log1p(float(sizes[obj_id])) * 0.01

        if best_score is None or score < best_score:
            best_score = score
            best_id = obj_id

    if best_id is None:
        return np.zeros_like(mask, dtype=bool)

    return lab == int(best_id)


def _component_containing_seed_2d(mask2d, seed2d):
    lab, n = ndi.label(mask2d)

    if n <= 0:
        return np.zeros_like(mask2d, dtype=bool)

    ids = lab[seed2d]
    ids = ids[ids > 0]

    if ids.size > 0:
        vals, counts = np.unique(ids, return_counts=True)
        best = int(vals[np.argmax(counts)])
        return lab == best

    sizes = np.bincount(lab.ravel())

    if len(sizes) <= 1:
        return np.zeros_like(mask2d, dtype=bool)

    best = int(np.argmax(sizes[1:]) + 1)
    return lab == best


def _touches_xy_border(mask3d, guard=2):
    if mask3d is None or not np.any(mask3d):
        return False

    g = max(1, int(guard))

    return bool(
        np.any(mask3d[:, :g, :])
        or np.any(mask3d[:, -g:, :])
        or np.any(mask3d[:, :, :g])
        or np.any(mask3d[:, :, -g:])
    )


def _make_nucleus_xy_guard(
    norm,
    seed,
    guard_dilation_px=10,
):
    """
    MIP上で見える白い核の外形を2Dガード領域として作る。
    最終3Dマスクはこの外へ出ない。
    """
    mip = np.max(norm, axis=0)
    seed2d = np.max(seed, axis=0).astype(bool)

    sample = mip.ravel()

    try:
        otsu2d = float(threshold_otsu(sample))
    except Exception:
        otsu2d = float(np.percentile(sample, 65))

    thr2d = float(np.clip(otsu2d * 0.85, 0.03, 0.95))

    footprint = mip >= thr2d
    footprint |= seed2d

    footprint = ndi.binary_closing(
        footprint,
        structure=np.ones((5, 5), dtype=bool),
        iterations=1,
    )
    footprint = ndi.binary_fill_holes(footprint)

    footprint = _component_containing_seed_2d(
        footprint,
        seed2d,
    )

    if int(guard_dilation_px) > 0:
        footprint = ndi.binary_dilation(
            footprint,
            structure=np.ones((3, 3), dtype=bool),
            iterations=int(guard_dilation_px),
        )

    return footprint.astype(bool), {
        "xy_otsu": float(otsu2d),
        "xy_threshold": float(thr2d),
    }


def expand_seed_to_whole_nucleus(
    roi,
    labels,
    selected_object_id,
    threshold_factor=0.82,
    close_iterations=2,
    xy_guard_dilation_px=10,
):
    """
    StarDistはseedとしてのみ使用。
    最終体積はseedにつながる白いDAPI核全体から計算する。

    背景への漏れ対策:
    - MIPから白い核の2D外形ガードを作る
    - 3Dマスクはガード外へ出さない
    - それでもXY ROI端に触れる場合は閾値を自動的に上げて再試行する
    """
    if selected_object_id is None:
        return None, {}

    seed = labels == int(selected_object_id)

    if not np.any(seed):
        return None, {}

    x = np.asarray(roi, dtype=np.float32)

    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.8))

    if hi <= lo:
        return seed.copy(), {
            "otsu": None,
            "threshold": None,
            "fallback": "StarDist seed",
        }

    norm = np.clip((x - lo) / (hi - lo), 0, 1)

    xy_guard, xy_info = _make_nucleus_xy_guard(
        norm,
        seed,
        guard_dilation_px=int(xy_guard_dilation_px),
    )

    sample = norm.ravel()
    if sample.size > 1_000_000:
        step = max(1, sample.size // 1_000_000)
        sample = sample[::step]

    try:
        otsu = float(threshold_otsu(sample))
    except Exception:
        otsu = float(np.percentile(sample, 65))

    structure = ndi.generate_binary_structure(3, 1)

    factors = []
    base = float(threshold_factor)

    for delta in [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]:
        f = min(1.25, base + delta)
        if f not in factors:
            factors.append(f)

    seed_for_overlap = ndi.binary_dilation(
        seed,
        structure=ndi.generate_binary_structure(3, 2),
        iterations=2,
    )

    best_mask = None
    best_info = None

    for factor in factors:
        threshold = float(
            np.clip(
                otsu * factor,
                0.03,
                0.95,
            )
        )

        bright = norm >= threshold
        bright |= seed

        # ここが漏れ止めの本体。
        bright &= xy_guard[None, :, :]

        bright = ndi.binary_closing(
            bright,
            structure=structure,
            iterations=max(0, int(close_iterations)),
        )

        for z in range(bright.shape[0]):
            bright[z] = ndi.binary_fill_holes(bright[z])

        bright &= xy_guard[None, :, :]

        whole = _largest_component_overlapping_seed(
            bright,
            seed_for_overlap,
        )

        whole |= seed
        whole &= xy_guard[None, :, :]

        if not np.any(whole):
            continue

        xy_border_touch = _touches_xy_border(
            whole,
            guard=2,
        )

        roi_fraction = (
            float(np.count_nonzero(whole))
            / float(whole.size)
        )

        info = {
            "otsu": float(otsu),
            "threshold": float(threshold),
            "threshold_factor_requested": float(threshold_factor),
            "threshold_factor_used": float(factor),
            "whole_voxels": int(np.count_nonzero(whole)),
            "seed_voxels": int(np.count_nonzero(seed)),
            "roi_fraction": float(roi_fraction),
            "xy_border_touch": bool(xy_border_touch),
            "xy_guard_fraction": float(
                np.count_nonzero(xy_guard)
                / xy_guard.size
            ),
            **xy_info,
        }

        best_mask = whole.astype(bool)
        best_info = info

        if not xy_border_touch:
            break

    if best_mask is None:
        best_mask = seed.copy()
        best_info = {
            "otsu": float(otsu),
            "threshold": None,
            "threshold_factor_requested": float(threshold_factor),
            "threshold_factor_used": None,
            "whole_voxels": int(np.count_nonzero(seed)),
            "seed_voxels": int(np.count_nonzero(seed)),
            "roi_fraction": float(
                np.count_nonzero(seed) / seed.size
            ),
            "xy_border_touch": False,
            **xy_info,
        }

    return best_mask.astype(bool), best_info



def _best_overlapping_component_2d(mask2d, reference2d, dilation_px=3):
    """Return the 2D component that continues from the previous Z slice."""
    if not np.any(mask2d) or not np.any(reference2d):
        return np.zeros_like(mask2d, dtype=bool)

    ref = ndi.binary_dilation(
        reference2d,
        structure=np.ones((3, 3), dtype=bool),
        iterations=max(1, int(dilation_px)),
    )

    lab, n = ndi.label(mask2d)
    if n <= 0:
        return np.zeros_like(mask2d, dtype=bool)

    ids = lab[ref]
    ids = ids[ids > 0]
    if ids.size == 0:
        return np.zeros_like(mask2d, dtype=bool)

    vals, counts = np.unique(ids, return_counts=True)
    best = int(vals[np.argmax(counts)])
    comp = lab == best

    overlap = int(np.count_nonzero(comp & ref))
    min_overlap = max(2, int(round(np.count_nonzero(reference2d) * 0.01)))
    if overlap < min_overlap:
        return np.zeros_like(mask2d, dtype=bool)

    return comp


def _connected_hysteresis_from_seed(
    smoothed,
    seed,
    xy_guard,
    high_thr,
    low_thr,
    close_iterations=2,
):
    """
    Grow only the weak-signal voxels that are connected to the StarDist seed.

    This is a seed-constrained hysteresis segmentation:
      - high_thr marks reliable DAPI signal
      - low_thr permits faint nuclear edge signal
      - propagation starts from the StarDist seed, so isolated background noise
        is not accepted simply because it exceeds low_thr.
    """
    seed = np.asarray(seed, dtype=bool)
    guard3d = xy_guard[None, :, :]

    strong = (smoothed >= float(high_thr)) & guard3d
    weak = (smoothed >= float(low_thr)) & guard3d

    # The StarDist seed must always remain valid.
    strong |= seed
    weak |= seed

    # Join tiny salt-and-pepper gaps before propagation.  This is deliberately
    # modest; propagation is still constrained to the seed-connected region.
    structure3d = ndi.generate_binary_structure(3, 2)
    if int(close_iterations) > 0:
        weak = ndi.binary_closing(
            weak,
            structure=structure3d,
            iterations=int(close_iterations),
        )
        weak |= seed

    # Seed-constrained hysteresis. This is the key change for dotted masks.
    grown = ndi.binary_propagation(
        seed,
        structure=structure3d,
        mask=weak,
    )

    # Make sure reliable signal immediately connected to the grown object is
    # included, then fill holes slice-by-slice for a solid nuclear cross-section.
    grown |= (strong & ndi.binary_dilation(grown, structure=structure3d, iterations=1))

    for z in range(grown.shape[0]):
        if np.any(grown[z]):
            grown[z] = ndi.binary_fill_holes(grown[z])

    grown &= guard3d
    grown |= seed
    return grown.astype(bool)


def expand_seed_mask_full_z(
    roi_full_z,
    seed_full_z,
    threshold_factor=0.82,
    xy_guard_dilation_px=10,
    edge_sensitivity=0.65,
    max_gap_slices=2,
):
    """
    Full-Z whole-nucleus segmentation designed for both smooth and noisy DAPI.

    Key points:
      1) StarDist is used only as a seed.
      2) A light Gaussian filter suppresses salt-and-pepper noise.
      3) Seed-constrained hysteresis grows through faint edge signal without
         accepting disconnected background speckles.
      4) Remaining faint Z tails are followed slice-by-slice by continuity.

    edge_sensitivity: 0.0..1.0
      Higher -> lower weak threshold, slightly stronger smoothing/closing,
      therefore more permissive at faint XY/Z edges.
    """
    if seed_full_z is None or not np.any(seed_full_z):
        return None, {}

    x = np.asarray(roi_full_z)
    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.8))
    if hi <= lo:
        return seed_full_z.astype(bool).copy(), {
            "fallback": "StarDist seed",
            "seed_voxels": int(np.count_nonzero(seed_full_z)),
            "whole_voxels": int(np.count_nonzero(seed_full_z)),
            "z_tracked_slices": 0,
        }

    sensitivity = float(np.clip(edge_sensitivity, 0.0, 1.0))

    # float32 only for this XY-cropped ROI, never for the whole uploaded stack.
    norm = np.asarray(x, dtype=np.float32)
    norm = np.clip((norm - lo) / (hi - lo), 0, 1)
    seed = np.asarray(seed_full_z, dtype=bool)

    # Light smoothing turns the dotted/noisy appearance into a continuous
    # intensity field while preserving the macroscopic nuclear boundary.
    smooth_sigma = 0.8 + 0.8 * sensitivity
    smoothed = ndi.gaussian_filter(
        norm,
        sigma=(0.55, smooth_sigma, smooth_sigma),
        mode="nearest",
    ).astype(np.float32, copy=False)

    # Make the guard a little more permissive as edge sensitivity rises.
    # It remains a 2D safety fence against runaway background growth.
    effective_guard = int(round(int(xy_guard_dilation_px) + 8 * sensitivity))
    xy_guard, xy_info = _make_nucleus_xy_guard(
        smoothed,
        seed,
        guard_dilation_px=effective_guard,
    )

    sample = smoothed.ravel()
    if sample.size > 1_000_000:
        step = max(1, sample.size // 1_000_000)
        sample = sample[::step]

    try:
        otsu = float(threshold_otsu(sample))
    except Exception:
        otsu = float(np.percentile(sample, 65))

    high_thr = float(np.clip(otsu * float(threshold_factor), 0.03, 0.95))

    # At default sensitivity 0.65, faint-edge threshold is ~51% of high_thr.
    # Range is deliberately bounded to avoid accepting nearly pure background.
    weak_ratio = float(np.clip(0.72 - 0.32 * sensitivity, 0.38, 0.72))
    low_thr = float(np.clip(high_thr * weak_ratio, 0.018, high_thr))
    close_iterations = 1 + int(round(2 * sensitivity))

    whole = _connected_hysteresis_from_seed(
        smoothed=smoothed,
        seed=seed,
        xy_guard=xy_guard,
        high_thr=high_thr,
        low_thr=low_thr,
        close_iterations=close_iterations,
    )

    # Keep only the 3D component that actually belongs to the seed.
    seed_overlap = ndi.binary_dilation(
        seed,
        structure=ndi.generate_binary_structure(3, 2),
        iterations=2,
    )
    whole = _largest_component_overlapping_seed(whole, seed_overlap)
    whole |= seed
    whole &= xy_guard[None, :, :]

    z_present = np.where(np.any(whole, axis=(1, 2)))[0]
    if z_present.size == 0:
        whole = seed.copy()
        z_present = np.where(np.any(whole, axis=(1, 2)))[0]

    start_z = int(z_present.min())
    end_z = int(z_present.max())
    tracked = 0

    # Follow any remaining faint tail through Z.  The threshold becomes a
    # little more permissive with edge sensitivity, but a component must
    # spatially continue from the preceding nuclear slice.
    z_tail_thr = float(np.clip(low_thr * (0.95 - 0.18 * sensitivity), 0.015, low_thr))
    overlap_dilation = 3 + int(round(3 * sensitivity))

    for direction, first_z in [(-1, start_z - 1), (1, end_z + 1)]:
        z = first_z
        ref_z = start_z if direction < 0 else end_z
        reference = whole[ref_z].copy()
        gap = 0

        while 0 <= z < whole.shape[0]:
            candidate = (smoothed[z] >= z_tail_thr) & xy_guard
            candidate = ndi.binary_closing(
                candidate,
                structure=np.ones((3, 3), dtype=bool),
                iterations=close_iterations,
            )
            candidate = ndi.binary_fill_holes(candidate)

            comp = _best_overlapping_component_2d(
                candidate,
                reference,
                dilation_px=overlap_dilation,
            )

            if np.any(comp):
                prev_area = max(1, int(np.count_nonzero(reference)))
                area = int(np.count_nonzero(comp))

                # Allow natural tapering/expansion, reject sudden runaway growth.
                max_growth = 2.8 + 0.8 * sensitivity
                if area <= max(prev_area * max_growth, prev_area + 250):
                    whole[z] = comp
                    reference = comp
                    tracked += 1
                    gap = 0
                else:
                    gap += 1
            else:
                gap += 1

            if gap > int(max_gap_slices):
                break
            z += direction

    # Final small closing makes the displayed contour a single solid outline
    # instead of hundreds of tiny dots, while staying inside the XY guard.
    final_structure = ndi.generate_binary_structure(3, 1)
    whole = ndi.binary_closing(whole, structure=final_structure, iterations=1)
    for z in range(whole.shape[0]):
        if np.any(whole[z]):
            whole[z] = ndi.binary_fill_holes(whole[z])

    whole |= seed
    whole &= xy_guard[None, :, :]

    info = {
        "otsu": float(otsu),
        "threshold": float(high_thr),
        "tail_threshold": float(z_tail_thr),
        "weak_threshold": float(low_thr),
        "edge_sensitivity": float(sensitivity),
        "smoothing_sigma_xy": float(smooth_sigma),
        "threshold_factor_requested": float(threshold_factor),
        "threshold_factor_used": float(threshold_factor),
        "seed_voxels": int(np.count_nonzero(seed)),
        "whole_voxels": int(np.count_nonzero(whole)),
        "roi_fraction": float(np.count_nonzero(whole) / max(1, whole.size)),
        "xy_border_touch": bool(_touches_xy_border(whole, guard=2)),
        "xy_guard_fraction": float(np.count_nonzero(xy_guard) / xy_guard.size),
        "effective_xy_guard_dilation": int(effective_guard),
        "z_tracked_slices": int(tracked),
        **xy_info,
    }

    del norm, smoothed
    return whole.astype(bool), info

def mask_touches_z_boundary(mask, guard_slices=1):
    if mask is None or not np.any(mask):
        return False, False

    guard_slices = max(1, int(guard_slices))
    return (
        bool(np.any(mask[:guard_slices])),
        bool(np.any(mask[-guard_slices:])),
    )


# ============================================================
# Measurement
# ============================================================

def measure_binary_mask(
    mask,
    pixel_x,
    pixel_y,
    z_spacing,
    z_offset,
    y_offset,
    x_offset,
):
    mask = np.asarray(mask, dtype=bool)

    voxel_count = int(np.count_nonzero(mask))

    if voxel_count == 0:
        return None

    voxel_volume = (
        float(pixel_x)
        * float(pixel_y)
        * float(z_spacing)
    )

    volume = voxel_count * voxel_volume

    coords = np.argwhere(mask)
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    area_pixels_by_z = np.count_nonzero(
        mask,
        axis=(1, 2),
    )

    areas = (
        area_pixels_by_z.astype(np.float64)
        * float(pixel_x)
        * float(pixel_y)
    )

    cz, cy, cx = ndi.center_of_mass(mask)

    return {
        "Volume (µm³)": float(volume),
        "Voxel count": voxel_count,
        "Z start": int(z_min + z_offset + 1),
        "Z end": int(z_max + z_offset + 1),
        "Z slices": int(z_max - z_min + 1),
        "Max area (µm²)": float(areas.max()),
        "Mean area (µm²)": float(
            areas[area_pixels_by_z > 0].mean()
            if np.any(area_pixels_by_z > 0)
            else 0.0
        ),
        "Center Z": float(cz + z_offset + 1),
        "Center Y (original px)": float(cy + y_offset),
        "Center X (original px)": float(cx + x_offset),
    }


# ============================================================
# Full ROI pipeline
# ============================================================

def analyze_candidates(
    raw,
    candidates,
    prob_thresh,
    nms_thresh,
    pixel_x,
    pixel_y,
    z_spacing,
    min_volume,
    tile_z,
    tile_yx,
    progress_callback=None,
    z_expand_step=8,
    max_z_expansions=2,
    whole_nucleus_threshold_factor=0.82,
    xy_guard_dilation_px=10,
    edge_sensitivity=0.65,
):
    """
    Memory-safe pipeline.

    1) StarDist runs ONLY on the compact candidate Z ROI to find a nucleus seed.
    2) The seed is mapped into an XY-cropped ROI spanning the FULL Z-stack.
    3) Lightweight DAPI tracking follows the nucleus toward both Z ends,
       including faint upper/lower slices.
    4) Final volume is measured from this full-Z whole-nucleus mask.

    No resize / resampling.
    """
    import gc

    roi_results = []
    table_rows = []
    total = max(1, len(candidates))

    for idx, c0 in enumerate(candidates, start=1):
        c = dict(c0)
        z0_seed, z1_seed = int(c["z0"]), int(c["z1"])
        y0, y1 = int(c["y0"]), int(c["y1"])
        x0, x1 = int(c["x0"]), int(c["x1"])

        seed_roi = raw[z0_seed:z1_seed, y0:y1, x0:x1]
        target_y = float(c["center_y"] - y0)
        target_x = float(c["center_x"] - x0)

        if progress_callback is not None:
            progress_callback(idx, total, c, seed_roi.shape, 0)

        t0 = time.perf_counter()
        (
            labels,
            details,
            used_tiles,
            memory_errors,
            used_prob_thresh,
            zero_detection_attempts,
        ) = run_stardist_roi(
            seed_roi,
            prob_thresh,
            nms_thresh,
            tile_z,
            tile_yx,
            tile_yx,
        )
        elapsed = time.perf_counter() - t0

        detected_objects = int(labels.max())
        chosen_id = choose_best_object(
            labels,
            seed_roi.shape,
            target_y=target_y,
            target_x=target_x,
        )

        if chosen_id is None:
            roi_results.append({
                "candidate": c["candidate"],
                "roi": c,
                "image": seed_roi,
                "labels": None,
                "whole_mask": None,
                "selected_object_id": None,
                "elapsed_sec": elapsed,
                "details": None,
                "used_tiles": used_tiles,
                "memory_retries": len(memory_errors),
                "z_expansions": 0,
                "detected_objects": detected_objects,
                "used_prob_thresh": float(used_prob_thresh),
                "zero_detection_attempts": len(zero_detection_attempts),
                "mask_info": {},
            })
            del labels, details
            gc.collect()
            continue

        seed_small = labels == int(chosen_id)

        # Full Z is used only for the cheap intensity-based tracking step.
        # StarDist is NOT run again on this larger array.
        full_roi = raw[:, y0:y1, x0:x1]
        seed_full = np.zeros(full_roi.shape, dtype=bool)
        seed_full[z0_seed:z1_seed] = seed_small

        whole_mask, mask_info = expand_seed_mask_full_z(
            roi_full_z=full_roi,
            seed_full_z=seed_full,
            threshold_factor=float(whole_nucleus_threshold_factor),
            xy_guard_dilation_px=int(xy_guard_dilation_px),
            edge_sensitivity=float(edge_sensitivity),
            max_gap_slices=2,
        )

        # From here on, ROI Z coordinates are full-stack coordinates.
        c["z0"] = 0
        c["z1"] = int(raw.shape[0])

        if whole_mask is None or not np.any(whole_mask):
            roi_results.append({
                "candidate": c["candidate"],
                "roi": c,
                "image": full_roi,
                "labels": None,
                "whole_mask": None,
                "selected_object_id": int(chosen_id),
                "elapsed_sec": elapsed,
                "details": None,
                "used_tiles": used_tiles,
                "memory_retries": len(memory_errors),
                "z_expansions": 0,
                "detected_objects": detected_objects,
                "used_prob_thresh": float(used_prob_thresh),
                "zero_detection_attempts": len(zero_detection_attempts),
                "mask_info": mask_info,
            })
            del labels, details, seed_small, seed_full
            gc.collect()
            continue

        measurement = measure_binary_mask(
            mask=whole_mask,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            z_spacing=z_spacing,
            z_offset=0,
            y_offset=y0,
            x_offset=x0,
        )
        if measurement is None:
            del labels, details, seed_small, seed_full, whole_mask
            gc.collect()
            continue

        passes_min_volume = measurement["Volume (µm³)"] >= float(min_volume)
        nucleus_no = len(table_rows) + 1
        top_touch, bottom_touch = mask_touches_z_boundary(whole_mask, guard_slices=1)
        seed_voxels = int(mask_info.get("seed_voxels", np.count_nonzero(seed_small)))
        whole_voxels = int(mask_info.get("whole_voxels", measurement["Voxel count"]))
        expansion_ratio = whole_voxels / max(seed_voxels, 1)

        row = {
            "Nucleus": nucleus_no,
            "Candidate ROI": c["candidate"],
            "StarDist detected": detected_objects,
            "Used probability": round(float(used_prob_thresh), 3),
            "Zero-detection retries": int(len(zero_detection_attempts)),
            "Pass minimum volume": bool(passes_min_volume),
            "Volume (µm³)": round(measurement["Volume (µm³)"], 3),
            "Voxel count": measurement["Voxel count"],
            "StarDist seed voxels": seed_voxels,
            "Whole/seed ratio": round(float(expansion_ratio), 2),
            "DAPI threshold": round(float(mask_info.get("threshold", np.nan)), 4),
            "Tail threshold": round(float(mask_info.get("tail_threshold", np.nan)), 4),
            "Weak threshold": round(float(mask_info.get("weak_threshold", np.nan)), 4),
            "Edge sensitivity": round(float(mask_info.get("edge_sensitivity", edge_sensitivity)), 2),
            "Threshold factor used": mask_info.get("threshold_factor_used", np.nan),
            "XY border touch": bool(mask_info.get("xy_border_touch", False)),
            "Z tracked slices": int(mask_info.get("z_tracked_slices", 0)),
            "Z start": measurement["Z start"],
            "Z end": measurement["Z end"],
            "Z slices": measurement["Z slices"],
            "Max area (µm²)": round(measurement["Max area (µm²)"], 3),
            "Mean area (µm²)": round(measurement["Mean area (µm²)"], 3),
            "Center Z": round(measurement["Center Z"], 2),
            "Center Y (original px)": round(measurement["Center Y (original px)"], 2),
            "Center X (original px)": round(measurement["Center X (original px)"], 2),
            "ROI shape": f"{full_roi.shape[0]}×{full_roi.shape[1]}×{full_roi.shape[2]}",
            "n_tiles": "×".join(str(x) for x in used_tiles) if used_tiles is not None else "",
            "Memory retries": int(len(memory_errors)),
            "Z expansions": 0,
            "Z edge touch": bool(top_touch or bottom_touch),
            "Inference time (s)": round(elapsed, 2),
        }
        table_rows.append(row)

        # Keep only what the result viewer needs. In particular, do NOT retain
        # StarDist's int32 label volume or `details` for every ROI.
        roi_results.append({
            "nucleus": nucleus_no,
            "candidate": c["candidate"],
            "roi": c,
            "image": full_roi,
            "labels": None,
            "whole_mask": whole_mask,
            "selected_object_id": int(chosen_id),
            "elapsed_sec": elapsed,
            "details": None,
            "used_tiles": used_tiles,
            "memory_retries": len(memory_errors),
            "z_expansions": 0,
            "detected_objects": detected_objects,
            "used_prob_thresh": float(used_prob_thresh),
            "zero_detection_attempts": len(zero_detection_attempts),
            "mask_info": mask_info,
        })

        del labels, details, seed_small, seed_full
        gc.collect()

    df = pd.DataFrame(table_rows)
    if not df.empty:
        df = df.sort_values("Nucleus").reset_index(drop=True)
    return roi_results, df


# ============================================================
# Visualization
# ============================================================

def make_roi_overlay(
    roi_record,
    local_z,
):
    import matplotlib.pyplot as plt

    image = roi_record["image"]
    whole_mask = roi_record.get("whole_mask")

    local_z = int(
        np.clip(
            local_z,
            0,
            image.shape[0] - 1,
        )
    )

    bg = normalize_for_display(
        image[local_z]
    )

    if whole_mask is None:
        mask = np.zeros_like(
            image[local_z],
            dtype=bool,
        )
    else:
        mask = np.asarray(
            whole_mask[local_z],
            dtype=bool,
        )

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(bg, cmap="gray")

    if np.any(mask):
        color = object_color(
            roi_record["nucleus"]
        )

        rgba = np.zeros(
            (*mask.shape, 4),
            dtype=np.float32,
        )

        rgba[..., 0] = color[0]
        rgba[..., 1] = color[1]
        rgba[..., 2] = color[2]
        rgba[..., 3] = mask.astype(np.float32) * 0.42

        ax.imshow(rgba)

        ax.contour(
            mask,
            levels=[0.5],
            colors=[color],
            linewidths=2.0,
        )

    global_z = (
        roi_record["roi"]["z0"]
        + local_z
        + 1
    )

    ax.set_title(
        f"核{roi_record['nucleus']} / global Z={global_z} "
        "（色付き領域＝体積計算領域）"
    )
    ax.axis("off")
    fig.tight_layout()

    return fig


def make_roi_mip_overlay(roi_record):
    import matplotlib.pyplot as plt

    image = roi_record["image"]
    whole_mask = roi_record.get("whole_mask")

    bg = normalize_for_display(
        np.max(image, axis=0)
    )

    if whole_mask is None:
        mask_mip = np.zeros_like(
            bg,
            dtype=bool,
        )
    else:
        mask_mip = np.max(
            whole_mask,
            axis=0,
        ).astype(bool)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(bg, cmap="gray")

    if np.any(mask_mip):
        color = object_color(
            roi_record["nucleus"]
        )

        rgba = np.zeros(
            (*mask_mip.shape, 4),
            dtype=np.float32,
        )
        rgba[..., 0] = color[0]
        rgba[..., 1] = color[1]
        rgba[..., 2] = color[2]
        rgba[..., 3] = mask_mip.astype(np.float32) * 0.32

        ax.imshow(rgba)
        ax.contour(
            mask_mip,
            levels=[0.5],
            colors=[color],
            linewidths=2.0,
        )

    ax.set_title(
        f"核{roi_record['nucleus']} MIP "
        "（色付き領域＝体積計算領域）"
    )
    ax.axis("off")
    fig.tight_layout()
    return fig


def mask_to_mesh(
    mask,
    pixel_x,
    pixel_y,
    z_spacing,
):
    if not np.any(mask):
        return None

    coords = np.argwhere(mask)
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)

    pad = 1

    z0 = max(0, int(zmin) - pad)
    y0 = max(0, int(ymin) - pad)
    x0 = max(0, int(xmin) - pad)

    z1 = min(mask.shape[0], int(zmax) + pad + 1)
    y1 = min(mask.shape[1], int(ymax) + pad + 1)
    x1 = min(mask.shape[2], int(xmax) + pad + 1)

    crop = mask[
        z0:z1,
        y0:y1,
        x0:x1,
    ]

    if min(crop.shape) < 2:
        return None

    verts, faces, _, _ = marching_cubes(
        crop.astype(np.uint8),
        level=0.5,
        spacing=(
            float(z_spacing),
            float(pixel_y),
            float(pixel_x),
        ),
    )

    verts[:, 0] += z0 * float(z_spacing)
    verts[:, 1] += y0 * float(pixel_y)
    verts[:, 2] += x0 * float(pixel_x)

    return (
        verts[:, 2],
        verts[:, 1],
        verts[:, 0],
        faces,
    )


def make_single_3d_figure(
    roi_record,
    pixel_x,
    pixel_y,
    z_spacing,
):
    mask = roi_record.get("whole_mask")

    if mask is None:
        return go.Figure()

    mesh = mask_to_mesh(
        mask,
        pixel_x,
        pixel_y,
        z_spacing,
    )

    fig = go.Figure()

    if mesh is None:
        return fig

    x, y, z, faces = mesh

    rgb = rgb255(
        object_color(
            roi_record["nucleus"]
        )
    )

    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=f"rgb({rgb[0]},{rgb[1]},{rgb[2]})",
            opacity=0.75,
            flatshading=False,
            showscale=False,
            name=f"核{roi_record['nucleus']}",
        )
    )

    fig.update_layout(
        title=f"核{roi_record['nucleus']} 3D",
        scene=dict(
            xaxis_title="X (µm)",
            yaxis_title="Y (µm)",
            zaxis_title="Z (µm)",
            aspectmode="data",
        ),
        margin=dict(
            l=0,
            r=0,
            t=45,
            b=0,
        ),
    )

    return fig


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("① DAPI / image")

channel = st.sidebar.selectbox(
    "DAPI channel",
    ["R", "G", "B"],
    index=2,
)


st.sidebar.header("② Voxel size")

pixel_x = st.sidebar.number_input(
    "Pixel X (µm)",
    min_value=0.0001,
    value=0.093,
    step=0.001,
    format="%.4f",
)

pixel_y = st.sidebar.number_input(
    "Pixel Y (µm)",
    min_value=0.0001,
    value=0.093,
    step=0.001,
    format="%.4f",
)

z_spacing = st.sidebar.number_input(
    "Z spacing (µm)",
    min_value=0.0001,
    value=0.400,
    step=0.010,
    format="%.3f",
)


st.sidebar.header("③ 核候補ROI")

expected_count = st.sidebar.number_input(
    "予想される核数",
    min_value=1,
    max_value=30,
    value=5,
    step=1,
)

xy_margin = st.sidebar.slider(
    "XY余白 (pixel)",
    min_value=32,
    max_value=256,
    value=64,
    step=16,
)

z_margin = st.sidebar.slider(
    "Z余白 (slice)",
    min_value=2,
    max_value=30,
    value=6,
    step=1,
)

z_expand_step = st.sidebar.slider(
    "Z境界接触時の追加幅 (slice)",
    min_value=2,
    max_value=20,
    value=6,
    step=1,
)

min_candidate_area = st.sidebar.number_input(
    "候補の最小2D面積 (pixel)",
    min_value=20,
    max_value=100000,
    value=300,
    step=50,
)

st.sidebar.caption(
    "余白は核を切らないための安全域です。画像は縮小しません。"
)


st.sidebar.header("④ StarDist 3D")

prob_thresh = st.sidebar.slider(
    "Probability threshold",
    0.05,
    0.95,
    0.35,
    0.01,
)

nms_thresh = st.sidebar.slider(
    "NMS overlap threshold",
    0.05,
    0.90,
    0.30,
    0.01,
)

min_volume = st.sidebar.number_input(
    "Minimum volume (µm³)",
    min_value=0.0,
    value=0.0,
    step=5.0,
    help=(
        "まず0で解析してください。体積計算後に小さい物体を除外できます。"
    ),
)


st.sidebar.header("⑤ 核全体の認識")

whole_nucleus_threshold_factor = st.sidebar.slider(
    "白い核の境界しきい値",
    min_value=0.50,
    max_value=1.20,
    value=0.82,
    step=0.02,
    help=(
        "小さく認識される場合は下げます。"
        "背景まで広がる場合は上げます。"
        "StarDistは核の中心を決めるseedとして使い、"
        "最終体積は白いDAPI核全体から計算します。"
    ),
)

st.sidebar.caption(
    "推奨開始値: 0.82。"
    "今回のように白い核全体を体積として認識するための設定です。"
)

xy_guard_dilation_px = st.sidebar.slider(
    "核外形ガード余白 (px)",
    min_value=0,
    max_value=30,
    value=10,
    step=2,
    help=(
        "MIPで見える白い核外形から許容する余白です。"
        "背景まで広がる場合は小さく、核の端が切れる場合は大きくします。"
    ),
)

edge_sensitivity_percent = st.sidebar.slider(
    "核の端の感度",
    min_value=0,
    max_value=100,
    value=65,
    step=5,
    help=(
        "淡い核外周とZ上下端をどこまで拾うかを調整します。"
        "高くすると端を拾いやすくなりますが、背景まで広がる場合は下げてください。"
        "まず65、端が不足する場合は75→85の順で試してください。"
    ),
)
edge_sensitivity = edge_sensitivity_percent / 100.0

st.sidebar.caption(
    "端が粒々になる画像では 70〜85 を推奨。"
    "輪郭が明瞭な画像では 50〜65 から試してください。"
)


st.sidebar.header("⑥ Tiling")

tile_z = st.sidebar.select_slider(
    "Z tile数",
    options=[1, 2, 4, 8],
    value=4,
)

tile_yx = st.sidebar.select_slider(
    "XY tile数",
    options=[1, 2, 3, 4],
    value=3,
)

st.sidebar.caption(
    "Streamlit向けに4×3×3を初期値にしています。メモリ使用量を抑える設定です。"
)


# ============================================================
# Upload
# ============================================================

st.header("① Z-stack TIFを読み込む")

uploaded_files = st.file_uploader(
    "Z-stackの2D TIFをすべて選択してください",
    type=["tif", "tiff"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info(
        "DAPIのZ-stack TIFをアップロードしてください。"
    )
    st.stop()

uploaded_files = sorted(
    uploaded_files,
    key=lambda x: natural_key(x.name),
)

file_signature = tuple(
    (f.name, f.size)
    for f in uploaded_files
)

if st.session_state.file_signature != file_signature:
    st.session_state.file_signature = file_signature
    st.session_state.raw_stack = None
    st.session_state.candidates = None
    st.session_state.roi_results = None
    st.session_state.result_table = None
    st.session_state.prediction_done = False
    st.session_state.selected_nucleus = None
    st.session_state.excluded_ids = set()
    st.session_state.selected_roi_ids = None
    st.session_state.timings = None


# ============================================================
# Load raw stack
# ============================================================

if st.session_state.raw_stack is None:
    with st.spinner("Z-stackを読み込んでいます..."):
        try:
            st.session_state.raw_stack = load_uploaded_stack(
                uploaded_files,
                channel,
            )
        except Exception as e:
            st.exception(e)
            st.stop()

raw = st.session_state.raw_stack

c1, c2, c3, c4 = st.columns(4)
c1.metric("Z", raw.shape[0])
c2.metric("Y", raw.shape[1])
c3.metric("X", raw.shape[2])
c4.metric(
    "総voxel",
    f"{raw.size / 1e6:.1f} M",
)


# ============================================================
# Candidate detection
# ============================================================

st.header("② 核候補を高速検出")

detect_button = st.button(
    "🔎 核候補ROIを検出",
    width="stretch",
)

if detect_button or st.session_state.candidates is None:
    t0 = time.perf_counter()

    with st.spinner("MIPから核候補を探しています..."):
        candidates, mip, candidate_mask = (
            detect_nucleus_candidates(
                raw,
                expected_count=int(expected_count),
                xy_margin=int(xy_margin),
                z_margin=int(z_margin),
                min_component_area=int(min_candidate_area),
            )
        )

    detect_time = time.perf_counter() - t0

    st.session_state.candidates = candidates
    st.session_state.prediction_done = False
    st.session_state.roi_results = None
    st.session_state.result_table = None
    st.session_state.selected_roi_ids = [
        int(c["candidate"]) for c in candidates
    ]

    if st.session_state.timings is None:
        st.session_state.timings = {}

    st.session_state.timings["candidate_detection_sec"] = detect_time

else:
    candidates = st.session_state.candidates
    mip = np.max(raw, axis=0)


if not candidates:
    st.error(
        "核候補を検出できませんでした。"
        "「候補の最小2D面積」を小さくして再検出してください。"
    )
    st.stop()


fig_candidates = make_candidate_figure(
    mip,
    candidates,
)

st.pyplot(
    fig_candidates,
    width="stretch",
)

import matplotlib.pyplot as plt
plt.close(fig_candidates)


# ROI table
candidate_rows = []

full_voxels = int(raw.size)
roi_voxels_total = 0

for c in candidates:
    shape = (
        c["z1"] - c["z0"],
        c["y1"] - c["y0"],
        c["x1"] - c["x0"],
    )

    nvox = int(
        shape[0]
        * shape[1]
        * shape[2]
    )

    roi_voxels_total += nvox

    candidate_rows.append(
        {
            "ROI": c["candidate"],
            "Z range": f"{c['z0'] + 1}–{c['z1']}",
            "Y range": f"{c['y0']}–{c['y1'] - 1}",
            "X range": f"{c['x0']}–{c['x1'] - 1}",
            "Shape Z×Y×X": (
                f"{shape[0]}×{shape[1]}×{shape[2]}"
            ),
            "Voxel (M)": round(nvox / 1e6, 2),
        }
    )

candidate_df = pd.DataFrame(candidate_rows)

st.dataframe(
    candidate_df,
    width="stretch",
    hide_index=True,
)

ratio = roi_voxels_total / float(full_voxels)

c1, c2, c3 = st.columns(3)
c1.metric(
    "候補ROI数",
    len(candidates),
)
c2.metric(
    "ROI総voxel",
    f"{roi_voxels_total / 1e6:.1f} M",
)
c3.metric(
    "全面解析との比率",
    f"{ratio * 100:.1f}%",
)

st.caption(
    "同じ場所が複数ROIに含まれる場合があるため、"
    "比率は厳密な処理時間比ではありません。"
)


# ============================================================
# Select ROIs to analyze
# ============================================================

st.subheader("解析するROIを選択")

all_roi_ids = [
    int(c["candidate"])
    for c in candidates
]

if st.session_state.selected_roi_ids is None:
    st.session_state.selected_roi_ids = all_roi_ids.copy()

# Remove stale IDs if candidate detection changed.
st.session_state.selected_roi_ids = [
    int(x)
    for x in st.session_state.selected_roi_ids
    if int(x) in all_roi_ids
]

selected_roi_ids = st.multiselect(
    "③でStarDist解析するROI",
    options=all_roi_ids,
    default=st.session_state.selected_roi_ids,
    format_func=lambda x: f"ROI {x}",
    help=(
        "不要なROIは選択を外してください。"
        "選択したROIだけが次のStarDist 3D解析に進みます。"
    ),
)

st.session_state.selected_roi_ids = [
    int(x) for x in selected_roi_ids
]

selected_candidates = [
    c
    for c in candidates
    if int(c["candidate"]) in st.session_state.selected_roi_ids
]

selected_voxels_total = 0

for c in selected_candidates:
    selected_voxels_total += (
        (c["z1"] - c["z0"])
        * (c["y1"] - c["y0"])
        * (c["x1"] - c["x0"])
    )

c1, c2, c3 = st.columns(3)

c1.metric(
    "③へ進むROI",
    len(selected_candidates),
)

c2.metric(
    "選択ROI総voxel",
    f"{selected_voxels_total / 1e6:.1f} M",
)

c3.metric(
    "全面解析との比率",
    f"{selected_voxels_total / float(full_voxels) * 100:.1f}%",
)

if not selected_candidates:
    st.warning(
        "解析するROIが選択されていません。"
        "少なくとも1つ選択してください。"
    )
else:
    removed_ids = [
        x
        for x in all_roi_ids
        if x not in st.session_state.selected_roi_ids
    ]

    if removed_ids:
        st.info(
            "③では次のROIを解析しません: "
            + ", ".join(f"ROI {x}" for x in removed_ids)
        )


# ============================================================
# ROI StarDist
# ============================================================

st.header("③ 各ROIだけStarDist 3D解析")

st.info(
    "ここからStarDistを実行します。"
    "StarDistは小さいseed ROIだけを解析し、その後は軽量処理で全Z方向へ核を追跡します。"
    "0個の場合だけProbability thresholdを0.30→0.25→0.20→0.15まで自動で下げて再試行します。"
)

run_button = st.button(
    "🧠 選択したROIだけ3D核を認識して体積計算",
    type="primary",
    width="stretch",
    disabled=(len(selected_candidates) == 0),
)

if run_button:
    progress = st.progress(0.0)
    status_text = st.empty()
    t_start = time.perf_counter()

    def update_progress(i, total, c, shape, expansion_count):
        status_text.write(
            f"ROI {i}/{total} をStarDist解析中 "
            f"— shape={shape} "
            f"— Z拡張={expansion_count}回"
        )
        progress.progress(
            min(1.0, (i - 1) / max(1, total))
        )

    try:
        # Load model before inference so model-loading status is explicit.
        status_text.write(
            "StarDist 3Dモデルを読み込んでいます..."
        )
        load_stardist_model()

        roi_results, result_table = analyze_candidates(
            raw=raw,
            candidates=selected_candidates,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            z_spacing=z_spacing,
            min_volume=min_volume,
            tile_z=tile_z,
            tile_yx=tile_yx,
            progress_callback=update_progress,
            z_expand_step=int(z_expand_step),
            max_z_expansions=2,
            whole_nucleus_threshold_factor=float(
                whole_nucleus_threshold_factor
            ),
            xy_guard_dilation_px=int(xy_guard_dilation_px),
            edge_sensitivity=float(edge_sensitivity),
        )

        total_time = time.perf_counter() - t_start

        progress.progress(1.0)
        status_text.success(
            f"完了しました。総解析時間: {total_time:.1f} 秒"
        )

        st.session_state.roi_results = roi_results
        st.session_state.result_table = result_table
        st.session_state.prediction_done = True

        if st.session_state.timings is None:
            st.session_state.timings = {}

        st.session_state.timings["roi_analysis_sec"] = total_time

        if not result_table.empty:
            st.session_state.selected_nucleus = int(
                result_table.iloc[0]["Nucleus"]
            )

    except Exception as e:
        st.error(
            "ROI StarDist 3Dの実行に失敗しました。"
        )
        st.exception(e)
        st.stop()


if not st.session_state.prediction_done:
    st.stop()


# ============================================================
# Results
# ============================================================

roi_results = st.session_state.roi_results
results = st.session_state.result_table

st.header("④ 体積結果")

if results is None or results.empty:
    st.error(
        "体積結果を作れませんでした。"
        "下のROI診断で、StarDistが本当に0個だったのかを確認してください。"
    )

    diag_rows = []

    for r in roi_results or []:
        diag_rows.append(
            {
                "ROI": r.get("candidate"),
                "StarDist detected": r.get(
                    "detected_objects",
                    0,
                ),
                "Used probability": r.get(
                    "used_prob_thresh",
                    prob_thresh,
                ),
                "Zero-detection retries": r.get(
                    "zero_detection_attempts",
                    0,
                ),
                "ROI Z range": (
                    f"{r['roi']['z0'] + 1}–"
                    f"{r['roi']['z1']}"
                    if r.get("roi")
                    else ""
                ),
                "n_tiles": (
                    "×".join(
                        str(x)
                        for x in r.get(
                            "used_tiles",
                            (),
                        )
                    )
                    if r.get("used_tiles")
                    else ""
                ),
            }
        )

    if diag_rows:
        st.dataframe(
            pd.DataFrame(diag_rows),
            width="stretch",
            hide_index=True,
        )

    st.info(
        "StarDist detected が0なら認識失敗です。"
        "1以上なら、別の後処理条件が原因です。"
    )
    st.stop()

c1, c2, c3 = st.columns(3)

c1.metric(
    "測定された核",
    len(results),
)

c2.metric(
    "XY解像度",
    "100% / 変更なし",
)

total_inference = float(
    results["Inference time (s)"].sum()
)

c3.metric(
    "ROI推論合計",
    f"{total_inference:.1f} 秒",
)


st.dataframe(
    results[
        [
            "Nucleus",
            "Volume (µm³)",
            "Voxel count",
            "StarDist seed voxels",
            "Whole/seed ratio",
            "DAPI threshold",
            "Tail threshold",
            "Weak threshold",
            "Edge sensitivity",
            "Threshold factor used",
            "XY border touch",
            "Z start",
            "Z end",
            "Z slices",
            "Z tracked slices",
            "ROI shape",
            "StarDist detected",
            "Used probability",
            "Zero-detection retries",
            "Pass minimum volume",
            "n_tiles",
            "Memory retries",
            "Z expansions",
            "Z edge touch",
            "Inference time (s)",
        ]
    ],
    width="stretch",
    hide_index=True,
)


for _, row in results.iterrows():
    st.write(
        f"**核{int(row['Nucleus'])}** "
        f"→ **{row['Volume (µm³)']:.3f} µm³** "
        f"（{row['Inference time (s)']:.1f} 秒）"
    )


# ============================================================
# Inspect one nucleus
# ============================================================

st.header("⑦ 認識結果を確認")

nuclei = [
    int(x)
    for x in results["Nucleus"].tolist()
]

selected_nucleus = st.selectbox(
    "核",
    nuclei,
    index=0,
)

selected_record = None

for r in roi_results:
    if r.get("nucleus") == int(selected_nucleus):
        selected_record = r
        break

if selected_record is not None:
    roi = selected_record["roi"]

    st.subheader("核全体の認識（MIP）")
    fig_mip_overlay = make_roi_mip_overlay(
        selected_record
    )
    st.pyplot(
        fig_mip_overlay,
        width="stretch",
    )
    plt.close(fig_mip_overlay)

    st.caption(
        "色付き領域が、実際にvoxel数を数えて体積計算している核領域です。"
    )

    global_z_start = roi["z0"] + 1
    global_z_end = roi["z1"]

    default_z = int(
        round(
            (
                global_z_start
                + global_z_end
            )
            / 2
        )
    )

    z_view = st.slider(
        "表示するglobal Z",
        global_z_start,
        global_z_end,
        default_z,
        1,
    )

    local_z = z_view - 1 - roi["z0"]

    fig_overlay = make_roi_overlay(
        selected_record,
        local_z,
    )

    st.pyplot(
        fig_overlay,
        width="stretch",
    )

    plt.close(fig_overlay)

    st.caption(
        f"ROI {selected_record['candidate']} / "
        f"Z={roi['z0'] + 1}–{roi['z1']}, "
        f"Y={roi['y0']}–{roi['y1'] - 1}, "
        f"X={roi['x0']}–{roi['x1'] - 1}"
    )


# ============================================================
# Optional 3D
# ============================================================

st.header("⑧ 3Dモデル（必要なときだけ）")

show_3d = st.checkbox(
    "選択中の核を3D表示",
    value=False,
)

if show_3d and selected_record is not None:
    fig3d = make_single_3d_figure(
        selected_record,
        pixel_x,
        pixel_y,
        z_spacing,
    )

    st.plotly_chart(
        fig3d,
        width="stretch",
    )

else:
    st.caption(
        "体積計算には3D meshは不要なので、通常はOFFで構いません。"
    )


# ============================================================
# Exclusion
# ============================================================

st.header("⑨ 誤認識した核を除外")

exclude_text = st.text_input(
    "除外する核番号",
    placeholder="例：2,5",
)

if st.button("除外リストへ追加"):
    try:
        ids = [
            int(x.strip())
            for x in exclude_text.split(",")
            if x.strip()
        ]

        valid = set(nuclei)

        for x in ids:
            if x in valid:
                st.session_state.excluded_ids.add(x)

    except ValueError:
        st.error(
            "例：2,5 のように入力してください。"
        )

if st.session_state.excluded_ids:
    st.warning(
        "除外中: "
        + ", ".join(
            f"核{x}"
            for x in sorted(
                st.session_state.excluded_ids
            )
        )
    )

    if st.button("除外を解除"):
        st.session_state.excluded_ids = set()
        st.rerun()


final_df = results[
    ~results["Nucleus"].isin(
        st.session_state.excluded_ids
    )
].copy()

if float(min_volume) > 0:
    final_df = final_df[
        final_df["Volume (µm³)"] >= float(min_volume)
    ].copy()


# ============================================================
# Final result and CSV
# ============================================================

st.header("⑩ 最終結果")

st.dataframe(
    final_df,
    width="stretch",
    hide_index=True,
)

csv = final_df.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    "📥 CSVを保存",
    data=csv,
    file_name="3D_nuclear_volume_ROI_fast.csv",
    mime="text/csv",
)


# ============================================================
# CPU/GPU info
# ============================================================

with st.expander("CPU / GPU確認"):
    if st.button("GPU認識を確認"):
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")

            if gpus:
                st.success(
                    f"TensorFlowがGPUを認識しています: {gpus}"
                )
            else:
                st.info(
                    "TensorFlowはGPUを認識していません。CPU実行です。"
                )

        except Exception as e:
            st.warning(
                "TensorFlowのGPU情報を取得できませんでした。"
            )
            st.exception(e)


# ============================================================
# Technical notes
# ============================================================

with st.expander("解析条件"):
    st.write(
        f"""
**Raw shape:** {raw.shape}

**Raw total voxels:** {raw.size:,}

**Candidate ROI total voxels:** {roi_voxels_total:,}

**Selected ROI total voxels:** {selected_voxels_total:,}

**Selected ROI / full:** {selected_voxels_total / float(full_voxels) * 100:.2f} %

**Pixel X:** {pixel_x:.4f} µm

**Pixel Y:** {pixel_y:.4f} µm

**Z spacing:** {z_spacing:.4f} µm

**XY resize:** なし

**Z resize:** なし

**Expected nuclei:** {expected_count}

**XY margin:** {xy_margin} px

**Z margin:** {z_margin} slices

**Probability threshold:** {prob_thresh:.2f}

**NMS threshold:** {nms_thresh:.2f}

**Whole nucleus threshold factor:** {whole_nucleus_threshold_factor:.2f}

**XY guard dilation:** {xy_guard_dilation_px} px

**Edge sensitivity:** {edge_sensitivity_percent} / 100

**Volume formula:**
voxel count × Pixel X × Pixel Y × Z spacing

候補検出はROIを決めるためだけに使用します。
StarDist 3Dは目的核を決めるseedとして使い、最終的な核境界と体積はseedにつながる白いDAPI核全体から計算します。
"""
    )


# ============================================================
# Reset
# ============================================================

st.header("⑪ リセット")

if st.button("🔄 すべてリセット"):
    st.session_state.raw_stack = None
    st.session_state.candidates = None
    st.session_state.roi_results = None
    st.session_state.result_table = None
    st.session_state.prediction_done = False
    st.session_state.selected_nucleus = None
    st.session_state.excluded_ids = set()
    st.session_state.selected_roi_ids = None
    st.session_state.timings = None
    st.rerun()
