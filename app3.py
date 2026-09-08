import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import re
import time
import inspect
import colorsys
import io

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
    page_title="3D Nuclear Volume Analyzer - Cellpose 2D + Z Stitch Fast",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 3D Nuclear Volume Analyzer - Cellpose 2D + Z Stitch Fast")
st.caption(
    "StarDistは使用しません。MIPで核候補を高速に見つけ、候補をまとめたROIの各Z面を"
    "Cellpose 2Dで一括解析し、Cellposeのstitch機能でZ方向につないで3D体積を計算します。"
)


# ============================================================
# Session state
# ============================================================

DEFAULTS = {
    "raw_stack": None,
    "file_signature": None,
    "candidates": None,
    "candidate_mip": None,
    "result_table": None,
    "analysis_record": None,
    "prediction_done": False,
    "excluded_ids": set(),
    "mask_review_pdf": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# Helpers
# ============================================================

def natural_key(name):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", str(name))
    ]


def normalize_for_display(img):
    x = np.asarray(img, dtype=np.float32)
    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.8))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((x - lo) / (hi - lo), 0, 1)


def object_color(object_id):
    hue = (int(object_id) * 0.618033988749895) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.85, 1.0)


def rgb255(rgb):
    return tuple(int(round(x * 255)) for x in rgb)


# ============================================================
# Loading
# ============================================================

def load_uploaded_stack(uploaded_files, channel):
    uploaded_files = sorted(uploaded_files, key=lambda x: natural_key(x.name))
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
            raise ValueError(f"{f.name}: 対応できないshapeです: {img.shape}")

        slices.append(np.asarray(img))

    if not slices:
        raise ValueError("画像がありません。")

    shapes = [x.shape for x in slices]
    if len(set(shapes)) != 1:
        raise ValueError("すべてのZ画像でXYサイズを一致させてください。")

    return np.stack(slices, axis=0).astype(np.float32)


# ============================================================
# Fast 2D candidate detection
# Boundary is NOT determined here; this only defines where Cellpose should run.
# ============================================================

def detect_nucleus_candidates(
    stack,
    expected_count=5,
    xy_margin=64,
    min_component_area=300,
    max_component_area_fraction=0.30,
):
    zdim, h, w = stack.shape
    mip = np.max(stack, axis=0)
    norm = normalize_for_display(mip)

    sm = gaussian(norm, sigma=2.0, preserve_range=True)
    try:
        otsu = float(threshold_otsu(sm))
    except Exception:
        otsu = 0.25

    # Candidate finder only. Slightly permissive so nuclei are not missed.
    thr = max(0.03, otsu * 0.78)
    mask = sm > thr
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5), dtype=bool))
    mask = ndi.binary_fill_holes(mask)

    lab, n = ndi.label(mask)
    if n == 0:
        return [], mip, mask

    sizes = np.bincount(lab.ravel())
    max_area = int(h * w * max_component_area_fraction)
    comps = []

    for lid in range(1, n + 1):
        area = int(sizes[lid])
        if area < int(min_component_area) or area > max_area:
            continue

        ys, xs = np.where(lab == lid)
        if xs.size == 0:
            continue

        comps.append({
            "area_px": area,
            "y_min": int(ys.min()),
            "y_max": int(ys.max()),
            "x_min": int(xs.min()),
            "x_max": int(xs.max()),
            "center_y": float(ys.mean()),
            "center_x": float(xs.mean()),
        })

    comps.sort(key=lambda d: d["area_px"], reverse=True)
    keep_n = max(int(expected_count) + 4, int(expected_count))
    comps = comps[:keep_n]

    candidates = []
    for i, c in enumerate(comps, start=1):
        candidates.append({
            "candidate": i,
            "y0": max(0, c["y_min"] - int(xy_margin)),
            "y1": min(h, c["y_max"] + int(xy_margin) + 1),
            "x0": max(0, c["x_min"] - int(xy_margin)),
            "x1": min(w, c["x_max"] + int(xy_margin) + 1),
            "center_y": c["center_y"],
            "center_x": c["center_x"],
            "area_px": c["area_px"],
        })

    return candidates, mip, mask


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
            c["x0"] + 5,
            c["y0"] + 18,
            f"ROI {c['candidate']}",
            color="white",
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor=color, pad=2),
        )

    ax.set_title("Detected nuclear candidate ROIs")
    ax.axis("off")
    fig.tight_layout()
    return fig


def union_bbox(candidates, image_shape, extra_margin=16):
    """One XY box containing all selected candidates. Full Z is always used."""
    zdim, h, w = image_shape
    if not candidates:
        return (0, zdim, 0, h, 0, w)

    y0 = max(0, min(c["y0"] for c in candidates) - int(extra_margin))
    y1 = min(h, max(c["y1"] for c in candidates) + int(extra_margin))
    x0 = max(0, min(c["x0"] for c in candidates) - int(extra_margin))
    x1 = min(w, max(c["x1"] for c in candidates) + int(extra_margin))
    return (0, zdim, y0, y1, x0, x1)


# ============================================================
# Cellpose 2D + Z stitching
# ============================================================

@st.cache_resource(show_spinner=False)
def load_cellpose_model(model_name="cpsam_v2", prefer_gpu=True):
    """Load Cellpose once and keep it cached across Streamlit reruns."""
    from cellpose import models

    try:
        return models.CellposeModel(
            gpu=bool(prefer_gpu),
            pretrained_model=str(model_name),
        )
    except TypeError:
        # Compatibility fallback for older Cellpose releases.
        return models.CellposeModel(
            gpu=bool(prefer_gpu),
            model_type="nuclei",
        )


def _cellpose_version():
    try:
        import cellpose
        return str(getattr(cellpose, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _prepare_cellpose_input(roi):
    """
    Convert Z,Y,X grayscale stack to Z,Y,X,3 for Cellpose-SAM.
    Values are normalized once across the whole stack so dim edge planes are
    not independently boosted relative to bright central planes.
    """
    x = np.asarray(roi, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Cellpose input must be Z,Y,X; got {x.shape}")

    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.8))
    if hi <= lo:
        hi = lo + 1.0

    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)
    cp = np.zeros((*x.shape, 3), dtype=np.float32)
    cp[..., 0] = x
    return cp


def run_cellpose_2d_stitch(
    roi,
    model_name="cpsam_v2",
    prefer_gpu=True,
    cellprob_threshold=0.0,
    flow_threshold=0.4,
    stitch_threshold=0.25,
    min_size=15,
    batch_size=32,
    resample=False,
):
    """
    Fast volume segmentation:
      1) run Cellpose in 2D on every XY plane in one eval() call
      2) stitch adjacent 2D masks into 3D objects using Cellpose stitch_threshold

    No StarDist, no manual 3D threshold expansion, no XY-shape extrusion.
    """
    model = load_cellpose_model(
        model_name=str(model_name),
        prefer_gpu=bool(prefer_gpu),
    )
    cp_input = _prepare_cellpose_input(roi)

    supported = set(inspect.signature(model.eval).parameters.keys())
    kwargs = {
        "do_3D": False,
        "stitch_threshold": float(stitch_threshold),
        "cellprob_threshold": float(cellprob_threshold),
        "flow_threshold": float(flow_threshold),
        "min_size": int(min_size),
        "batch_size": int(batch_size),
        "z_axis": 0,
        "channel_axis": 3,
        "resample": bool(resample),
        "diameter": None,
        "augment": False,
    }

    # In stitching mode, keep one normalization across the full Z stack.
    if "normalize" in supported:
        kwargs["normalize"] = {"normalize": True, "norm3D": True}

    kwargs = {k: v for k, v in kwargs.items() if k in supported}

    t0 = time.perf_counter()
    output = model.eval(cp_input, **kwargs)
    elapsed = time.perf_counter() - t0

    masks = output[0] if isinstance(output, tuple) else output
    masks = np.squeeze(np.asarray(masks))

    if masks.shape != roi.shape:
        raise RuntimeError(
            f"Cellpose stitched mask shape mismatch: input={roi.shape}, mask={masks.shape}. "
            "元voxel gridを維持できないため処理を停止しました。"
        )

    return masks.astype(np.int32, copy=False), {
        "elapsed_sec": float(elapsed),
        "version": _cellpose_version(),
        "model": str(model_name),
        "mode": "2D + stitch",
        "stitch_threshold": float(stitch_threshold),
        "kwargs": kwargs,
    }


# ============================================================
# Match Cellpose labels to candidate nuclei
# ============================================================

def label_properties(labels):
    max_id = int(labels.max())
    if max_id <= 0:
        return []

    counts = np.bincount(labels.ravel(), minlength=max_id + 1)
    objs = ndi.find_objects(labels, max_label=max_id)
    out = []

    for lid in range(1, max_id + 1):
        if counts[lid] <= 0:
            continue
        sl = objs[lid - 1]
        if sl is None:
            continue
        local = labels[sl] == lid
        cz, cy, cx = ndi.center_of_mass(local)
        out.append({
            "label_id": lid,
            "voxels": int(counts[lid]),
            "centroid_z": float(cz + sl[0].start),
            "centroid_y": float(cy + sl[1].start),
            "centroid_x": float(cx + sl[2].start),
            "slice": sl,
        })
    return out


def match_labels_to_candidates(labels, candidates, bbox):
    """
    Assign one unique Cellpose object to each MIP candidate using XY centroid distance.
    The candidate detector does not alter the Cellpose mask; it only associates labels.
    """
    z0, z1, y0, y1, x0, x1 = bbox
    props = label_properties(labels)
    if not props:
        return []

    pairs = []
    for ci, c in enumerate(candidates):
        ty = float(c["center_y"] - y0)
        tx = float(c["center_x"] - x0)
        for p in props:
            d = float(np.hypot(p["centroid_y"] - ty, p["centroid_x"] - tx))
            pairs.append((d, ci, p["label_id"]))

    # Greedy unique assignment by smallest distance.
    pairs.sort(key=lambda x: x[0])
    used_c = set()
    used_l = set()
    matches = []

    for dist, ci, lid in pairs:
        if ci in used_c or lid in used_l:
            continue
        c = candidates[ci]
        # Require centroid to be reasonably near the candidate ROI.
        roi_diag = np.hypot(c["y1"] - c["y0"], c["x1"] - c["x0"])
        if dist > max(20.0, 0.75 * roi_diag):
            continue
        used_c.add(ci)
        used_l.add(lid)
        matches.append({
            "candidate_index": ci,
            "candidate": c,
            "label_id": int(lid),
            "distance_px": float(dist),
        })

    matches.sort(key=lambda m: m["candidate"]["candidate"])
    return matches


# ============================================================
# Measurements
# ============================================================

def measure_mask(mask, pixel_x, pixel_y, z_spacing, bbox):
    mask = np.asarray(mask, dtype=bool)
    voxels = int(np.count_nonzero(mask))
    if voxels == 0:
        return None

    z0, z1, y0, y1, x0, x1 = bbox
    voxel_volume = float(pixel_x) * float(pixel_y) * float(z_spacing)
    volume = voxels * voxel_volume

    coords = np.argwhere(mask)
    zz0, yy0, xx0 = coords.min(axis=0)
    zz1, yy1, xx1 = coords.max(axis=0)
    cz, cy, cx = ndi.center_of_mass(mask)

    area_px_by_z = np.count_nonzero(mask, axis=(1, 2))
    areas = area_px_by_z.astype(np.float64) * float(pixel_x) * float(pixel_y)

    return {
        "Volume (µm³)": float(volume),
        "Voxel count": voxels,
        "Z start": int(zz0 + z0 + 1),
        "Z end": int(zz1 + z0 + 1),
        "Z slices": int(zz1 - zz0 + 1),
        "Max area (µm²)": float(areas.max()),
        "Mean area (µm²)": float(areas[area_px_by_z > 0].mean()),
        "Center Z": float(cz + z0 + 1),
        "Center Y (original px)": float(cy + y0),
        "Center X (original px)": float(cx + x0),
        "Touches Z edge": bool(np.any(mask[0]) or np.any(mask[-1])),
        "Touches XY crop edge": bool(
            np.any(mask[:, 0, :]) or np.any(mask[:, -1, :]) or
            np.any(mask[:, :, 0]) or np.any(mask[:, :, -1])
        ),
    }


def analyze_grouped_once(
    raw,
    candidates,
    pixel_x,
    pixel_y,
    z_spacing,
    model_name,
    prefer_gpu,
    cellprob_threshold,
    flow_threshold,
    stitch_threshold,
    cellpose_min_size,
    batch_size,
    resample,
    bbox_extra_margin,
):
    bbox = union_bbox(candidates, raw.shape, extra_margin=int(bbox_extra_margin))
    z0, z1, y0, y1, x0, x1 = bbox
    roi = raw[z0:z1, y0:y1, x0:x1]

    labels, cp_info = run_cellpose_2d_stitch(
        roi=roi,
        model_name=model_name,
        prefer_gpu=prefer_gpu,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        stitch_threshold=stitch_threshold,
        min_size=cellpose_min_size,
        batch_size=batch_size,
        resample=resample,
    )

    matches = match_labels_to_candidates(labels, candidates, bbox)
    rows = []
    records = []

    for nucleus_no, m in enumerate(matches, start=1):
        lid = int(m["label_id"])
        mask = labels == lid
        meas = measure_mask(mask, pixel_x, pixel_y, z_spacing, bbox)
        if meas is None:
            continue

        row = {
            "Nucleus": nucleus_no,
            "Candidate ROI": int(m["candidate"]["candidate"]),
            "Cellpose object ID": lid,
            "Candidate/object distance (px)": round(float(m["distance_px"]), 2),
            "Volume (µm³)": round(meas["Volume (µm³)"], 3),
            "Voxel count": int(meas["Voxel count"]),
            "Z start": meas["Z start"],
            "Z end": meas["Z end"],
            "Z slices": meas["Z slices"],
            "Max area (µm²)": round(meas["Max area (µm²)"], 3),
            "Mean area (µm²)": round(meas["Mean area (µm²)"], 3),
            "Center Z": round(meas["Center Z"], 2),
            "Center Y (original px)": round(meas["Center Y (original px)"], 2),
            "Center X (original px)": round(meas["Center X (original px)"], 2),
            "Z edge touch": bool(meas["Touches Z edge"]),
            "XY crop edge touch": bool(meas["Touches XY crop edge"]),
        }
        rows.append(row)
        records.append({
            "nucleus": nucleus_no,
            "candidate": int(m["candidate"]["candidate"]),
            "label_id": lid,
            "mask": mask,
            "roi": roi,
            "bbox": bbox,
        })

    return records, pd.DataFrame(rows), {
        "bbox": bbox,
        "roi_shape": roi.shape,
        "labels": labels,
        "cellpose": cp_info,
        "detected_objects": int(labels.max()),
        "matched_objects": len(records),
    }


# ============================================================
# Visualization
# ============================================================

def make_slice_overlay(record, local_z):
    import matplotlib.pyplot as plt

    roi = record["roi"]
    mask = record["mask"]
    local_z = int(np.clip(local_z, 0, roi.shape[0] - 1))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(normalize_for_display(roi[local_z]), cmap="gray")

    m = mask[local_z]
    if np.any(m):
        color = object_color(record["nucleus"])
        rgba = np.zeros((*m.shape, 4), dtype=np.float32)
        rgba[..., :3] = color
        rgba[..., 3] = m.astype(np.float32) * 0.42
        ax.imshow(rgba)
        ax.contour(m, levels=[0.5], colors=[color], linewidths=2.0)

    global_z = record["bbox"][0] + local_z + 1
    ax.set_title(f"核{record['nucleus']} / global Z={global_z}")
    ax.axis("off")
    fig.tight_layout()
    return fig


def mask_to_mesh(mask, pixel_x, pixel_y, z_spacing):
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

    crop = mask[z0:z1, y0:y1, x0:x1]
    if min(crop.shape) < 2:
        return None

    verts, faces, _, _ = marching_cubes(
        crop.astype(np.uint8),
        level=0.5,
        spacing=(float(z_spacing), float(pixel_y), float(pixel_x)),
    )
    verts[:, 0] += z0 * float(z_spacing)
    verts[:, 1] += y0 * float(pixel_y)
    verts[:, 2] += x0 * float(pixel_x)
    return verts[:, 2], verts[:, 1], verts[:, 0], faces


def make_3d_figure(record, pixel_x, pixel_y, z_spacing):
    mesh = mask_to_mesh(record["mask"], pixel_x, pixel_y, z_spacing)
    fig = go.Figure()
    if mesh is None:
        return fig

    x, y, z, faces = mesh
    rgb = rgb255(object_color(record["nucleus"]))
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=f"rgb({rgb[0]},{rgb[1]},{rgb[2]})",
        opacity=0.75,
        flatshading=False,
        showscale=False,
        name=f"核{record['nucleus']}",
    ))
    fig.update_layout(
        title=f"核{record['nucleus']} 3D",
        scene=dict(
            xaxis_title="X (µm)",
            yaxis_title="Y (µm)",
            zaxis_title="Z (µm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=45, b=0),
    )
    return fig


def make_mask_review_pdf(records, pixel_x, pixel_y, z_spacing, slices_per_page=6):
    """
    Create an in-memory PDF containing masked XY slices for every detected nucleus.
    Each page shows up to 6 Z slices. Only slices containing that nucleus are included.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        for record in records:
            roi = record["roi"]
            mask = record["mask"]
            z_indices = np.where(np.any(mask, axis=(1, 2)))[0]
            if z_indices.size == 0:
                continue

            for p0 in range(0, len(z_indices), int(slices_per_page)):
                page_z = z_indices[p0:p0 + int(slices_per_page)]
                n = len(page_z)
                cols = 3
                rows_n = int(np.ceil(n / cols))
                fig, axes = plt.subplots(rows_n, cols, figsize=(11.69, 8.27))
                axes = np.atleast_1d(axes).ravel()

                for ax in axes:
                    ax.axis("off")

                for ax, local_z in zip(axes, page_z):
                    bg = normalize_for_display(roi[int(local_z)])
                    m = mask[int(local_z)]
                    ax.imshow(bg, cmap="gray")

                    color = object_color(record["nucleus"])
                    rgba = np.zeros((*m.shape, 4), dtype=np.float32)
                    rgba[..., :3] = color
                    rgba[..., 3] = m.astype(np.float32) * 0.42
                    ax.imshow(rgba)
                    if np.any(m):
                        ax.contour(m, levels=[0.5], colors=[color], linewidths=1.2)

                    global_z = record["bbox"][0] + int(local_z) + 1
                    ax.set_title(f"Nucleus {record['nucleus']} / Z={global_z}", fontsize=9)
                    ax.axis("off")

                fig.suptitle(
                    f"Cellpose 2D + Z stitch mask review - Nucleus {record['nucleus']}",
                    fontsize=13,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("① DAPI / image")
channel = st.sidebar.selectbox("DAPI channel", ["R", "G", "B"], index=2)

st.sidebar.header("② Voxel size")
pixel_x = st.sidebar.number_input("Pixel X (µm)", min_value=0.0001, value=0.0983, step=0.001, format="%.4f")
pixel_y = st.sidebar.number_input("Pixel Y (µm)", min_value=0.0001, value=0.0983, step=0.001, format="%.4f")
z_spacing = st.sidebar.number_input("Z spacing (µm)", min_value=0.0001, value=0.200, step=0.010, format="%.3f")

st.sidebar.header("③ 高速候補検出")
expected_count = st.sidebar.number_input("予想される核数", min_value=1, max_value=50, value=5, step=1)
xy_margin = st.sidebar.slider("候補XY余白 (px)", 16, 192, 48, 8)
min_candidate_area = st.sidebar.number_input("候補の最小2D面積 (pixel)", min_value=20, max_value=100000, value=300, step=50)
bbox_extra_margin = st.sidebar.slider("まとめROIの追加余白 (px)", 0, 96, 16, 8)

st.sidebar.header("④ Cellpose 2D + Z stitch")
cellpose_model_name = st.sidebar.text_input("Cellpose pretrained model", value="cpsam_v2")
prefer_gpu = st.sidebar.checkbox("GPUを優先", value=True)
cellpose_cellprob_threshold = st.sidebar.slider("Cell probability threshold", -6.0, 6.0, 0.0, 0.1)
cellpose_flow_threshold = st.sidebar.slider("Flow threshold", 0.0, 1.5, 0.4, 0.05)
cellpose_stitch_threshold = st.sidebar.slider(
    "Z stitch IoU threshold", 0.05, 0.90, 0.25, 0.05,
    help="隣接Z面のmaskを同じ核としてつなぐIoU基準。Cellpose utils.stitch3Dの標準値は0.25です。",
)
cellpose_batch_size = st.sidebar.select_slider(
    "Batch size", options=[4, 8, 16, 32, 64], value=32,
    help="GPUメモリに余裕があれば大きいほど速くなる場合があります。MemoryErrorなら下げてください。",
)
cellpose_resample = st.sidebar.checkbox(
    "高精度resample（遅くなる）", value=False,
    help="OFFを高速設定にしています。境界精度を比較したい場合だけONで再解析してください。",
)
cellpose_min_size = st.sidebar.number_input("Cellpose minimum mask size (2D pixel)", min_value=0, value=15, step=5)

st.sidebar.caption(
    "true 3D推論は使わず、各XY面を2Dで一括推論してからZ方向にstitchします。核を切り替えても再推論しません。"
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
    st.info("DAPIのZ-stack TIFをアップロードしてください。")
    st.stop()

uploaded_files = sorted(uploaded_files, key=lambda x: natural_key(x.name))
file_signature = tuple((f.name, f.size) for f in uploaded_files)

if st.session_state.file_signature != file_signature:
    st.session_state.file_signature = file_signature
    st.session_state.raw_stack = None
    st.session_state.candidates = None
    st.session_state.candidate_mip = None
    st.session_state.result_table = None
    st.session_state.analysis_record = None
    st.session_state.prediction_done = False
    st.session_state.excluded_ids = set()
    st.session_state.mask_review_pdf = None

if st.session_state.raw_stack is None:
    with st.spinner("Z-stackを読み込んでいます..."):
        try:
            st.session_state.raw_stack = load_uploaded_stack(uploaded_files, channel)
        except Exception as e:
            st.exception(e)
            st.stop()

raw = st.session_state.raw_stack
c1, c2, c3, c4 = st.columns(4)
c1.metric("Z", raw.shape[0])
c2.metric("Y", raw.shape[1])
c3.metric("X", raw.shape[2])
c4.metric("総voxel", f"{raw.size / 1e6:.1f} M")


# ============================================================
# Candidate detection
# ============================================================

st.header("② 核候補をMIPで高速検出")
detect_button = st.button("🔎 核候補ROIを検出", use_container_width=True)

if detect_button or st.session_state.candidates is None:
    with st.spinner("MIPから核候補を探しています..."):
        candidates, mip, _ = detect_nucleus_candidates(
            raw,
            expected_count=int(expected_count),
            xy_margin=int(xy_margin),
            min_component_area=int(min_candidate_area),
        )
    st.session_state.candidates = candidates
    st.session_state.candidate_mip = mip
    st.session_state.prediction_done = False
    st.session_state.result_table = None
    st.session_state.analysis_record = None
else:
    candidates = st.session_state.candidates
    mip = st.session_state.candidate_mip

if not candidates:
    st.error("核候補を検出できませんでした。候補の最小2D面積を小さくして再検出してください。")
    st.stop()

fig = make_candidate_figure(mip, candidates)
st.pyplot(fig, use_container_width=True)
import matplotlib.pyplot as plt
plt.close(fig)

candidate_ids = [int(c["candidate"]) for c in candidates]
selected_ids = st.multiselect(
    "Cellpose 2D + stitchで解析する候補ROI",
    candidate_ids,
    default=candidate_ids[: min(int(expected_count), len(candidate_ids))],
)
selected_candidates = [c for c in candidates if int(c["candidate"]) in set(selected_ids)]

if not selected_candidates:
    st.warning("解析するROIを1つ以上選択してください。")
    st.stop()

bbox_preview = union_bbox(selected_candidates, raw.shape, extra_margin=int(bbox_extra_margin))
z0, z1, y0, y1, x0, x1 = bbox_preview
shape_preview = (z1 - z0, y1 - y0, x1 - x0)

p1, p2, p3 = st.columns(3)
p1.metric("選択候補", len(selected_candidates))
p2.metric("Cellpose実行回数", "1回")
p3.metric("まとめROI", f"{shape_preview[0]}×{shape_preview[1]}×{shape_preview[2]}")


# ============================================================
# One grouped Cellpose inference
# ============================================================

st.header("③ Cellpose 2D + Z stitchを一括実行")
st.write(
    "選択した核候補を含むROIの各Z面をCellpose 2Dで一括処理し、隣接スライスのmaskをIoUでZ方向につなぎます。"
    "true 3D推論は行わず、核1→核2の表示切替でも再推論しません。"
)

analyze_button = st.button("🧠 Cellpose 2D + stitchで体積計算", type="primary", use_container_width=True)

if analyze_button:
    try:
        with st.spinner("Cellposeモデルを読み込み、全Z面を2D解析してstitchしています..."):
            t0 = time.perf_counter()
            records, result_table, info = analyze_grouped_once(
                raw=raw,
                candidates=selected_candidates,
                pixel_x=float(pixel_x),
                pixel_y=float(pixel_y),
                z_spacing=float(z_spacing),
                model_name=str(cellpose_model_name),
                prefer_gpu=bool(prefer_gpu),
                cellprob_threshold=float(cellpose_cellprob_threshold),
                flow_threshold=float(cellpose_flow_threshold),
                stitch_threshold=float(cellpose_stitch_threshold),
                cellpose_min_size=int(cellpose_min_size),
                batch_size=int(cellpose_batch_size),
                resample=bool(cellpose_resample),
                bbox_extra_margin=int(bbox_extra_margin),
            )
            total = time.perf_counter() - t0

        st.session_state.analysis_record = {"records": records, "info": info, "total_sec": total}
        st.session_state.mask_review_pdf = None
        st.session_state.result_table = result_table
        st.session_state.prediction_done = True
        st.success(f"完了しました。2D + stitch総解析時間 {total:.1f} 秒です。")
    except Exception as e:
        st.error("Cellpose 2D + stitchの実行に失敗しました。")
        st.exception(e)
        st.stop()

if not st.session_state.prediction_done:
    st.stop()


# ============================================================
# Results
# ============================================================

results = st.session_state.result_table
analysis = st.session_state.analysis_record
records = analysis["records"]
info = analysis["info"]

st.header("④ 体積結果")

if results is None or results.empty:
    st.error("候補核に対応するCellpose objectを取得できませんでした。")
    st.write(f"Cellpose detected objects: {info.get('detected_objects', 0)}")
    st.stop()

st.dataframe(results, use_container_width=True, hide_index=True)

r1, r2, r3, r4 = st.columns(4)
r1.metric("Cellpose detected", info["detected_objects"])
r2.metric("候補と対応", info["matched_objects"])
r3.metric("解析方式", "2D+stitch")
r4.metric("Cellpose推論時間", f"{info['cellpose']['elapsed_sec']:.1f} s")

st.caption(
    f"Model: {info['cellpose']['model']} / Cellpose version: {info['cellpose']['version']} / "
    f"stitch IoU: {info['cellpose']['stitch_threshold']:.2f}"
)


# ============================================================
# Review each nucleus (NO re-inference here)
# ============================================================

st.header("⑤ 核ごとの確認（ここではAI再計算しません）")

nuclei = [int(r["nucleus"]) for r in records]
selected_nucleus = st.selectbox("表示する核", nuclei)
record = next(r for r in records if int(r["nucleus"]) == int(selected_nucleus))

mask = record["mask"]
z_has = np.where(np.any(mask, axis=(1, 2)))[0]
if z_has.size > 0:
    default_z = int(round(float(z_has.mean())))
else:
    default_z = record["roi"].shape[0] // 2

local_z = st.slider(
    "Z slice",
    min_value=0,
    max_value=record["roi"].shape[0] - 1,
    value=int(default_z),
    step=1,
)

col_a, col_b = st.columns(2)
with col_a:
    fig2 = make_slice_overlay(record, local_z)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
with col_b:
    fig3 = make_3d_figure(record, pixel_x, pixel_y, z_spacing)
    st.plotly_chart(fig3, use_container_width=True)

st.info(
    "核を切り替えてもCellposeは再実行しません。すでに2D解析＋Z stitchingで得たlabel volumeから表示だけを切り替えます。"
)


# ============================================================
# CSV
# ============================================================

st.header("⑥ CSV")
csv = results.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "📥 CSVを保存",
    data=csv,
    file_name="3D_nuclear_volume_Cellpose2D_stitch_fast.csv",
    mime="text/csv",
)


st.header("⑦ マスク確認PDF")
st.write(
    "各核について、Cellpose maskが存在する全ZスライスをPDFにまとめます。"
    "PDF作成ではAI解析をやり直しません。"
)

@st.cache_data(show_spinner=False)
def _cached_mask_pdf(pdf_key, _records, px, py, zs):
    # pdf_key is used only to invalidate cache when analysis changes.
    return make_mask_review_pdf(_records, px, py, zs, slices_per_page=6)

pdf_key = (
    tuple((r["nucleus"], r["label_id"], int(np.count_nonzero(r["mask"]))) for r in records),
    float(pixel_x), float(pixel_y), float(z_spacing),
)

if st.button("📄 マスク確認PDFを作成", use_container_width=True):
    with st.spinner("マスク確認PDFを作成しています..."):
        pdf_bytes = _cached_mask_pdf(
            pdf_key, records, float(pixel_x), float(pixel_y), float(z_spacing)
        )
    st.session_state["mask_review_pdf"] = pdf_bytes

if st.session_state.get("mask_review_pdf"):
    st.download_button(
        "📥 マスク確認PDFを保存",
        data=st.session_state["mask_review_pdf"],
        file_name="Cellpose2D_stitch_mask_review.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# ============================================================
# Technical info / license
# ============================================================

with st.expander("解析条件"):
    st.write(f"""
**Raw shape:** {raw.shape}

**Grouped Cellpose ROI:** {info['roi_shape']}

**Segmentation mode:** Cellpose 2D + Z stitch

**Pixel X:** {pixel_x:.4f} µm

**Pixel Y:** {pixel_y:.4f} µm

**Z spacing:** {z_spacing:.4f} µm

**XY resize:** なし

**Z resize:** なし

**Cellpose model:** {cellpose_model_name}

**Cell probability threshold:** {cellpose_cellprob_threshold:.2f}

**Flow threshold:** {cellpose_flow_threshold:.2f}

**Z stitch IoU threshold:** {cellpose_stitch_threshold:.2f}

**Batch size:** {cellpose_batch_size}

**Resample:** {cellpose_resample}

**Volume formula:** voxel count × Pixel X × Pixel Y × Z spacing
""")

with st.expander("Cellposeライセンス / 研究利用"):
    st.markdown(
        """
Cellpose software is distributed under the **BSD 3-Clause License**.
研究利用・改変・組み込みはライセンス条件に従って可能です。
論文では、実際に使用したCellposeのバージョンとモデルを記録し、対応するCellpose論文を引用してください。

このアプリでは、MIP候補検出はCellposeを実行する範囲を絞るためだけに使用し、最終的な核境界は各Z面のCellpose 2D maskをCellposeのstitch機能でZ方向に連結したlabel maskから取得します。
"""
    )
