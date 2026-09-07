import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ["TQDM_DISABLE"] = "1"

# Streamlit Cloud では tqdm の出力先が切れて BrokenPipeError になることがあるため、
# tqdm 自体も無効化しておく。
try:
    import tqdm as _tqdm_module
    _orig_tqdm = _tqdm_module.tqdm

    def _silent_tqdm(*args, **kwargs):
        kwargs["disable"] = True
        return _orig_tqdm(*args, **kwargs)

    _tqdm_module.tqdm = _silent_tqdm
except Exception:
    pass

import gc, re, time
from io import BytesIO
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import tifffile
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import marching_cubes
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go

st.set_page_config(page_title='3D Nuclear Volume Analyzer - AI Edge', page_icon='🧬', layout='wide')
st.title('🧬 3D Nuclear Volume Analyzer - AI Edge')
st.caption('StarDist 3D（AI）で核をseed検出し、背景補正＋2段階しきい値＋Z連続性で淡い外縁まで追跡します。')

DEFAULTS={'raw':None,'sig':None,'cands':None,'records':None,'table':None,'done':False,'selected':None,'sam_click':None,'sam_2d':None,'sam_3d':None,'sam_target_nucleus':None,'sam_target_z':None,'report_pdf':None,'report_pdf_name':None}
for k,v in DEFAULTS.items():
    if k not in st.session_state: st.session_state[k]=v

def natural_key(s):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)',str(s))]

def norm_disp(x):
    x=np.asarray(x,np.float32); lo=np.percentile(x,1); hi=np.percentile(x,99.8)
    if hi<=lo: hi=lo+1
    return np.clip((x-lo)/(hi-lo),0,1)

def load_stack(files,channel):
    files=sorted(files,key=lambda f:natural_key(f.name)); out=[]
    for f in files:
        a=tifffile.imread(f)
        if a.ndim==2: pass
        elif a.ndim==3 and a.shape[-1] in (3,4): a=a[...,{'R':0,'G':1,'B':2}[channel]]
        elif a.ndim==3 and a.shape[0]==1: a=a[0]
        else: raise ValueError(f'{f.name}: 未対応shape {a.shape}')
        out.append(np.asarray(a))
    if len(set(x.shape for x in out))!=1: raise ValueError('全Z画像のXYサイズを一致させてください。')
    a=np.stack(out,axis=0)
    return a.astype(np.float32) if a.dtype==np.float64 else a

def detect_candidates(stack,expected=5,xy_margin=64,z_margin=8,min_area=300):
    zdim,h,w=stack.shape; mip=np.max(stack,axis=0); sm=gaussian(norm_disp(mip),2,preserve_range=True)
    try: otsu=float(threshold_otsu(sm))
    except: otsu=.25
    m=sm>max(.04,otsu*.8)
    m=ndi.binary_opening(m,np.ones((3,3),bool)); m=ndi.binary_closing(m,np.ones((5,5),bool)); m=ndi.binary_fill_holes(m)
    lab,n=ndi.label(m); sizes=np.bincount(lab.ravel()) if n else np.array([0])
    comps=[]
    for i in range(1,n+1):
        area=int(sizes[i])
        if area<min_area or area>h*w*.25: continue
        ys,xs=np.where(lab==i)
        comps.append((area,int(ys.min()),int(ys.max()),int(xs.min()),int(xs.max()),float(ys.mean()),float(xs.mean())))
    comps=sorted(comps,reverse=True)[:max(expected+3,expected)]
    out=[]
    for j,(area,ymin,ymax,xmin,xmax,cy,cx) in enumerate(comps,1):
        y0=max(0,ymin-xy_margin); y1=min(h,ymax+xy_margin+1); x0=max(0,xmin-xy_margin); x1=min(w,xmax+xy_margin+1)
        r=stack[:,y0:y1,x0:x1]; zs=np.percentile(r.reshape(zdim,-1),99.5,axis=1).astype(np.float32); zs=ndi.gaussian_filter1d(zs,1)
        base=float(np.percentile(zs,20)); peak=float(zs.max())
        if peak<=base: z0,z1=0,zdim
        else:
            p=np.where(zs>=base+.18*(peak-base))[0]
            if len(p)==0: z0,z1=0,zdim
            else:
                z0=max(0,int(p.min())-z_margin); z1=min(zdim,int(p.max())+z_margin+1)
                if z1-z0<12:
                    mid=(z0+z1)//2; z0=max(0,mid-10); z1=min(zdim,mid+11)
        out.append(dict(candidate=j,z0=z0,z1=z1,y0=y0,y1=y1,x0=x0,x1=x1,center_y=cy,center_x=cx,area_px=area))
    return out,mip

@st.cache_resource(show_spinner=False)
def model():
    from stardist.models import StarDist3D
    return StarDist3D.from_pretrained('3D_demo')

def run_sd(seed_roi, prob, nms, tz, tyx):
    """
    StarDist 3Dを小さいseed ROIに実行。
    Streamlit Cloudでtqdmの出力先が切れてBrokenPipeErrorになる場合があるため、
    BrokenPipeErrorを捕捉し、進捗表示を無効化した状態で再試行する。
    """
    from csbdeep.utils import normalize

    x = normalize(
        np.asarray(seed_roi),
        1,
        99.8,
        axis=(0, 1, 2),
    ).astype(np.float32, copy=False)

    probs = [
        float(prob),
        min(float(prob), 0.30),
        min(float(prob), 0.25),
        min(float(prob), 0.20),
        min(float(prob), 0.15),
    ]
    probs = list(
        dict.fromkeys(
            round(max(0.05, float(p)), 3)
            for p in probs
        )
    )

    tile_plans = [
        (int(tz), int(tyx), int(tyx)),
        (2, 2, 2),
        (4, 2, 2),
        (4, 3, 3),
        (8, 3, 3),
        (8, 4, 4),
    ]

    # duplicateを除外
    uniq = []
    seen = set()
    for nt in tile_plans:
        nt = tuple(max(1, int(v)) for v in nt)
        if nt not in seen:
            seen.add(nt)
            uniq.append(nt)
    tile_plans = uniq

    memory_retries = 0
    zero_retries = 0
    broken_pipe_retries = 0

    try:
        for p in probs:
            for nt in tile_plans:
                try:
                    labels, _ = model().predict_instances(
                        x,
                        axes="ZYX",
                        prob_thresh=float(p),
                        nms_thresh=float(nms),
                        n_tiles=tuple(map(int, nt)),
                        verbose=False,
                    )

                    labels = labels.astype(np.int32, copy=False)

                    if int(labels.max()) > 0:
                        return (
                            labels,
                            nt,
                            float(p),
                            int(memory_retries),
                            int(zero_retries),
                        )

                    zero_retries += 1

                    # 同じprobでtileだけ増やしても、0検出は改善しにくいため
                    # 次のprob thresholdへ進む
                    break

                except BrokenPipeError:
                    broken_pipe_retries += 1
                    gc.collect()

                    # tqdmを再度明示的に無効化
                    try:
                        import tqdm as _tqdm_module
                        _orig = getattr(_tqdm_module, "_chatgpt_orig_tqdm", None)
                        if _orig is None:
                            _orig = _tqdm_module.tqdm
                            _tqdm_module._chatgpt_orig_tqdm = _orig

                        def _silent_tqdm_local(*args, **kwargs):
                            kwargs["disable"] = True
                            return _orig(*args, **kwargs)

                        _tqdm_module.tqdm = _silent_tqdm_local
                    except Exception:
                        pass

                    # 何度も同じBrokenPipeが出る場合は次tileへ
                    if broken_pipe_retries >= 3:
                        continue
                    continue

                except Exception as e:
                    msg = str(e).lower()

                    if (
                        isinstance(e, MemoryError)
                        or "out of memory" in msg
                        or "resourceexhausted" in msg
                        or "unable to allocate" in msg
                        or "oom" in msg
                    ):
                        memory_retries += 1
                        gc.collect()
                        continue

                    raise

        return (
            np.zeros(seed_roi.shape, dtype=np.int32),
            tile_plans[-1],
            float(probs[-1]),
            int(memory_retries),
            int(zero_retries),
        )

    finally:
        try:
            del x
        except Exception:
            pass
        gc.collect()

def choose_object(labels,shape,ty,tx):
    m=int(labels.max())
    if m<1:return None
    cnt=np.bincount(labels.ravel(),minlength=m+1); objs=ndi.find_objects(labels,max_label=m); H,W=shape[1:]
    score=[]
    for i in range(1,m+1):
        sl=objs[i-1]
        if sl is None or cnt[i]==0: continue
        c=labels[sl]==i; _,cy,cx=ndi.center_of_mass(c); cy+=sl[1].start; cx+=sl[2].start
        d=np.hypot(cy-ty,cx-tx)/max(1,np.hypot(H,W)); score.append((d-np.log1p(cnt[i])*.015,-cnt[i],i))
    return int(sorted(score)[0][2]) if score else None

def component_with_seed(mask,seed):
    lab,n=ndi.label(mask)
    if n<1:return np.zeros_like(mask,bool)
    ids=lab[seed]; ids=ids[ids>0]
    if not ids.size:return np.zeros_like(mask,bool)
    v,c=np.unique(ids,return_counts=True); return lab==int(v[np.argmax(c)])

def xy_guard(sm,seed,dilate):
    mip=np.max(sm,axis=0); seed2=np.max(seed,axis=0)
    try:o=float(threshold_otsu(mip.ravel()))
    except:o=float(np.percentile(mip,65))
    g=(mip>=max(.01,o*.65))|seed2; g=ndi.binary_closing(g,np.ones((5,5),bool),iterations=2); g=ndi.binary_fill_holes(g)
    s=component_with_seed(g,seed2)
    if np.any(s): g=s
    if dilate>0:g=ndi.binary_dilation(g,np.ones((3,3),bool),iterations=int(dilate))
    return g

def overlap_comp(mask,ref,dilate=4):
    if not np.any(mask) or not np.any(ref):return np.zeros_like(mask,bool)
    rr=ndi.binary_dilation(ref,np.ones((3,3),bool),iterations=max(1,int(dilate))); lab,n=ndi.label(mask)
    if n<1:return np.zeros_like(mask,bool)
    ids=lab[rr]; ids=ids[ids>0]
    if not ids.size:return np.zeros_like(mask,bool)
    v,c=np.unique(ids,return_counts=True); comp=lab==int(v[np.argmax(c)])
    if np.count_nonzero(comp&rr)<max(2,int(np.count_nonzero(ref)*.005)):return np.zeros_like(mask,bool)
    return comp

def segment(full,seed,sens,bg_sigma,sigma_xy,high_scale,low_ratio,guard_px,gaps):
    sens=float(np.clip(sens,0,1)); x=np.asarray(full,np.float32); lo=np.percentile(x,1); hi=np.percentile(x,99.8); hi=hi if hi>lo else lo+1
    x=np.clip((x-lo)/(hi-lo),0,1)
    if bg_sigma>0:
        bg=ndi.gaussian_filter(x,(0,float(bg_sigma),float(bg_sigma)),mode='nearest'); x=np.clip(x-bg,0,None); del bg
        h2=np.percentile(x,99.8); x=x/(h2 if h2>0 else 1)
    sm=ndi.gaussian_filter(x,(.45,float(sigma_xy),float(sigma_xy)),mode='nearest').astype(np.float32,copy=False); del x
    g=xy_guard(sm,seed,int(round(guard_px+8*sens)))
    sample=sm.ravel(); sample=sample[::max(1,sample.size//1_000_000)] if sample.size>1_000_000 else sample
    try:o=float(threshold_otsu(sample))
    except:o=float(np.percentile(sample,65))
    high=float(np.clip(o*high_scale*(1.05-.20*sens),.02,.95)); low=float(np.clip(high*(low_ratio-.16*sens),.008,high)); tail=float(np.clip(low*(.94-.18*sens),.006,low))
    close=1+round(2*sens); dil=3+round(4*sens); whole=np.zeros_like(seed,bool); zs=np.where(np.any(seed,axis=(1,2)))[0]
    for z in range(int(zs.min()),int(zs.max())+1):
        s=seed[z]
        if not np.any(s) and z>0:s=whole[z-1]
        if not np.any(s):continue
        cand=(sm[z]>=low)&g; cand|=s; cand=ndi.binary_closing(cand,np.ones((3,3),bool),iterations=close); cand=ndi.binary_fill_holes(cand)
        comp=component_with_seed(cand,s); whole[z]=comp if np.any(comp) else s
    whole|=seed; p=np.where(np.any(whole,axis=(1,2)))[0]; tracked=0
    for direction,start in [(-1,int(p.min())-1),(1,int(p.max())+1)]:
        ref=whole[int(p.min()) if direction<0 else int(p.max())].copy(); gap=0; z=start
        while 0<=z<whole.shape[0]:
            cand=(sm[z]>=tail)&g; cand=ndi.binary_closing(cand,np.ones((3,3),bool),iterations=close); cand=ndi.binary_fill_holes(cand); comp=overlap_comp(cand,ref,dil)
            if np.any(comp) and np.count_nonzero(comp)<=max(np.count_nonzero(ref)*(3+1.5*sens),np.count_nonzero(ref)+500):
                whole[z]=comp; ref=comp; tracked+=1; gap=0
            else: gap+=1
            if gap>gaps:break
            z+=direction
    whole|=seed; whole&=g[None,:,:]
    for z in range(whole.shape[0]):
        if np.any(whole[z]):whole[z]=ndi.binary_fill_holes(whole[z])
    del sm; gc.collect()
    return whole,dict(high=high,low=low,tail=tail,tracked=tracked,seed_vox=int(np.count_nonzero(seed)),whole_vox=int(np.count_nonzero(whole)))

def measure(mask,px,py,dz):
    n=int(np.count_nonzero(mask)); c=np.argwhere(mask); z0=int(c[:,0].min()); z1=int(c[:,0].max())
    return dict(volume=n*px*py*dz,vox=n,zstart=z0+1,zend=z1+1,zs=z1-z0+1)

def overlay(raw,rec,z=None,mip=False):
    r=rec['roi']; img=raw[:,r['y0']:r['y1'],r['x0']:r['x1']]; m=rec['mask']
    if mip: bg=norm_disp(np.max(img,0)); mm=np.max(m,0); title=f"核{rec['nucleus']} MIP"
    else: zi=int(z)-1; bg=norm_disp(img[zi]); mm=m[zi]; title=f"核{rec['nucleus']} / Z={zi+1}"
    fig,ax=plt.subplots(figsize=(7,7)); ax.imshow(bg,cmap='gray')
    if np.any(mm):
        rgba=np.zeros((*mm.shape,4),np.float32); rgba[...,2]=1; rgba[...,3]=mm*.32; ax.imshow(rgba); ax.contour(mm,levels=[.5],colors=['cyan'],linewidths=2)
    ax.set_title(title); ax.axis('off'); fig.tight_layout(); return fig


def make_3d_mask_figure(mask, px, py, dz, title="3D nuclear mask"):
    """
    最終bool maskから等値面を作り、Plotlyで回転可能な3D表示を返す。
    軸は実寸 µm。
    """
    m = np.asarray(mask, dtype=np.uint8)

    if m.ndim != 3 or not np.any(m):
        return None

    # marching_cubesが外周を閉じられるよう1 voxel pad
    padded = np.pad(m, 1, mode="constant")

    try:
        verts, faces, _, _ = marching_cubes(
            padded.astype(np.float32),
            level=0.5,
            spacing=(float(dz), float(py), float(px)),
        )
    except Exception:
        return None

    # padした1 voxel分を座標から戻す
    verts[:, 0] -= float(dz)
    verts[:, 1] -= float(py)
    verts[:, 2] -= float(px)

    # ブラウザ負荷を抑えるため、面数が非常に多い場合のみ間引く
    max_faces = 120_000
    if faces.shape[0] > max_faces:
        step = int(np.ceil(faces.shape[0] / max_faces))
        faces = faces[::step]

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=verts[:, 2],
                y=verts[:, 1],
                z=verts[:, 0],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.65,
                flatshading=False,
                hoverinfo="skip",
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (µm)",
            yaxis_title="Y (µm)",
            zaxis_title="Z (µm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=45),
        height=650,
    )
    return fig


def _pdf_overlay_rgb(img2d, mask2d):
    """
    PDF用のRGB overlay。
    grayscaleをRGB化し、mask内を青く薄く重ね、境界をシアン相当にする。
    """
    g = (norm_disp(img2d) * 255).astype(np.uint8)
    rgb = np.repeat(g[..., None], 3, axis=2).astype(np.float32)

    m = np.asarray(mask2d, dtype=bool)
    if np.any(m):
        alpha = 0.32
        blue = np.zeros_like(rgb)
        blue[..., 2] = 255
        rgb[m] = (1 - alpha) * rgb[m] + alpha * blue[m]

        edge = m ^ ndi.binary_erosion(m)
        rgb[edge, 0] = 0
        rgb[edge, 1] = 255
        rgb[edge, 2] = 255

    return np.clip(rgb, 0, 255).astype(np.uint8)


def build_mask_comparison_pdf(
    raw,
    records,
    results,
    file_names,
    px,
    py,
    dz,
    selected_nuclei=None,
    include_all_z=False,
):
    """
    PDF:
      - 1ページ目: 解析サマリー
      - 以降: 1ページに2 Z slices
        Original / Binary mask / Overlay の3列比較
    """
    if selected_nuclei is None:
        selected_nuclei = [int(r["nucleus"]) for r in records]
    else:
        selected_nuclei = [int(x) for x in selected_nuclei]

    buf = BytesIO()

    with PdfPages(buf) as pdf:
        # ---------------- Summary page ----------------
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        ax = fig.add_axes([0.04, 0.06, 0.92, 0.88])
        ax.axis("off")

        ax.text(
            0.0, 0.98,
            "3D Nuclear Volume - Image / Mask Comparison Report",
            fontsize=18, fontweight="bold", va="top",
        )
        ax.text(
            0.0, 0.92,
            f"Voxel size: X={float(px):.4f} um, Y={float(py):.4f} um, Z={float(dz):.4f} um",
            fontsize=10, va="top",
        )

        table_rows = []
        for nuc in selected_nuclei:
            row = results[results["Nucleus"] == nuc]
            if row.empty:
                continue
            rr = row.iloc[0]
            table_rows.append([
                str(nuc),
                f"{float(rr['Volume (µm³)']):.3f}",
                str(int(rr["Voxel count"])),
                str(int(rr["Z start"])),
                str(int(rr["Z end"])),
                str(rr.get("Mask source", "Standard")),
            ])

        if table_rows:
            tbl = ax.table(
                cellText=table_rows,
                colLabels=[
                    "Nucleus", "Volume (um3)", "Voxels",
                    "Z start", "Z end", "Mask source"
                ],
                loc="upper left",
                cellLoc="center",
                bbox=[0.0, 0.30, 1.0, 0.53],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)

        ax.text(
            0.0, 0.22,
            "Each following row shows: Original image | Binary mask | Overlay.",
            fontsize=10,
        )
        ax.text(
            0.0, 0.17,
            "Cyan/blue areas are the voxels used for volume calculation.",
            fontsize=10,
        )
        ax.text(
            0.0, 0.12,
            "Volume = mask voxel count x Pixel X x Pixel Y x Z spacing.",
            fontsize=10,
        )

        pdf.savefig(fig, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---------------- Image/mask pages ----------------
        for nuc in selected_nuclei:
            rec = next(
                (r for r in records if int(r["nucleus"]) == int(nuc)),
                None,
            )
            if rec is None:
                continue

            roi = rec["roi"]
            mask3d = np.asarray(rec["mask"], dtype=bool)

            if include_all_z:
                z_indices = list(range(raw.shape[0]))
            else:
                z_indices = np.where(np.any(mask3d, axis=(1, 2)))[0].tolist()

            if not z_indices:
                continue

            # 2 slices / page, each slice has 3 panels
            for page_start in range(0, len(z_indices), 2):
                page_z = z_indices[page_start:page_start + 2]

                fig, axes = plt.subplots(
                    2, 3,
                    figsize=(11.69, 8.27),
                    squeeze=False,
                )

                for row_i in range(2):
                    for col_i in range(3):
                        axes[row_i, col_i].axis("off")

                for row_i, zg in enumerate(page_z):
                    img = raw[
                        zg,
                        roi["y0"]:roi["y1"],
                        roi["x0"]:roi["x1"],
                    ]
                    m = mask3d[zg]

                    filename = (
                        file_names[zg]
                        if zg < len(file_names)
                        else f"Z_{zg + 1}"
                    )

                    axes[row_i, 0].imshow(norm_disp(img), cmap="gray")
                    axes[row_i, 0].set_title(
                        f"Original - Z {zg + 1}\n{filename}",
                        fontsize=9,
                    )

                    axes[row_i, 1].imshow(m, cmap="gray", vmin=0, vmax=1)
                    axes[row_i, 1].set_title(
                        f"Binary mask - Z {zg + 1}",
                        fontsize=9,
                    )

                    axes[row_i, 2].imshow(_pdf_overlay_rgb(img, m))
                    axes[row_i, 2].set_title(
                        f"Overlay - Z {zg + 1}",
                        fontsize=9,
                    )

                    for col_i in range(3):
                        axes[row_i, col_i].axis("off")

                source_row = results[results["Nucleus"] == int(nuc)]
                source = (
                    str(source_row.iloc[0].get("Mask source", "Standard"))
                    if not source_row.empty
                    else "Standard"
                )

                fig.suptitle(
                    f"Nucleus {nuc} - mask source: {source}",
                    fontsize=14,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig, dpi=150)
                plt.close(fig)

    buf.seek(0)
    return buf.getvalue()



# ============================================================
# MobileSAM fallback (only for difficult images)
# ============================================================

MOBILE_SAM_URL = (
    "https://raw.githubusercontent.com/ChaoningZhang/"
    "MobileSAM/master/weights/mobile_sam.pt"
)


def ensure_mobile_sam_checkpoint():
    checkpoint = Path("/tmp/mobile_sam.pt")
    if checkpoint.exists() and checkpoint.stat().st_size > 10_000_000:
        return checkpoint

    tmp = checkpoint.with_suffix(".download")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    urllib.request.urlretrieve(MOBILE_SAM_URL, tmp)

    if not tmp.exists() or tmp.stat().st_size < 10_000_000:
        try:
            tmp.unlink()
        except Exception:
            pass
        raise RuntimeError(
            "MobileSAMの重みを正しく取得できませんでした。"
            "GitHub側のダウンロードが一時的に失敗している可能性があります。"
        )

    tmp.replace(checkpoint)
    return checkpoint


def _to_rgb_uint8(img2d):
    g = (norm_disp(img2d) * 255.0).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


def mobile_sam_predict_once(img2d, point_x, point_y, current_mask=None):
    try:
        model.clear()
    except Exception:
        pass
    gc.collect()

    import torch
    from mobile_sam import sam_model_registry, SamPredictor

    checkpoint = ensure_mobile_sam_checkpoint()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mobile = sam_model_registry["vit_t"](checkpoint=str(checkpoint))
    mobile.to(device=device)
    mobile.eval()

    predictor = SamPredictor(mobile)
    rgb = _to_rgb_uint8(img2d)
    predictor.set_image(rgb)

    p = np.array([[float(point_x), float(point_y)]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)

    masks, scores, _ = predictor.predict(
        point_coords=p,
        point_labels=labels,
        multimask_output=True,
    )

    h, w = img2d.shape
    click_x = int(np.clip(round(point_x), 0, w - 1))
    click_y = int(np.clip(round(point_y), 0, h - 1))

    best_mask = None
    best_rank = None
    total_px = float(h * w)

    for m, sc in zip(masks, scores):
        m = np.asarray(m, dtype=bool)
        area_frac = float(np.count_nonzero(m)) / max(1.0, total_px)
        contains = bool(m[click_y, click_x])
        plausible = 0.003 <= area_frac <= 0.85

        iou = 0.0
        if current_mask is not None and np.any(current_mask):
            inter = np.count_nonzero(m & current_mask)
            union = np.count_nonzero(m | current_mask)
            if union:
                iou = inter / float(union)

        rank = (
            (1.0 if contains else 0.0)
            + (0.5 if plausible else -0.5)
            + float(sc)
            + 0.20 * float(iou)
        )

        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_mask = m

    if best_mask is None:
        raise RuntimeError("MobileSAMがマスク候補を返しませんでした。")

    best_mask = ndi.binary_closing(
        best_mask,
        structure=np.ones((3, 3), dtype=bool),
        iterations=1,
    )
    best_mask = ndi.binary_fill_holes(best_mask)

    del predictor, mobile, masks
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_mask.astype(bool), str(device)


def make_sam_click_image(img2d, current_mask=None, click=None):
    rgb = _to_rgb_uint8(img2d).copy()

    if current_mask is not None and np.any(current_mask):
        edge = current_mask ^ ndi.binary_erosion(current_mask)
        rgb[edge] = np.array([0, 170, 255], dtype=np.uint8)

    if click is not None:
        x, y = click
        x = int(np.clip(round(x), 0, rgb.shape[1] - 1))
        y = int(np.clip(round(y), 0, rgb.shape[0] - 1))
        yy, xx = np.ogrid[:rgb.shape[0], :rgb.shape[1]]
        dot = (xx - x) ** 2 + (yy - y) ** 2 <= 5 ** 2
        rgb[dot] = np.array([255, 40, 40], dtype=np.uint8)

    return Image.fromarray(rgb)


def make_mask_comparison_figure(img2d, standard_mask, sam_mask, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, mask, name in [
        (axes[0], standard_mask, "通常認識"),
        (axes[1], sam_mask, "MobileSAM補助"),
    ]:
        ax.imshow(norm_disp(img2d), cmap="gray")
        if mask is not None and np.any(mask):
            ax.contour(mask, levels=[0.5], colors=["cyan"], linewidths=2)
            rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
            rgba[..., 2] = 1.0
            rgba[..., 3] = mask.astype(np.float32) * 0.25
            ax.imshow(rgba)
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _slice_threshold_candidate(sm2d, guide, reference, sensitivity):
    vals = sm2d[guide]
    if vals.size < 20:
        return np.zeros_like(guide, dtype=bool)

    try:
        t = float(threshold_otsu(vals))
    except Exception:
        t = float(np.percentile(vals, 55))

    factor = float(np.clip(0.90 - 0.30 * sensitivity, 0.55, 0.90))
    thr = max(0.005, t * factor)

    cand = (sm2d >= thr) & guide
    cand = ndi.binary_closing(
        cand,
        structure=np.ones((3, 3), dtype=bool),
        iterations=1 + int(round(sensitivity)),
    )
    cand = ndi.binary_fill_holes(cand)

    return overlap_comp(
        cand,
        reference,
        dilate=4 + int(round(3 * sensitivity)),
    )


def propagate_mobile_sam_mask(
    full_roi,
    anchor_mask,
    anchor_z,
    sensitivity=0.70,
    bg_sigma=12.0,
    sigma_xy=1.0,
    max_gap=2,
    guide_dilation=14,
):
    sens = float(np.clip(sensitivity, 0.0, 1.0))
    full = np.asarray(full_roi)

    x = np.asarray(full, dtype=np.float32)
    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.8))
    if hi <= lo:
        hi = lo + 1.0

    x = np.clip((x - lo) / (hi - lo), 0, 1)

    if bg_sigma > 0:
        bg = ndi.gaussian_filter(
            x,
            sigma=(0.0, float(bg_sigma), float(bg_sigma)),
            mode="nearest",
        )
        x = np.clip(x - bg, 0, None)
        del bg
        h2 = float(np.percentile(x, 99.8))
        if h2 > 0:
            x /= h2

    sm = ndi.gaussian_filter(
        x,
        sigma=(0.35, float(sigma_xy), float(sigma_xy)),
        mode="nearest",
    ).astype(np.float32, copy=False)
    del x

    anchor = np.asarray(anchor_mask, dtype=bool)
    anchor_z = int(np.clip(anchor_z, 0, full.shape[0] - 1))

    guide = ndi.binary_dilation(
        anchor,
        structure=np.ones((3, 3), dtype=bool),
        iterations=max(1, int(guide_dilation)),
    )
    guide = ndi.binary_fill_holes(guide)

    out = np.zeros(full.shape, dtype=bool)
    out[anchor_z] = anchor
    tracked = 0

    for direction in (-1, 1):
        z = anchor_z + direction
        ref = anchor.copy()
        gap = 0

        while 0 <= z < full.shape[0]:
            comp = _slice_threshold_candidate(
                sm[z],
                guide,
                ref,
                sens,
            )

            if np.any(comp):
                prev_area = max(1, np.count_nonzero(ref))
                area = np.count_nonzero(comp)

                if area <= max(
                    prev_area * (3.2 + sens),
                    prev_area + 600,
                ):
                    out[z] = comp
                    ref = comp
                    gap = 0
                    tracked += 1
                else:
                    gap += 1
            else:
                gap += 1

            if gap > int(max_gap):
                break

            z += direction

    for z in range(out.shape[0]):
        if np.any(out[z]):
            out[z] = ndi.binary_fill_holes(out[z])

    del sm
    gc.collect()
    return out.astype(bool), tracked


def update_result_with_fallback(results, records, nucleus, new_mask, px, py, dz):
    rec = next(
        r for r in records
        if int(r["nucleus"]) == int(nucleus)
    )

    rec["mask"] = np.asarray(new_mask, dtype=bool)
    m = measure(rec["mask"], float(px), float(py), float(dz))

    row_idx = results.index[results["Nucleus"] == int(nucleus)]
    if len(row_idx):
        i = row_idx[0]
        results.loc[i, "Volume (µm³)"] = round(m["volume"], 3)
        results.loc[i, "Voxel count"] = int(m["vox"])
        results.loc[i, "Z start"] = int(m["zstart"])
        results.loc[i, "Z end"] = int(m["zend"])
        results.loc[i, "Z slices"] = int(m["zs"])
        results.loc[i, "Mask source"] = "MobileSAM fallback"

    return results, records



# Sidebar
st.sidebar.header('① DAPI / image'); channel=st.sidebar.selectbox('DAPI channel',['R','G','B'],index=2)
st.sidebar.header('② Voxel size'); px=st.sidebar.number_input('Pixel X (µm)',.0001,value=.093,format='%.4f'); py=st.sidebar.number_input('Pixel Y (µm)',.0001,value=.093,format='%.4f'); dz=st.sidebar.number_input('Z spacing (µm)',.0001,value=.400,format='%.3f')
st.sidebar.header('③ ROI'); expected=st.sidebar.number_input('予想される核数',1,30,5); xy_margin=st.sidebar.slider('XY余白',32,256,64,16); z_margin=st.sidebar.slider('Seed用Z余白',2,30,8); min_area=st.sidebar.number_input('候補の最小2D面積',20,100000,300,50)
st.sidebar.header('④ StarDist AI'); prob=st.sidebar.slider('Probability threshold',.05,.95,.35,.01); nms=st.sidebar.slider('NMS overlap threshold',.05,.90,.30,.01); tz=st.sidebar.select_slider('Z tile数',[1,2,4,8],value=4); tyx=st.sidebar.select_slider('XY tile数',[1,2,3,4],value=3)
st.sidebar.header('⑤ 淡い外縁'); sens=st.sidebar.slider('核の端の感度',0.,1.,.65,.05); bg_sigma=st.sidebar.slider('背景補正',0.,30.,15.,1.); sigma_xy=st.sidebar.slider('粒ノイズ平滑化',0.,3.,1.2,.1); high_scale=st.sidebar.slider('核中心しきい値',.4,1.2,.82,.02); low_ratio=st.sidebar.slider('淡い外縁しきい値',.25,.8,.48,.01); guard_px=st.sidebar.slider('核外形ガード余白',0,40,12,2); gaps=st.sidebar.slider('Z方向の途切れ許容',0,5,2)

st.header('① Z-stack TIFを読み込む')
files=st.file_uploader('2D TIFをすべて選択',type=['tif','tiff'],accept_multiple_files=True)
if not files: st.stop()
sig=tuple((f.name,f.size) for f in files)
if st.session_state.sig!=sig:
    st.session_state.sig=sig; st.session_state.raw=None; st.session_state.cands=None; st.session_state.records=None; st.session_state.table=None; st.session_state.done=False; st.session_state.selected=None; st.session_state.sam_click=None; st.session_state.sam_2d=None; st.session_state.sam_3d=None; st.session_state.sam_target_nucleus=None; st.session_state.sam_target_z=None; st.session_state.report_pdf=None; st.session_state.report_pdf_name=None
if st.session_state.raw is None:
    with st.spinner('読み込み中...'): st.session_state.raw=load_stack(files,channel)
raw=st.session_state.raw
c1,c2,c3,c4=st.columns(4); c1.metric('Z',raw.shape[0]); c2.metric('Y',raw.shape[1]); c3.metric('X',raw.shape[2]); c4.metric('Raw memory',f'{raw.nbytes/1024**2:.1f} MB')

st.header('② 核候補')
if st.button('🔎 核候補ROIを検出',width='stretch') or st.session_state.cands is None:
    cands,mip=detect_candidates(raw,int(expected),int(xy_margin),int(z_margin),int(min_area)); st.session_state.cands=cands; st.session_state.selected=[c['candidate'] for c in cands]
else: cands=st.session_state.cands; mip=np.max(raw,0)
if not cands: st.error('核候補がありません。'); st.stop()
fig,ax=plt.subplots(figsize=(8,8)); ax.imshow(norm_disp(mip),cmap='gray')
from matplotlib.patches import Rectangle
for c in cands:
    ax.add_patch(Rectangle((c['x0'],c['y0']),c['x1']-c['x0'],c['y1']-c['y0'],fill=False,edgecolor='cyan',linewidth=2)); ax.text(c['x0']+5,c['y0']+15,f"ROI {c['candidate']}",color='white')
ax.axis('off'); st.pyplot(fig); plt.close(fig)
ids=[c['candidate'] for c in cands]; selected=st.multiselect('解析するROI',ids,default=[x for x in (st.session_state.selected or []) if x in ids]); st.session_state.selected=selected; selected_c=[c for c in cands if c['candidate'] in selected]

st.header('③ AI seed + 外縁追跡')
if st.button('🧠 選択ROIを解析',type='primary',width='stretch',disabled=not selected_c):
    prog=st.progress(0.); status=st.empty(); rows=[]; records=[]
    try:
        status.write('StarDistモデル準備中...'); model()
        for idx,c in enumerate(selected_c,1):
            status.write(f"ROI {idx}/{len(selected_c)} 解析中"); prog.progress((idx-1)/len(selected_c))
            seed_roi=raw[c['z0']:c['z1'],c['y0']:c['y1'],c['x0']:c['x1']]
            t0=time.perf_counter(); labels,nt,used_prob,mem,zero=run_sd(seed_roi,prob,nms,tz,tyx)
            oid=choose_object(labels,seed_roi.shape,c['center_y']-c['y0'],c['center_x']-c['x0'])
            if oid is None: del labels; gc.collect(); continue
            seed_small=labels==oid; detected=int(labels.max()); del labels; gc.collect()
            full=raw[:,c['y0']:c['y1'],c['x0']:c['x1']]; seed=np.zeros(full.shape,bool); seed[c['z0']:c['z1']]=seed_small; del seed_small
            mask,info=segment(full,seed,sens,bg_sigma,sigma_xy,high_scale,low_ratio,guard_px,gaps); del seed; gc.collect()
            if not np.any(mask): continue
            m=measure(mask,float(px),float(py),float(dz)); nuc=len(rows)+1
            rows.append({'Nucleus':nuc,'Candidate ROI':c['candidate'],'Volume (µm³)':round(m['volume'],3),'Voxel count':m['vox'],'Z start':m['zstart'],'Z end':m['zend'],'Z slices':m['zs'],'Z tracked slices':info['tracked'],'StarDist seed voxels':info['seed_vox'],'Whole/seed ratio':round(info['whole_vox']/max(info['seed_vox'],1),2),'High threshold':round(info['high'],4),'Low threshold':round(info['low'],4),'Tail threshold':round(info['tail'],4),'StarDist detected':detected,'Used probability':used_prob,'n_tiles':'×'.join(map(str,nt)),'Memory retries':mem,'Time (s)':round(time.perf_counter()-t0,2),'Mask source':'Standard'})
            records.append({'nucleus':nuc,'candidate':c['candidate'],'roi':{'y0':c['y0'],'y1':c['y1'],'x0':c['x0'],'x1':c['x1']},'mask':mask})
        st.session_state.records=records; st.session_state.table=pd.DataFrame(rows); st.session_state.report_pdf=None; st.session_state.report_pdf_name=None; st.session_state.done=True; prog.progress(1.); status.success('完了')
    except Exception as e: st.exception(e); st.stop()
if not st.session_state.done: st.stop()

results=st.session_state.table; records=st.session_state.records or []
st.header('④ 体積結果')
if results is None or results.empty: st.error('解析できた核がありません。'); st.stop()
st.dataframe(results,width='stretch',hide_index=True)

st.header('⑤ 認識結果を確認')
nuclei=[r['nucleus'] for r in records]; n=st.selectbox('核',nuclei); rec=next(r for r in records if r['nucleus']==n)
fig=overlay(raw,rec,mip=True); st.pyplot(fig); plt.close(fig)
p=np.where(np.any(rec['mask'],axis=(1,2)))[0]; default=int(round((p.min()+p.max())/2))+1 if len(p) else raw.shape[0]//2
z=st.slider('表示するglobal Z',1,int(raw.shape[0]),int(default)); fig=overlay(raw,rec,z=z); st.pyplot(fig); plt.close(fig)


st.subheader("3Dマスク表示")
st.caption(
    "体積計算に実際に使っている最終3Dマスクを立体表示します。"
    "ドラッグで回転、ホイールで拡大縮小できます。"
)
fig3d = make_3d_mask_figure(
    rec["mask"],
    px=float(px),
    py=float(py),
    dz=float(dz),
    title=f"Nucleus {n} - 3D mask",
)
if fig3d is None:
    st.warning("この核は3D表面を生成できませんでした。")
else:
    st.plotly_chart(fig3d, width="stretch", config={"displaylogo": False})



st.header("⑥ 認識不良時だけ MobileSAM AI補助")

st.info(
    "通常認識がうまくいかない核だけに使います。"
    "核の中央を1回クリックすると、無料の軽量AI MobileSAMが2D外形を提案します。"
    "その外形をアンカーに上下Zへ追跡し、通常マスクと比較してから採用できます。"
)

sam_nucleus = st.selectbox(
    "MobileSAMで補助する核",
    nuclei,
    key="sam_nucleus_selector",
)

sam_rec = next(
    r for r in records
    if int(r["nucleus"]) == int(sam_nucleus)
)

sam_roi = sam_rec["roi"]
sam_present = np.where(np.any(sam_rec["mask"], axis=(1, 2)))[0]

if len(sam_present):
    sam_default_z = int(round((sam_present.min() + sam_present.max()) / 2)) + 1
else:
    sam_default_z = max(1, int(raw.shape[0] // 2))

sam_z = st.slider(
    "MobileSAMに見せるglobal Z",
    min_value=1,
    max_value=int(raw.shape[0]),
    value=int(sam_default_z),
    step=1,
    key="sam_z_slider",
)

roi_slice = raw[
    sam_z - 1,
    sam_roi["y0"]:sam_roi["y1"],
    sam_roi["x0"]:sam_roi["x1"],
]
standard_2d = sam_rec["mask"][sam_z - 1]

target_changed = (
    st.session_state.sam_target_nucleus != int(sam_nucleus)
    or st.session_state.sam_target_z != int(sam_z)
)

if target_changed:
    st.session_state.sam_click = None
    st.session_state.sam_2d = None
    st.session_state.sam_3d = None
    st.session_state.sam_target_nucleus = int(sam_nucleus)
    st.session_state.sam_target_z = int(sam_z)

st.write(
    "**下の画像で、核だと確信できる中央付近を1回クリックしてください。** "
    "青線は現在の通常認識です。"
)

click_image = make_sam_click_image(
    roi_slice,
    current_mask=standard_2d,
    click=st.session_state.sam_click,
)

orig_w = int(roi_slice.shape[1])
display_w = min(700, orig_w)

clicked = streamlit_image_coordinates(
    click_image,
    width=display_w,
    cursor="crosshair",
    key=f"sam_click_{sam_nucleus}_{sam_z}",
)

if clicked is not None:
    scale = orig_w / float(display_w)
    cx = float(clicked["x"]) * scale
    cy = float(clicked["y"]) * scale
    st.session_state.sam_click = (cx, cy)

if st.session_state.sam_click is not None:
    cx, cy = st.session_state.sam_click
    st.caption(f"クリック位置: ROI内 X={cx:.1f}, Y={cy:.1f}")

    if st.button(
        "🤖 MobileSAMで2D核外形を提案",
        type="secondary",
        width="stretch",
    ):
        with st.spinner(
            "MobileSAMを読み込み、核外形を推定しています。"
            "初回は約41 MBの重みを取得します..."
        ):
            sam_mask_2d, sam_device = mobile_sam_predict_once(
                roi_slice,
                point_x=cx,
                point_y=cy,
                current_mask=standard_2d,
            )
            st.session_state.sam_2d = sam_mask_2d
            st.session_state.sam_3d = None
            st.success(f"MobileSAM 2D推定完了（device={sam_device}）")

if st.session_state.sam_2d is not None:
    fig_compare = make_mask_comparison_figure(
        roi_slice,
        standard_2d,
        st.session_state.sam_2d,
        title=f"核{sam_nucleus} / global Z={sam_z}",
    )
    st.pyplot(fig_compare, width="stretch")
    plt.close(fig_compare)

    st.caption(
        "左が今までの認識、右がMobileSAMです。"
        "右の輪郭が核外形として妥当なら、次に3D補助マスクを作成します。"
    )

    sam_sens = st.slider(
        "MobileSAM Z追跡の端感度",
        0.0,
        1.0,
        0.70,
        0.05,
        key="sam_sens",
    )
    sam_gap = st.slider(
        "MobileSAM Z追跡の途切れ許容",
        0,
        5,
        2,
        1,
        key="sam_gap",
    )
    sam_guide = st.slider(
        "MobileSAM XYガイド余白",
        4,
        30,
        14,
        2,
        key="sam_guide",
    )

    if st.button(
        "🧬 このMobileSAM外形から3D補助マスクを作成",
        width="stretch",
    ):
        full_roi = raw[
            :,
            sam_roi["y0"]:sam_roi["y1"],
            sam_roi["x0"]:sam_roi["x1"],
        ]

        with st.spinner("上下のZ sliceへ連続追跡しています..."):
            sam_mask_3d, tracked = propagate_mobile_sam_mask(
                full_roi,
                anchor_mask=st.session_state.sam_2d,
                anchor_z=int(sam_z - 1),
                sensitivity=float(sam_sens),
                bg_sigma=float(bg_sigma),
                sigma_xy=float(sigma_xy),
                max_gap=int(sam_gap),
                guide_dilation=int(sam_guide),
            )

        st.session_state.sam_3d = sam_mask_3d
        st.success(f"3D補助マスク作成完了（追加追跡 {tracked} slices）")

if st.session_state.sam_3d is not None:
    temp_rec = {
        "nucleus": int(sam_nucleus),
        "roi": sam_roi,
        "mask": st.session_state.sam_3d,
    }

    c_std, c_sam = st.columns(2)

    with c_std:
        st.write("**通常マスク MIP**")
        f1 = overlay(raw, sam_rec, mip=True)
        st.pyplot(f1, width="stretch")
        plt.close(f1)

    with c_sam:
        st.write("**MobileSAM補助 3D MIP**")
        f2 = overlay(raw, temp_rec, mip=True)
        st.pyplot(f2, width="stretch")
        plt.close(f2)

    m_std = measure(
        sam_rec["mask"],
        float(px),
        float(py),
        float(dz),
    )
    m_sam = measure(
        st.session_state.sam_3d,
        float(px),
        float(py),
        float(dz),
    )

    cc1, cc2 = st.columns(2)
    cc1.metric("通常マスク体積", f"{m_std['volume']:.3f} µm³")
    cc2.metric("MobileSAM補助体積", f"{m_sam['volume']:.3f} µm³")

    if st.button(
        "✅ このMobileSAM補助マスクを採用",
        type="primary",
        width="stretch",
    ):
        results, records = update_result_with_fallback(
            results.copy(),
            records,
            nucleus=int(sam_nucleus),
            new_mask=st.session_state.sam_3d,
            px=px,
            py=py,
            dz=dz,
        )

        st.session_state.table = results
        st.session_state.records = records
        st.session_state.sam_2d = None
        st.session_state.sam_3d = None
        st.session_state.sam_click = None
        st.session_state.report_pdf = None
        st.session_state.report_pdf_name = None

        st.success(f"核{sam_nucleus}をMobileSAM補助マスクに置き換えました。")
        st.rerun()



st.header("⑦ 画像 / マスク比較PDF")

st.caption(
    "アップロードした元画像、体積計算に使った二値マスク、"
    "元画像への重ね合わせをPDFで保存できます。"
)

pdf_scope = st.radio(
    "PDFに含める核",
    ["現在選択中の核だけ", "解析した全核"],
    horizontal=True,
    key="pdf_scope",
)

pdf_all_z = st.checkbox(
    "マスクが存在しないZ sliceも含める",
    value=False,
    help=(
        "OFFでは、核マスクが存在するZ sliceだけをPDFにします。"
        "ONではZ-stack全枚を含めるためPDFが大きくなります。"
    ),
)

if pdf_scope == "現在選択中の核だけ":
    pdf_nuclei = [int(n)]
    pdf_suffix = f"nucleus_{int(n)}"
else:
    pdf_nuclei = [int(x) for x in nuclei]
    pdf_suffix = "all_nuclei"

if st.button(
    "📄 比較PDFを作成",
    width="stretch",
):
    with st.spinner("元画像とマスクの比較PDFを作成しています..."):
        sorted_file_names = [
            f.name for f in sorted(files, key=lambda f: natural_key(f.name))
        ]
        st.session_state.report_pdf = build_mask_comparison_pdf(
            raw=raw,
            records=records,
            results=results,
            file_names=sorted_file_names,
            px=float(px),
            py=float(py),
            dz=float(dz),
            selected_nuclei=pdf_nuclei,
            include_all_z=bool(pdf_all_z),
        )
        st.session_state.report_pdf_name = (
            f"nuclear_mask_comparison_{pdf_suffix}.pdf"
        )
    st.success("比較PDFを作成しました。")

if st.session_state.report_pdf is not None:
    st.download_button(
        "📥 比較PDFをダウンロード",
        data=st.session_state.report_pdf,
        file_name=(
            st.session_state.report_pdf_name
            or "nuclear_mask_comparison.pdf"
        ),
        mime="application/pdf",
        width="stretch",
    )


st.header('⑧ CSV')
csv=results.to_csv(index=False).encode('utf-8-sig'); st.download_button('📥 CSVを保存',csv,'3D_nuclear_volume_AI_edge.csv','text/csv')

with st.expander('メモリ情報'):
    mb=sum(r['mask'].nbytes for r in records)/1024**2
    st.write(f'Raw: {raw.nbytes/1024**2:.1f} MB / 保存mask: {mb:.1f} MB')
    st.caption('StarDistの巨大labelsは解析後に保持せず、最終bool maskだけを保存します。')

if st.button('🔄 リセット'):
    for k in DEFAULTS: st.session_state[k]=DEFAULTS[k]
    gc.collect(); st.rerun()
