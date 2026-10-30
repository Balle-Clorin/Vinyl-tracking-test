"""
Tracking Ability Summary — Streamlit App
=========================================
Upload one or more WAV files recorded from a tracking-ability test record.
The app segments each file into tracks, measures THD per track, and shows:

  Tab 1 — Raw THD summary (THD % vs groove amplitude)
  Tab 2 — Normalised THD summary (amplitude + radius normalisation)

Supported records
-----------------
  ClearAudio  333 Hz   50–100 µm   ref: 50 µm / 125 mm
  Ortofon     315 Hz   50–100 µm   ref: 50 µm / 125 mm  (Ortofon inner radii)
  CBS STR-112 300 Hz   22.4–89.6 µm  ref: 22.4 µm / 124 mm
"""

import io
import gc
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, sosfilt
import streamlit as st

# ── record definitions ────────────────────────────────────────────────────────

RECORDS = {
    "Ortofon 315 Hz": {
        "fund_hz":  315,
        "ref_um":    50.0,
        "ref_r_mm": 125.0,
        "radius_mm": {50: 99.0, 60: 92.0, 70: 90.0,
                      80: 88.0, 90: 85.5, 100: 83.5},
    },
    "ClearAudio 333 Hz": {
        "fund_hz":  333,
        "ref_um":    50.0,
        "ref_r_mm": 125.0,
        "radius_mm": {50: 125.0, 60: 120.0, 70: 114.0,
                      80: 108.0, 90: 103.0, 100: 97.0},
    },
    "CBS STR-112 300 Hz": {
        "fund_hz":  300,
        "ref_um":   22.4,
        "ref_r_mm": 124.0,
        "radius_mm": {22.4: 124.0, 31.7: 120.0, 44.8: 116.6,
                      63.4: 113.0, 89.6: 107.0},
    },
}

COL_L    = "#4C9BE8"
COL_R    = "#E8734C"
COL_BG   = "#1A1A2E"
COL_FG   = "#E8E8F0"
COL_GRID = "#2E2E4E"

# ── DSP helpers ───────────────────────────────────────────────────────────────

def highpass(sig, sr, freq=100, order=4):
    sos = butter(order, freq, btype="high", fs=sr, output="sos")
    return sosfilt(sos, sig)


def find_tracks(data, sr, gap_thresh=0.02, min_gap_s=0.3, min_track_s=2.0):
    if data.ndim == 1:
        data = data[:, np.newaxis]
    win_s = 0.010
    win   = max(1, int(win_s * sr))
    n     = len(data) // win
    rms   = np.array([
        np.sqrt(np.mean(data[i * win:(i + 1) * win, :] ** 2))
        for i in range(n)
    ])
    silent_frac = np.mean(rms < gap_thresh)
    if silent_frac < 0.005:
        gap_thresh = np.percentile(rms, 2) * 3.0
    silent         = rms < gap_thresh
    min_gap_wins   = max(1, int(min_gap_s   / win_s))
    min_track_wins = max(1, int(min_track_s / win_s))
    gap_ranges = []
    i = 0
    while i < n:
        if silent[i]:
            j = i
            while j < n and silent[j]:
                j += 1
            if (j - i) >= min_gap_wins:
                gap_ranges.append((i, j))
            i = j
        else:
            i += 1
    seg_starts = [0]                        + [g[1] for g in gap_ranges]
    seg_ends   = [g[0] for g in gap_ranges] + [n]
    tracks = []
    for s_w, e_w in zip(seg_starts, seg_ends):
        if (e_w - s_w) >= min_track_wins:
            tracks.append((s_w * win, min(e_w * win, len(data))))
    return tracks


def trim_track(seg, sr, trim_s=0.5):
    n = int(trim_s * sr)
    return seg[n:-n] if len(seg) > 2 * n else seg


def detect_fundamental(sig, sr, fund_nominal, search_bw=20, zero_pad=8):
    N    = len(sig)
    Npad = N * zero_pad
    win  = np.hanning(N)
    spec = np.fft.rfft(sig * win, n=Npad)
    freq = np.fft.rfftfreq(Npad, 1 / sr)
    mags = np.abs(spec)
    mask = (freq >= fund_nominal - search_bw) & (freq <= fund_nominal + search_bw)
    return freq[mask][np.argmax(mags[mask])] if mask.any() else fund_nominal


def harmonic_magnitudes(sig, sr, fund, n_harmonics=8):
    f_m  = detect_fundamental(sig, sr, fund)
    N    = len(sig)
    Npad = N * 8
    win  = np.hanning(N)
    spec = np.fft.rfft(sig * win, n=Npad)
    mags = np.abs(spec) * 2 / np.sum(win)
    freq = np.fft.rfftfreq(Npad, 1 / sr)
    bw   = min(max(sr / Npad * 4, 1.0), f_m * 0.40)
    result = []
    for h in range(1, n_harmonics + 1):
        f0   = f_m * h
        mask = (freq >= f0 - bw) & (freq <= f0 + bw)
        result.append(np.sqrt(np.sum(mags[mask] ** 2)) if mask.any() else 0.0)
    return f_m, np.array(result)


def calc_thd(hm):
    h1 = hm[0] if hm[0] > 0 else 1e-9
    return np.sqrt(np.sum(hm[1:] ** 2)) / h1 * 100


def normalise_thd(thd, level_um, r_mm, ref_um, ref_r_mm):
    return thd * (ref_um / level_um) * (r_mm / ref_r_mm)


# ── plotting ──────────────────────────────────────────────────────────────────

def set_style():
    plt.rcParams.update({
        "figure.facecolor": COL_BG, "axes.facecolor":  COL_BG,
        "axes.edgecolor":   COL_FG, "axes.labelcolor": COL_FG,
        "xtick.color":      COL_FG, "ytick.color":     COL_FG,
        "text.color":       COL_FG, "grid.color":      COL_GRID,
        "grid.linewidth":   0.6,    "font.family":     "monospace",
        "figure.dpi":       110,
        "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 11, "figure.titlesize": 15,
    })


def _summary_axes(axes, level_arr, thd_L, thd_R, results,
                  n_harmonics, ylabel, title_suffix, rec_label):
    cmap_h = plt.cm.tab10

    ax = axes[0, 0]
    ax.plot(level_arr, thd_L, "o-", color=COL_L, lw=2, label="Left")
    ax.plot(level_arr, thd_R, "s-", color=COL_R, lw=2, label="Right")
    ax.set_xlabel("Groove amplitude (µm)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"THD vs groove amplitude{title_suffix}")
    ax.legend(); ax.grid(True)

    for ax, key, h1_key, ch in [
        (axes[0, 1], "hm_L", "h1_L", "Left"),
        (axes[1, 0], "hm_R", "h1_R", "Right"),
    ]:
        for h in range(2, min(n_harmonics + 1, 9)):
            dBc_vals = [
                20 * np.log10(max(r[key][h - 1] / r[h1_key], 1e-9))
                for r in results
            ]
            ax.plot(level_arr, dBc_vals, "o-", color=cmap_h(h - 2),
                    lw=1.5, label=f"H{h}")
        ax.set_xlabel("Groove amplitude (µm)")
        ax.set_ylabel("dBc")
        ax.set_title(f"{ch} — harmonics vs level")
        ax.legend(fontsize=9, ncol=1, loc="upper left",
                  bbox_to_anchor=(1.01, 1), borderaxespad=0,
                  framealpha=0.7, handlelength=1.2,
                  handletextpad=0.4, markerscale=0.7)
        ax.grid(True)

    ax = axes[1, 1]
    imb    = thd_L - thd_R
    colors = [COL_L if v >= 0 else COL_R for v in imb]
    ax.bar(level_arr, imb, color=colors, width=level_arr[1] - level_arr[0] if len(level_arr) > 1 else 4)
    ax.axhline(0, color=COL_FG, lw=0.8)
    ax.set_xlabel("Groove amplitude (µm)")
    ax.set_ylabel(f"{ylabel} L − R")
    ax.set_title(f"Channel imbalance{title_suffix}\n(blue=L worse, orange=R worse)")
    ax.grid(True, axis="y")


def make_summary_fig(level_arr, thd_L, thd_R, results, n_harmonics,
                     rec_label, ylabel, title_suffix, norm_note=""):
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    suptitle = f"Tracking Ability — {rec_label}{title_suffix}"
    if norm_note:
        suptitle += f"\n{norm_note}"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    _summary_axes(axes, level_arr, thd_L, thd_R, results,
                  n_harmonics, ylabel, title_suffix, rec_label)
    fig.tight_layout(rect=[0, 0, 0.92, 0.94])
    return fig


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=COL_BG)
    buf.seek(0)
    plt.close(fig)
    gc.collect()
    return buf


# ── analysis pipeline ─────────────────────────────────────────────────────────

def analyse_wav(wav_bytes, rec, filename, progress_cb, description=""):
    fund     = rec["fund_hz"]
    levels   = sorted(rec["radius_mm"].keys())
    ref_um   = rec["ref_um"]
    ref_r_mm = rec["ref_r_mm"]

    data, sr = sf.read(io.BytesIO(wav_bytes))
    if data.ndim == 1:
        data = np.column_stack([data, data])

    tracks = find_tracks(data, sr)
    n_use  = min(len(tracks), len(levels))
    if n_use == 0:
        return None, "No tracks detected — check GAP_THRESH or file content."

    results = []
    for i in range(n_use):
        progress_cb(i / n_use, f"Analysing track {i+1}/{n_use} ({levels[i]} µm)…")
        s, e  = tracks[i]
        L_raw = trim_track(data[s:e, 0], sr)
        R_raw = trim_track(data[s:e, 1], sr)
        _, hm_L = harmonic_magnitudes(L_raw, sr, fund)
        _, hm_R = harmonic_magnitudes(R_raw, sr, fund)
        thd_L = calc_thd(hm_L)
        thd_R = calc_thd(hm_R)
        results.append({
            "level_um": levels[i],
            "thd_L": thd_L, "thd_R": thd_R,
            "hm_L":  hm_L,  "hm_R":  hm_R,
            "h1_L":  hm_L[0] or 1e-9,
            "h1_R":  hm_R[0] or 1e-9,
        })
        del L_raw, R_raw
        gc.collect()

    progress_cb(1.0, "Building plots…")

    level_arr  = np.array([r["level_um"] for r in results])
    thd_L_raw  = np.array([r["thd_L"]    for r in results])
    thd_R_raw  = np.array([r["thd_R"]    for r in results])
    thd_L_norm = np.array([
        normalise_thd(r["thd_L"], r["level_um"],
                      rec["radius_mm"].get(r["level_um"], ref_r_mm),
                      ref_um, ref_r_mm)
        for r in results
    ])
    thd_R_norm = np.array([
        normalise_thd(r["thd_R"], r["level_um"],
                      rec["radius_mm"].get(r["level_um"], ref_r_mm),
                      ref_um, ref_r_mm)
        for r in results
    ])

    rec_label = next(k for k, v in RECORDS.items() if v is rec)

    desc_suffix = f"\n{description}" if description else ""
    wav_name    = filename
    fig_raw = make_summary_fig(
        level_arr, thd_L_raw, thd_R_raw, results, 8,
        rec_label, ylabel="THD (%)", title_suffix="  —  Raw THD",
        norm_note=f"File: {wav_name}{desc_suffix}",
    )
    norm_note = (f"THD_norm = THD × ({ref_um}/µm) × (r_mm/{ref_r_mm:.0f})    "
                 f"ref: {ref_um} µm / {ref_r_mm:.0f} mm\nFile: {wav_name}{desc_suffix}")
    fig_norm = make_summary_fig(
        level_arr, thd_L_norm, thd_R_norm, results, 8,
        rec_label, ylabel="Normalised THD (%)",
        title_suffix="  —  Normalised THD", norm_note=norm_note,
    )

    raw_png  = fig_to_bytes(fig_raw)
    norm_png = fig_to_bytes(fig_norm)
    del data
    gc.collect()
    return (raw_png, norm_png, results, level_arr,
            thd_L_raw, thd_R_raw, thd_L_norm, thd_R_norm), None


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Tracking Ability Analyser",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 Tracking Ability Analyser")
st.caption("Upload a WAV recording of a tracking-ability test record. "
           "The app measures THD per track and shows raw and normalised summaries.")

# Sidebar — record selection
with st.sidebar:
    st.header("Configuration")
    rec_name    = st.selectbox("Test record", list(RECORDS.keys()))
    rec         = RECORDS[rec_name]
    description = st.text_input("Description", placeholder="e.g. My cartridge — setup notes here")

    st.markdown("---")
    st.markdown(f"**Frequency:** {rec['fund_hz']} Hz")
    st.markdown(f"**Reference:** {rec['ref_um']} µm / {rec['ref_r_mm']:.0f} mm")
    st.markdown("**Radius table (µm → mm):**")
    for um, r in sorted(rec["radius_mm"].items()):
        st.markdown(f"&nbsp;&nbsp;{um} µm → {r} mm")

    st.markdown("---")
    st.markdown(
        "**Normalisation formula:**\n\n"
        "THD_norm = THD × (ref_µm / track_µm) × (track_r_mm / ref_r_mm)"
    )

# File upload
uploaded = st.file_uploader(
    "Upload WAV file", type=["wav"],
    help="Stereo or mono WAV, any sample rate. "
         "Tracks are auto-detected from inter-track silences.",
)

if uploaded is not None:
    wav_bytes = uploaded.read()
    desc_line = f" — {description}" if description else ""
    st.info(f"File: **{uploaded.name}**  ({len(wav_bytes)/1e6:.1f} MB){desc_line}")

    progress = st.progress(0, text="Starting…")

    def progress_cb(frac, msg):
        progress.progress(frac, text=msg)

    result, err = analyse_wav(wav_bytes, rec, uploaded.name, progress_cb, description)
    progress.empty()

    if err:
        st.error(err)
    else:
        (raw_png, norm_png, results, level_arr,
         thd_L_raw, thd_R_raw, thd_L_norm, thd_R_norm) = result

        # Summary table
        with st.expander("📊 THD values per track", expanded=True):
            import pandas as pd
            rows = []
            for r, nl, nr in zip(results, thd_L_norm, thd_R_norm):
                rows.append({
                    "Amplitude (µm)": r["level_um"],
                    "THD L raw (%)":  f"{r['thd_L']:.2f}",
                    "THD R raw (%)":  f"{r['thd_R']:.2f}",
                    "THD L norm (%)": f"{nl:.2f}",
                    "THD R norm (%)": f"{nr:.2f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Two tabs: normalised (default) | raw
        tab_norm, tab_raw = st.tabs(["Normalised THD", "Raw THD"])

        with tab_raw:
            st.image(raw_png, use_container_width=True)
            st.download_button("⬇ Download PNG", raw_png,
                               file_name="summary_raw.png",
                               mime="image/png")

        with tab_norm:
            st.image(norm_png, use_container_width=True)
            st.download_button("⬇ Download PNG", norm_png,
                               file_name="summary_normalised.png",
                               mime="image/png")

        del wav_bytes, result
        gc.collect()
else:
    st.markdown(
        """
        ### How to use
        1. Select the **test record** in the sidebar.
        2. Upload a **stereo WAV** recording of the test record.
        3. The app auto-detects the tracks and shows two summary plots.

        #### Normalisation
        The normalised plot corrects for:
        - **Amplitude**: higher groove amplitudes produce more THD by geometry alone
          → corrected by `ref_µm / track_µm`
        - **Groove radius**: inner grooves are harder to track at the same amplitude
          → corrected by `track_r_mm / ref_r_mm`

        If the cartridge tracks perfectly linearly, all curves collapse to a flat line.
        """
    )
