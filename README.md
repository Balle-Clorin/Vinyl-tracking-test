# Tracking Ability Analyser

Streamlit web app for vinyl tracking distortion analysis from test record WAV recordings.

## Features

- Auto-detects tracks from inter-track silences (robust to gaps as short as 0.3 s)
- Measures THD (H1–H8) per track, left and right channel independently
- **Tab 1 — Raw THD**: THD % vs groove amplitude + harmonic profiles + L/R imbalance
- **Tab 2 — Normalised THD**: same plots with amplitude and groove-radius normalisation

## Supported test records

| Record | Frequency | Amplitudes | Reference |
|--------|-----------|------------|-----------|
| ClearAudio | 333 Hz | 50–100 µm | 50 µm / 125 mm |
| Ortofon | 315 Hz | 50–100 µm | 50 µm / 125 mm |
| CBS STR-112 | 300 Hz | 22.4–89.6 µm | 22.4 µm / 124 mm |

## Normalisation formula

```
THD_norm = THD_raw × (ref_µm / track_µm) × (track_r_mm / ref_r_mm)
```

- **Amplitude factor** `ref_µm / track_µm`: removes expected THD rise with groove velocity
- **Radius factor** `track_r_mm / ref_r_mm`: inner grooves are harder to track — corrects for position on record

If the cartridge tracks perfectly linearly, all normalised curves collapse to a flat horizontal line.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repository (e.g. `Balle-Clorin/tracking-analyser`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect the repo, set main file to `app.py`
4. Deploy
