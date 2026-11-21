# Procedural Blender 4.2 PBR Cutaway Renderer (Public Render-Only)

Stateless, public-safe 3D procedural mechanical cutaway renderer using **Blender 4.2.3 LTS (Cycles CPU + OpenImageDenoise + AgX High Contrast + Fog Glow)**.

- **Contains NO secrets, NO API keys, NO narration audio, NO channel metadata, and NO episode HUD text.** Episode titles/teasers live only in the episode audio pack (see below), never in this repo.
- Outputs silent 6-second seamless 1080p (24fps) H.264 `.mp4` loops (`act_1_shot_1.mp4` .. `act_5_shot_3.mp4`) as GitHub Actions artifacts, rendered in 48-frame chunks (3 chunks per shot, up to 20 parallel jobs).

## Workflows
1. **`Render Single PBR Shot (1080p)`** — Renders 1 specific `(episode, act, shot, chunk)` 48-frame chunk (e.g., `ep05`, Act `1`, Shot `1`, Chunk `1`). Used to measure render time and verify the pipeline before a full episode.
2. **`Render Full 15-Shot Episode PBR (1080p)`** — Renders all 15 shots in 45 parallel chunk jobs, merges chunks into 15 clips, then (when an audio pack URL is supplied) assembles the final **1440p master** in the `assemble_master` job and publishes it as an artifact (`<episode>-final-1440p-master`, retention 30 days).

## Episode audio pack (supplied at dispatch, never committed)

The `assemble_master` job needs the episode audio pack as a zip downloaded from a Google Drive direct-download link (`audio_pack_url` input). The pack contains:

- `seg_1.wav` .. `seg_5.wav` — Leda narration segments (24 kHz mono)
- `acts_meta.json` — segment list, gap, act boundaries, total duration
- `hud.json` — act pills + bridge card text for this episode
- `subtitles/*.srt` — EN + ES/DE/JA/ID subtitles (verified distinct)
- `UPLOAD.md`, `thumbnail_*.png`, `SCHEDULE.json` — used by the QC gate

The assembly script (`scripts/assemble_multishot_1080p.py`) mixes narration + procedural SFX ambience, masters audio to **-15 LUFS**, draws 1080p HUD overlays, and encodes the final master at **2560×1440** (`lanczos` + CAS sharpen, libx264 CRF 17 preset slow, AAC 192k) so YouTube processes it with the **VP9** codec. `scripts/qc_check.py` gates the master (resolution, duration, loudness, true peak, subtitles, thumbnails, UPLOAD.md) before the artifact is published.
