# Procedural Blender 4.2 PBR Cutaway Renderer (Public Render-Only)

Stateless, public-safe 3D procedural mechanical cutaway renderer using **Blender 4.2.3 LTS (Cycles CPU + OpenImageDenoise + AgX High Contrast + Fog Glow)**.

- **Contains NO secrets, NO API keys, NO narration audio, NO scripts, and NO channel metadata.**
- Outputs silent 6-second seamless 720p (24fps) H.264 `.mp4` loops (`act_1_shot_1.mp4` .. `act_5_shot_3.mp4`) as GitHub Actions artifacts.

## Workflows
1. **`Render Single PBR Shot (720p)`** — Renders 1 specific `(episode, act, shot)` clip (e.g., `ep03`, Act `1`, Shot `2`).
2. **`Render Full 15-Shot Episode PBR (720p)`** — Renders all 15 shots in parallel across 15 runners and bundles them into a single `<episode>-all-15-pbr-clips` artifact for easy one-click download to VM.
