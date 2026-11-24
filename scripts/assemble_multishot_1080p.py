#!/usr/bin/env python3
"""
Inside The Machine — Universal Multi-Shot Blender PBR 1080p Assembler & 1440p Master (v6)
Public-repo port of the v5 VM assembler. Differences from v5:
  - HUD pills / CTA / bridge cards are loaded from a JSON file in the episode
    audio pack (--hud-json) instead of the hardcoded EPISODE_HUD dict, so
    unreleased episode titles/teasers never appear in the public repo.
  - Overlays are drawn at 1920x1080 (clips are 1080p now).
  - Final master is upscaled to 2560x1440 (YouTube VP9) with
    scale=lanczos + CAS, libx264 CRF 17 preset slow, AAC 192k.
  - Audio mastering unchanged: 2-pass -15.0 LUFS / <= -1.0 dBTP.
  - Per-act intermediates: libx264 preset medium CRF 18 (no maxrate cap).
  - YouTube-ready 1440p H.264 MP4 (VP9 on YouTube) — zero manual re-encode!
"""
import argparse
import json
import os
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def load_hud(hud_json: Path, episode: str) -> dict:
    """Load the episode HUD (act pills + bridge card) from the audio pack.

    The JSON is episode-keyed like the old EPISODE_HUD dict, or just the
    episode entry itself: {"acts": {"1": [title, sub, spec], ...},
    "bridge": [label, title, sub]}. Act keys are converted to int.
    """
    raw = json.loads(Path(hud_json).read_text(encoding="utf-8"))
    entry = raw.get(episode) or raw
    acts = {int(k): tuple(v) for k, v in entry["acts"].items()}
    return {"acts": acts, "bridge": tuple(entry["bridge"])}


def load_font(size: int, bold: bool = True):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def build_act_overlay_png(out_png: Path, episode: str, act: int, hud: dict, show_cta: bool = False, show_bridge: bool = False):
    W, H = 1920, 1080
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    f_wm = load_font(18, bold=True)
    f_title = load_font(27, bold=True)
    f_sub = load_font(20, bold=False)
    f_spec = load_font(18, bold=True)
    f_badge = load_font(18, bold=True)

    d.text((W - 285, 33), "INSIDE THE MACHINE", font=f_wm, fill=(248, 250, 252, 85))

    ep_cfg = hud
    title, sub, spec = ep_cfg["acts"].get(act, (f"{episode.upper()} ACT {act}", "", ""))
    d.rounded_rectangle([39, 33, 885, 138], radius=12, fill=(8, 13, 24, 195), outline=(56, 189, 248, 120), width=2)
    d.rectangle([39, 33, 47, 138], fill=(56, 189, 248, 255))
    d.text((66, 45), title, font=f_title, fill=(248, 250, 252, 255))
    d.text((66, 81), sub, font=f_sub, fill=(255, 209, 102, 255))
    d.text((66, 110), spec, font=f_spec, fill=(148, 163, 184, 245))

    if show_cta:
        d.rounded_rectangle([W - 578, H - 117, W - 39, H - 33], radius=10, fill=(10, 16, 30, 228), outline=(255, 209, 102, 235), width=3)
        d.rounded_rectangle([W - 558, H - 96, W - 413, H - 54], radius=6, fill=(255, 209, 102, 255))
        d.text((W - 543, H - 86), "SUBSCRIBE", font=f_badge, fill=(10, 14, 24, 255))
        d.text((W - 396, H - 98), "Inside The Machine 3D", font=f_spec, fill=(248, 250, 252, 255))
        d.text((W - 396, H - 69), "New cutaways Tue & Fri", font=f_sub, fill=(148, 163, 184, 255))

    if show_bridge:
        b_lbl, b_title, b_sub = ep_cfg["bridge"]
        d.rounded_rectangle([W - 668, H - 153, W - 39, H - 33], radius=12, fill=(10, 16, 30, 235), outline=(56, 189, 248, 245), width=3)
        d.rectangle([W - 668, H - 153, W - 39, H - 147], fill=(56, 189, 248, 255))
        d.text((W - 645, H - 135), b_lbl, font=f_badge, fill=(56, 189, 248, 255))
        d.text((W - 645, H - 104), b_title, font=f_spec, fill=(248, 250, 252, 255))
        d.text((W - 645, H - 72), b_sub, font=f_sub, fill=(203, 213, 225, 255))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png, "PNG")


def synthesize_ambience(out_wav: Path, duration_sec: float, episode: str, sr: int = 48000):
    n = int(round(duration_sec * sr))
    t = np.linspace(0.0, duration_sec, n, endpoint=False, dtype=np.float32)
    rng = np.random.default_rng(42)

    if episode == "ep03":
        hum = 0.55 * np.sin(2.0 * np.pi * 60.0 * t) + 0.28 * np.sin(2.0 * np.pi * 120.0 * t) + 0.10 * np.sin(2.0 * np.pi * 180.0 * t)
        sig = hum * (0.78 + 0.22 * np.sin(2.0 * np.pi * 0.16 * t)) * 0.036
    elif episode == "ep04":
        white = rng.standard_normal(n).astype(np.float32)
        k = np.exp(-np.linspace(0, 5, 320, dtype=np.float32)); k /= k.sum()
        air = np.convolve(white, k, mode="same")
        drone = 0.4 * np.sin(2.0 * np.pi * 58.0 * t) + 0.2 * np.sin(2.0 * np.pi * 116.0 * t)
        sig = (air * 1.1 + drone * 0.25) * 0.038
    elif episode == "ep05":
        bp = (t * 8.0) % 1.0
        env = np.exp(-bp * 55.0)
        tick = np.sin(2.0 * np.pi * 2400.0 * t) * 0.45 + np.sin(2.0 * np.pi * 4100.0 * t) * 0.25
        bed = 0.22 * np.sin(2.0 * np.pi * 84.0 * t)
        sig = (tick * env * 0.55 + bed * 0.25) * 0.040
    elif episode == "ep06":
        white = rng.standard_normal(n).astype(np.float32)
        k = np.exp(-np.linspace(0, 5, 220, dtype=np.float32)); k /= k.sum()
        flow = np.convolve(white, k, mode="same")
        spool = 0.35 * np.sin(2.0 * np.pi * 145.0 * t) + 0.15 * np.sin(2.0 * np.pi * 290.0 * t)
        sig = (flow * 1.2 + spool * 0.16) * (0.82 + 0.18 * np.sin(2.0 * np.pi * 0.12 * t)) * 0.038
    elif episode == "ep07":
        hull = 0.55 * np.sin(2.0 * np.pi * 52.0 * t) + 0.25 * np.sin(2.0 * np.pi * 104.0 * t)
        ping = np.sin(2.0 * np.pi * 920.0 * t) * np.exp(-(t % 12.0) * 3.8) * 0.18
        sig = (hull * 0.5 + ping) * 0.036
    elif episode == "ep08":
        white = rng.standard_normal(n).astype(np.float32)
        k = np.exp(-np.linspace(0, 5, 260, dtype=np.float32)); k /= k.sum()
        glide = np.convolve(white, k, mode="same")
        mag = 0.40 * np.sin(2.0 * np.pi * 110.0 * t) + 0.22 * np.sin(2.0 * np.pi * 165.0 * t)
        sig = (glide * 0.9 + mag * 0.28) * (0.80 + 0.20 * np.sin(2.0 * np.pi * 0.15 * t)) * 0.036
    elif episode == "ep09":
        ch = 0.45 * np.sin(2.0 * np.pi * 68.0 * t) + 0.20 * np.sin(2.0 * np.pi * 136.0 * t)
        penv = np.exp(-((t * 15.0) % 1.0) * 18.0) * (0.5 + 0.5 * np.sin(2.0 * np.pi * 0.10 * t))
        sol = np.sin(2.0 * np.pi * 240.0 * t) * penv * 0.14
        sig = (ch * 0.45 + sol) * 0.038
    else:
        white = rng.standard_normal(n).astype(np.float32)
        k = np.exp(-np.linspace(0, 6, 480, dtype=np.float32)); k /= k.sum()
        water = np.convolve(white, k, mode="same")
        stator = 0.42 * np.sin(2.0 * np.pi * 60.0 * t) + 0.20 * np.sin(2.0 * np.pi * 120.0 * t)
        sig = (water * 1.45 + stator * 0.22) * (0.84 + 0.16 * np.sin(2.0 * np.pi * 0.12 * t)) * 0.038

    fade = min(int(2.0 * sr), n // 4)
    sig[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    sig[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    pcm = np.clip(sig * 32767.0, -32767, 32767).astype(np.int16)
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def run(cmd: list[str], capture: bool = False) -> str:
    print("+", " ".join(cmd), flush=True)
    if capture:
        p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return p.stdout
    subprocess.run(cmd, check=True)
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--clips-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--episode", required=True)
    p.add_argument("--shot-hold-sec", type=float, default=12.0)
    p.add_argument("--hud-json", required=True, help="HUD JSON from the episode audio pack")
    p.add_argument("--crf", type=int, default=17, help="Final 1440p CRF (fallback: 18)")
    p.add_argument("--preset", default="slow", help="Final 1440p preset (fallback: medium)")
    p.add_argument("--no-cas", action="store_true", help="Drop the CAS sharpening filter (runner ffmpeg without cas)")
    args = p.parse_args()

    hud = load_hud(args.hud_json, args.episode)

    audio_dir = Path(args.audio_dir)
    clips_dir = Path(args.clips_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    work.mkdir(exist_ok=True)

    meta = json.loads((audio_dir / "acts_meta.json").read_text(encoding="utf-8"))
    gap_sec = float(meta["gap_sec"])
    act_boundaries = meta["act_boundaries"]
    total_dur = float(meta["total_duration"])

    # Gap silence MUST be generated at the segments' native sample rate.
    # The concat demuxer does not resample: a 48 kHz gap between 24 kHz
    # Leda segments plays back as 2.0 s, so every act boundary drifted
    # +1.00 s (EP06: acts started +1/+2/+3/+4 s late and the final clause
    # was cut off by -shortest). Read the rate from the first segment's header.
    with wave.open(str(audio_dir / meta["segment_files"][0]), "rb") as wf:
        seg_sr = wf.getframerate()
    silence_wav = work / "gap.wav"
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r={seg_sr}:cl=mono", "-t", str(gap_sec), "-c:a", "pcm_s16le", str(silence_wav)])

    aconcat = work / "audio_concat.txt"
    alines = []
    for i, fname in enumerate(meta["segment_files"]):
        alines.append(f"file '{(audio_dir / fname).resolve()}'")
        if i < len(meta["segment_files"]) - 1:
            alines.append(f"file '{silence_wav.resolve()}'")
    aconcat.write_text("\n".join(alines) + "\n", encoding="utf-8")

    narration_wav = work / "narration_full.wav"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(aconcat), "-c:a", "pcm_s16le", str(narration_wav)])

    sfx_wav = work / f"{args.episode}_sfx.wav"
    synthesize_ambience(sfx_wav, total_dur, episode=args.episode)

    raw_mix = work / "raw_mix.wav"
    run([
        "ffmpeg", "-y", "-i", str(narration_wav), "-i", str(sfx_wav),
        "-filter_complex", "[0:a]volume=1.0[v];[1:a]volume=0.22[s];[v][s]amix=inputs=2:duration=first:dropout_transition=2,pan=stereo|c0=c0|c1=c0[out]",
        "-map", "[out]", "-ar", "48000", "-c:a", "pcm_s16le", str(raw_mix),
    ])

    pass1 = run(["ffmpeg", "-y", "-i", str(raw_mix), "-af", "loudnorm=I=-15:TP=-1.2:LRA=9:print_format=json", "-f", "null", "-"], capture=True)
    m = re.search(r"\{\s*\"input_i\".*?\}", pass1, re.DOTALL)
    if not m:
        raise RuntimeError("Failed to parse loudnorm pass-1 JSON")
    st = json.loads(m.group(0))

    master_wav = work / "master_audio.wav"
    ln2 = (
        f"loudnorm=I=-15:TP=-1.2:LRA=9:linear=true:"
        f"measured_I={st['input_i']}:measured_TP={st['input_tp']}:"
        f"measured_LRA={st['input_lra']}:measured_thresh={st['input_thresh']}:"
        f"offset={st['target_offset']}"
    )
    run(["ffmpeg", "-y", "-i", str(raw_mix), "-af", ln2, "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(master_wav)])

    final_act_lines = []
    for act in range(1, 6):
        slot_dur = round(act_boundaries[act] - act_boundaries[act - 1], 3)
        shot_files = [clips_dir / f"act_{act}_shot_{s}.mp4" for s in (1, 2, 3)]
        for sf in shot_files:
            if not sf.exists():
                raise FileNotFoundError(f"Missing rendered Blender shot: {sf}")

        seq_lines = []
        acc, idx = 0.0, 0
        repeats = max(1, int(round(args.shot_hold_sec / 6.0)))
        while acc < slot_dur + 6.0:
            sf = shot_files[idx % 3]
            for _ in range(repeats):
                seq_lines.append(f"file '{sf.resolve()}'")
                acc += 6.0
            idx += 1

        act_seq_txt = work / f"act_{act}_seq.txt"
        act_seq_txt.write_text("\n".join(seq_lines) + "\n", encoding="utf-8")

        act_raw = work / f"act_{act}_raw.mp4"
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(act_seq_txt), "-t", str(slot_dur), "-c", "copy", str(act_raw)])

        ov_base = work / f"ov_act_{act}_base.png"
        build_act_overlay_png(ov_base, args.episode, act, hud, show_cta=False, show_bridge=False)

        act_final = work / f"act_{act}_final.mp4"
        if act == 2:
            ov_cta = work / "ov_act_2_cta.png"
            build_act_overlay_png(ov_cta, args.episode, 2, hud, show_cta=True, show_bridge=False)
            t0, t1 = round(slot_dur * 0.42, 2), round(slot_dur * 0.78, 2)
            fc = f"[0:v][1:v]overlay=0:0:enable='not(between(t,{t0},{t1}))'[v1];[v1][2:v]overlay=0:0:enable='between(t,{t0},{t1})'[vout]"
            run([
                "ffmpeg", "-y", "-i", str(act_raw), "-i", str(ov_base), "-i", str(ov_cta),
                "-filter_complex", fc, "-map", "[vout]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-r", "24", "-pix_fmt", "yuv420p", str(act_final),
            ])
        elif act == 5:
            ov_br = work / "ov_act_5_bridge.png"
            build_act_overlay_png(ov_br, args.episode, 5, hud, show_cta=False, show_bridge=True)
            t0 = round(max(1.0, slot_dur - 22.0), 2)
            fc = f"[0:v][1:v]overlay=0:0:enable='lt(t,{t0})'[v1];[v1][2:v]overlay=0:0:enable='gte(t,{t0})'[vout]"
            run([
                "ffmpeg", "-y", "-i", str(act_raw), "-i", str(ov_base), "-i", str(ov_br),
                "-filter_complex", fc, "-map", "[vout]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-r", "24", "-pix_fmt", "yuv420p", str(act_final),
            ])
        else:
            run([
                "ffmpeg", "-y", "-i", str(act_raw), "-i", str(ov_base),
                "-filter_complex", "[0:v][1:v]overlay=0:0[vout]", "-map", "[vout]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-r", "24", "-pix_fmt", "yuv420p", str(act_final),
            ])
        final_act_lines.append(f"file '{act_final.resolve()}'")

    vconcat = work / "video_concat.txt"
    vconcat.write_text("\n".join(final_act_lines) + "\n", encoding="utf-8")

    vf = "scale=2560:1440:flags=lanczos,cas=strength=0.25" if not args.no_cas else "scale=2560:1440:flags=lanczos"
    final_mp4 = out_dir / f"{args.episode}_final_1440p.mp4"
    run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(vconcat),
        "-i", str(master_wav), "-map", "0:v:0", "-map", "1:a:0",
        "-vf", vf, "-c:v", "libx264", "-crf", str(args.crf), "-preset", args.preset,
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart", "-shortest", str(final_mp4),
    ])
    print("Final Blender PBR 1440p MP4 assembled:", final_mp4, flush=True)


if __name__ == "__main__":
    main()
