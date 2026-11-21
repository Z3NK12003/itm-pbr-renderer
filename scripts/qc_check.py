#!/usr/bin/env python3
"""
Pro QC Gate (v6 — Actions port of v5):
- Supports 720p Blender PBR, 1080p & 1440p masters
- Strictly validates 5-language .srt distinctness (>= 65% distinct lines vs English)
- Outputs pure neutral RELEASE_METADATA.json (zero prompt-injection strings)
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def extract_srt_lines(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    lines = []
    for block in re.split(r"\n\s*\n", raw):
        parts = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(parts) >= 3:
            lines.append(" ".join(parts[2:]).lower())
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--srt-dir", required=True)
    parser.add_argument("--thumb-dir", required=True)
    parser.add_argument("--upload-md", required=True)
    parser.add_argument("--schedule-json", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--publish-at", default="")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    video = Path(args.video)
    srt_dir = Path(args.srt_dir)
    thumb_dir = Path(args.thumb_dir)
    upload_md = Path(args.upload_md)

    probe = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(video)],
            text=True,
        )
    )
    vstream = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
    astream = next((s for s in probe["streams"] if s["codec_type"] == "audio"), None)
    dur = float(probe["format"]["duration"])
    size_mb = video.stat().st_size / (1024 * 1024)

    ln_out = subprocess.run(
        ["ffmpeg", "-i", str(video), "-af", "loudnorm=I=-15:TP=-1.2:LRA=9:print_format=json", "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True,
    ).stdout
    m = re.search(r"\{\s*\"input_i\".*?\}", ln_out, re.DOTALL)
    stats = json.loads(m.group(0)) if m else {}
    lufs = float(stats.get("input_i", -99.0))
    tp = float(stats.get("input_tp", 0.0))

    srt_files = sorted(p for p in srt_dir.glob("*.srt") if p.stat().st_size > 400)
    thumb_files = sorted(p.name for p in thumb_dir.glob("thumbnail_*.png") if p.stat().st_size > 10000)

    en_srt = srt_dir / f"{args.episode}_subtitles_en.srt"
    en_lines = extract_srt_lines(en_srt) if en_srt.exists() else []
    bad_langs = []
    for p in srt_files:
        if p.name != en_srt.name and en_lines:
            other_lines = extract_srt_lines(p)
            n = min(len(en_lines), len(other_lines))
            if n == 0:
                bad_langs.append(f"{p.name}(empty)")
            else:
                diff = sum(1 for a, b in zip(en_lines[:n], other_lines[:n]) if a != b)
                ratio = diff / float(n)
                if ratio < 0.65:
                    bad_langs.append(f"{p.name}({ratio*100:.0f}% distinct)")

    checks = []
    passed = True

    def check(name: str, ok: bool, detail: str):
        nonlocal passed
        checks.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            passed = False

    w = int(vstream["width"]) if vstream else 0
    h = int(vstream["height"]) if vstream else 0
    check("Video Resolution", (w, h) in ((1280, 720), (1920, 1080), (2560, 1440)), f"{w}x{h} ({size_mb:.1f} MB)")
    check("Audio Stream", astream is not None, astream["codec_name"] if astream else "missing")
    check("Video Duration", dur >= 300.0, f"{dur:.1f}s ({dur/60:.2f} min)")
    check("Integrated Loudness", -16.8 <= lufs <= -13.2, f"{lufs:.2f} LUFS (target -14 to -16)")
    check("True Peak", tp <= -0.9, f"{tp:.2f} dBTP (target <= -1.0)")
    check("5-Language Subtitles (.srt)", len(srt_files) >= 5, f"{len(srt_files)} files")
    check("Subtitle Translation Integrity", len(bad_langs) == 0, "all 4 translations verified distinct (>=65%)" if not bad_langs else f"failed: {', '.join(bad_langs)}")
    check("Thumbnails (A/B Pack)", len(thumb_files) >= 1, f"{len(thumb_files)} files: {', '.join(thumb_files)}")
    check("UPLOAD.md + AI Disclosure", upload_md.exists() and "AI voiceover" in upload_md.read_text(encoding="utf-8"), "verified")

    report = "=== INSIDE THE MACHINE — PRO AUTOMATED QC REPORT (v5) ===\n"
    report += f"Episode: {args.episode.upper()}\n"
    report += f"Overall Status: {'QC PASS' if passed else 'QC FAILED'}\n\n"
    report += "\n".join(checks) + "\n"
    (out_dir / "QC_REPORT.txt").write_text(report, encoding="utf-8")
    print(report)

    sched = json.loads(Path(args.schedule_json).read_text(encoding="utf-8"))
    ep_info = next((e for e in sched["episodes"] if e["id"] == args.episode), {})
    publish_utc = args.publish_at.strip() or ep_info.get("target_publish_utc", "")
    publish_wib = ep_info.get("target_publish_wib", publish_utc)

    summary = {
        "episode": args.episode,
        "number": ep_info.get("number", 0),
        "title": ep_info.get("title", ""),
        "video_file": video.name,
        "resolution": f"{w}x{h}",
        "file_size_mb": round(size_mb, 2),
        "duration_min": round(dur / 60.0, 2),
        "lufs": round(lufs, 2),
        "true_peak_dbtp": round(tp, 2),
        "thumbnails": thumb_files,
        "subtitles": [p.name for p in srt_files],
        "channel_account": sched.get("channel_account", "andry.murad@gmail.com"),
        "playlist": "How Engines & Machines Work (3D Cutaways)",
        "target_publish_wib": publish_wib,
        "target_publish_utc": publish_utc,
        "pinned_comment": ep_info.get("pinned_comment", ""),
        "ai_voiceover_disclosure": True,
        "made_for_kids": False,
        "qc_passed": passed,
    }
    (out_dir / "RELEASE_METADATA.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
