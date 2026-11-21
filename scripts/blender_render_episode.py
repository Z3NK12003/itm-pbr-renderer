#!/usr/bin/env python3
"""
Inside The Machine — Unified Blender 4.2 PBR Multi-Shot 720p Episode Renderer (v5)
Supports EP03 through EP10, each with bespoke 3D cutaway mechanical geometry and
15 unique parametric cinema camera rigs (5 Acts x 3 Shots, 6s seamless loops @ 24fps).

Usage:
  blender -b --python-exit-code 1 --python scripts/blender_render_episode.py -- \
    --episode ep04 --act 1 --shot 1 --frames 144 --fps 24 --samples 14 --out build/frames
"""
import argparse
import math
import os
import sys
from pathlib import Path

# Ensure scripts/ directory is importable inside Blender's bundled Python
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy
import blender_pbr_core as core


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--episode", required=True, choices=["ep01", "ep02", "ep03", "ep04", "ep05", "ep06", "ep07", "ep08", "ep09", "ep10"])
    p.add_argument("--act", type=int, required=True, choices=[1, 2, 3, 4, 5])
    p.add_argument("--shot", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--frames", type=int, default=144)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--samples", type=int, default=14)
    p.add_argument("--device", default="AUTO", choices=["AUTO", "OPTIX", "CUDA", "CPU"])
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def configure_compute_device(scene, device_pref="AUTO"):
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if device_pref == "CPU":
        scene.cycles.device = "CPU"
        print("[DEVICE] Forced CPU", flush=True)
        return "CPU"
    order = ("OPTIX", "CUDA") if device_pref == "AUTO" else (device_pref,)
    for backend in order:
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == backend]
            if gpus:
                for d in prefs.devices:
                    d.use = (d.type == backend)
                scene.cycles.device = "GPU"
                print(f"[DEVICE] Active GPU backend={backend} devices={[d.name for d in gpus]}", flush=True)
                return backend
        except Exception as e:
            print(f"[DEVICE WARN] {backend}: {e}", flush=True)
    scene.cycles.device = "CPU"
    print("[DEVICE] Fallback to CPU", flush=True)
    return "CPU"


# ==============================================================================
# EP01 — WANKEL ROTARY ENGINE (EPITROCHOID HOUSING, REULEAUX ROTOR, 3:1 ECCENTRIC)
# ==============================================================================
def build_ep01(scene, act, shot, total_frames):
    _, rim, acc = core.setup_studio_lights(
        floor_z=-3.4,
        key_color=(0.95, 0.97, 1.0),
        rim_color=(0.16, 0.78, 1.0),
        accent_color=(1.0, 0.44, 0.12),
    )
    mat_housing = core.make_pbr_mat("AlumHousing", (0.58, 0.64, 0.72, 1.0), metallic=0.92, roughness=0.25, noise_scale=20.0, rough_var=0.09, bump_strength=0.05)
    mat_dark_iron = core.make_pbr_mat("CastIronRotor", (0.14, 0.16, 0.20, 1.0), metallic=0.88, roughness=0.32, noise_scale=24.0, rough_var=0.11, bump_strength=0.07)
    mat_chrome = core.make_pbr_mat("ChromeLiner", (0.86, 0.89, 0.94, 1.0), metallic=0.98, roughness=0.12, noise_scale=26.0, rough_var=0.05, bump_strength=0.02)
    mat_copper = core.make_pbr_mat("ApexSealBronze", (0.88, 0.48, 0.18, 1.0), metallic=0.95, roughness=0.20, noise_scale=24.0, rough_var=0.08, bump_strength=0.04)
    mat_steel = core.make_pbr_mat("EccentricSteel", (0.72, 0.76, 0.82, 1.0), metallic=0.95, roughness=0.18, noise_scale=22.0, rough_var=0.07, bump_strength=0.04)
    mat_intake = core.make_glow_mat("IntakeCyan", (0.14, 0.82, 1.0, 1.0), 8.5, light_tree=False)
    mat_spark = core.make_glow_mat("SparkPlugPlasma", (0.45, 0.85, 1.0, 1.0), 14.0, light_tree=False)
    mat_combust = core.make_glow_mat("CombustionFlame", (1.0, 0.36, 0.06, 1.0), 14.0, light_tree=False)
    mat_exhaust = core.make_glow_mat("ExhaustAmber", (1.0, 0.56, 0.14, 1.0), 8.0, light_tree=False)

    # Rear Housing Plate + Cutaway Epitrochoid Peanut-Shaped Rotor Chamber
    core.add_box((0.0, 1.15, 0.0), (7.6, 0.38, 6.4), mat_housing, bevel=0.08)
    R_maj, e_ecc = 2.15, 0.36
    seg_n = 72
    for i in range(seg_n):
        th = (i / float(seg_n)) * 2.0 * math.pi
        # True epitrochoid profile (R + e*cos(2*th))
        x = R_maj * math.cos(th) + e_ecc * math.cos(3.0 * th) * 0.12 + 0.52 * math.cos(th) * (1.0 + 0.24 * math.cos(2.0 * th))
        z = R_maj * math.sin(th) * (1.0 - 0.16 * math.cos(2.0 * th))
        ang_tan = math.degrees(th)
        core.add_box((x * 1.12, 0.0, z * 1.12), (0.36, 1.65, 0.28), mat_housing, rot_deg=(0, ang_tan, 0), bevel=0.015)
        core.add_box((x * 0.98, 0.0, z * 0.98), (0.14, 1.52, 0.22), mat_chrome, rot_deg=(0, ang_tan, 0), bevel=0.008)

    # Cooling Jacket Fins around Outer Housing
    for i in range(16):
        th = (i / 16.0) * 2.0 * math.pi
        core.add_box((math.cos(th) * 3.35, 0.15, math.sin(th) * 2.75), (0.48, 1.4, 0.12), mat_housing, rot_deg=(0, math.degrees(th), 0), bevel=0.01)

    # Intake Port (Top-Left) & Exhaust Port (Bottom-Left) + Twin Spark Plugs (Right)
    core.add_cyl((-3.4, 0.0, 1.15), 0.42, 1.9, mat_steel, rot_deg=(0, 72, 0))
    core.add_cyl((-3.4, 0.0, -1.15), 0.42, 1.9, mat_dark_iron, rot_deg=(0, 108, 0))
    spark_tips = []
    for sz in (0.52, -0.52):
        core.add_cyl((3.15, 0.0, sz), 0.16, 1.35, mat_steel, rot_deg=(0, 90, 0))
        core.add_cyl((3.55, 0.0, sz), 0.22, 0.45, mat_chrome, rot_deg=(0, 90, 0))
        sp_mat = core.make_glow_mat(f"Spark_{sz}", (0.55, 0.90, 1.0, 1.0), 1.0, light_tree=False)
        core.mark_fast_glow(core.add_sphere((2.38, 0.0, sz), 0.18, sp_mat, fast_glow=True))
        spark_tips.append(sp_mat)

    # Stationary Phasing Gear on Rear Housing
    core.add_gear_wheel("PhasingGear", (0.0, 0.72, 0.0), 0.62, 0.26, 18, mat_steel, rot_deg=(90, 0, 0))

    # Eccentric Output Shaft (Spins 3x faster than rotor!)
    ecc_shaft = bpy.data.objects.new("EccentricShaft", None)
    bpy.context.collection.objects.link(ecc_shaft)
    core.add_cyl((0.0, -0.1, 0.0), 0.34, 4.4, mat_steel, rot_deg=(90, 0, 0)).parent = ecc_shaft
    core.add_cyl((e_ecc, 0.0, 0.0), 0.68, 1.30, mat_chrome, rot_deg=(90, 0, 0)).parent = ecc_shaft
    core.add_box((0.0, -1.75, 0.0), (1.35, 0.22, 0.48), mat_dark_iron, bevel=0.02).parent = ecc_shaft  # Counterweight

    # Triangular Reuleaux Rotor + 3 Spring-Loaded Bronze Apex Seals + Combustion Recesses
    rotor = bpy.data.objects.new("WankelRotor", None)
    bpy.context.collection.objects.link(rotor)
    core.add_cyl((0.0, 0.0, 0.0), 1.18, 1.28, mat_dark_iron, rot_deg=(90, 0, 0), verts=36).parent = rotor
    core.add_gear_wheel("RotorInternalGear", (0.0, 0.50, 0.0), 0.92, 0.22, 27, mat_copper, rot_deg=(90, 0, 0)).parent = rotor

    for k in range(3):
        ang = k * (2.0 * math.pi / 3.0)
        # 3 Curved Triangular Flanks + Apex Tips + Bronze Apex Seals
        ax, az = math.cos(ang) * 1.72, math.sin(ang) * 1.72
        core.add_box((math.cos(ang) * 0.98, 0.0, math.sin(ang) * 0.98), (1.45, 1.26, 0.85), mat_dark_iron, rot_deg=(0, math.degrees(-ang), 0), bevel=0.05).parent = rotor
        core.add_box((ax, 0.0, az), (0.24, 1.38, 0.14), mat_copper, rot_deg=(0, math.degrees(-ang), 0), bevel=0.01).parent = rotor
        # Side Seal Arcs
        mid_a = ang + math.pi / 3.0
        core.add_box((math.cos(mid_a) * 1.05, -0.66, math.sin(mid_a) * 1.05), (1.35, 0.06, 0.10), mat_chrome, rot_deg=(0, math.degrees(-mid_a + math.pi / 2.0), 0), bevel=0.005).parent = rotor

    # Pulsing Combustion Pocket on Right Chamber + Intake/Exhaust Flow Beads
    flame_pocket = core.mark_fast_glow(core.add_sphere((2.05, 0.0, 0.0), 0.55, mat_combust, fast_glow=True))
    intake_beads = [(core.add_sphere((-3.2, 0.0, 1.15), 0.12, mat_intake, fast_glow=True), i / 10.0) for i in range(10)]
    exhaust_beads = [(core.add_sphere((-2.4, 0.0, -1.15), 0.12, mat_exhaust, fast_glow=True), i / 10.0) for i in range(10)]

    core.animate_light_pulse(acc, 320.0, 170.0, 6, total_frames)

    for f in core.periodic_frames(total_frames, step=2):
        u = (f - 1) / float(total_frames)
        tp = 2.0 * math.pi
        # Exact 3:1 Wankel Kinematics: Eccentric shaft spins 6 turns, Rotor spins 2 turns (100% seamless loop!)
        shaft_ang = u * tp * 6.0
        rotor_ang = u * tp * 2.0
        ecc_shaft.rotation_euler.y = shaft_ang
        ecc_shaft.keyframe_insert("rotation_euler", index=1, frame=f)

        rotor.location = (math.cos(-shaft_ang) * e_ecc, 0.0, math.sin(-shaft_ang) * e_ecc)
        rotor.rotation_euler.y = rotor_ang
        rotor.keyframe_insert("location", frame=f)
        rotor.keyframe_insert("rotation_euler", index=1, frame=f)

        # 6 Ignition Pulses per loop (3 faces x 2 full rotor turns)
        ign_phase = (u * 6.0) % 1.0
        spark_on = 16.0 if ign_phase < 0.22 else 0.5
        for sp_mat in spark_tips:
            bsdf = sp_mat.node_tree.nodes.get("Principled BSDF")
            bsdf.inputs["Emission Strength"].default_value = spark_on
            bsdf.inputs["Emission Strength"].keyframe_insert("default_value", frame=f)

        f_sc = 0.25 + 1.05 * math.sin(min(1.0, ign_phase * 2.2) * math.pi) if ign_phase < 0.55 else 0.15
        flame_pocket.scale = (f_sc * 0.7, f_sc * 1.1, f_sc * 1.35)
        flame_pocket.keyframe_insert("scale", frame=f)

        for bd, off in intake_beads:
            t = (off + u * 3.0) % 1.0
            bd.location = (-3.9 + t * 2.2, 0.0, 1.35 - t * 0.35)
            bd.keyframe_insert("location", frame=f)
        for bd, off in exhaust_beads:
            t = (off + u * 3.0) % 1.0
            bd.location = (-1.7 - t * 2.4, 0.0, -0.95 - t * 0.45)
            bd.keyframe_insert("location", frame=f)

    rigs = {
        (1, 1): ((-1.2, -9.8, 1.4), (1.3, 0.5, 0.3), (0.0, 0.0, 0.0), 42.0, 3.0),
        (1, 2): ((5.4, -7.8, 0.8), (1.1, 0.4, 0.25), (0.2, 0.0, 0.0), 48.0, 2.6),
        (1, 3): ((-4.8, -8.2, 3.6), (1.3, 0.5, 0.25), (0.0, 0.0, 0.0), 40.0, 3.2),
        (2, 1): ((0.0, -6.4, 0.4), (0.9, 0.35, 0.2), (0.0, 0.0, 0.0), 54.0, 2.4),
        (2, 2): ((-3.2, -5.6, 1.2), (0.8, 0.3, 0.2), (-0.4, 0.0, 0.2), 58.0, 2.2),
        (2, 3): ((2.8, -5.4, -0.8), (0.8, 0.3, 0.2), (0.4, 0.0, 0.0), 56.0, 2.4),
        (3, 1): ((3.4, -5.0, 0.5), (0.8, 0.3, 0.2), (1.8, 0.0, 0.0), 58.0, 2.2),
        (3, 2): ((1.6, -5.8, -1.2), (0.8, 0.3, 0.18), (1.4, 0.0, 0.0), 54.0, 2.4),
        (3, 3): ((4.6, -4.4, 1.8), (0.8, 0.35, 0.2), (1.8, 0.0, 0.1), 56.0, 2.2),
        (4, 1): ((-1.4, -5.2, 0.6), (0.8, 0.3, 0.2), (0.0, 0.0, 0.0), 60.0, 2.2),
        (4, 2): ((2.2, -4.8, 1.4), (0.7, 0.3, 0.18), (0.8, 0.0, 0.6), 62.0, 2.0),
        (4, 3): ((-3.6, -5.8, -0.6), (0.9, 0.35, 0.2), (-1.4, 0.0, -0.4), 52.0, 2.5),
        (5, 1): ((-4.5, -8.8, 2.2), (1.3, 0.5, 0.3), (0.0, 0.0, 0.0), 44.0, 2.8),
        (5, 2): ((4.2, -8.4, -1.0), (1.2, 0.45, 0.25), (0.2, 0.0, 0.0), 46.0, 2.8),
        (5, 3): ((0.8, -10.4, 2.8), (1.5, 0.55, 0.3), (0.0, 0.0, 0.0), 38.0, 3.4),
    }
    core.setup_cinema_camera(scene, *rigs[(act, shot)], total_frames)


# ==============================================================================
# EP02 — SATURN V F-1 ROCKET ENGINE (55,000 HP TURBOPUMP, INJECTOR, BELL NOZZLE)
# ==============================================================================
def build_ep02(scene, act, shot, total_frames):
    _, rim, acc = core.setup_studio_lights(
        floor_z=-5.4,
        key_color=(0.95, 0.96, 1.0),
        rim_color=(0.16, 0.78, 1.0),
        accent_color=(1.0, 0.42, 0.08),
    )
    mat_inconel = core.make_pbr_mat("InconelX750", (0.28, 0.27, 0.29, 1.0), metallic=0.92, roughness=0.30, noise_scale=22.0, rough_var=0.11, bump_strength=0.07)
    mat_steel = core.make_pbr_mat("CryoStainless", (0.72, 0.76, 0.82, 1.0), metallic=0.95, roughness=0.20, noise_scale=20.0, rough_var=0.08, bump_strength=0.04)
    mat_copper = core.make_pbr_mat("InjectorCopper", (0.86, 0.42, 0.14, 1.0), metallic=0.96, roughness=0.20, noise_scale=24.0, rough_var=0.08, bump_strength=0.04)
    mat_gold = core.make_pbr_mat("ThermalFoilGold", (0.90, 0.66, 0.16, 1.0), metallic=0.92, roughness=0.26, noise_scale=28.0, rough_var=0.14, bump_strength=0.14)
    mat_lox = core.make_glow_mat("LOX_CryoBlue", (0.14, 0.80, 1.0, 1.0), 9.0, light_tree=False)
    mat_rp1 = core.make_glow_mat("RP1_Kerosene", (1.0, 0.64, 0.12, 1.0), 8.5, light_tree=False)
    mat_flame = core.make_glow_mat("F1_ExhaustPlasma", (1.0, 0.46, 0.08, 1.0), 15.0, light_tree=False)

    # Top Gimbal Mount & Cryogenic LOX Dome
    core.add_cyl((0.0, 0.0, 3.85), 0.72, 0.85, mat_steel)
    core.add_sphere((0.0, 0.0, 2.75), 1.48, mat_steel, segs=28, rings=16)

    # Copper Baffle Injector Plate (Anti-Combustion Instability Baffles)
    core.add_cyl((0.0, 0.0, 2.05), 1.38, 0.28, mat_copper, verts=42)
    for k in range(6):
        a = (k / 6.0) * math.pi
        core.add_box((0.0, 0.0, 1.88), (2.55, 0.06, 0.24), mat_copper, rot_deg=(0, 0, math.degrees(a)), bevel=0.005)
    core.add_torus((0.0, 0.0, 1.88), 0.72, 0.05, mat_copper, maj_seg=32, min_seg=10)

    # Main Combustion Chamber + Regenerative-Cooled Ribbed Bell Nozzle (Half-Cutaway)
    core.add_cyl((0.0, 0.45, 0.95), 1.42, 1.95, mat_inconel, verts=42)
    core.add_cone((0.0, 0.0, -1.65), 2.55, 0.96, 3.35, mat_inconel, verts=48)
    for z_r, rad in [(-0.25, 1.12), (-0.85, 1.38), (-1.45, 1.72), (-2.05, 2.06), (-2.65, 2.38), (-3.15, 2.60)]:
        core.add_torus((0.0, 0.0, z_r), rad, 0.08, mat_steel, maj_seg=48, min_seg=10)

    # Iconic F-1 Turbine Exhaust Manifold Wrapping Around Nozzle Waist!
    core.add_torus((0.0, 0.0, -1.05), 1.85, 0.36, mat_gold, maj_seg=42, min_seg=16)

    # Side-Mounted 55,000 HP Turbopump Assembly + Gas Generator (Left Side)
    core.add_cyl((-2.55, 0.0, 1.85), 0.75, 1.65, mat_steel, rot_deg=(0, 90, 0))
    core.add_cyl((-2.55, 0.0, 0.65), 0.42, 1.10, mat_inconel)  # Gas Generator
    core.add_cyl((-1.85, 0.0, -0.25), 0.30, 1.85, mat_gold, rot_deg=(0, 38, 0))  # Exhaust duct to manifold
    impeller = core.add_gear_wheel("TurbopumpImpeller", (-2.55, -0.45, 1.85), 0.66, 0.32, 16, mat_copper, rot_deg=(90, 0, 0))

    # Glowing 3,300 C Combustion Core + Pulsing Mach-Diamond Exhaust Plume
    core_flame = core.mark_fast_glow(core.add_cyl((0.0, -0.15, 0.85), 1.05, 1.75, mat_flame, verts=28))
    plume = core.mark_fast_glow(core.add_cone((0.0, 0.0, -4.25), 0.65, 2.15, 3.1, mat_flame, verts=32))
    diamonds = [core.mark_fast_glow(core.add_sphere((0.0, 0.0, -3.55 - i * 0.75), 0.48 - i * 0.08, mat_lox, fast_glow=True)) for i in range(3)]

    # Cryogenic LOX (Blue) & RP-1 Kerosene (Amber) Propellant Feed Beads
    lox_beads = [(core.add_sphere((0.35, -0.4, 3.2), 0.11, mat_lox, fast_glow=True), i / 12.0) for i in range(12)]
    rp1_beads = [(core.add_sphere((-2.5, -0.4, 1.8), 0.11, mat_rp1, fast_glow=True), i / 12.0) for i in range(12)]

    core.animate_light_pulse(acc, 380.0, 180.0, 12, total_frames)

    for f in core.periodic_frames(total_frames, step=2):
        u = (f - 1) / float(total_frames)
        tp = 2.0 * math.pi
        # High-speed 5,500 RPM visual spin (8 full cycles per 6s loop)
        impeller.rotation_euler.y = u * tp * 8.0
        impeller.keyframe_insert("rotation_euler", index=1, frame=f)

        pf = 0.90 + 0.16 * math.sin(u * tp * 14.0)
        plume.scale = (1.0, 1.0, pf)
        plume.keyframe_insert("scale", frame=f)
        cf = 0.95 + 0.08 * math.sin(u * tp * 10.0)
        core_flame.scale = (cf, cf, 1.0)
        core_flame.keyframe_insert("scale", frame=f)

        for idx, dm in enumerate(diamonds):
            ds = 0.85 + 0.20 * math.sin(u * tp * 12.0 + idx)
            dm.scale = (ds, ds, ds * 1.3)
            dm.keyframe_insert("scale", frame=f)

        for bd, off in lox_beads:
            t = (off + u * 3.0) % 1.0
            bd.location = (0.35 * math.sin(t * tp), -0.55, 3.3 - t * 2.5)
            bd.keyframe_insert("location", frame=f)
        for bd, off in rp1_beads:
            t = (off + u * 3.0) % 1.0
            bd.location = (-2.6 + t * 2.4, -0.55, 1.85 - t * 0.95)
            bd.keyframe_insert("location", frame=f)

    rigs = {
        (1, 1): ((-4.5, -13.5, 0.2), (1.6, 0.5, 0.35), (-0.3, 0.0, -0.2), 40.0, 3.2),
        (1, 2): ((6.4, -10.8, -1.6), (1.4, 0.5, 0.3), (0.0, 0.0, -0.8), 44.0, 2.8),
        (1, 3): ((-2.8, -12.2, 3.8), (1.5, 0.6, 0.3), (-0.4, 0.0, 0.6), 38.0, 3.4),
        (2, 1): ((-4.8, -6.2, 2.0), (0.9, 0.35, 0.2), (-2.2, 0.0, 1.6), 52.0, 2.4),
        (2, 2): ((-2.4, -5.2, 1.6), (0.8, 0.3, 0.18), (-2.4, -0.2, 1.8), 58.0, 2.2),
        (2, 3): ((-5.8, -5.8, 0.6), (0.9, 0.35, 0.2), (-2.1, 0.0, 1.0), 50.0, 2.5),
        (3, 1): ((1.6, -5.6, 2.2), (0.8, 0.35, 0.2), (0.0, 0.0, 1.85), 56.0, 2.2),
        (3, 2): ((-2.0, -5.2, 1.4), (0.8, 0.3, 0.18), (0.0, 0.0, 1.6), 58.0, 2.2),
        (3, 3): ((3.4, -5.4, 0.8), (0.9, 0.35, 0.2), (0.0, 0.0, 1.2), 52.0, 2.4),
        (4, 1): ((3.8, -7.8, -0.8), (1.1, 0.4, 0.25), (0.0, 0.0, -1.2), 48.0, 2.6),
        (4, 2): ((-3.6, -7.4, -1.2), (1.0, 0.4, 0.2), (-0.3, 0.0, -1.1), 50.0, 2.5),
        (4, 3): ((0.4, -8.5, -2.6), (1.1, 0.45, 0.2), (0.0, 0.0, -2.0), 46.0, 2.8),
        (5, 1): ((4.8, -11.5, -2.2), (1.4, 0.5, 0.3), (0.0, 0.0, -2.2), 42.0, 3.0),
        (5, 2): ((-5.2, -12.2, 0.8), (1.5, 0.5, 0.3), (-0.2, 0.0, -0.6), 40.0, 3.2),
        (5, 3): ((6.8, -13.8, 1.4), (1.6, 0.6, 0.35), (-0.2, 0.0, -0.2), 38.0, 3.4),
    }
    core.setup_cinema_camera(scene, *rigs[(act, shot)], total_frames)


# ==============================================================================
# EP03 — SOLAR INVERTER (v5 Upgraded PBR + Flicker Rim Light + Fast Glow MIS)
# ==============================================================================
def build_ep03(scene, act, shot, total_frames):
    _, rim, acc = core.setup_studio_lights(floor_z=-0.75)
    mat_chassis = core.make_pbr_mat("ChassisMetal", (0.055, 0.070, 0.098, 1.0), metallic=0.88, roughness=0.28, noise_scale=18.0, rough_var=0.10, bump_strength=0.06)
    mat_pcb = core.make_pbr_mat("PCB_Emerald", (0.014, 0.068, 0.054, 1.0), metallic=0.15, roughness=0.30, coat_weight=0.45, noise_scale=28.0, rough_var=0.08, bump_strength=0.04)
    mat_copper = core.make_pbr_mat("BrushedCopper", (0.86, 0.39, 0.14, 1.0), metallic=0.96, roughness=0.20, noise_scale=24.0, rough_var=0.09, bump_strength=0.05)
    mat_ferrite = core.make_pbr_mat("FerriteCore", (0.06, 0.065, 0.075, 1.0), metallic=0.35, roughness=0.46, noise_scale=22.0, rough_var=0.12, bump_strength=0.08)
    mat_silver = core.make_pbr_mat("BrushedAlum", (0.78, 0.82, 0.88, 1.0), metallic=0.95, roughness=0.18, noise_scale=20.0, rough_var=0.07, bump_strength=0.04)
    mat_solar = core.make_pbr_mat("SolarSilicon", (0.02, 0.08, 0.28, 1.0), metallic=0.55, roughness=0.16, coat_weight=0.7, noise_scale=30.0, rough_var=0.05, bump_strength=0.03)
    mat_cap = core.make_pbr_mat("CapSleeve", (0.03, 0.09, 0.22, 1.0), metallic=0.25, roughness=0.28, coat_weight=0.5, noise_scale=16.0, rough_var=0.06, bump_strength=0.03)
    mat_film = core.make_pbr_mat("FilmCapGold", (0.88, 0.62, 0.10, 1.0), metallic=0.20, roughness=0.24, coat_weight=0.6, noise_scale=16.0, rough_var=0.06, bump_strength=0.03)
    mat_silicon = core.make_pbr_mat("SiliconBody", (0.035, 0.04, 0.05, 1.0), metallic=0.15, roughness=0.38, noise_scale=24.0, rough_var=0.10, bump_strength=0.06)
    mat_cyan = core.make_glow_mat("CyanGlow", (0.16, 0.78, 1.0, 1.0), 9.5, light_tree=False)
    mat_gold = core.make_glow_mat("GoldGlow", (1.0, 0.76, 0.22, 1.0), 6.8, light_tree=False)
    mat_orange = core.make_glow_mat("OrangeGlow", (1.0, 0.42, 0.10, 1.0), 8.5, light_tree=False)

    core.add_box((0.3, 0.0, -0.42), (12.8, 5.8, 0.55), mat_chassis, bevel=0.06)
    for i in range(18):
        core.add_box((-5.1 + i * 0.62, 2.95, 0.25), (0.14, 0.65, 1.55), mat_chassis, bevel=0.015)
    core.add_box((0.3, -0.05, 0.0), (12.0, 5.0, 0.16), mat_pcb, bevel=0.025)
    for y_tr in (-1.3, 0.0, 1.3):
        tr = core.add_box((0.2, y_tr, 0.095), (10.8, 0.09, 0.03), mat_copper if y_tr != 0.0 else mat_cyan, bevel=0.0)
        if y_tr == 0.0:
            core.mark_fast_glow(tr)
    core.add_box((-7.2, 0.0, 1.25), (0.18, 3.2, 2.2), mat_solar, rot_deg=(0, -22, 14), bevel=0.02)
    core.add_box((-4.1, 1.35, 0.22), (1.25, 1.25, 0.26), mat_silicon, bevel=0.02)
    core.build_wound_toroid("BoostInductor", (-3.8, -0.95, 0.62), 0.72, 0.22, 20, mat_ferrite, mat_copper)
    flux_ring = core.mark_fast_glow(core.add_torus((-3.8, -0.95, 0.62), 1.02, 0.035, mat_cyan, maj_seg=36, min_seg=10))

    for yc in (-1.4, 0.0, 1.4):
        core.add_cyl((-1.95, yc, 0.86), 0.42, 1.55, mat_cap)
        core.add_cyl((-1.95, yc, 1.65), 0.39, 0.08, mat_silver)

    mosfets = {}
    for lbl, mx, my in [("S1", -0.35, 1.25), ("S2", 1.25, 1.25), ("S3", -0.35, -1.25), ("S4", 1.25, -1.25)]:
        core.add_box((mx, my, 0.78), (0.78, 0.34, 0.96), mat_silicon, bevel=0.025)
        core.add_box((mx, my + 0.18, 0.95), (0.74, 0.08, 1.28), mat_silver, bevel=0.01)
        for lx in (-0.22, 0.0, 0.22):
            core.add_cyl((mx + lx, my, 0.18), 0.032, 0.36, mat_silver, verts=10)
        gm = core.make_glow_mat(f"Glow_{lbl}", (1.0, 0.74, 0.20, 1.0), 1.0, light_tree=False)
        core.mark_fast_glow(core.add_box((mx, my - 0.18, 0.85), (0.56, 0.03, 0.18), gm, bevel=0.0))
        mosfets[lbl] = gm

    core.build_wound_toroid("LC_Filter", (3.35, 0.15, 0.82), 0.98, 0.30, 26, mat_ferrite, mat_copper)
    core.add_box((4.95, 0.15, 0.78), (0.72, 1.55, 1.32), mat_film, bevel=0.04)
    core.add_box((5.95, 0.15, 0.32), (0.65, 1.1, 0.45), mat_silicon, bevel=0.02)
    relay_arm = core.mark_fast_glow(core.add_box((5.95, 0.15, 0.78), (0.92, 0.14, 0.10), mat_cyan, bevel=0.01))

    pwm_bars, sine_nodes, grid_nodes = [], [], []
    npts = 30
    for i in range(npts):
        x = -4.5 + (9.8 * i) / (npts - 1)
        if act in (3, 4):
            b = core.mark_fast_glow(core.add_box((x, 0.85, 2.55), (0.10, 0.08, 0.4), mat_gold, bevel=0.0))
            pwm_bars.append((b, i))
        if act in (1, 3, 4, 5):
            sine_nodes.append((core.add_sphere((x, -0.30, 2.55), 0.085, mat_cyan, fast_glow=True), i))
        if act == 5:
            grid_nodes.append((core.add_sphere((x, 0.35, 2.55), 0.075, mat_orange, fast_glow=True), i))

    beads = [(core.add_sphere((-5.0, 0.0, 0.18), 0.075, mat_cyan, fast_glow=True), i / 14.0) for i in range(14)]

    # Dynamic rim-light flicker synced with H-Bridge MOSFET switching in Act 3!
    if act == 3:
        core.animate_light_pulse(acc, 290.0, 140.0, 8, total_frames)
    else:
        core.animate_light_pulse(rim, 420.0, 75.0, 2, total_frames)

    for f in core.periodic_frames(total_frames, step=2):
        u = (f - 1) / float(total_frames)
        tp = 2.0 * math.pi
        phase = u * tp * 3.0
        fs = 1.0 + 0.14 * math.sin(u * tp * 4.0)
        flux_ring.scale = (fs, fs, fs)
        flux_ring.keyframe_insert("scale", frame=f)

        diag_a = math.sin(u * tp * 8.0) >= 0.0
        for lbl, gm in mosfets.items():
            bsdf = gm.node_tree.nodes.get("Principled BSDF")
            bsdf.inputs["Emission Strength"].default_value = 14.0 if ((lbl in ("S1", "S4")) == diag_a) else 0.4
            bsdf.inputs["Emission Strength"].keyframe_insert("default_value", frame=f)

        for b, idx in pwm_bars:
            sv = math.sin((idx / (npts - 1)) * tp * 2.0 - phase)
            on = abs(sv) > (0.18 + 0.55 * ((idx % 4) / 4.0))
            b.scale = (0.10, 0.08, max(0.06, abs(sv) * 0.95) if on else 0.04)
            b.location.z = 2.55 + (0.45 if sv >= 0 else -0.45) * (1.0 if on else 0.1)
            b.keyframe_insert("scale", frame=f)
            b.keyframe_insert("location", index=2, frame=f)

        for d, idx in sine_nodes:
            d.location.z = 2.55 + math.sin((idx / (npts - 1)) * tp * 2.0 - phase) * 0.82
            d.keyframe_insert("location", index=2, frame=f)

        for gd, idx in grid_nodes:
            drift = (0.65 * math.sin(u * tp)) if (act == 5 and shot == 2) else 0.0
            gd.location.z = 2.55 + math.sin((idx / (npts - 1)) * tp * 2.0 - phase + drift) * 0.82
            gd.keyframe_insert("location", index=2, frame=f)

        if act == 5 and shot == 2:
            relay_arm.rotation_euler = (0.0, math.radians(-38.0 * (0.5 - 0.5 * math.cos(u * tp))), 0.0)
            relay_arm.keyframe_insert("rotation_euler", frame=f)

        for bd, off in beads:
            t = (off + u * 2.0) % 1.0
            bd.location.x = -5.2 + t * 10.8
            bd.location.y = 0.22 * math.sin(t * tp * 3.0)
            bd.keyframe_insert("location", frame=f)

    rigs = {
        (1, 1): ((-6.5, -9.8, 5.0), (1.8, 0.5, 0.4), (0.0, 0.0, 1.1), 40.0, 3.2),
        (1, 2): ((5.2, -8.2, 2.4), (1.4, 0.6, 0.3), (-0.4, 0.0, 1.0), 48.0, 2.8),
        (1, 3): ((-0.5, -6.5, 9.2), (1.5, 0.8, 0.2), (0.2, 0.0, 0.4), 38.0, 4.0),
        (2, 1): ((-4.8, -5.2, 3.4), (1.0, 0.4, 0.3), (-3.8, 0.1, 0.6), 52.0, 2.6),
        (2, 2): ((-3.4, -3.9, 1.8), (0.8, 0.3, 0.2), (-3.8, -0.9, 0.65), 60.0, 2.2),
        (2, 3): ((-0.8, -4.8, 2.4), (0.9, 0.4, 0.2), (-2.3, -0.1, 0.85), 54.0, 2.6),
        (3, 1): ((0.5, -4.8, 2.5), (1.2, 0.4, 0.25), (0.45, 0.0, 0.9), 50.0, 2.4),
        (3, 2): ((-1.8, -5.8, 1.5), (1.0, 0.4, 0.2), (0.6, 0.2, 1.65), 45.0, 2.6),
        (3, 3): ((3.2, -4.4, 2.8), (0.9, 0.4, 0.2), (0.45, 0.1, 0.85), 55.0, 2.5),
        (4, 1): ((3.6, -4.6, 2.5), (1.1, 0.4, 0.25), (3.8, 0.15, 0.85), 52.0, 2.4),
        (4, 2): ((1.6, -7.2, 3.1), (1.3, 0.5, 0.2), (2.4, 0.2, 1.75), 44.0, 2.8),
        (4, 3): ((5.4, -3.8, 1.7), (0.8, 0.3, 0.2), (3.4, 0.15, 0.82), 62.0, 2.2),
        (5, 1): ((4.2, -7.4, 3.2), (1.3, 0.5, 0.25), (3.4, 0.1, 1.8), 44.0, 2.8),
        (5, 2): ((6.4, -4.1, 2.2), (0.8, 0.3, 0.2), (5.8, 0.15, 0.75), 56.0, 2.4),
        (5, 3): ((7.2, -9.8, 5.2), (1.8, 0.6, 0.35), (0.4, 0.0, 1.2), 38.0, 3.5),
    }
    core.setup_cinema_camera(scene, *rigs[(act, shot)], total_frames)


# ==============================================================================
# EP04 — NASA MARS ROVER & FLYING SKY CRANE (EDL / 8 HYDRAZINE THRUSTERS)
# ==============================================================================
def build_ep04(scene, act, shot, total_frames):
    _, rim, acc = core.setup_studio_lights(
        floor_z=-2.8,
        key_color=(1.0, 0.90, 0.80),
        rim_color=(1.0, 0.48, 0.18),
        accent_color=(0.35, 0.78, 1.0),
        add_floor=False,
    )
    mat_mars = core.make_pbr_mat("MartianRegolith", (0.36, 0.13, 0.065, 1.0), metallic=0.05, roughness=0.86, noise_scale=3.5, rough_var=0.10, bump_strength=0.35, bump_dist=0.12)
    mat_gold_foil = core.make_pbr_mat("MLI_GoldFoil", (0.92, 0.68, 0.14, 1.0), metallic=0.94, roughness=0.22, noise_scale=28.0, rough_var=0.14, bump_strength=0.18)
    mat_ti = core.make_pbr_mat("AerospaceTitanium", (0.68, 0.72, 0.78, 1.0), metallic=0.92, roughness=0.26, noise_scale=18.0, rough_var=0.09, bump_strength=0.05)
    mat_white = core.make_pbr_mat("AeroWhiteCoat", (0.84, 0.86, 0.88, 1.0), metallic=0.12, roughness=0.32, coat_weight=0.35, noise_scale=14.0, rough_var=0.07, bump_strength=0.04)
    mat_dark = core.make_pbr_mat("CarbonWheel", (0.06, 0.07, 0.08, 1.0), metallic=0.72, roughness=0.34, noise_scale=22.0, rough_var=0.10, bump_strength=0.06)
    mat_plasma = core.make_glow_mat("PlasmaGlow", (1.0, 0.42, 0.08, 1.0), 12.0, light_tree=False)
    mat_jet = core.make_glow_mat("HydrazineJet", (0.98, 0.72, 0.25, 1.0), 14.0, light_tree=False)
    mat_radar = core.make_glow_mat("KaRadar", (0.18, 0.82, 1.0, 1.0), 7.5, light_tree=False)

    # Textured Martian Crater Surface
    bpy.ops.mesh.primitive_plane_add(size=95.0, location=(0.0, 0.0, -2.8))
    bpy.context.active_object.data.materials.append(mat_mars)
    for rx, ry, rs in [(-4.2, 2.5, 0.65), (5.0, -1.8, 0.8), (-2.5, -4.0, 0.45), (3.2, 3.8, 0.55)]:
        core.add_sphere((rx, ry, -2.65), rs, mat_mars, segs=12, rings=8)

    # Sky Crane Descent Stage (Hovering above Rover)
    crane = bpy.data.objects.new("SkyCrane", None)
    bpy.context.collection.objects.link(crane)
    crane.location = (0.0, 0.0, 2.8)
    for p in [
        core.add_box((0, 0, 0), (2.4, 2.4, 0.75), mat_gold_foil, bevel=0.04),
        core.add_cyl((-0.9, 0.9, 0.0), 0.42, 0.85, mat_ti),
        core.add_cyl((0.9, 0.9, 0.0), 0.42, 0.85, mat_ti),
        core.add_cyl((-0.9, -0.9, 0.0), 0.42, 0.85, mat_ti),
        core.add_cyl((0.9, -0.9, 0.0), 0.42, 0.85, mat_ti),
    ]:
        p.parent = crane

    # 8 Canted Hydrazine MLE Thruster Pods + Glowing Mach-Diamond Plumes
    plumes = []
    for cx, cy in [(-1.65, 1.65), (1.65, 1.65), (-1.65, -1.65), (1.65, -1.65)]:
        arm = core.add_box((cx * 0.65, cy * 0.65, -0.05), (1.35, 0.24, 0.22), mat_ti, rot_deg=(0, 0, math.degrees(math.atan2(cy, cx))), bevel=0.02)
        arm.parent = crane
        for d_off in (-0.18, 0.18):
            nx = cx + d_off * (-math.copysign(1.0, cy))
            ny = cy + d_off * math.copysign(1.0, cx)
            nozzle = core.add_cone((nx, ny, -0.35), 0.19, 0.09, 0.42, mat_dark, rot_deg=(math.copysign(16, -cy), math.copysign(16, cx), 0))
            nozzle.parent = crane
            plume = core.mark_fast_glow(core.add_cone((nx * 1.15, ny * 1.15, -1.45), 0.06, 0.26, 2.1, mat_jet, rot_deg=(math.copysign(16, -cy), math.copysign(16, cx), 0)))
            plume.parent = crane
            plumes.append(plume)

    # 1-Ton 6-Wheel Mars Rover (Perseverance/Curiosity Cutaway Architecture)
    rover = bpy.data.objects.new("MarsRover", None)
    bpy.context.collection.objects.link(rover)
    rover.location = (0.0, 0.0, -1.25)
    for p in [
        core.add_box((0, 0, 0.25), (2.2, 1.45, 0.72), mat_white, bevel=0.04),
        core.add_box((-0.95, 0, 0.35), (0.65, 0.45, 0.48), mat_gold_foil, rot_deg=(0, 22, 0), bevel=0.02),  # MMRTG nuclear battery
        core.add_cyl((0.75, 0.45, 0.95), 0.08, 0.95, mat_ti),  # MastCam mast
        core.add_box((0.75, 0.45, 1.48), (0.38, 0.32, 0.24), mat_white, bevel=0.02),  # MastCam head
        core.mark_fast_glow(core.add_cyl((0.96, 0.45, 1.48), 0.07, 0.08, mat_radar, rot_deg=(0, 90, 0))),
    ]:
        p.parent = rover
    # 6 Rocker-Bogie Aluminum Cleated Wheels
    for wx in (-0.95, 0.0, 0.95):
        for wy in (-0.95, 0.95):
            w = core.add_cyl((wx, wy, -0.45), 0.32, 0.26, mat_dark, rot_deg=(90, 0, 0), verts=24)
            w.parent = rover
            strut = core.add_cyl((wx * 0.85, wy * 0.72, -0.05), 0.045, 0.78, mat_ti, rot_deg=(math.copysign(24, wy), 0, 0))
            strut.parent = rover

    # 3 Nylon Bridle Tethers + Umbilical Cord
    tethers = []
    for tx, ty in [(-0.75, 0.55), (0.75, 0.55), (0.0, -0.62)]:
        t = core.add_cyl((tx, ty, 0.85), 0.022, 3.8, mat_ti, verts=10)
        tethers.append(t)

    # Act 1 Heat Shield Plasma Shell / Act 2 Parachute Risers (conditionally visible)
    heat_shield = core.add_cone((0.0, 0.0, -2.05), 2.5, 0.8, 0.85, mat_plasma if act == 1 else mat_dark, rot_deg=(180, 0, 0))
    heat_shield.hide_render = (act not in (1, 2))
    if act == 1:
        core.mark_fast_glow(heat_shield)

    # Ka-Band Radar Beams sweeping down to Mars surface
    radar_rings = [core.mark_fast_glow(core.add_torus((0, 0, -0.5 - i * 0.65), 0.45 + i * 0.45, 0.02, mat_radar, maj_seg=32, min_seg=8)) for i in range(4)]

    core.animate_light_pulse(rim, 460.0, 180.0, 12, total_frames)

    for f in core.periodic_frames(total_frames, step=2):
        u = (f - 1) / float(total_frames)
        tp = 2.0 * math.pi
        hover = 0.22 * math.sin(u * tp)
        crane.location.z = (3.4 + 0.55 * (0.5 - 0.5 * math.cos(u * tp))) if act == 5 else (2.75 + hover)
        crane.keyframe_insert("location", index=2, frame=f)

        rover_z = (-1.95 if act == 5 else (-1.20 + hover * 0.6 - 0.35 * (0.5 - 0.5 * math.cos(u * tp))))
        rover.location.z = rover_z
        rover.keyframe_insert("location", index=2, frame=f)

        # Thruster plume flicker
        ps = 0.88 + 0.22 * math.sin(u * tp * 14.0)
        for pl in plumes:
            pl.scale = (1.0, 1.0, ps)
            pl.keyframe_insert("scale", frame=f)

        # Tethers stretch cleanly between crane and rover (retract/cut in Act 5)
        span = max(0.2, crane.location.z - rover_z - 0.5)
        for t in tethers:
            t.scale = (1.0, 1.0, (0.05 if act == 5 and u > 0.45 else span / 3.8))
            t.location.z = (crane.location.z + rover_z) * 0.5
            t.keyframe_insert("scale", frame=f)
            t.keyframe_insert("location", index=2, frame=f)

        for idx, rr in enumerate(radar_rings):
            rt = (u * 2.0 + idx * 0.25) % 1.0
            rr.location.z = 2.0 - rt * 4.5
            sc = 0.4 + rt * 2.2
            rr.scale = (sc, sc, 1.0)
            rr.keyframe_insert("location", index=2, frame=f)
            rr.keyframe_insert("scale", frame=f)

        if act == 2:
            heat_shield.location.z = -1.6 - 1.1 * (0.5 - 0.5 * math.cos(u * tp))
            heat_shield.keyframe_insert("location", index=2, frame=f)

    rigs = {
        (1, 1): ((-6.5, -8.8, 1.2), (1.4, 0.5, 0.4), (0.0, 0.0, -0.5), 40.0, 3.0),
        (1, 2): ((5.5, -6.5, -1.6), (1.2, 0.5, 0.3), (0.0, 0.0, -1.4), 48.0, 2.6),
        (1, 3): ((0.0, -10.5, 4.2), (1.5, 0.6, 0.3), (0.0, 0.0, 0.2), 38.0, 3.5),
        (2, 1): ((-5.8, -8.5, -0.4), (1.2, 0.4, 0.3), (0.0, 0.0, -1.2), 45.0, 2.8),
        (2, 2): ((4.8, -6.2, 2.6), (1.0, 0.4, 0.25), (0.0, 0.0, 1.8), 50.0, 2.6),
        (2, 3): ((-3.2, -9.2, -2.0), (1.2, 0.5, 0.2), (0.0, 0.0, -0.8), 42.0, 3.0),
        (3, 1): ((-5.2, -6.2, 2.8), (1.2, 0.4, 0.25), (0.0, 0.0, 2.2), 48.0, 2.5),
        (3, 2): ((4.4, -4.8, 1.4), (0.9, 0.35, 0.2), (1.4, 0.0, 2.2), 56.0, 2.2),
        (3, 3): ((0.5, -8.4, 4.6), (1.3, 0.5, 0.3), (0.0, 0.0, 1.6), 42.0, 3.0),
        (4, 1): ((-5.6, -7.8, 0.5), (1.3, 0.5, 0.3), (0.0, 0.0, 0.5), 42.0, 2.8),
        (4, 2): ((3.8, -4.6, -1.0), (0.9, 0.35, 0.2), (0.0, 0.0, -1.2), 54.0, 2.4),
        (4, 3): ((-4.2, -5.8, 1.8), (1.0, 0.4, 0.25), (0.0, 0.0, 0.8), 48.0, 2.6),
        (5, 1): ((-5.8, -7.2, -1.2), (1.2, 0.5, 0.25), (0.0, 0.0, -1.5), 44.0, 2.8),
        (5, 2): ((5.4, -8.5, 1.8), (1.4, 0.5, 0.35), (0.0, 0.0, 1.4), 40.0, 3.0),
        (5, 3): ((6.8, -9.6, 0.8), (1.6, 0.6, 0.3), (0.0, 0.0, 0.2), 38.0, 3.4),
    }
    core.setup_cinema_camera(scene, *rigs[(act, shot)], total_frames)


# ==============================================================================
# EP05 — SWISS MECHANICAL WATCH MOVEMENT (ESCAPEMENT, RUBY JEWELS, 4 HZ BALANCE)
# ==============================================================================
def build_ep05(scene, act, shot, total_frames):
    _, rim, acc = core.setup_studio_lights(floor_z=-1.0, key_color=(1.0, 0.96, 0.90), rim_color=(0.35, 0.72, 1.0), accent_color=(1.0, 0.80, 0.30))
    mat_brass = core.make_pbr_mat("WatchBrassGold", (0.92, 0.72, 0.22, 1.0), metallic=0.97, roughness=0.18, noise_scale=30.0, rough_var=0.07, bump_strength=0.04)
    mat_rhodium = core.make_pbr_mat("RhodiumPlate", (0.76, 0.80, 0.86, 1.0), metallic=0.95, roughness=0.22, noise_scale=24.0, rough_var=0.08, bump_strength=0.05)
    mat_blue_steel = core.make_pbr_mat("BluedSteel", (0.08, 0.22, 0.72, 1.0), metallic=0.94, roughness=0.16, noise_scale=26.0, rough_var=0.06, bump_strength=0.03)
    mat_ruby = core.make_pbr_mat("SyntheticRuby", (0.85, 0.04, 0.18, 1.0), metallic=0.08, roughness=0.08, coat_weight=0.9, noise_scale=18.0, rough_var=0.03, bump_strength=0.01)
    mat_ruby_glow = core.make_glow_mat("RubyHighlight", (1.0, 0.12, 0.32, 1.0), 5.5, light_tree=False)
    mat_cyan = core.make_glow_mat("TickPulse", (0.20, 0.82, 1.0, 1.0), 6.0, light_tree=False)

    # Circular Machined Mainplate + Bridges
    core.add_cyl((0.0, 0.0, -0.35), 4.6, 0.28, mat_rhodium, verts=64)
    core.add_torus((0.0, 0.0, 0.05), 4.65, 0.26, mat_rhodium, maj_seg=64, min_seg=18)

    # Mainspring Barrel (Left) + Going Gear Train (Center, Third, Fourth)
    barrel = core.add_gear_wheel("MainspringBarrel", (-2.1, 0.6, 0.20), 1.35, 0.42, 32, mat_brass)
    center_w = core.add_gear_wheel("CenterWheel", (0.0, 0.0, 0.38), 1.05, 0.20, 28, mat_brass)
    third_w = core.add_gear_wheel("ThirdWheel", (1.55, 0.85, 0.52), 0.78, 0.18, 22, mat_rhodium)
    fourth_w = core.add_gear_wheel("FourthWheel", (2.45, -0.35, 0.62), 0.68, 0.16, 20, mat_brass)
    escape_w = core.add_gear_wheel("EscapeWheel", (1.65, -1.65, 0.72), 0.62, 0.14, 15, mat_blue_steel)

    # Ruby Bearing Jewels at Every Pivot
    for jx, jy, jz in [(-2.1, 0.6, 0.46), (0.0, 0.0, 0.56), (1.55, 0.85, 0.68), (2.45, -0.35, 0.76), (1.65, -1.65, 0.85), (-0.65, -1.95, 1.12)]:
        core.add_cyl((jx, jy, jz), 0.16, 0.10, mat_ruby, verts=20)
        core.mark_fast_glow(core.add_torus((jx, jy, jz + 0.04), 0.17, 0.025, mat_ruby_glow, maj_seg=20, min_seg=8))

    # Swiss Lever Pallet Fork with Twin Crimson Ruby Pallets
    pallet = bpy.data.objects.new("PalletFork", None)
    bpy.context.collection.objects.link(pallet)
    pallet.location = (0.55, -1.80, 0.76)
    core.add_box((0.0, 0.0, 0.0), (1.15, 0.12, 0.10), mat_rhodium, bevel=0.01).parent = pallet
    core.add_box((0.52, 0.24, 0.0), (0.12, 0.36, 0.12), mat_ruby, rot_deg=(0, 0, -25), bevel=0.01).parent = pallet
    core.add_box((0.52, -0.24, 0.0), (0.12, 0.36, 0.12), mat_ruby, rot_deg=(0, 0, 25), bevel=0.01).parent = pallet

    # 4 Hz Oscillating Balance Wheel + Breathing Spiral Blued Hairspring
    balance = bpy.data.objects.new("BalanceWheel", None)
    bpy.context.collection.objects.link(balance)
    balance.location = (-0.65, -1.95, 0.92)
    core.add_torus((0, 0, 0), 1.32, 0.11, mat_brass, maj_seg=48, min_seg=14).parent = balance
    for a_deg in (0, 120, 240):
        core.add_box((0, 0, 0), (2.5, 0.10, 0.08), mat_brass, rot_deg=(0, 0, a_deg), bevel=0.01).parent = balance

    # Spiral Hairspring Curve
    cd = bpy.data.curves.new("HairspringCurve", type="CURVE")
    cd.dimensions = "3D"
    cd.bevel_depth = 0.022
    sp = cd.splines.new("POLY")
    steps = 110
    sp.points.add(steps)
    for i in range(steps + 1):
        t = i / float(steps)
        ang = t * 5.0 * 2.0 * math.pi
        r = 0.18 + t * 0.85
        sp.points[i].co = (math.cos(ang) * r, math.sin(ang) * r, 0.16, 1.0)
    hairspring = bpy.data.objects.new("Hairspring", cd)
    bpy.context.collection.objects.link(hairspring)
    hairspring.data.materials.append(mat_blue_steel)
    hairspring.location = (-0.65, -1.95, 0.92)

    # Sweep Hands above Center Pivot
    hand_sec = core.add_box((0.0, 0.0, 1.25), (3.4, 0.06, 0.04), mat_blue_steel, bevel=0.005)
    hand_min = core.add_box((0.0, 0.0, 1.12), (2.6, 0.14, 0.05), mat_rhodium, bevel=0.01)

    core.animate_light_pulse(acc, 280.0, 95.0, 6, total_frames)

    for f in core.periodic_frames(total_frames, step=2):
        u = (f - 1) / float(total_frames)
        tp = 2.0 * math.pi
        # Exact integer-cycle rotations so loop is 100% seamless
        barrel.rotation_euler.z = u * tp * 1.0
        center_w.rotation_euler.z = -u * tp * 2.0
        third_w.rotation_euler.z = u * tp * 3.0
        fourth_w.rotation_euler.z = -u * tp * 4.0
        escape_w.rotation_euler.z = u * tp * 6.0
        hand_sec.rotation_euler.z = -u * tp * 1.0
        hand_min.rotation_euler.z = -u * tp * 1.0 * 0.25 + 0.5  # reset cleanly via oscillation or full cycle
        # Make hand_min seamless by oscillating slightly or full 1 turn on Act 5
        hand_min.rotation_euler.z = -u * tp * 1.0 + 1.1

        # 4 Hz Balance Wheel sinusoidal oscillation (6 full cycles per 6s loop = clean crisp visual beat)
        osc = math.sin(u * tp * 6.0)
        balance.rotation_euler.z = osc * 1.9
        hs = 1.0 + 0.12 * osc
        hairspring.scale = (hs, hs, 1.0)
        pallet.rotation_euler.z = (0.22 if osc >= 0 else -0.22)

        for obj in (barrel, center_w, third_w, fourth_w, escape_w, balance, pallet, hand_sec, hand_min):
            obj.keyframe_insert("rotation_euler", index=2, frame=f)
        hairspring.keyframe_insert("scale", frame=f)

    rigs = {
        (1, 1): ((-4.5, -7.2, 6.2), (1.4, 0.5, 0.35), (0.0, -0.2, 0.4), 45.0, 2.8),
        (1, 2): ((5.2, -5.4, 3.6), (1.2, 0.5, 0.25), (0.2, -0.4, 0.5), 52.0, 2.5),
        (1, 3): ((0.0, -2.5, 9.8), (1.0, 0.6, 0.2), (0.0, -0.2, 0.3), 42.0, 3.6),
        (2, 1): ((-4.2, -3.5, 3.8), (0.9, 0.4, 0.25), (-1.2, 0.4, 0.3), 55.0, 2.4),
        (2, 2): ((-1.2, -4.4, 2.2), (0.8, 0.3, 0.2), (0.4, 0.3, 0.4), 60.0, 2.2),
        (2, 3): ((2.8, -3.8, 3.2), (0.9, 0.35, 0.2), (1.2, 0.4, 0.5), 56.0, 2.4),
        (3, 1): ((2.6, -4.2, 2.1), (0.7, 0.3, 0.15), (1.1, -1.7, 0.75), 68.0, 2.0),
        (3, 2): ((0.4, -4.5, 1.8), (0.6, 0.25, 0.15), (0.8, -1.75, 0.75), 72.0, 2.0),
        (3, 3): ((3.6, -2.6, 3.2), (0.8, 0.3, 0.2), (1.3, -1.65, 0.72), 62.0, 2.2),
        (4, 1): ((-2.8, -5.2, 2.6), (0.9, 0.35, 0.2), (-0.65, -1.95, 0.95), 60.0, 2.2),
        (4, 2): ((1.4, -4.8, 2.1), (0.8, 0.3, 0.15), (-0.65, -1.95, 0.95), 66.0, 2.0),
        (4, 3): ((-0.6, -3.6, 4.6), (0.7, 0.35, 0.2), (-0.65, -1.95, 0.92), 54.0, 2.5),
        (5, 1): ((-5.0, -7.0, 4.8), (1.4, 0.5, 0.3), (0.0, -0.3, 0.5), 44.0, 3.0),
        (5, 2): ((4.8, -6.2, 2.8), (1.2, 0.45, 0.25), (0.0, -0.3, 0.6), 48.0, 2.6),
        (5, 3): ((0.6, -8.5, 6.4), (1.5, 0.6, 0.3), (0.0, -0.2, 0.4), 40.0, 3.4),
    }
    core.setup_cinema_camera(scene, *rigs[(act, shot)], total_frames)


# ==============================================================================
# EP06 — HIGH-BYPASS TURBOFAN JET ENGINE (N1/N2 SPOOL, COMBUSTOR, THRUST REVERSER)
# ==============================================================================
def build_ep06(scene, act, shot, total_frames):
    _, rim, acc = core.setup_studio_lights(floor_z=-3.2, rim_color=(0.18, 0.78, 1.0), accent_color=(1.0, 0.45, 0.12))
    mat_ti = core.make_pbr_mat("TitaniumBlade", (0.64, 0.70, 0.78, 1.0), metallic=0.95, roughness=0.22, noise_scale=22.0, rough_var=0.08, bump_strength=0.05)
    mat_casing = core.make_pbr_mat("NacelleCarbon", (0.08, 0.10, 0.14, 1.0), metallic=0.72, roughness=0.30, noise_scale=16.0, rough_var=0.10, bump_strength=0.06)
    mat_inconel = core.make_pbr_mat("InconelHot", (0.36, 0.28, 0.24, 1.0), metallic=0.90, roughness=0.34, noise_scale=24.0, rough_var=0.12, bump_strength=0.08)
    mat_cyan = core.make_glow_mat("ColdBypass", (0.16, 0.80, 1.0, 1.0), 8.5, light_tree=False)
    mat_flame = core.make_glow_mat("CombustorFlame", (1.0, 0.36, 0.06, 1.0), 14.0, light_tree=False)

    # Half-Cutaway Outer Bypass Nacelle + Translating Thrust Reverser Sleeve
    for x_ring, r_ring in [(-4.8, 2.55), (-3.2, 2.65), (-1.2, 2.55), (1.0, 2.35)]:
        core.add_torus((x_ring, 0.0, 0.0), r_ring, 0.12, mat_casing, rot_deg=(0, 90, 0), maj_seg=48, min_seg=14)
    core.add_box((-1.5, 2.35, 0.0), (7.6, 0.28, 3.8), mat_casing, bevel=0.05)  # Rear half-shell backdrop
    reverser = core.add_torus((2.2, 0.0, 0.0), 2.15, 0.22, mat_ti, rot_deg=(0, 90, 0), maj_seg=42, min_seg=12)

    # N1 Inner Low-Pressure Spool (Giant Front Fan + Inner Shaft + Rear LPT)
    n1 = bpy.data.objects.new("N1_Spool", None)
    bpy.context.collection.objects.link(n1)
    core.add_cyl((0.0, 0.0, 0.0), 0.22, 10.6, mat_ti, rot_deg=(0, 90, 0)).parent = n1
    core.add_cone((-5.4, 0.0, 0.0), 0.65, 0.04, 1.15, mat_casing, rot_deg=(0, -90, 0)).parent = n1
    # 18 Giant Swept Fan Blades
    for i in range(18):
        a = (i / 18.0) * 2.0 * math.pi
        b = core.add_box((-4.5, math.cos(a) * 1.45, math.sin(a) * 1.45), (0.38, 1.72, 0.08), mat_ti, rot_deg=(math.degrees(a), 0, 22), bevel=0.01)
        b.parent = n1
    # Rear LPT Discs on N1
    for lx in (2.8, 3.5, 4.2):
        core.add_cyl((lx, 0.0, 0.0), 1.15, 0.24, mat_inconel, rot_deg=(0, 90, 0)).parent = n1

    # N2 Outer Hollow High-Pressure Spool (6-Stage Axial Compressor + HPT)
    n2 = bpy.data.objects.new("N2_Spool", None)
    bpy.context.collection.objects.link(n2)
    core.add_cyl((-0.4, 0.0, 0.0), 0.42, 4.6, mat_inconel, rot_deg=(0, 90, 0)).parent = n2
    for idx, cx in enumerate([-2.6, -2.0, -1.4, -0.8, -0.2]):
        rad = 1.35 - idx * 0.11
        core.add_cyl((cx, 0.0, 0.0), rad, 0.18, mat_ti, rot_deg=(0, 90, 0)).parent = n2
        for k in range(12):
            a = (k / 12.0) * 2.0 * math.pi
            core.add_box((cx, math.cos(a) * rad, math.sin(a) * rad), (0.16, 0.35, 0.05), mat_ti, rot_deg=(math.degrees(a), 0, 18), bevel=0.005).parent = n2
    # High-Pressure Turbine (HPT) right behind combustor
    core.add_cyl((1.95, 0.0, 0.0), 1.02, 0.26, mat_inconel, rot_deg=(0, 90, 0)).parent = n2

    # 1,900 C Annular Combustor Ring + Pulsing Flame Core
    combustor = core.mark_fast_glow(core.add_torus((0.85, 0.0, 0.0), 0.92, 0.26, mat_flame, rot_deg=(0, 90, 0), maj_seg=36, min_seg=14))

    # Cold Bypass Cyan Streams (Outer) & Hot Core Streams (Inner)
    bypass_beads = [(core.add_sphere((-4.8, 1.95 * math.cos(i), 1.95 * math.sin(i)), 0.09, mat_cyan, fast_glow=True), i / 18.0, i) for i in range(18)]
    hot_beads = [(core.add_sphere((0.8, 0.7 * math.cos(i), 0.7 * math.sin(i)), 0.08, mat_flame, fast_glow=True), i / 12.0, i) for i in range(12)]

    core.animate_light_pulse(acc, 340.0, 140.0, 10, total_frames)

    for f in core.periodic_frames(total_frames, step=2):
        u = (f - 1) / float(total_frames)
        tp = 2.0 * math.pi
        # Dual-spool different speeds (N1 = 2 turns, N2 = 5 turns -> visually unmistakable!)
        n1.rotation_euler.x = u * tp * 2.0
        n2.rotation_euler.x = u * tp * 5.0
        n1.keyframe_insert("rotation_euler", index=0, frame=f)
        n2.keyframe_insert("rotation_euler", index=0, frame=f)

        cs = 1.0 + 0.08 * math.sin(u * tp * 10.0)
        combustor.scale = (cs, cs, cs)
        combustor.keyframe_insert("scale", frame=f)

        if act == 5:
            reverser.location.x = 2.2 + 0.95 * (0.5 - 0.5 * math.cos(u * tp))
            reverser.keyframe_insert("location", index=0, frame=f)

        for bd, off, idx in bypass_beads:
            t = (off + u * 2.0) % 1.0
            ang = (idx / 18.0) * tp
            bd.location = (-4.8 + t * 9.8, math.cos(ang) * 1.95, math.sin(ang) * 1.95)
            bd.keyframe_insert("location", frame=f)
        for bd, off, idx in hot_beads:
            t = (off + u * 3.0) % 1.0
            ang = (idx / 12.0) * tp
            bd.location = (0.6 + t * 4.8, math.cos(ang) * 0.78, math.sin(ang) * 0.78)
            bd.keyframe_insert("location", frame=f)

    rigs = {
        (1, 1): ((-7.8, -9.2, 2.8), (1.5, 0.5, 0.35), (-0.5, 0.0, 0.0), 40.0, 3.0),
        (1, 2): ((-8.2, -5.2, 1.2), (1.1, 0.4, 0.25), (-4.2, 0.0, 0.0), 48.0, 2.6),
        (1, 3): ((1.5, -10.4, 3.8), (1.4, 0.6, 0.3), (-0.4, 0.0, 0.0), 38.0, 3.4),
        (2, 1): ((-2.8, -6.0, 1.8), (1.0, 0.4, 0.25), (-1.4, 0.0, 0.0), 52.0, 2.4),
        (2, 2): ((0.4, -5.4, 1.0), (0.9, 0.35, 0.2), (-1.2, 0.0, 0.0), 56.0, 2.2),
        (2, 3): ((-4.5, -5.6, 2.8), (1.0, 0.4, 0.2), (-1.8, 0.0, 0.0), 48.0, 2.6),
        (3, 1): ((0.8, -4.8, 1.4), (0.8, 0.35, 0.2), (0.85, 0.0, 0.0), 58.0, 2.2),
        (3, 2): ((-1.2, -5.2, 0.8), (0.8, 0.3, 0.15), (0.85, 0.0, 0.0), 56.0, 2.4),
        (3, 3): ((2.6, -4.6, 2.2), (0.8, 0.35, 0.2), (0.85, 0.0, 0.0), 55.0, 2.4),
        (4, 1): ((2.2, -4.5, 1.3), (0.8, 0.3, 0.15), (2.0, 0.0, 0.0), 60.0, 2.2),
        (4, 2): ((4.4, -4.8, 1.0), (0.8, 0.35, 0.2), (2.4, 0.0, 0.0), 56.0, 2.4),
        (4, 3): ((1.4, -5.5, 2.8), (0.9, 0.35, 0.2), (2.2, 0.0, 0.0), 52.0, 2.5),
        (5, 1): ((5.8, -7.8, 2.2), (1.3, 0.5, 0.3), (2.2, 0.0, 0.0), 44.0, 2.8),
        (5, 2): ((3.4, -6.2, 1.4), (1.0, 0.4, 0.2), (2.5, 0.0, 0.0), 50.0, 2.6),
        (5, 3): ((7.2, -9.8, 3.6), (1.6, 0.6, 0.3), (0.4, 0.0, 0.0), 38.0, 3.2),
    }
    core.setup_cinema_camera(scene, *rigs[(act, shot)], total_frames)


# ==============================================================================
# EP07 — NUCLEAR SUBMARINE LIFE SUPPORT (PWR, RO DESALINATION, O2 ELECTROLYSIS, CO2)
# ==============================================================================
def build_ep07(scene, act, shot, total_frames):
    _, rim, acc = core.setup_studio_lights(floor_z=-2.8, key_color=(0.82, 0.94, 1.0), rim_color=(0.08, 0.72, 0.95), accent_color=(1.0, 0.72, 0.22))
    mat_hull = core.make_pbr_mat("SubHullSteel", (0.045, 0.065, 0.095, 1.0), metallic=0.86, roughness=0.34, noise_scale=12.0, rough_var=0.12, bump_strength=0.07)
    mat_steel = core.make_pbr_mat("ReactorSteel", (0.62, 0.68, 0.74, 1.0), metallic=0.92, roughness=0.24, noise_scale=20.0, rough_var=0.08, bump_strength=0.05)
    mat_copper = core.make_pbr_mat("PipeCopper", (0.84, 0.42, 0.16, 1.0), metallic=0.94, roughness=0.22, noise_scale=22.0, rough_var=0.08, bump_strength=0.04)
    mat_cherenkov = core.make_glow_mat("ReactorBlue", (0.14, 0.58, 1.0, 1.0), 11.0, light_tree=False)
    mat_o2 = core.make_glow_mat("OxygenCyan", (0.20, 0.92, 1.0, 1.0), 9.0, light_tree=False)
    mat_h2 = core.make_glow_mat("HydrogenGold", (1.0, 0.68, 0.16, 1.0), 8.0, light_tree=False)

    # Ribbed Cutaway Submarine Pressure Hull Backdrop
    core.add_box((0.0, 2.1, 0.0), (13.0, 0.32, 4.4), mat_hull, bevel=0.05)
    for x_rib in range(-5, 6, 2):
        core.add_torus((float(x_rib), 0.0, 0.0), 2.35, 0.12, mat_hull, rot_deg=(0, 90, 0), maj_seg=42, min_seg=12)
    core.add_box((0.0, 0.0, -1.65), (12.4, 3.6, 0.22), mat_hull, bevel=0.03)

    # Bay 1 (Left): Pressurized Water Reactor (PWR) Vessel + Cherenkov Core
    core.add_cyl((-4.2, 0.0, -0.1), 1.05, 2.5, mat_steel)
    core_glow = core.mark_fast_glow(core.add_cyl((-4.2, -0.35, -0.1), 0.65, 1.65, mat_cherenkov))
    core.add_cyl((-2.5, 0.0, 0.2), 0.55, 2.1, mat_copper)  # Steam Generator

    # Bay 2 (Center-Left): 60-Bar Reverse Osmosis & Vacuum Flash Desalination Banks
    for y_ro in (-0.55, 0.55):
        for z_ro in (-0.4, 0.55):
            core.add_cyl((-0.5, y_ro, z_ro), 0.32, 2.1, mat_steel, rot_deg=(0, 90, 0))

    # Bay 3 (Center-Right): Electrolytic O2 Generator Stack (H2O -> O2 + H2)
    core.add_box((2.1, 0.0, -0.1), (1.6, 1.4, 2.2), mat_hull, bevel=0.04)
    for px in (1.65, 2.1, 2.55):
        core.add_box((px, 0.0, -0.1), (0.08, 1.1, 1.8), mat_copper, bevel=0.01)

    # Bay 4 (Right): Dual-Tower MEA CO2 Scrubber & Hopcalite Burner
    core.add_cyl((4.3, -0.45, 0.0), 0.48, 2.4, mat_steel)
    core.add_cyl((5.3, 0.45, 0.0), 0.48, 2.4, mat_copper)

    # Rising O2 (Cyan) & H2 (Amber) Micro-Bubbles + Coolant Loop Beads
    o2_bubbles = [(core.add_sphere((1.85, -0.3 + (i % 3) * 0.3, -1.0), 0.085, mat_o2, fast_glow=True), i / 12.0) for i in range(12)]
    h2_bubbles = [(core.add_sphere((2.45, -0.2 + (i % 3) * 0.25, -1.0), 0.075, mat_h2, fast_glow=True), i / 10.0) for i in range(10)]
    flow_beads = [(core.add_sphere((-3.2, -0.8, 0.0), 0.08, mat_o2, fast_glow=True), i / 14.0) for i in range(14)]

    core.animate_light_pulse(rim, 430.0, 95.0, 4, total_frames)

    for f in core.periodic_frames(total_frames, step=2):
        u = (f - 1) / float(total_frames)
        tp = 2.0 * math.pi
        gs = 1.0 + 0.07 * math.sin(u * tp * 4.0)
        core_glow.scale = (gs, gs, 1.0)
        core_glow.keyframe_insert("scale", frame=f)

        for bd, off in o2_bubbles:
            t = (off + u * 2.0) % 1.0
            bd.location.z = -1.1 + t * 2.6
            bd.keyframe_insert("location", index=2, frame=f)
        for bd, off in h2_bubbles:
            t = (off + u * 2.0) % 1.0
            bd.location.z = -1.1 + t * 2.6
            bd.keyframe_insert("location", index=2, frame=f)
        for bd, off in flow_beads:
            t = (off + u * 2.0) % 1.0
            bd.location.x = -4.6 + t * 10.2
            bd.location.z = -0.9 + 0.35 * math.sin(t * tp * 3.0)
            bd.keyframe_insert("location", frame=f)

    rigs = {
        (1, 1): ((-6.5, -10.5, 2.8), (1.6, 0.5, 0.35), (0.4, 0.0, 0.0), 38.0, 3.2),
        (1, 2): ((5.8, -8.8, 1.6), (1.3, 0.5, 0.3), (0.0, 0.0, 0.0), 44.0, 2.8),
        (1, 3): ((0.0, -11.5, 4.4), (1.5, 0.6, 0.25), (0.4, 0.0, -0.2), 36.0, 3.5),
        (2, 1): ((-4.2, -5.2, 1.2), (0.9, 0.35, 0.2), (-3.8, 0.0, 0.0), 52.0, 2.4),
        (2, 2): ((-2.2, -4.6, 0.8), (0.8, 0.3, 0.2), (-3.5, 0.0, 0.0), 56.0, 2.2),
        (2, 3): ((-5.4, -4.8, 2.4), (0.9, 0.4, 0.2), (-3.8, 0.0, 0.1), 50.0, 2.5),
        (3, 1): ((-0.5, -4.8, 1.2), (0.9, 0.35, 0.2), (-0.5, 0.0, 0.1), 54.0, 2.4),
        (3, 2): ((1.2, -4.5, 0.6), (0.8, 0.3, 0.15), (-0.5, 0.0, 0.1), 58.0, 2.2),
        (3, 3): ((-2.2, -5.0, 2.2), (0.9, 0.35, 0.2), (-0.5, 0.0, 0.0), 50.0, 2.5),
        (4, 1): ((2.1, -4.6, 1.0), (0.8, 0.35, 0.2), (2.1, 0.0, 0.0), 56.0, 2.2),
        (4, 2): ((3.6, -4.2, 0.4), (0.7, 0.3, 0.15), (2.1, 0.0, 0.1), 60.0, 2.2),
        (4, 3): ((0.6, -4.8, 2.1), (0.8, 0.35, 0.2), (2.1, 0.0, 0.1), 54.0, 2.4),
        (5, 1): ((4.8, -5.2, 1.4), (0.9, 0.4, 0.2), (4.7, 0.0, 0.0), 52.0, 2.5),
        (5, 2): ((6.5, -7.2, 2.2), (1.2, 0.45, 0.25), (3.4, 0.0, 0.0), 46.0, 2.8),
        (5, 3): ((7.4, -10.2, 3.2), (1.6, 0.55, 0.3), (0.5, 0.0, 0.0), 38.0, 3.2),
    }
    core.setup_cinema_camera(scene, *rigs[(act, shot)], total_frames)


# ==============================================================================
# EP08 — 600 KM/H SUPERCONDUCTING MAGLEV TRAIN (L0 SERIES, -269C NbTi, FIGURE-8)
# ==============================================================================
def build_ep08(scene, act, shot, total_frames):
    _, rim, acc = core.setup_studio_lights(floor_z=-2.8, rim_color=(0.15, 0.80, 1.0), accent_color=(1.0, 0.32, 0.18))
    mat_conc = core.make_pbr_mat("GuidewayConcrete", (0.18, 0.21, 0.26, 1.0), metallic=0.08, roughness=0.78, noise_scale=8.0, rough_var=0.14, bump_strength=0.18)
    mat_aero = core.make_pbr_mat("L0_WhiteAero", (0.88, 0.91, 0.95, 1.0), metallic=0.35, roughness=0.16, coat_weight=0.75, noise_scale=18.0, rough_var=0.05, bump_strength=0.02)
    mat_blue = core.make_pbr_mat("JR_BlueStripe", (0.02, 0.32, 0.78, 1.0), metallic=0.45, roughness=0.20, coat_weight=0.6)
    mat_dark = core.make_pbr_mat("BogieSteel", (0.07, 0.09, 0.12, 1.0), metallic=0.86, roughness=0.32, noise_scale=20.0, rough_var=0.10, bump_strength=0.05)
    mat_cryo = core.make_glow_mat("CryoHelium", (0.16, 0.84, 1.0, 1.0), 9.5, light_tree=False)
    mat_gold = core.make_glow_mat("NbTiCoil", (1.0, 0.76, 0.18, 1.0), 8.0, light_tree=False)

    # U-Shaped Concrete Guideway
    core.add_box((0.0, 0.0, -2.2), (34.0, 4.8, 0.7), mat_conc, bevel=0.05)
    core.add_box((0.0, 2.35, -0.7), (34.0, 0.55, 2.8), mat_conc, bevel=0.05)

    # Passive Figure-8 Null-Flux Aluminum Loops + Animated LSM N/S Propulsion Wave Coils
    lsm_mats = []
    coil_strip = bpy.data.objects.new("GuidewayCoils", None)
    bpy.context.collection.objects.link(coil_strip)
    for i in range(-7, 8):
        x = i * 1.8
        pm = core.make_glow_mat(f"LSM_{i}", (0.16, 0.75, 1.0, 1.0), 5.0, light_tree=False)
        lsm_mats.append((pm, i))
        core.mark_fast_glow(core.add_box((x, 2.02, -0.65), (1.05, 0.14, 1.45), pm, bevel=0.01)).parent = coil_strip
        core.mark_fast_glow(core.add_torus((x, 1.88, -0.25), 0.32, 0.045, mat_cryo, rot_deg=(90, 0, 0), maj_seg=22, min_seg=8)).parent = coil_strip
        core.mark_fast_glow(core.add_torus((x, 1.88, -0.98), 0.32, 0.045, mat_gold, rot_deg=(90, 0, 0), maj_seg=22, min_seg=8)).parent = coil_strip

    # Streamlined L0 Series Train + Superconducting Cryostat Bogie
    train = bpy.data.objects.new("MaglevTrain", None)
    bpy.context.collection.objects.link(train)
    core.add_box((-1.5, 0.0, 0.85), (9.5, 2.8, 2.0), mat_aero, bevel=0.14).parent = train
    core.add_cone((5.8, 0.0, 0.58), 1.45, 0.12, 5.2, mat_aero, rot_deg=(0, 90, 0)).parent = train
    core.add_box((-1.2, 0.0, 0.65), (10.2, 2.86, 0.28), mat_blue, bevel=0.02).parent = train
    core.add_box((0.0, 0.0, -0.68), (5.4, 2.9, 1.0), mat_dark, bevel=0.05).parent = train

    # -269C Liquid Helium Cryostats + NbTi Racetrack Coils on Both Sides
    for sy in (-1.52, 1.52):
        for cx in (-1.45, 1.45):
            core.mark_fast_glow(core.add_box((cx, sy, -0.68), (2.0, 0.24, 0.78), mat_cryo, bevel=0.03)).parent = train
            c = core.mark_fast_glow(core.add_torus((cx, sy + math.copysign(0.14, sy), -0.68), 0.30, 0.06, mat_gold, rot_deg=(90, 0, 0), maj_seg=26, min_seg=10))
            c.scale = (1.85, 0.8, 1.0)
            c.parent = train

    # Retractable Aircraft Rubber Tires (animated in Act 5)
    wheels = bpy.data.objects.new("RetractableGear", None)
    bpy.context.collection.objects.link(wheels)
    wheels.parent = train
    for wx in (-1.75, 1.75):
        for wy in (-0.95, 0.95):
            core.add_cyl((wx, wy, -1.45), 0.38, 0.26, mat_dark, rot_deg=(90, 0, 0), verts=22).parent = wheels

    core.animate_light_pulse(rim, 440.0, 120.0, 6, total_frames)

    for f in core.periodic_frames(total_frames, step=2):
        u = (f - 1) / float(total_frames)
        tp = 2.0 * math.pi
        # Exact seamless coil scroll (4 coil pitches per loop)
        coil_strip.location.x = -((u * 4.0) % 1.0) * 1.8
        coil_strip.keyframe_insert("location", index=0, frame=f)

        # Traveling N/S magnetic wave along LSM coils
        for pm, idx in lsm_mats:
            ph = math.sin(idx * 0.7 - u * tp * 6.0)
            bsdf = pm.node_tree.nodes.get("Principled BSDF")
            col = (1.0, 0.22, 0.14, 1.0) if ph >= 0 else (0.14, 0.78, 1.0, 1.0)
            bsdf.inputs["Emission Color"].default_value = col
            bsdf.inputs["Emission Strength"].default_value = 3.0 + 7.5 * abs(ph)
            bsdf.inputs["Emission Color"].keyframe_insert("default_value", frame=f)
            bsdf.inputs["Emission Strength"].keyframe_insert("default_value", frame=f)

        if act == 5:
            lift = 0.5 - 0.5 * math.cos(u * tp)
            train.location.z = -0.25 + 0.55 * lift
            wheels.location.z = 0.55 * lift
        else:
            train.location.z = 0.30 + 0.06 * math.sin(u * tp * 3.0)
            wheels.location.z = 0.55
        train.keyframe_insert("location", index=2, frame=f)
        wheels.keyframe_insert("location", index=2, frame=f)

    rigs = {
        (1, 1): ((6.8, -11.2, 2.2), (1.6, 0.5, 0.3), (1.2, 0.0, 0.0), 40.0, 3.0),
        (1, 2): ((9.2, -5.8, 0.2), (1.1, 0.4, 0.2), (2.2, 0.0, -0.5), 48.0, 2.6),
        (1, 3): ((-5.5, -9.8, 4.4), (1.4, 0.5, 0.3), (0.8, 0.0, 0.0), 38.0, 3.4),
        (2, 1): ((0.0, -5.6, -0.2), (1.0, 0.35, 0.2), (0.0, -0.6, -0.68), 54.0, 2.4),
        (2, 2): ((-2.8, -4.6, -0.5), (0.8, 0.3, 0.15), (-0.8, -1.2, -0.68), 60.0, 2.2),
        (2, 3): ((3.2, -5.0, 0.8), (0.9, 0.35, 0.2), (0.4, -0.8, -0.68), 52.0, 2.5),
        (3, 1): ((0.5, -6.5, 0.4), (1.1, 0.4, 0.2), (0.0, 1.4, -0.65), 50.0, 2.5),
        (3, 2): ((-3.4, -5.4, -0.3), (0.9, 0.35, 0.18), (0.0, 1.6, -0.65), 55.0, 2.4),
        (3, 3): ((3.8, -5.8, 1.5), (1.0, 0.4, 0.2), (0.2, 1.4, -0.65), 48.0, 2.6),
        (4, 1): ((3.4, -7.8, 1.4), (1.3, 0.45, 0.25), (0.5, 1.4, -0.55), 46.0, 2.8),
        (4, 2): ((-3.8, -7.2, 1.2), (1.2, 0.45, 0.2), (0.0, 1.5, -0.6), 48.0, 2.6),
        (4, 3): ((1.0, -9.5, 3.6), (1.4, 0.5, 0.25), (0.5, 0.8, -0.4), 40.0, 3.2),
        (5, 1): ((3.8, -6.2, -0.8), (1.0, 0.4, 0.2), (0.4, 0.0, -1.0), 50.0, 2.5),
        (5, 2): ((-3.6, -6.8, -0.6), (1.0, 0.4, 0.2), (0.0, 0.0, -0.9), 48.0, 2.6),
        (5, 3): ((7.5, -10.8, 2.6), (1.6, 0.55, 0.3), (1.0, 0.0, 0.0), 38.0, 3.2),
    }
    core.setup_cinema_camera(scene, *rigs[(act, shot)], total_frames)


# ==============================================================================
# EP09 — CAR ABS BRAKE SYSTEM (ROTOR, TONE RING, 8-SOLENOID HCU, 15 HZ PULSE)
# ==============================================================================
def build_ep09(scene, act, shot, total_frames):
    _, rim, acc = core.setup_studio_lights(floor_z=-3.6, rim_color=(0.18, 0.78, 1.0), accent_color=(1.0, 0.25, 0.14))
    mat_disc = core.make_pbr_mat("BrakeSteel", (0.64, 0.68, 0.74, 1.0), metallic=0.94, roughness=0.22, noise_scale=26.0, rough_var=0.08, bump_strength=0.05)
    mat_caliper = core.make_pbr_mat("CaliperRed", (0.78, 0.06, 0.06, 1.0), metallic=0.55, roughness=0.24, coat_weight=0.75, noise_scale=18.0, rough_var=0.06, bump_strength=0.04)
    mat_alum = core.make_pbr_mat("HCU_Alum", (0.48, 0.54, 0.62, 1.0), metallic=0.88, roughness=0.28, noise_scale=20.0, rough_var=0.09, bump_strength=0.05)
    mat_copper = core.make_pbr_mat("SolenoidCopper", (0.86, 0.40, 0.14, 1.0), metallic=0.95, roughness=0.20, noise_scale=24.0, rough_var=0.08, bump_strength=0.04)
    mat_tire = core.make_pbr_mat("WetRubber", (0.04, 0.045, 0.052, 1.0), metallic=0.05, roughness=0.62, noise_scale=16.0, rough_var=0.14, bump_strength=0.14)
    mat_high = core.make_glow_mat("FluidHigh", (1.0, 0.44, 0.08, 1.0), 10.0, light_tree=False)
    mat_low = core.make_glow_mat("FluidLow", (0.16, 0.80, 1.0, 1.0), 9.5, light_tree=False)

    # Left: Wheel Hub + Ventilated Brake Rotor + 24-Tooth Tone Ring + Red Caliper
    wheel_hub = bpy.data.objects.new("WheelHub", None)
    bpy.context.collection.objects.link(wheel_hub)
    wheel_hub.location = (-3.6, 0.0, 0.2)

    rotor = bpy.data.objects.new("RotorSpin", None)
    bpy.context.collection.objects.link(rotor)
    rotor.parent = wheel_hub
    core.add_cyl((0, 0, 0), 2.15, 0.24, mat_disc, rot_deg=(90, 0, 0), verts=48).parent = rotor
    core.add_torus((0, 0, 0), 2.95, 0.58, mat_tire, rot_deg=(90, 0, 0), maj_seg=48, min_seg=18).parent = rotor
    tone = core.add_gear_wheel("ToneRing", (0, -0.22, 0), 0.88, 0.14, 24, mat_disc, rot_deg=(90, 0, 0))
    tone.parent = rotor

    core.mark_fast_glow(core.add_cyl((1.38, -0.25, 0.0), 0.14, 0.85, mat_low, rot_deg=(0, 90, 0))).parent = wheel_hub
    core.add_box((1.85, 0.0, 0.72), (1.05, 0.78, 1.85), mat_caliper, bevel=0.06).parent = wheel_hub
    pad = core.add_box((1.72, -0.24, 0.72), (0.68, 0.14, 1.25), mat_copper, bevel=0.02)
    pad.parent = wheel_hub

    # Right: Cutaway 8-Solenoid Hydraulic Control Unit (HCU) Manifold
    hcu = bpy.data.objects.new("HCU_Block", None)
    bpy.context.collection.objects.link(hcu)
    hcu.location = (3.3, 0.0, 0.2)
    core.add_box((0, 0.65, 0), (5.0, 1.1, 4.0), mat_alum, bevel=0.06).parent = hcu
    core.add_cyl((-1.2, 0, 1.0), 0.52, 1.05, mat_copper).parent = hcu
    core.add_cyl((1.2, 0, 1.0), 0.52, 1.05, mat_copper).parent = hcu
    inlet_p = core.mark_fast_glow(core.add_cyl((-1.2, 0, 0.5), 0.22, 0.80, mat_high))
    inlet_p.parent = hcu
    outlet_p = core.mark_fast_glow(core.add_cyl((1.2, 0, 0.5), 0.22, 0.80, mat_low))
    outlet_p.parent = hcu
    core.add_cyl((1.2, 0, -1.1), 0.64, 1.35, mat_disc).parent = hcu
    accum_p = core.mark_fast_glow(core.add_cyl((1.2, -0.25, -1.1), 0.50, 0.34, mat_low))
    accum_p.parent = hcu
    pump_cam = core.mark_fast_glow(core.add_cyl((-1.2, -0.15, -1.1), 0.56, 0.45, mat_high, rot_deg=(90, 0, 0)))
    pump_cam.parent = hcu

    # Pulsing Hydraulic Fluid Beads between HCU & Caliper
    beads = [(core.add_sphere((0.0, -0.2, 1.2), 0.14, mat_high, fast_glow=True), i / 12.0) for i in range(12)]

    core.animate_light_pulse(acc, 310.0, 130.0, 12, total_frames)

    for f in core.periodic_frames(total_frames, step=2):
        u = (f - 1) / float(total_frames)
        tp = 2.0 * math.pi
        rotor.rotation_euler.y = -u * tp * 4.0
        rotor.keyframe_insert("rotation_euler", index=1, frame=f)
        pump_cam.rotation_euler.y = u * tp * 12.0
        pump_cam.keyframe_insert("rotation_euler", index=1, frame=f)

        # 12 Hz 3-Phase Build / Hold / Release cycle
        ph = (u * 12.0) % 1.0
        inlet_p.location.z = 0.72 if ph < 0.34 else 0.28
        outlet_p.location.z = 0.22 if ph >= 0.67 else 0.55
        accum_p.location.z = -1.40 if ph >= 0.67 else -1.10
        pad.location.y = -0.34 if ph >= 0.67 else -0.18
        for obj in (inlet_p, outlet_p, accum_p):
            obj.keyframe_insert("location", index=2, frame=f)
        pad.keyframe_insert("location", index=1, frame=f)

        if act == 5:
            wheel_hub.rotation_euler.z = math.sin(u * tp * 2.0) * 0.32
            wheel_hub.keyframe_insert("rotation_euler", index=2, frame=f)

        for bd, off in beads:
            t = (off + u * 3.0) % 1.0
            bd.location = (2.1 - t * 3.9, -0.22, 1.3 + 0.45 * math.sin(t * math.pi))
            bd.keyframe_insert("location", frame=f)

    rigs = {
        (1, 1): ((-0.4, -13.2, 1.4), (1.6, 0.5, 0.35), (-0.2, 0.0, 0.1), 40.0, 3.2),
        (1, 2): ((-5.8, -8.4, -0.4), (1.2, 0.4, 0.25), (-3.4, 0.0, 0.0), 48.0, 2.6),
        (1, 3): ((1.2, -12.8, 3.6), (1.4, 0.5, 0.3), (-0.2, 0.0, 0.2), 38.0, 3.4),
        (2, 1): ((-3.8, -5.6, 0.6), (0.8, 0.3, 0.2), (-3.2, -0.2, 0.2), 58.0, 2.2),
        (2, 2): ((-1.8, -4.8, 0.2), (0.7, 0.25, 0.15), (-2.6, -0.2, 0.2), 62.0, 2.2),
        (2, 3): ((-5.2, -5.2, 1.8), (0.8, 0.35, 0.2), (-3.4, -0.2, 0.2), 54.0, 2.4),
        (3, 1): ((3.3, -6.4, 0.6), (0.9, 0.35, 0.2), (3.3, 0.0, 0.2), 52.0, 2.4),
        (3, 2): ((1.4, -5.4, 1.0), (0.8, 0.3, 0.18), (3.1, 0.0, 0.4), 56.0, 2.2),
        (3, 3): ((5.4, -5.6, -0.4), (0.8, 0.35, 0.2), (3.3, 0.0, -0.3), 54.0, 2.4),
        (4, 1): ((0.2, -9.8, 1.2), (1.2, 0.4, 0.25), (0.2, 0.0, 0.3), 46.0, 2.8),
        (4, 2): ((-1.4, -6.5, 0.8), (0.9, 0.35, 0.2), (-1.4, -0.1, 0.6), 52.0, 2.4),
        (4, 3): ((2.4, -7.2, 0.5), (1.0, 0.4, 0.2), (1.8, 0.0, 0.2), 50.0, 2.5),
        (5, 1): ((-2.8, -9.6, 2.2), (1.3, 0.45, 0.25), (-2.4, 0.0, 0.1), 44.0, 2.8),
        (5, 2): ((0.5, -11.5, 1.8), (1.4, 0.5, 0.3), (-0.4, 0.0, 0.2), 42.0, 3.0),
        (5, 3): ((4.8, -12.2, 2.8), (1.6, 0.55, 0.3), (-0.2, 0.0, 0.1), 38.0, 3.4),
    }
    core.setup_cinema_camera(scene, *rigs[(act, shot)], total_frames)


# ==============================================================================
# EP10 — HYDROELECTRIC DAM, SPIRAL SCROLL CASE & 60-POLE FRANCIS TURBINE
# ==============================================================================
def build_ep10(scene, act, shot, total_frames):
    _, rim, acc = core.setup_studio_lights(floor_z=-4.8, rim_color=(0.08, 0.74, 1.0), accent_color=(1.0, 0.78, 0.24))
    mat_conc = core.make_pbr_mat("DamConcrete", (0.24, 0.27, 0.32, 1.0), metallic=0.06, roughness=0.82, noise_scale=6.0, rough_var=0.14, bump_strength=0.22)
    mat_steel = core.make_pbr_mat("TurbineSteel", (0.64, 0.70, 0.78, 1.0), metallic=0.94, roughness=0.22, noise_scale=22.0, rough_var=0.08, bump_strength=0.05)
    mat_copper = core.make_pbr_mat("StatorCopper", (0.86, 0.40, 0.14, 1.0), metallic=0.95, roughness=0.20, noise_scale=24.0, rough_var=0.08, bump_strength=0.04)
    mat_brass = core.make_pbr_mat("WicketBrass", (0.90, 0.70, 0.22, 1.0), metallic=0.94, roughness=0.22, noise_scale=24.0, rough_var=0.08, bump_strength=0.04)
    mat_water = core.make_glow_mat("HydroWater", (0.08, 0.68, 1.0, 1.0), 6.5, light_tree=False)
    mat_gold = core.make_glow_mat("PowerGold", (1.0, 0.78, 0.22, 1.0), 9.0, light_tree=False)

    # Thick Stepped Concrete Dam Wall (Left) + Deep Pressurized Reservoir
    core.add_box((-5.6, 0.6, -2.2), (4.6, 5.0, 4.8), mat_conc, bevel=0.08)
    core.add_box((-6.2, 0.6, 1.8), (3.0, 5.0, 4.2), mat_conc, bevel=0.08)
    core.mark_fast_glow(core.add_box((-9.4, 0.6, -0.4), (5.0, 4.6, 7.6), mat_water, bevel=0.04))

    # Inclined Steel Penstock + 360-deg Tapering Spiral Scroll Case (Volute)
    core.add_cyl((-2.2, 0.0, -0.4), 0.66, 6.2, mat_steel, rot_deg=(0, 52, 0))
    for i in range(22):
        t = i / 22.0
        a = t * math.pi * 1.85
        r_orb = 2.05 - t * 0.30
        pr = 0.56 * (1.0 - t * 0.55)
        core.add_sphere((2.6 + math.cos(a) * r_orb, math.sin(a) * r_orb, -2.2), pr, mat_water, segs=12, rings=8, fast_glow=True)

    # 20 Synchronized Pivoting Brass Wicket Gates
    gates = []
    for g in range(20):
        a = (g / 20.0) * 2.0 * math.pi
        v = core.add_box((2.6 + math.cos(a) * 1.42, math.sin(a) * 1.42, -2.2), (0.42, 0.09, 0.80), mat_brass, rot_deg=(0, 0, math.degrees(a)), bevel=0.01)
        gates.append((v, a))

    # Vertical Rotating Assembly: Francis Runner + Forged Shaft + 60-Pole Generator Rotor
    rotor = bpy.data.objects.new("HydroRotor", None)
    bpy.context.collection.objects.link(rotor)
    rotor.location = (2.6, 0.0, 0.0)
    core.add_cyl((0, 0, 0.0), 0.30, 6.0, mat_steel).parent = rotor
    core.add_cyl((0, 0, -1.75), 1.05, 0.30, mat_steel).parent = rotor
    for b in range(14):
        a = (b / 14.0) * 2.0 * math.pi
        core.add_box((math.cos(a) * 0.85, math.sin(a) * 0.85, -2.25), (0.50, 0.10, 0.90), mat_steel, rot_deg=(0, 18, math.degrees(a) + 30), bevel=0.01).parent = rotor

    # Giant Multi-Pole Synchronous Generator Rotor (Top) + Stationary Copper Stator Ring
    core.add_cyl((0, 0, 2.0), 1.65, 1.50, mat_steel).parent = rotor
    for p in range(24):
        a = (p / 24.0) * 2.0 * math.pi
        core.mark_fast_glow(core.add_box((math.cos(a) * 1.78, math.sin(a) * 1.78, 2.0), (0.24, 0.22, 1.40), mat_gold if p % 2 == 0 else mat_water, rot_deg=(0, 0, math.degrees(a)), bevel=0.01)).parent = rotor

    core.add_torus((2.6, 0.0, 2.6), 2.28, 0.24, mat_copper, maj_seg=48, min_seg=14)
    core.add_torus((2.6, 0.0, 1.4), 2.28, 0.24, mat_copper, maj_seg=48, min_seg=14)
    core.add_cone((2.6, 0.0, -3.7), 1.75, 1.05, 2.0, mat_water, verts=32)

    # Flowing Penstock Water Beads + 60 Hz AC Grid Output Sine Wave
    wbeads = [(core.add_sphere((-4.5, -0.3, 1.4), 0.18, mat_water, fast_glow=True), i / 14.0) for i in range(14)]
    ac_dots = [(core.add_sphere((5.0 + i * 0.20, 0.0, 2.0), 0.09, mat_gold, fast_glow=True), i) for i in range(24)]

    core.animate_light_pulse(rim, 440.0, 95.0, 4, total_frames)

    for f in core.periodic_frames(total_frames, step=2):
        u = (f - 1) / float(total_frames)
        tp = 2.0 * math.pi
        rotor.rotation_euler.z = u * tp * 3.0
        rotor.keyframe_insert("rotation_euler", index=2, frame=f)

        twist = (0.25 + 0.65 * (0.5 - 0.5 * math.cos(u * tp))) if act == 3 else 0.55
        for v, base_a in gates:
            v.rotation_euler.z = base_a + twist
            v.keyframe_insert("rotation_euler", index=2, frame=f)

        for bd, off in wbeads:
            t = (off + u * 2.0) % 1.0
            bd.location = (-4.6 + t * 5.4, -0.25, 1.5 - t * 3.7)
            bd.keyframe_insert("location", frame=f)

        for d, idx in ac_dots:
            d.location.z = 2.0 + math.sin(idx * 0.42 - u * tp * 6.0) * 0.85
            d.keyframe_insert("location", index=2, frame=f)

    rigs = {
        (1, 1): ((-1.2, -16.5, 2.4), (1.8, 0.6, 0.35), (-1.0, 0.0, -0.3), 38.0, 3.4),
        (1, 2): ((5.4, -13.2, 0.8), (1.5, 0.5, 0.3), (0.2, 0.0, -0.5), 42.0, 3.0),
        (1, 3): ((-6.2, -14.8, 4.2), (1.6, 0.6, 0.3), (-1.2, 0.0, -0.2), 38.0, 3.4),
        (2, 1): ((-1.4, -9.8, 0.6), (1.2, 0.4, 0.25), (-0.4, 0.0, -1.2), 48.0, 2.6),
        (2, 2): ((2.6, -7.8, -0.2), (1.0, 0.4, 0.2), (2.4, 0.0, -2.2), 52.0, 2.4),
        (2, 3): ((-4.0, -8.5, 1.8), (1.1, 0.4, 0.2), (-1.8, 0.0, -0.6), 48.0, 2.6),
        (3, 1): ((2.6, -6.8, -0.6), (1.0, 0.35, 0.2), (2.6, 0.0, -2.2), 54.0, 2.4),
        (3, 2): ((5.6, -5.8, -1.4), (0.8, 0.3, 0.18), (2.6, 0.0, -2.2), 58.0, 2.2),
        (3, 3): ((0.4, -6.4, -0.2), (0.9, 0.35, 0.2), (2.6, 0.0, -2.2), 52.0, 2.4),
        (4, 1): ((3.4, -7.4, 2.4), (1.0, 0.4, 0.2), (3.0, 0.0, 2.0), 50.0, 2.5),
        (4, 2): ((6.8, -6.2, 2.0), (0.9, 0.35, 0.2), (3.4, 0.0, 2.0), 54.0, 2.4),
        (4, 3): ((0.8, -7.2, 3.6), (1.0, 0.4, 0.2), (2.8, 0.0, 2.0), 48.0, 2.6),
        (5, 1): ((5.4, -10.8, 2.8), (1.4, 0.5, 0.25), (3.4, 0.0, 1.2), 44.0, 2.8),
        (5, 2): ((2.2, -13.5, 1.8), (1.5, 0.55, 0.3), (1.4, 0.0, 0.4), 40.0, 3.0),
        (5, 3): ((7.8, -14.8, 3.2), (1.8, 0.6, 0.35), (0.6, 0.0, 0.2), 38.0, 3.4),
    }
    core.setup_cinema_camera(scene, *rigs[(act, shot)], total_frames)


BUILDERS = {
    "ep01": build_ep01,
    "ep02": build_ep02,
    "ep03": build_ep03,
    "ep04": build_ep04,
    "ep05": build_ep05,
    "ep06": build_ep06,
    "ep07": build_ep07,
    "ep08": build_ep08,
    "ep09": build_ep09,
    "ep10": build_ep10,
}


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    core.clear_scene()
    scene = bpy.context.scene
    world_col = (0.022, 0.010, 0.008, 1.0) if args.episode == "ep04" else (0.008, 0.015, 0.030, 1.0)
    core.setup_render_v5(scene, fps=args.fps, samples=args.samples, world_color=world_col)
    configure_compute_device(scene, args.device)
    BUILDERS[args.episode](scene, args.act, args.shot, args.frames)

    scene.frame_start = 1
    scene.frame_end = args.frames
    scene.render.filepath = os.path.join(args.out, "frame_")
    print(f"[v5 Blender PBR 720p] {args.episode.upper()} Act {args.act} Shot {args.shot} | {args.frames}f @ {args.fps}fps | {args.samples} samples + OIDN")
    bpy.ops.render.render(animation=True)
    print(f"[DONE] {args.episode.upper()} Act {args.act} Shot {args.shot}")


if __name__ == "__main__":
    main()
