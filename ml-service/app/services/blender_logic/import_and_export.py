#!/usr/bin/env python3
import bpy, json, sys, argparse, mathutils, traceback
from pathlib import Path

def log(msg, level="INFO"):
    print(f"[{level}] {msg}", flush=True)
    if level == "ERROR":
        print(f"[{level}] {msg}", file=sys.stderr, flush=True)

def main():
    try:
        return _run()
    except Exception as e:
        log(f"CRITICAL: {type(e).__name__}: {e}", "ERROR")
        for line in traceback.format_exc().splitlines():
            log(line, "ERROR")
        return 1

def _run():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",     required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--armature", default="Armature")
    parser.add_argument("--format",   default="GLB", choices=["GLB", "FBX"])
    args = parser.parse_args(argv)

    log(f"Starting: {args.json} -> {args.output}")

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    fps = float(data.get("ticksPerSecond", 30))
    if fps <= 0: fps = 30.0
    frames_data = data.get("frames", [])
    total_frames = len(frames_data)
    if not total_frames:
        raise ValueError("No frames")

    log(f"fps={fps}, frames={total_frames}")

    armature_obj = bpy.data.objects.get(args.armature)
    if not armature_obj:
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE':
                armature_obj = obj
                break
    if not armature_obj:
        raise RuntimeError("Armature not found")

    bpy.context.scene.render.fps = int(fps)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = max(0, total_frames - 1)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    action = bpy.data.actions.new(name="ImportedAnimation")
    if armature_obj.animation_data is None:
        armature_obj.animation_data_create()
    armature_obj.animation_data.action = action

    def resolve(raw):
        for v in [raw, raw.replace(":", "_"), raw.split(":")[-1]]:
            if v in armature_obj.pose.bones:
                return v
        return None

    for i, fd in enumerate(frames_data):
        frame_idx = int(fd.get("time", i))
        bpy.context.scene.frame_set(frame_idx)
        for bd in fd.get("bones", []):
            bn = resolve(bd.get("name", ""))
            if not bn: continue
            pb = armature_obj.pose.bones[bn]
            rot = bd.get("rotation")
            if rot:
                try:
                    q = mathutils.Quaternion((float(rot.get("w",1)), float(rot.get("x",0)), float(rot.get("y",0)), float(rot.get("z",0))))
                    pb.rotation_mode = 'QUATERNION'
                    pb.rotation_quaternion = q
                    pb.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)
                except: pass
            pos = bd.get("position")
            if pos:
                try:
                    px,py,pz = float(pos.get("x",0)), float(pos.get("y",0)), float(pos.get("z",0))
                    if px or py or pz:
                        pb.location = mathutils.Vector((px,py,pz))
                        pb.keyframe_insert(data_path="location", frame=frame_idx)
                except: pass
        if (i+1) % 50 == 0 or (i+1) == total_frames:
            log(f"  {i+1}/{total_frames}")

    try:
        for layer in action.layers:
            for strip in layer.strips:
                for cb in strip.channelbags:
                    for fc in cb.fcurves:
                        for kp in fc.keyframe_points:
                            kp.interpolation = 'SINE'
                        fc.update()
    except AttributeError:
        for fc in action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'SINE'
            fc.update()

    try:
        bpy.ops.graph.select_all(action='SELECT')
        bpy.ops.graph.smooth()
    except Exception as e:
        log(f"fcurve smooth skipped: {e}", "WARNING")

    for old in list(bpy.data.actions):
        if old == action: continue
        if any(x in old.name.lower() for x in ["mixamo", "imported"]):
            try: bpy.data.actions.remove(old)
            except: pass

    bpy.ops.object.mode_set(mode='OBJECT')

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    log(f"Exporting {args.format}...")

    if args.format.upper() == "GLB":
        bpy.ops.export_scene.gltf(
            filepath=str(args.output), export_format='GLB',
            export_animations=True, export_frame_range=True,
            export_frame_step=1, export_apply=False,
            export_def_bones=True, export_optimize_animation_size=True,
            export_force_sampling=True,
        )
    else:
        bpy.ops.export_scene.fbx(
            filepath=str(args.output), use_anim=True,
            bake_anim=True, bake_anim_use_all_bones=True, bake_anim_step=1,
        )

    p = Path(args.output)
    if p.exists():
        log(f"Done: {p} ({p.stat().st_size/1024/1024:.2f} MB)")
    return 0

if __name__ == "__main__":
    sys.exit(main())