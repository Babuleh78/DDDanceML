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
    parser.add_argument("--json",       required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--armature",   default="Armature")
    parser.add_argument("--format",     default="GLB", choices=["GLB", "FBX"])
    parser.add_argument("--anim-only",  action="store_true")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Точное количество кадров для экспорта")
    args = parser.parse_args(argv)

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    fps = float(data.get("ticksPerSecond", 30))
    if fps <= 0: fps = 30.0

    frames_data = data.get("frames", [])
    if not frames_data:
        raise ValueError("No frames")

    # Нормализуем time к 0, 1, 2...
    frames_data = [{**f, "time": i} for i, f in enumerate(frames_data)]

    # Используем переданное num_frames или длину массива
    if args.num_frames is not None:
        total_frames = args.num_frames
        log(f"Using explicit num_frames={total_frames}")
    else:
        total_frames = len(frames_data)
        log(f"Using frames array length={total_frames}")

    log(f"fps={fps}, total_frames={total_frames}")

    # Поиск арматуры
    armature_obj = bpy.data.objects.get(args.armature)
    if not armature_obj:
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE':
                armature_obj = obj
                break
    if not armature_obj:
        raise RuntimeError("Armature not found")

    # Очищаем старые экшены
    if armature_obj.animation_data:
        armature_obj.animation_data.action = None
    for old_act in list(bpy.data.actions):
        try:
            bpy.data.actions.remove(old_act, do_unlink=True)
        except:
            pass

    # Жёстко выставляем диапазон — ДО keyframing
    bpy.context.scene.render.fps = int(fps)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = total_frames - 1
    bpy.context.scene.frame_preview_start = 0
    bpy.context.scene.frame_preview_end = total_frames - 1

    log(f"Timeline: 0 -> {total_frames - 1}")

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    action = bpy.data.actions.new(name="ImportedAnimation")
    if armature_obj.animation_data is None:
        armature_obj.animation_data_create()
    armature_obj.animation_data.action = action

    def resolve(raw):
        if not raw: return None
        for v in [raw, raw.replace(":", "_"), raw.split(":")[-1]]:
            if v in armature_obj.pose.bones:
                return v
        return None

    # Keyframing — только кадры в диапазоне [0, total_frames)
    for i, fd in enumerate(frames_data):
        if i >= total_frames:
            break
        frame_idx = i
        bpy.context.scene.frame_set(frame_idx)
        for bd in fd.get("bones", []):
            bn = resolve(bd.get("name", ""))
            if not bn: continue
            pb = armature_obj.pose.bones[bn]
            rot = bd.get("rotation")
            if rot:
                try:
                    q = mathutils.Quaternion((
                        float(rot.get("w", 1)), float(rot.get("x", 0)),
                        float(rot.get("y", 0)), float(rot.get("z", 0))
                    ))
                    pb.rotation_mode = 'QUATERNION'
                    pb.rotation_quaternion = q
                    pb.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)
                except Exception as e:
                    log(f"Rot err {bn}: {e}", "WARNING")
            pos = bd.get("position")
            if pos:
                try:
                    pb.location = mathutils.Vector((
                        float(pos.get("x", 0)),
                        float(pos.get("y", 0)),
                        float(pos.get("z", 0)),
                    ))
                    pb.keyframe_insert(data_path="location", frame=frame_idx)
                except Exception as e:
                    log(f"Pos err {bn}: {e}", "WARNING")
        if (i + 1) % 50 == 0 or (i + 1) == total_frames:
            log(f"  Keyframed {i+1}/{total_frames}")

    # Интерполяция
    try:
        for fc in action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'
            fc.update()
    except Exception as e:
        log(f"Interpolation skipped: {e}", "WARNING")

    bpy.ops.object.mode_set(mode='OBJECT')

    # Очистка сцены для anim_only
    if args.anim_only:
        for obj in list(bpy.data.objects):
            if obj.type in ('MESH', 'CAMERA', 'LIGHT', 'EMPTY', 'CURVE', 'SURFACE', 'FONT'):
                bpy.data.objects.remove(obj, do_unlink=True)
        for old_act in list(bpy.data.actions):
            if old_act != action:
                try:
                    bpy.data.actions.remove(old_act, do_unlink=True)
                except:
                    pass
        try:
            bpy.ops.outliner.orphans_purge(do_recursive=True)
        except:
            pass

    # Экспорт
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    log(f"Exporting {args.format}, anim_only={args.anim_only}, "
        f"frames=0..{total_frames - 1}")

    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    if args.format.upper() == "GLB":
        bpy.ops.export_scene.gltf(
            filepath=str(args.output),
            export_format='GLB',
            export_animations=True,
            export_frame_range=True,      # использует scene.frame_start/end
            export_frame_step=1,
            export_def_bones=True,
            export_optimize_animation_size=True,
            export_force_sampling=True,
            export_apply=False,
            export_nla_strips=False,
            export_anim_slide_to_zero=True,
            export_current_frame=False,
        )
    else:
        bpy.ops.export_scene.fbx(
            filepath=str(args.output),
            use_anim=True,
            bake_anim=True,
            bake_anim_use_all_bones=True,
            bake_anim_step=1,
        )

    p = Path(args.output)
    if p.exists():
        log(f"Done: {p} ({p.stat().st_size / 1024:.1f} KB), "
            f"duration={(total_frames / fps):.2f}s")
    return 0

if __name__ == "__main__":
    sys.exit(main())