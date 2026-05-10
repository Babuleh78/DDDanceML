import json
import cv2
import glm
import math
import matplotlib
import os
from .mixamo import Mixamo
from .model_node import ModelNode, json_to_glm_vec, json_to_glm_quat, calc_transform
from .shashura import ShashuraFilter
from scipy.signal import savgol_filter as _savgol
import copy
import numpy as np

_NORMAL_WINDOW   = 7
_EXTREMITY_WINDOW = 15
 

_EXTREMITY_INDICES = {
    8,
    9,
    10,
    11,
    14,
    15,
    16,
    17,
    20,
    21,
    24,
    25,
}
 

try:
    from .mediapipe_debugger import log_raw_pose, log_glm_list, save_debug_data
except ImportError:
    def log_raw_pose(*args, **kwargs): pass
    def log_glm_list(*args, **kwargs): pass
    def save_debug_data(*args, **kwargs): pass

def smooth_pose(shashura_filter, pose, time):
    t = np.ones_like(pose) * time
    return shashura_filter(t, pose)

def get_3d_len(left):
    return math.sqrt((left["x"])**2 + (left["y"])**2 + (left["z"])**2)


def set_axes(ax, azim=10, elev=10, xrange=1.0, yrange=1.0, zrange=1.0):
    ax.set_xlabel("Z")
    ax.set_ylabel("X")
    ax.set_zlabel("Y")
    ax.set_title('Vector')
    if xrange > 0.0:
        ax.set_xlim(-xrange, xrange)
        ax.set_ylim(-yrange, yrange)
        ax.set_zlim(-zrange, zrange)
    ax.view_init(elev=elev, azim=azim)


def get_dot(vec_list, group_lists):
    dots = []
    for group_list in group_lists:
        dot_group = {
            'x': [],
            'y': [],
            'z': []
        }
        for idx in group_list:
            dot_group['x'].append(vec_list[idx][2])
            dot_group['y'].append(vec_list[idx][0])
            dot_group['z'].append(vec_list[idx][1])
        dots.append(dot_group)
    return dots


def draw_list2(fig, vec_list=[], group_lists=[[]], azim=10, range=1.0):
    ax1 = matplotlib.pyplot.axes(projection='3d')
    set_axes(ax1, elev=10, azim=azim, xrange=range, yrange=range, zrange=range)
    dots = get_dot(vec_list, group_lists)
    for dot in dots:
        ax1.plot(dot['x'], dot['y'], dot['z'], marker='o')

    fig.canvas.draw()


def draw_list3(fig, vec_list=[],vec2_list=[], group_lists=[[]], azim=10, range=1.0):
    ax1 =  matplotlib.pyplot.axes(projection='3d')
    set_axes(ax1, elev=10, azim=azim, xrange=range, yrange=range, zrange=range)
    dots = get_dot(vec_list, group_lists)
    for dot in dots:
        ax1.plot(dot['x'], dot['y'], dot['z'], marker='.')

    dots2 = get_dot(vec2_list, group_lists)
    for dot in dots2:
        ax1.plot(dot['x'], dot['y'], dot['z'], marker='+', color='r')

    fig.canvas.draw()


def find_bones(bones, name):
    for bone in bones:
        idx = bone["name"].find(":")
        bone_name = bone["name"][idx+1:]
        if bone_name == name:
            return bone
    return None

def find_model_json(model_json, name):
    idx = model_json["name"].find(":")
    model_name = model_json["name"][idx+1:]
    
    if  model_name == name:
        return [True, model_json]
    else:
        for child in model_json["child"]:
            is_find, result = find_model_json(child, name)
            if is_find:
                return [is_find, result]
        return [False, None]


def get_name_idx_map():
    mediapipe_names = [
        "nose",
        "left_eye_inner",
        "left_eye", 
        "left_eye_outer",
        "right_eye_inner", 
        "right_eye", 
        "right_eye_outer", 
        "left_ear", 
        "right_ear", 
        "mouth_left",
        "mouth_right", 
        "left_shoulder", 
        "right_shoulder",
        "left_elbow", 
        "right_elbow", 
        "left_wrist", 
        "right_wrist", 
        "left_pinky",
        "right_pinky", 
        "left_index", 
        "right_index", 
        "left_thumb",
        "right_thumb", 
        "left_hip", 
        "right_hip", 
        "left_knee", 
        "right_knee", 
        "left_ankle", 
        "right_ankle", 
        "left_heel", 
        "right_heel", 
        "left_foot_index", 
        "right_foot_index"]

    name_idx_map = {}
    for idx in range(0, len(mediapipe_names)):
        name_idx_map[mediapipe_names[idx]] = idx
    return name_idx_map


def get_mixamo_names():
    return [
        ['Hips', 0, -1],  
        ['Spine', 1, 0],
        ['Spine1', 2, 1],
        ['Spine2', 3, 2],

        ['Neck', 4, 3], 
        ['Head', 5, 4],  

        ['LeftArm', 6, 3, "left_shoulder"],
        ['LeftForeArm', 7, 6, "left_elbow"],
        ['LeftHand', 8, 7, "left_wrist"],
        ['LeftHandThumb1', 9, 8, "left_thumb"],
        ['LeftHandIndex1', 10, 8, "left_index"],
        ['LeftHandPinky1', 11, 8, "left_pinky"],

        ['RightArm', 12, 3, "right_shoulder"],
        ['RightForeArm', 13, 12, "right_elbow"],
        ['RightHand', 14, 13, "right_wrist"],
        ['RightHandThumb1', 15, 14, "right_thumb"],
        ['RightHandIndex1', 16, 14, "right_index"],
        ['RightHandPinky1', 17, 14, "right_pinky"],

        ['LeftUpLeg', 18, 0, "left_hip"],
        ['LeftLeg', 19, 18, "left_knee"],
        ['LeftFoot', 20, 19, "left_ankle"],
        ['LeftToeBase', 21, 20, "left_foot_index"],

        ['RightUpLeg', 22, 0, "right_hip"],
        ['RightLeg', 23, 22, "right_knee"],
        ['RightFoot', 24, 23, "right_ankle"],
        ['RightToeBase', 25, 24, "right_foot_index"]
    ]


def get_mixamo_name_idx_map():
    mixamo_names = get_mixamo_names()
    mixamo_name_idx_map = {}
    for name in mixamo_names:
        mixamo_name_idx_map[name[0]] = name[1]
    return mixamo_name_idx_map

def get_mixamo_name_mediapipe_name_map():
    mm_name_mp_name_map = {}
    mixamo_names = get_mixamo_names()
    for idx in range(6, len(mixamo_names)):
        mm_name_mp_name_map[mixamo_names[idx][0]] = mixamo_names[idx][3]
    return mm_name_mp_name_map

def init_bindpose(bindpose_json, model_json):
    name = model_json["name"]
    position = json_to_glm_vec(model_json["position"])
    rotate = json_to_glm_quat(model_json["rotation"])
    scale = json_to_glm_vec(model_json["scale"])
    transform = np.array(calc_transform(position, rotate, scale))
    bindpose_json[name] = transform.flatten().tolist()
    childlist = model_json["child"]
    for child in childlist:
        init_bindpose(bindpose_json, child)

def mediapipe_to_mixamo(mp_manager,
                        mixamo_dict_string,
                        video_path):
    
    mm_name_idx_map = get_mixamo_name_idx_map()
    mixamo_json = None
    
    mixamo_json = json.loads(mixamo_dict_string)
    is_find, hip_node = find_model_json(mixamo_json["node"], Mixamo.Hips.name)
    if not is_find:
        return [False, None]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if not fps or fps <= 0:
        fps = 30.0
    
    time_factor = 1.0
    if mp_manager.fps > 0:
        time_factor = mp_manager.fps / fps
        fps = mp_manager.fps

    anim_result_json = {
        "fileName": os.path.basename(video_path),
        "duration": 0,
        "width":  cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "ticksPerSecond": fps,
        "bindpose": {},
        "frames": []
    }
    init_bindpose(anim_result_json["bindpose"], hip_node)

    try:
        root_node = ModelNode()
        root_node.set_mixamo(hip_node, mm_name_idx_map)
        root_node.normalize_spine()

        mediapipe_to_mixamo2(mp_manager,
                             anim_result_json,
                             cap,
                             mixamo_json,
                             root_node,
                             time_factor)
        
        if anim_result_json["frames"]:
            anim_result_json["duration"] = anim_result_json["frames"][-1]["time"]

    except ZeroDivisionError as e:
        if cap.isOpened():
            cap.release()
        return [False, None]
    except Exception as e:
        if cap.isOpened():
            cap.release()
        return [False, None]
        
    if cap.isOpened():
        cap.release()
    return [True, anim_result_json]

def glm_list_to_numpy(glm_list):
    result = np.zeros((len(glm_list), 3), dtype=np.float32)
    for i, v in enumerate(glm_list):
        if v is not None:
            result[i] = [v.x, v.y, v.z]
    return result


def numpy_to_glm_list(array, original_glm_list):
    result = list(original_glm_list)
    for i, v in enumerate(original_glm_list):
        if v is not None:
            result[i] = glm.vec3(float(array[i, 0]),
                                 float(array[i, 1]),
                                 float(array[i, 2]))
    return result

def mediapipe_to_mixamo2(mp_manager,
                         anim_result_json,
                         cap,
                         mixamo_bindingpose_json,
                         mixamo_bindingpose_root_node,
                         time_factor):
    mp_name_idx_map = get_name_idx_map()
    mm_mp_map = get_mixamo_name_mediapipe_name_map()
    mm_name_idx_map = get_mixamo_name_idx_map()
    mp_idx_mm_idx_map = dict()
    for mm_name in mm_mp_map.keys():
        mp_name = mm_mp_map[mm_name]
        mp_idx = mp_name_idx_map[mp_name]
        mm_idx = mm_name_idx_map[mm_name]
        mp_idx_mm_idx_map[mp_idx] = mm_idx

    _, model_right_up_leg = find_model_json(mixamo_bindingpose_json["node"], Mixamo.RightUpLeg.name)
    __, model_right_leg = find_model_json(mixamo_bindingpose_json["node"], Mixamo.RightLeg.name)

    model_scale = 100
    if _ != False:
        model_scale = get_3d_len(model_right_up_leg["position"])*2.0
    model_scale2 =1.0
    if __ !=False:
        model_scale2 = get_3d_len(model_right_leg["position"])
    hip_move_list = []
    origin = None
    factor = 0.0
    factor_list = []

    hip_world_origin = None

    original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = original_width
    height = original_height

    frame_num = -1
    matplotlib.pyplot.ion()
    matplotlib.pyplot.close()
    fig = None
    if mp_manager.is_show_result:
        fig = matplotlib.pyplot.figure()
        matplotlib.pyplot.show()

    try:
        max_frame_num = mp_manager.max_frame_num
        is_show_result = mp_manager.is_show_result
        min_visibility = mp_manager.min_visibility
        is_hips_move = mp_manager.is_hips_move
        shashura_filter = ShashuraFilter(min_cutoff=1.0, beta=0.1)

        hip_world_origin = None
        hips_scale_origin = None
        while cap.isOpened():

            success, cap_image = cap.read()
            frame_num += 1
            if not success or max_frame_num < frame_num:
                break
            height1, width1, _ = cap_image.shape
            cap_image = cv2.resize(
                cap_image, (int(width1 * (640 / height1)), 640))
            height2, width2, _ = cap_image.shape
            cap_image, glm_list, visibility_list, hip2d_left, hip2d_right, leg2d = detect_pose_to_glm_pose(
                mp_manager, cap_image, mp_idx_mm_idx_map)
            
            log_glm_list(frame_num, glm_list, visibility_list)
            
            if glm_list[0] != None:
                time =  math.floor(frame_num*time_factor)
                pose_array = glm_list_to_numpy(glm_list)
                t = frame_num / mp_manager.fps
                filtered_array = shashura_filter(
                    np.ones_like(pose_array) * t,
                    pose_array
                )
                glm_list = numpy_to_glm_list(filtered_array, glm_list)
                    
                bones_json = {
                    "time": time,
                    "bones": []
                }
                mixamo_bindingpose_root_node.normalize(glm_list)
                mixamo_bindingpose_root_node.calc_animation(glm_list)
                mixamo_bindingpose_root_node.tmp_to_json(bones_json, visibility_list, min_visibility)
                raw_hip = glm_list[Mixamo.Hips]  
                if hip_world_origin is None:
                    hip_world_origin = glm.vec3(raw_hip.x, raw_hip.y, raw_hip.z)

                hips_bone = find_bones(bones_json["bones"], Mixamo.Hips.name)
               
                bones_json["landmarks"] = [
                {
                    "x": float(glm_list[i].x),
                    "y": float(glm_list[i].y),
                    "z": float(glm_list[i].z),
                    "visibility": float(visibility_list[i]) if visibility_list[i] is not None else 0.0
                }
                if glm_list[i] is not None else None
                    for i in range(len(glm_list))
                ]
                anim_result_json["frames"].append(bones_json)
                if is_show_result:
                    rg = []
                    rv = [None] * len(glm_list)
                    mixamo_bindingpose_root_node.get_vec_and_group_list(
                        rv, rg, is_apply_animation_transform=True)
                    matplotlib.pyplot.clf()
                    draw_list3(fig,rv, glm_list, rg)
                if is_hips_move:
                    scale_factor_x = original_width / width2
                    scale_factor_y = original_height / height2
                    hip2d_left.x *= original_width * scale_factor_x
                    hip2d_left.y *= original_height * scale_factor_y
                    hip2d_left.z *= original_width * scale_factor_x
                    hip2d_right.x *= original_width * scale_factor_x
                    hip2d_right.y *= original_height * scale_factor_y
                    hip2d_right.z *= original_width * scale_factor_x
                    leg2d.x *= original_width * scale_factor_x
                    leg2d.y *= original_height * scale_factor_y
                    leg2d.z *= original_width * scale_factor_x

                    if origin == None:
                        origin = avg_vec3(hip2d_left, hip2d_right)
                        hips_scale_origin = glm.distance(hip2d_left, hip2d_right)
                    else:
                        hips2d_scale = glm.distance(hip2d_left, hip2d_right)
                        leg2d_scale = glm.distance(leg2d, hip2d_right)
                        factor_list.append(model_scale2/leg2d_scale)
                        factor = max(factor, model_scale/hips2d_scale)
                        hip_move_list.append([
                            len(anim_result_json["frames"]) - 1,
                            avg_vec3(hip2d_left, hip2d_right),
                            hips2d_scale
    ])

            key = cv2.waitKey(5)
            if key & 0xFF == 27:
                break
        factor_list.sort()
        if len(factor_list) > 0:
            factor_list_avg = sum(factor_list) / len(factor_list)
            idx_80 = min(int(len(factor_list) * 0.8), len(factor_list) - 1)
            factor_list_avg = max(factor_list_avg, factor_list[idx_80])
            factor = max(factor, factor_list_avg)
        else:
            factor = 1.0 if factor <= 0 else factor
            
        if mp_manager.factor != 0.0:
           factor = mp_manager.factor
        
        for hips_bone in hip_move_list:
            set_hips_position(
                find_bones(anim_result_json["frames"][hips_bone[0]]["bones"], 
                        Mixamo.Hips.name)["position"],
                origin,
                hips_bone[1],
                factor,
                hips_scale_origin=hips_scale_origin,
                current_scale=glm.length(glm.vec3(
                    hips_bone[1].x - origin.x,
                    hips_bone[1].y - origin.y,
                    hips_bone[1].z - origin.z
                )) if len(hips_bone) < 3 else hips_bone[2],
                video_width=original_width,
                video_height=original_height
            )
        if anim_result_json["frames"][0]["time"] != 0.0:
            tmp_json = copy.deepcopy(anim_result_json["frames"][0])
            tmp_json["time"] = 0.0
            anim_result_json["frames"].append(tmp_json)

        from app.services.py_module.mp_helper import postprocess_frames
        anim_result_json["frames"] = postprocess_frames(
            anim_result_json["frames"],
            min_visibility=min_visibility,
        )
        
        cap.release()
        cv2.destroyAllWindows()
        
        save_debug_data()

    except Exception as e:
        if cap.isOpened():
            cap.release()
            cv2.destroyAllWindows()


def detect_pose_to_glm_pose(mp_manager, image, mp_idx_mm_idx_map):
    output_image = image.copy()

    image.flags.writeable = False

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = mp_manager.get_pose().process(image_rgb)

    image.flags.writeable = True

    glm_list = [None]*26
    visibility_list = [None]*26
    hip2d_left, hip2d_right = glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 0.0, 0.0)

    if results.pose_world_landmarks:
        landmark = results.pose_world_landmarks.landmark

        glm_list[Mixamo.Hips] = glm.vec3(
            (landmark[mp_manager.mp_pose.PoseLandmark.LEFT_HIP].x + 
            landmark[mp_manager.mp_pose.PoseLandmark.RIGHT_HIP].x) * 0.5,
            -(landmark[mp_manager.mp_pose.PoseLandmark.LEFT_HIP].y + 
            landmark[mp_manager.mp_pose.PoseLandmark.RIGHT_HIP].y) * 0.5,
            -(landmark[mp_manager.mp_pose.PoseLandmark.LEFT_HIP].z + 
            landmark[mp_manager.mp_pose.PoseLandmark.RIGHT_HIP].z) * 0.5,
        )
        visibility_list[Mixamo.Hips] = (landmark[mp_manager.mp_pose.PoseLandmark.LEFT_HIP].visibility +
                                        landmark[mp_manager.mp_pose.PoseLandmark.RIGHT_HIP].visibility)*0.5
        glm_list[Mixamo.Neck] = avg_vec3(
            landmark[mp_manager.mp_pose.PoseLandmark.LEFT_SHOULDER], landmark[mp_manager.mp_pose.PoseLandmark.RIGHT_SHOULDER])
        visibility_list[Mixamo.Neck] = (landmark[mp_manager.mp_pose.PoseLandmark.LEFT_SHOULDER].visibility +
                                        landmark[mp_manager.mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility)*0.5
        glm_list[Mixamo.Spine1] = avg_vec3(
            glm_list[Mixamo.Hips], glm_list[Mixamo.Neck])
        visibility_list[Mixamo.Spine1] = (
            visibility_list[Mixamo.Hips] + visibility_list[Mixamo.Neck])*0.5
        glm_list[Mixamo.Spine] = avg_vec3(
            glm_list[Mixamo.Hips], glm_list[Mixamo.Spine1])
        visibility_list[Mixamo.Spine] = (
            visibility_list[Mixamo.Hips] + visibility_list[Mixamo.Spine1])*0.5
        glm_list[Mixamo.Spine2] = avg_vec3(
            glm_list[Mixamo.Spine1], glm_list[Mixamo.Neck])
        visibility_list[Mixamo.Spine2] = (
            visibility_list[Mixamo.Spine1] + visibility_list[Mixamo.Neck])*0.5
        glm_list[Mixamo.Head] = avg_vec3(
            landmark[mp_manager.mp_pose.PoseLandmark.LEFT_EAR], landmark[mp_manager.mp_pose.PoseLandmark.RIGHT_EAR])
        visibility_list[Mixamo.Head] = (landmark[mp_manager.mp_pose.PoseLandmark.LEFT_EAR].visibility +
                                        landmark[mp_manager.mp_pose.PoseLandmark.RIGHT_EAR].visibility)*0.5

        glm_list[Mixamo.Spine].y *= -1
        glm_list[Mixamo.Neck].y *= -1
        glm_list[Mixamo.Spine1].y *= -1
        glm_list[Mixamo.Spine2].y *= -1
        glm_list[Mixamo.Head].y *= -1

        glm_list[Mixamo.Neck].z *= -1
        glm_list[Mixamo.Spine].z *= -1
        glm_list[Mixamo.Spine1].z *= -1
        glm_list[Mixamo.Spine2].z *= -1
        glm_list[Mixamo.Head].z *= -1
        for mp_idx in mp_idx_mm_idx_map.keys():
            mm_idx = mp_idx_mm_idx_map[mp_idx]
            glm_list[mm_idx] = glm.vec3(
                landmark[mp_idx].x, -landmark[mp_idx].y, -landmark[mp_idx].z)
            visibility_list[mm_idx] = landmark[mp_idx].visibility

    leg2d = None
    if results.pose_landmarks:
        landmark = results.pose_landmarks.landmark
        hip2d_left.x = landmark[mp_manager.mp_pose.PoseLandmark.LEFT_HIP].x
        hip2d_left.y = landmark[mp_manager.mp_pose.PoseLandmark.LEFT_HIP].y
        hip2d_left.z = landmark[mp_manager.mp_pose.PoseLandmark.LEFT_HIP].z
        hip2d_right = glm.vec3(landmark[mp_manager.mp_pose.PoseLandmark.RIGHT_HIP].x,
                               landmark[mp_manager.mp_pose.PoseLandmark.RIGHT_HIP].y, landmark[mp_manager.mp_pose.PoseLandmark.RIGHT_HIP].z)
        leg2d = glm.vec3(landmark[26].x, landmark[26].y, landmark[26].z)
        

    mp_manager.mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                         connections=mp_manager.mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_manager.mp_drawing_styles.get_default_pose_landmarks_style())

    return output_image, glm_list, visibility_list, hip2d_left, hip2d_right, leg2d


def avg_vec3(v1, v2):
    v3 = glm.vec3((v1.x + v2.x) * 0.5,
                  (v1.y + v2.y) * 0.5,
                  (v1.z + v2.z) * 0.5)
    return v3



def set_hips_position(hips_bone_json, origin_hips, current_hips,
                      factor, hips_scale_origin=None, current_scale=None,
                      video_width=None, video_height=None):
    x = (current_hips.x - origin_hips.x) * factor
    y = (current_hips.y - origin_hips.y) * factor
 
    if (hips_scale_origin is not None
            and current_scale is not None
            and hips_scale_origin > 1e-6
            and current_scale > 1e-6):
        scale_ratio = (current_scale / hips_scale_origin) - 1.0
        Z_SCALE = factor * hips_scale_origin * 0.5 
        z_raw = scale_ratio * Z_SCALE
        Z_MAX = 3.0 * factor
        z = float(np.clip(z_raw, -Z_MAX, Z_MAX))
    else:
        z = 0.0 
 
    if video_width is not None and video_height is not None:
        x_max = video_width * 0.4
        y_max = video_height * 0.3
        x = np.clip(x, -x_max, x_max)
        y = np.clip(y, -y_max, y_max)
    
    hips_bone_json["x"] = x
    hips_bone_json["y"] = -y
    hips_bone_json["z"] = z




def _interpolate_invisible(positions: np.ndarray,
                            visibility: np.ndarray,
                            min_vis: float = 0.5) -> np.ndarray:
    result = positions.copy()
    N = len(positions)
    bad = visibility < min_vis
 
    if not bad.any():
        return result
 
    good_indices = np.where(~bad)[0]
    if len(good_indices) == 0:
        return result
 
    for i in np.where(bad)[0]:
        left_candidates  = good_indices[good_indices < i]
        right_candidates = good_indices[good_indices > i]
 
        if len(left_candidates) == 0:
            result[i] = result[right_candidates[0]]
        elif len(right_candidates) == 0:
            result[i] = result[left_candidates[-1]]
        else:
            l_idx = left_candidates[-1]
            r_idx = right_candidates[0]
            alpha = (i - l_idx) / (r_idx - l_idx)
            result[i] = (1 - alpha) * result[l_idx] + alpha * result[r_idx]
 
    return result


def _smooth_bone(positions: np.ndarray, bone_idx: int, fps: float) -> np.ndarray:
    N = len(positions)
    if N < 5:
        return positions
 
    if bone_idx in _EXTREMITY_INDICES:
        window = min(_EXTREMITY_WINDOW, N if N % 2 == 1 else N - 1)
    else:
        window = min(_NORMAL_WINDOW, N if N % 2 == 1 else N - 1)
 
    window = window if window % 2 == 1 else window - 1
    window = max(window, 5)
 
    try:
        return _savgol(positions, window_length=window, polyorder=2, axis=0)
    except Exception:
        return positions
    

def postprocess_frames(frames: list, min_visibility: float = 0.5) -> list:
    if not frames:
        return frames
 
    N = len(frames)
    n_bones = len(frames[0].get("landmarks", []) or [])
    if n_bones == 0:
        return frames
 
    positions   = np.zeros((N, n_bones, 3),  dtype=np.float32)
    visibilities = np.zeros((N, n_bones),    dtype=np.float32)
 
    for fi, frame in enumerate(frames):
        lms = frame.get("landmarks") or []
        for bi, lm in enumerate(lms):
            if lm is not None:
                positions[fi, bi]    = [lm["x"], lm["y"], lm["z"]]
                visibilities[fi, bi] = lm.get("visibility", 1.0)
 
    for bi in range(n_bones):
        positions[:, bi, :] = _interpolate_invisible(
            positions[:, bi, :], visibilities[:, bi], min_visibility
        )
 
    fps = 30.0
    for bi in range(n_bones):
        positions[:, bi, :] = _smooth_bone(positions[:, bi, :], bi, fps)
 
    for fi, frame in enumerate(frames):
        lms = frame.get("landmarks") or []
        for bi in range(min(n_bones, len(lms))):
            if lms[bi] is not None:
                lms[bi]["x"] = float(positions[fi, bi, 0])
                lms[bi]["y"] = float(positions[fi, bi, 1])
                lms[bi]["z"] = float(positions[fi, bi, 2])
            else:
                lms[bi] = {
                    "x": float(positions[fi, bi, 0]),
                    "y": float(positions[fi, bi, 1]),
                    "z": float(positions[fi, bi, 2]),
                    "visibility": float(visibilities[fi, bi]),
                }
 
    return frames