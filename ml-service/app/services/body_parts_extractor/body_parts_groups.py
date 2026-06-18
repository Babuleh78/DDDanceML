
BONE_NAMES = {
    0: "Hips",
    1: "Spine",
    2: "Spine1",
    3: "Spine2",
    4: "Neck",
    5: "Head",
    6: "LeftArm",
    7: "LeftForeArm",
    8: "LeftHand",
    9: "LeftHandThumb1",
    10: "LeftHandIndex1",
    11: "LeftHandPinky1",
    12: "RightArm",
    13: "RightForeArm",
    14: "RightHand",
    15: "RightHandThumb1",
    16: "RightHandIndex1",
    17: "RightHandPinky1",
    18: "LeftUpLeg",
    19: "LeftLeg",
    20: "LeftFoot",
    21: "LeftToeBase",
    22: "RightUpLeg",
    23: "RightLeg",
    24: "RightFoot",
    25: "RightToeBase",
}

BODY_PARTS_GROUPS = {
    "head": {
        "bones": [4, 5],
        "name": "Голова",
        "description": "Голова и шея"
    },
    "torso": {
        "bones": [0, 1, 2, 3],
        "name": "Торс",
        "description": "Туловище (бедра, позвоночник)"
    },
    "left_arm": {
        "bones": [6, 7, 8],
        "name": "Левая рука",
        "description": "Левая рука и предплечье"
    },
    "left_hand": {
        "bones": [9, 10, 11],
        "name": "Левая кисть",
        "description": "Пальцы левой кисти"
    },
    "right_arm": {
        "bones": [12, 13, 14],
        "name": "Правая рука",
        "description": "Правая рука и предплечье"
    },
    "right_hand": {
        "bones": [15, 16, 17],
        "name": "Правая кисть",
        "description": "Пальцы правой кисти"
    },
    "left_leg": {
        "bones": [18, 19, 20, 21],
        "name": "Левая нога",
        "description": "Левая нога (бедро, голень, стопа)"
    },
    "right_leg": {
        "bones": [22, 23, 24, 25],
        "name": "Правая нога",
        "description": "Правая нога (бедро, голень, стопа)"
    },
}

JOINT_ANGLES = {
    "left_elbow": {
        "points": (6, 7, 8),
        "name": "Угол левого локтя",
        "description": "Сгибание левой руки в локте"
    },
    "right_elbow": {
        "points": (12, 13, 14),
        "name": "Угол правого локтя",
        "description": "Сгибание правой руки в локте"
    },
    "left_knee": {
        "points": (18, 19, 20),
        "name": "Угол левого колена",
        "description": "Сгибание левой ноги в колене"
    },
    "right_knee": {
        "points": (22, 23, 24),
        "name": "Угол правого колена",
        "description": "Сгибание правой ноги в колене"
    },
    "left_shoulder": {
        "points": (3, 6, 7),
        "name": "Угол левого плеча",
        "description": "Подъём левой руки от корпуса"
    },
    "right_shoulder": {
        "points": (3, 12, 13),
        "name": "Угол правого плеча",
        "description": "Подъём правой руки от корпуса"
    },
    "left_hip_joint": {
        "points": (0, 18, 19),
        "name": "Угол левого бедра",
        "description": "Подъём левой ноги от таза"
    },
    "right_hip_joint": {
        "points": (0, 22, 23),
        "name": "Угол правого бедра",
        "description": "Подъём правой ноги от таза"
    },
    "torso_tilt": {
        "points": (0, 2, 4),
        "name": "Наклон корпуса",
        "description": "Наклон туловища относительно вертикали"
    },
}

SYMMETRY_PAIRS = [
    ("left_arm",  "right_arm"),
    ("left_leg",  "right_leg"),
    ("left_hand", "right_hand"),
]