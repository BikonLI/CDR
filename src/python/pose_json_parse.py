import json
from pathlib import Path
import os


def get_keypoint(path):
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            try:
                points_conf = data["people"][0]["pose_keypoints_2d"]
            except Exception:
                return None

        except Exception as e:
            return None
        
    points = [points_conf[i:i+3] for i in range(0, len(points_conf) - 3, 3)]
    
    rsholder = points[2][:2]
    lsholder = points[5][:2]
    rhip = points[9][:2]
    lhip = points[12][:2]
    mt = points[1][:2]
    md = points[8][:2]
    
    rhand = points[4][:2]
    lhand = points[7][:2]
    
    head = points[17][:2]
    head = points[18][:2]
    
    return {
        "rsholder": rsholder,
        "lsholder": lsholder,
        "rhip": rhip,
        "lhip": lhip,
        "mt": mt,
        "md": md,
        "rhand": rhand,
        "lhand": lhand,
        "head": head
    }
    

def raising_hand(pose: dict, thresh=30):        # 阈值越大，举手的人越少
    if pose is None:
        return 0
    rhand = pose["rhand"]
    lhand = pose["lhand"]
    rsholder = pose["rsholder"]
    lsholder = pose["lsholder"]
    head = pose["head"]
    
    r1 = 1 if lsholder[1] - head[1] > thresh else 0
    r2 = 1 if rsholder[1] - head[1] > thresh else 0
    
    total = r1 + r2
    
    return total
