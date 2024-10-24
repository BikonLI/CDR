"""该脚本重新复写本地设备的处理"""
import os
import subprocess
import time
import cv2
import json
from pathlib import Path
from typing import *
from ultralytics import YOLO
from OCR import predict
from slicenumberarea import sliceNumberArea as sna
from client import getJson
from pose_detect import getRectangle
from bayes_model_new import (
    update_probabilities1 as update_p,
    reset_priors as reset_p,
    get_most_likely_number1 as get_number
)

# 文件结构
# raw_video/
#   a1.mp4 ...
#   video_info.json # 该文件当中记录了视频处理信息
# frame_video/
#   1/ 00001.jpg ...
#   2/ 00002.jpg ...
# players/
#   1/ 
#       1/
#       2/
# pose_results/
#   1/ 00001_keypoints.json ...
# 
_VIDEO    = Literal["a1.mp4", "a2.mp4", "a3.mp4", "a4.mp4"]
_PROGRESS = Literal["extract", "track", "pose", "recognize", "update"]

class Config:
    
    def __init__(self) -> None:
        self.OPENPOSE_ROOT = Path(os.environ["OPENPOSE_ROOT"])
        self.RAW_VIDEO_DIR = Path("./raw_video")
        self.FRAME_VIDEO_DIR = Path("./frame_video")
        self.PLAYERS_DIR = Path("./players")
        self.POSE_RESULTS = Path("./pose_results")
        self.VIDEO_INFO_JSON = self.RAW_VIDEO_DIR / Path("video_info.json")
        self.ANALYZE = Path("./analyze")
        self.video: _VIDEO = None
        
        print("Several directories are created: ")
        print(self.OPENPOSE_ROOT.absolute())
        print(self.RAW_VIDEO_DIR.absolute())
        print(self.FRAME_VIDEO_DIR.absolute())
        print(self.PLAYERS_DIR.absolute())
        print(self.POSE_RESULTS.absolute())
        print(self.VIDEO_INFO_JSON.absolute())
        print(self.ANALYZE.absolute())
        
        self.OPENPOSE_ROOT.mkdir(exist_ok=True)
        self.RAW_VIDEO_DIR.mkdir(exist_ok=True)
        self.FRAME_VIDEO_DIR.mkdir(exist_ok=True)
        self.PLAYERS_DIR.mkdir(exist_ok=True)
        self.POSE_RESULTS.mkdir(exist_ok=True)
        if not self.VIDEO_INFO_JSON.exists():
            self.reset_info()
        
        self.ANALYZE.mkdir(exist_ok=True)
            
               
    def setVideo(self, video: _VIDEO):
        self.video = video
        with open(self.VIDEO_INFO_JSON, "r", encoding="utf-8") as f:
            video_info = json.load(f)
        
        self.folder_name: Path = video_info[self.video]["folder"]
        
        self.raw_video: Path = self.RAW_VIDEO_DIR / self.video
        self.frame_video: Path = self.FRAME_VIDEO_DIR / self.folder_name
        self.players: Path = self.PLAYERS_DIR / self.folder_name
        self.pose_result: Path = self.POSE_RESULTS / self.folder_name
        self.analyze: Path = self.ANALYZE / f"{self.folder_name}.txt"
        self.update: Path = self.ANALYZE / f"{self.folder_name}_final.txt"
        
        print("Several directories are created or detected: ")
        print(self.raw_video.absolute())
        print(self.frame_video.absolute())
        print(self.players.absolute())
        print(self.pose_result.absolute())
        print(self.analyze.absolute())
        print(self.update.absolute())
        
        self.frame_video.mkdir(exist_ok=True)
        self.players.mkdir(exist_ok=True)
        self.pose_result.mkdir(exist_ok=True)
        
        if not self.analyze.exists():
            with open(self.analyze, "w", encoding="utf--8") as f:
                pass
        
        if not self.update.exists():
            with open(self.update, "w", encoding="utf-8") as f:
                pass
         
    def extractIsDone(self):
        with open(self.VIDEO_INFO_JSON, "r", encoding="utf-8") as f:
            video_info = json.load(f)
            
        if "extract" in video_info[self.video]["progress"]:
            return True
        else:
            return False

    def trackIsDone(self):
        with open(self.VIDEO_INFO_JSON, "r", encoding="utf-8") as f:
            video_info = json.load(f)
            
        if "track" in video_info[self.video]["progress"]:
            return True
        else:
            return False 
        
    def poseIsDone(self):
        with open(self.VIDEO_INFO_JSON, "r", encoding="utf-8") as f:
            video_info = json.load(f)
            
        if "pose" in video_info[self.video]["progress"]:
            return True
        else:
            return False
        
    def updateIsDone(self):
        with open(self.VIDEO_INFO_JSON, "r", encoding="utf-8") as f:
            video_info = json.load(f)
            
        if "update" in video_info[self.video]["progress"]:
            return True
        else:
            return False
        
    def done(self, work: _PROGRESS):
        with open(self.VIDEO_INFO_JSON, "r", encoding="utf-8") as f:
            video_info = json.load(f)
            
        video_info[self.video]["progress"].append(work)
        
        with open(self.VIDEO_INFO_JSON, "w", encoding="utf-8") as f:
            json.dump(video_info, f)
        
    def recognizeIsDone(self):
        with open(self.VIDEO_INFO_JSON, "r", encoding="utf-8") as f:
            video_info = json.load(f)
            
        if "recognize" in video_info[self.video]["progress"]:
            return True
        else:
            return False 
        
    def reset_info(self):
        with open(self.VIDEO_INFO_JSON, "w", encoding="utf-8") as f:
            json.dump({
                "a1.mp4": {"progress": [], "folder": "0", "map": {}}, 
                "a2.mp4": {"progress": [], "folder": "1", "map": {}}, 
                "a3.mp4": {"progress": [], "folder": "2", "map": {}}, 
                "a4.mp4": {"progress": [], "folder": "3", "map": {}}, 
                }, f)
            
    def get_info(self):
        with open(self.VIDEO_INFO_JSON, "r", encoding="utf-8") as f:
            info = json.load(f)
        return info[self.video]
    
    def dump_info(self, video_info):
        with open(self.VIDEO_INFO_JSON, "r", encoding="utf-8") as f:
            info = json.load(f)
            
        info[self.video] = video_info
        with open(self.VIDEO_INFO_JSON, "w", encoding="utf-8") as f:
            json.dump(info, f)
            
        return 0
        
    

config = Config()
            
def extract():
    
    if config.extractIsDone():
        return 0
    
    video_path = config.raw_video
    output_folder = config.frame_video
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()

        if ret:
            frame_filename: Path = output_folder / Path(f"{frame_count:05}.jpg")
            if frame_count % 20 == 0:
                print(f"saving {frame_filename.absolute()}")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
        else:
            break

    cap.release()
    config.done("extract")
    
    return 0

def track():
    if config.trackIsDone():
        return 0
    
    model = YOLO("best.pt")
    frame_count = len_subdir(config.frame_video)
    
    for i in range(frame_count):
        pic_name = config.frame_video / Path(f"{i:05}.jpg")
        img = cv2.imread(pic_name)
        result = model.track(pic_name, persist=True)[0]
        
        xyxy_list = result.boxes.xyxy.tolist()
        id_list = result.boxes.id.tolist()
        
        for j in range(len(id_list)):
            xyxy = xyxy_list[j]
            trackId = int(id_list[j])
            
            id_folder = config.players / f"{trackId}"
            id_folder.mkdir(exist_ok=True)
            
            person_img = getRectangle(img, (xyxy[:2], xyxy[2:]))
            person_img_save_path = id_folder / f"{i}.jpg"
            cv2.imwrite(person_img_save_path, person_img)
            
            with open(config.analyze, "a", encoding="utf-8") as f:
                print(f"{i} {trackId} {xyxy[0]} {xyxy[1]} {xyxy[2]} {xyxy[3]}", file=f)
        
    config.done("track")
    return 0


def pose():
    if config.poseIsDone():
        return 0
    
    all_trackId = config.players.iterdir()
    openpose_exe_path = config.OPENPOSE_ROOT / "bin\\OpenPoseDemo.exe"
    
    for ids in all_trackId:
        people_img_folder = ids
        results_folder = config.pose_result / ids.name
        
        results_folder.mkdir(exist_ok=True)
        
        command = [
            openpose_exe_path,
            "--image_dir", people_img_folder,
            "--write_json", results_folder
        ]
        
        subprocess.run(command, cwd=config.OPENPOSE_ROOT)
        
        print(f"'{ids}' pose detect done!")
    
    config.done("pose")
    return 0


def recognize():
    
    if config.recognizeIsDone():
        return 0
    
    num_map_id = {}
    result_folders = config.pose_result.iterdir()
    
    for trackid_folder in result_folders:
        
        for jsonfile in trackid_folder.iterdir():
            json_name = jsonfile.name
        
            with open(jsonfile, "r", encoding="utf-8") as f:
                data = json.load(f)
                try: 
                    points_conf = data["people"][0]["pose_keypoints_2d"]
                except (KeyError, IndexError):
                    continue
            
            points = [points_conf[i:i+3] for i in range(0, len(points_conf) - 3, 3)]
            
            rsholder = points[2][:2]
            lsholder = points[5][:2]
            rhip = points[9][:2]
            lhip = points[12][:2]
            mt = points[1][:2]
            md = points[8][:2]
            
            pic_name = json_name.strip("_keypoints.json")
            img_path = config.players / pic_name
            
            point = [(int(_[0]), int(_[1])) for _ in (rsholder, lsholder, rhip, lhip, mt, md)]
        
            img = cv2.imread(img_path)
            rec_points = sna(point)
        
            img = getRectangle(img, rec_points)
            number = predict(img)
            update_p(number, .2)
        
        predicted_number = get_number()
        if not num_map_id.get(f"{predicted_number}"):
            num_map_id[f"{predicted_number}"] = [f"{trackid_folder}", ]
        else:
            num_map_id[f"{predicted_number}"].append(f"{trackid_folder}")
            
    video_info = config.get_info()
    video_info["map"] = num_map_id
    config.dump_info(video_info)
    config.done("recognize")
    
    return 0


def update():
    if config.updateIsDone():
        return 0
    
    num_map_id = config.get_info()["map"]
    
    with open(config.analyze, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    new_lines = []
    
    for line in lines:
        split_line = line.split(" ")
        number = mapping(num_map_id, split_line[1].strip())
        new_split_line = split_line.copy()
        new_split_line[1] = number
        
        new_line = ' '.join(new_split_line)
        new_lines.append(new_line)
        
    with open(config.update, "w", encoding="utf-8") as f:
        for new_line in new_lines:
            print(new_line, file=f)
    
    config.done("update")
    return 0
        

def process_all(video: _VIDEO):
    config.setVideo(video)
    
    extract()
    track()
    pose()
    recognize()
    update()
    
    return 0

   
    
# --- 工具函数    
def mapping(num_map_id: dict, id):
    for key, value in num_map_id.items():
        if id in value:
            return key
        
    return -1

def len_subdir(path):
    dir_path = path
    subdirectories = [d for d in dir_path.iterdir()]
    return len(subdirectories)
            
            
def main():
    while True:
        response = getJson()
        url = response.get("url")
        print(f"url=\"{url}\"")
        if url:
            process_all(url)
            break
            
        time.sleep(3)


if __name__ == "__main__":
    main()
        


