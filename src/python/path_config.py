import os
from pathlib import Path
from typing import *
import json

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
            with open(self.VIDEO_INFO_JSON, "w", encoding="utf-8") as f:
                pass
        
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
            
    def get_info(self) -> dict[dict]:
        with open(self.VIDEO_INFO_JSON, "r", encoding="utf-8") as f:
            info = json.load(f)
        
        return info.get(f"{self.video}") 
    
    def dump_info(self, video_info):
        with open(self.VIDEO_INFO_JSON, "r", encoding="utf-8") as f:
            info = json.load(f)
            
        info[self.video] = video_info
        with open(self.VIDEO_INFO_JSON, "w", encoding="utf-8") as f:
            json.dump(info, f)
            
        return 0
        
        
config = Config()   # 同一时间仅处理一个视频，确保单例 
