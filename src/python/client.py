"""客户端（算力部署）"""
import os
import cv2
import json
import time
from bilix.sites.bilibili import DownloaderBilibili
from OCR import extract_number_from_str
import requests
from typing import *


flag: Literal["AFT", "EOP", "AFP"] = "AFP"   # ["await for task", "end of processing", "await for processing"]

def extract_frames(video_path, output_folder):
    """
    从视频中提取帧并保存为 JPEG 格式。

    :param video_path: 视频文件的路径
    :param output_folder: 保存帧的文件夹
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("开始提取视频帧")
    else:
        print("视频帧已被提取") 
        return 1                  # 文件夹已存在，直接返回，不再进行提取帧

    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if ret:
            frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            print(f'Saved: {frame_filename}')
            frame_count += 1
        else:
            break

    cap.release()
    print('视频处理完成！')
    return 0                    # 对视频进行了处理


def downloadVideo(url: str):
    record_json = "video_downloads.json"
    if not os.path.exists(record_json):
        with open(record_json, "w", encoding="utf-8") as f:
            pass
    os.makedirs("./raw_videos", exist_ok=True)
    
    try:
        with open(record_json, "r", encoding="utf-8") as f:
            data: dict = json.load(f)
            videoName = data.get(url)
            if videoName is not None:
                return os.path.join("./raw_videos", videoName[0]), os.path.join("./frames", f"{videoName[1]}")
            
            else: raise NotImplementedError("使用三方链接尚不支持，目前仅可分析a1.mp4, a2.mp4, a3.mp4, a4.mp4。")
        
        downloader = DownloaderBilibili()
        downloader.download(url, output_dir="./raw_videos")
        
        videoName = max((int(extract_number_from_str(x)) for x in os.listdir("./raw_videos"))) + 1
        with open(record_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            data[url] = f"{videoName}."
            
        with open(record_json, "w", encoding="utf-8") as f:
            json.dump(data, f)
            
        return videoName
    except Exception as e:
        print(e)
        print("视频下载失败")
        return None
    

def getJson():
    response = requests.get("http://49.233.183.144:11451/geturl/")
    res = response.json()
    return res

def setFlag(flag: Literal["AFT", "EOP", "AFP"] = "AFP"):
    # params={"flag": flag}
    response = requests.get("http://49.233.183.144:11451/updateflag/", params={"flag": flag})
    print(response.text)
    return response.status_code

if __name__ == "__main__":
    # video, frame_folder = downloadVideo("a4.mp4")
    # extract_frames(video, frame_folder)
    # while True:
    #     result = getJson()
    #     if result.get("url"):
    #         break
    #     time.sleep(3)
        
    # print(result["url"])
    print(setFlag("EOP"))
