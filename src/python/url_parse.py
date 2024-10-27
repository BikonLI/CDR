import json
from pathlib import Path
import os
import sys
import requests
from path_config import config
import string
import random


if not (config.RAW_VIDEO_DIR / "mapping.json").exists():
    with open(config.RAW_VIDEO_DIR / "mapping.json", "w", encoding="utf-8") as f:
        json.dump({}, f)
        

def __get_video_format(url):
    
    for i in range(len(url) - 1, -1, -1):
        char = url[i]
        if char == '.':
            break
            
    format = url[i+1:]
    return format

def __download_video(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        return 0
    else:
        return -1        
    
def __generate_random_string(length=5):
    # 从字母表中选择随机字母
    letters = string.ascii_letters  # 包含大小写字母
    random_string = ''.join(random.choices(letters, k=length))
    return random_string
    
def video_download(url):
    # 映射关系 url -> video_name -> info

    video_format = __get_video_format(url)
    mapping_json = config.RAW_VIDEO_DIR / "mapping.json"

    with open(mapping_json, "r", encoding="utf-8") as f:
        url_map_path = json.load(f)
        video_path = url_map_path.get(url)
        
    if video_path is not None:
        return 0
    
    video_save_path = config.RAW_VIDEO_DIR / f"{__generate_random_string()}.{video_format.strip()}"
    rcode = __download_video(url, video_save_path)
    
    video_path = video_save_path.name
    
    if rcode != 0:
        print("视频下载失败！！！")
        return -1
    
    with open(config.VIDEO_INFO_JSON, "r", encoding="utf-8") as f:
        video_info: dict = json.load(f)
    
    max_value = 0
    for key, value in video_info.items():
        max_value = int(value["folder"]) if int(value["folder"]) > max_value else max_value
        
    video_info[video_path] = {"progress": [], "folder": f"{max_value}", "map": {}}
    
    with open(config.VIDEO_INFO_JSON, "w", encoding="utf-8") as f:
        json.dump(video_info, f)
        
    with open(mapping_json, "r", encoding="utf-8") as f:
        url_map_path = json.load(f)
        
    url_map_path[url] = video_path
    
    with open(mapping_json, "w", encoding="utf-8") as f:
        json.dump(url_map_path, f)
        
    print(f"video downloads successfully {video_path}")    
    return 0


if __name__ == "__main__":
    rc_code = video_download("https://www.ikcest.org/MediaStore/site/site2024/video/2024/07/08/a1.mp4")
    print(rc_code)
    "https://www.ikcest.org/MediaStore/site/site2024/video/2024/07/08/a2.mp4"
    
        
    
        
    
    
