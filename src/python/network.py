"""与文心智能体之间的桥梁"""
from flask import Flask, request, make_response
import cv2
import os
import json
from bilix.sites.bilibili import DownloaderBilibili

app = Flask(__name__)
@app.route("/detect/", methods=["POST"])
def detect():
    req = request.get_json()
    print(req)

    return {"frames": 1}, 200, {"Content-Type": "application/json"}

@app.route("/")
def test():
    return "connected"



def extract_frames(video_path, output_folder):
    """
    从视频中提取帧并保存为 JPEG 格式。

    :param video_path: 视频文件的路径
    :param output_folder: 保存帧的文件夹
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else: return 1                  # 文件夹已存在，直接返回，不再进行提取帧

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


def downloadVideo(url: str, dstPath: str):
    record_json = "video_downloads.json"
    if not os.path.exists(record_json):
        with open(record_json, "w", encoding="utf-8") as f:
            pass
    
    try:
        with open(record_json, "r", encoding="utf-8") as f:
            data: dict = json.load(f)
            dstPath = data.get("url")
            if dstPath is not None:
                return dstPath
        
        downloader = DownloaderBilibili()
        downloader.download(url, output_dir=dstPath)
        
        with open(record_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            data[url] = dstPath
            
        with open(record_json, "w", encoding="utf-8") as f:
            json.dump(data, f)
            
        return dstPath
    except Exception as e:
        print("视频下载失败")
        return -1


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=11451)


