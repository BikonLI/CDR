"""与文心智能体之间的桥梁，服务端，网络部署"""
from flask import Flask, request, make_response
import cv2
import os
import json
from bilix.sites.bilibili import DownloaderBilibili
from OCR import extract_number_from_str
import queue
from typing import *

que = queue.Queue()
flag: Literal["AFT", "EOP", "AFP"] = "AFT" # ["await for task", "end of processing", "await for processing"]

app = Flask(__name__)
@app.route("/detect/", methods=["POST"])
def detect():
    req = request.get_json()
    que.put(req)

    if flag == "AFT":
        # 等待任务，返回需要时间处理
        return {"prompt": "Needing time for processing your video, please wait. You can send the url again to see the progress!", "result": {}}, 200, {"Content-Type": "application/json"}
    elif flag == "AFP":
        return {"prompt": "Needing time for processing your video, please wait. You can send the url again to see the progress!", "result": {}}, 200, {"Content-Type": "application/json"}
    elif flag == "EOP":
        return {"prompt": "Finished!", "result": {}}, 200, {"Content-Type": "application/json"}

@app.route("/")
def test():
    return "connected"

@app.route("/geturl/")
def url():
    return que.get(timeout=1)





if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=11451)


