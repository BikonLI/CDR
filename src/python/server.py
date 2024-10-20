"""与文心智能体之间的桥梁，服务端，网络部署"""
from flask import Flask, request, make_response, render_template
import cv2
import os
import json
import queue
from typing import *

que = queue.Queue()
flag: Literal["AFT", "EOP", "AFP"] = "AFT" # ["await for task", "end of processing", "await for processing"]

news = ""


app = Flask(__name__)
@app.route("/detect/", methods=["POST"])
def detect():
    req: dict = request.get_json()
    url = req.get("VIDEO_URL")
    que.put(url)
    global flag

    if flag == "AFT":
        # 等待任务，返回需要时间处理
        return {"prompt": "Needing time for processing your video, please wait. You can send the url again to see the progress!", "result": {}}, 200, {"Content-Type": "application/json"}
    elif flag == "AFP":
        return {"prompt": "Needing time for processing your video, please wait. You can send the url again to see the progress!", "result": {}}, 200, {"Content-Type": "application/json"}
    elif flag == "EOP":
        return {"prompt": "Finished!", "result": {}}, 200, {"Content-Type": "application/json"}

@app.route("/")
def test():
    return render_template("README.html")

@app.route("/geturl/")
def url():
    url = ""
    if not que.empty():
        url = que.get(block=False)
    return {"url": url}


@app.route("/updateflag/")
def updateflag():
    global flag
    flag = request.args.get("flag", "AFT")
    print(flag)
    return "updated"

@app.route("/postnews/", methods=["POST"])
def getnews():
    data = request.get_json()
    global news
    news = data.get("news")
    if news is None:
        news = ""
    
    with open("messages.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines.append(f"{news}\n")
        
    with open("messages.txt", "w", encoding="utf-8") as f:
        f.writelines(lines)
        
    return "posed"

@app.route("/progress/")
def pushnews():
    return news



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=11451)


