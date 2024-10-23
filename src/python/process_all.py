import os
os.environ['YOLO_VERBOSE'] = str(False)
from ultralytics import YOLO
import cv2
import subprocess
import json
from line import Line, getVerticalLine
from OCR import getRectangle, predict
from timeout import *
from client import *
from bayes_model_new import update_probabilities1, reset_priors, get_most_likely_number1

def done(folder):
    path = os.path.join(folder, "marked.done")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{time.time()}")

def isDone(folder):
    path = os.path.join(folder, "marked.done")
    isDone = os.path.exists(path)
    print(isDone)
    if isDone:
        with open(path, "r", encoding="utf-8") as f:
            string = f.read()
            print(f'任务完成 {string}')
    return isDone

# 一个trackId的球员矩形照片放到一个文件夹当中
def analyze_video(folder: str):
    
    folders_list = []
    
    model = YOLO("best.pt")
    if isDone(folder):
        all_files_folders = os.listdir(folder)
        for file_folder in all_files_folders:
            try:
                int(file_folder)
                players_images_path = os.path.join(folder, file_folder)
                folders_list.append(players_images_path)
            except Exception as e:
                continue
        return folders_list
    
    for i in range(len(os.listdir(folder))):
        imgPath = os.path.join(folder, f"frame_{i:04}.jpg")

        if f"frame_{i:04}.jpg" not in os.listdir(folder):
            continue

        print(f"正在追踪 {imgPath}")
        
        img = cv2.imread(imgPath)
        result = model.track(img, persist=True, show=True)[0]
        
        xyxy_list = result.boxes.xyxy.tolist()
        id_list = result.boxes.id.tolist()
        
        for j in range(len(id_list)):
            people = xyxy_list[j]
            trackId = int(id_list[j])
            players_images_path = os.path.join(folder, f"{trackId}")
            pose_detect_result_dir = os.path.join(players_images_path, "results")
            
            os.makedirs(players_images_path, exist_ok=True)
            os.makedirs(pose_detect_result_dir, exist_ok=True)
            
            folders_list.append(players_images_path)
            
            people = (people[:2], people[2:])
            print(people)
            people_img = getRectangle(img, people)
            imgName = f"{i}.jpg"
            cv2.imwrite(os.path.join(players_images_path, imgName), people_img)
            
            print(os.path.join(folder, "analyze.txt"))
            with open(os.path.join(folder, "analyze.txt"), "a", encoding="utf-8") as f:
                print(f"{i + 1} {trackId} {float(people[0][0])} {float(people[0][1])} {float(people[1][0])} {float(people[1][1])}", file=f)
    done(folder)
    return folders_list
    
            
# --------------------------------- 对posedetect的智能体版本的重构          
        

OPENPOSE_ROOT = os.environ.get("OPENPOSE_ROOT")
record_progress_path = "progress.json"

def writeKeyPoints(imgFolder):
    """同时处理一个文件夹中的所有照片

    Args:
        imgFolder (str): 该路径必须为绝对路径
    """
    save_folder = os.path.join(imgFolder, "results")
    os.makedirs(save_folder, exist_ok=True)
    
    openpose_exe_path = os.path.join(OPENPOSE_ROOT, ".\\bin\\OpenPoseDemo.exe")
    
    if isDone(save_folder):
        return save_folder
    
    command = [
        openpose_exe_path,
        "--image_dir", imgFolder,
        "--write_json", save_folder
    ]
    subprocess.run(command, cwd=OPENPOSE_ROOT)
    print(f"results have been saved in {save_folder}")
    
    done(save_folder)
    return save_folder


def getKeyPoints(resultFolder, index, imgs_folder): 
    # 起始位置：poseresult文件夹， 照片文件夹， 结果文件坐标(起始位置), 打榜文件夹
    jsons = os.listdir(resultFolder)[index:]
    
    for jsonfile in jsons:
        name: str = jsonfile
        if name == "marked.done":
            continue
        jsonfile = os.path.join(resultFolder, jsonfile)
        
        with open(jsonfile, "r", encoding="utf-8") as f:
            data = json.load(f)
            try: 
                points_conf = data["people"][0]["pose_keypoints_2d"]
            except (KeyError, IndexError) as e:
                continue
            
                
        points = [points_conf[i:i+3] for i in range(0, len(points_conf) - 3, 3)]
        
        rsholder = points[2][:2]
        lsholder = points[5][:2]
        rhip = points[9][:2]
        lhip = points[12][:2]
        mt = points[1][:2]
        md = points[8][:2]
        
        name = name.strip("_keypoints.json")
        img = cv2.imread(os.path.join(imgs_folder, f"{name}.jpg"))
        print(os.path.join(imgs_folder, f"{name}.jpg"))
        
        point = [(int(_[0]), int(_[1])) for _ in (rsholder, lsholder, rhip, lhip, mt, md)]
        # drawPoints(point, img)    
        
        number = sliceNumberArea(img, point, .65)
        
        update_probabilities1(number, .1)
        index += 1
    
    number = get_most_likely_number1()
    reset_priors()
    return number
    
        

def sliceNumberArea(img, points: tuple, weight: float=.65, weight2: float=.95, weight3: float=.3):
    def findMiddlePoint(point0, point1, weight: float=.5):
        """传入一个系数，返回数字中心点坐标

        Args:
            point0 (tuple): 点
            
            point1 (tuple): 点
            
            weight (float): 系数
            

        Returns:
            tuple[int, int]: 中心点
        """
        x0, x1 = sorted((point0[0], point1[0]))
        x = x0 + (x1 - x0) * (1 - weight)
        y0, y1 = sorted((point0[1], point1[1]))
        y = y0 + (y1 - y0) * (1 - weight)

        return int(x), int(y)
    
    def genRectangle(lsholder, rsholder, rhip, middle):
        """生成一个矩形

        Args:
            point (tuple[int, int]): 中心点
            line0 (Line): 水平线
            line1 (Line): 垂直线
            line2 (Line): 肩膀线
            point0 (tuple[x, y]): 左肩
            point1 (tuple[x, y]): 右肩
        """
        lx, rx = sorted((lsholder[0], rsholder[0]))
        
        x_ratio = .1            # 框的左右界相对于肩膀宽度的比例。越大，框越宽。为0则与肩同宽。
        x0 = lx - (rx - lx) * x_ratio
        x1 = rx + (rx - lx) * x_ratio
        biasT = abs(lsholder[1] - middle[1])
        biasB = abs(rhip[1] - middle[1])
        
        top_ratio = .9          # 框上界相对于肩膀的比例，越大框上界越高
        bot_ratio = .8         # 框下届对于髋的比例，越大框下界越往下
        y0 = middle[1] - biasT * top_ratio         # 框上界的位置。
        y1 = middle[1] + biasB * bot_ratio
        
        dx = rsholder[0] - lsholder[0]
        dy = rsholder[1] - lsholder[1]
        
        biasRatio = .2     # 左右偏置比例，会改变框的位置，不改变框的大小。越大偏的越多。
        if dx * dy > 0:
            xBias = -(x1 - x0) * biasRatio
        else:
            xBias = (x1 - x0) * biasRatio
            
        x0 += xBias
        x1 += xBias
        
        xscale = .90     # 越小框越小，不会改变框位置
        yscale = .90
        
        x0 += (x1 - x0) * (1 - xscale) / 2
        x1 -= (x1 - x0) * (1 - xscale) / 2
        
        y0 += (y1 - y0) * (1 - yscale) / 2
        y1 -= (y1 - y0) * (1 - yscale) / 2
    
        return (int(x0), int(y0)), (int(x1), int(y1))
    
          
    rsholder, lsholder, rhip, lhip, mt, md = points
    
    if (rsholder[0] - lsholder[0]) < 0 and (rhip[0] - lhip[0]) < 0:
        pass
        return ""
    
    middlePoint = findMiddlePoint(mt, md, weight)
    hLine = getVerticalLine(middlePoint, Line(mt, md))
    
    wid1 = abs(rsholder[0] - lsholder[0])
    wid2 = abs(rhip[0] - lhip[0])
    wid = (wid1 + wid2) / 2 * weight2
    
    het1 = abs(rsholder[1] - lsholder[1])
    het2 = abs(rhip[1] - rhip[1])
    het = (het1 + het2) / 2 * weight3
    
    # drawPoints((middlePoint, rsholder, lsholder, rhip, lhip, mt, md), img)
    point1, point2 = genRectangle(lsholder, rsholder,  rhip, middlePoint)
    
    x0, y0 = point1
    x1, y1 = point2
    
    if abs(x1 - x0) < abs(y1 - y0) / 2:
        return ""
    
    img = cv2.rectangle(img, point1, point2, (255, 0, 0), 1)
    # cv2.imshow("name", img)
    # cv2.waitKey(50)
    img = getRectangle(img, (point1, point2))
    
    number = predict(img)
    number = extract_number_from_str(number)
    
    print(f"预测结果={number}")
    if number is None:
        return ""
    
    return number

def trackIdMatToNumber(mapping: dict, trackID):
    for key, values in mapping.items():
        if trackID in values:
            return key
    

def updata_results(folder, mapping: dict):
    with open(os.path.join(folder, "analyze.txt"), "r", encoding="utf-8") as f:
        all_frame_player = f.readlines()
    afp = []

    for line in all_frame_player:
        line = line.strip()
        if line:
            afp.append(line.split(" "))
    all_frame_player = afp
    for line in all_frame_player:
        trackId = int(line[1])
        number = trackIdMatToNumber(mapping, trackID=trackId)
        line[1] = str(number)
        
    with open(os.path.join(folder, "analyze.txt"), "w", encoding="utf-8") as f:
        for line in all_frame_player:
            line_str = ' '.join(line)
            print(line_str, file=f)
    
    return os.path.join(folder, "analyze.txt")


def processing(url):
    setFlag("AFP")
    folders = downloadVideo(url)
    
    number_trackid_map = {}
    if folders is not None:
        videoPath, frameSaveFolder = folders
        cwd = os.getcwd()
        frameSaveFolder = os.path.join(cwd, frameSaveFolder)
        print(f"帧保存绝对路径={frameSaveFolder}")
        rcode = extract_frames(videoPath, frameSaveFolder)
        playersFolders = analyze_video(frameSaveFolder)
        
        for player in playersFolders:
            pointsSaveFolder = writeKeyPoints(player)
            trackId = int(os.path.split(player)[-1])
            number = getKeyPoints(pointsSaveFolder, 0, player)
            print(f"number={number}")
            try:
                number_trackid_map[f"{number}"].append(trackId)
            except Exception:
                number_trackid_map[f"{number}"] = [trackId, ]
                
        print(f"映射={number_trackid_map}")
        results_path = updata_results(frameSaveFolder, number_trackid_map)
        return results_path
                
        

def main():
    while True:
        response = getJson()
        url = response.get("url")
        print(f"url=\"{url}\"")
        if url:
            print("任务被触发！")
            result_file = processing(url)

            return 0;
            
        time.sleep(3)
            
            
if __name__ == "__main__":
    main()
        

        
