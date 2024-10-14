import subprocess
import json
import os
import cv2
from line import Line, getVerticalLine

OPENPOSE_ROOT = os.environ.get("OPENPOSE_ROOT")

def writeKeyPoints(imgFolder):
    """同时处理一个文件夹中的所有照片

    Args:
        imgFolder (str): 该路径必须为绝对路径
    """
    folder_name = os.path.split(imgFolder)[-1]
    openpose_exe_path = os.path.join(OPENPOSE_ROOT, "bin\\OpenPoseDemo.exe")
    results_folder = os.path.join(OPENPOSE_ROOT, "results")
    save_folder = os.path.join(results_folder, folder_name)
    name_path = os.path.join(results_folder, "name.txt")
    os.makedirs(save_folder, exist_ok=True)
    
    try:
        with open(name_path, "r", encoding="utf-8") as f:
            folders = f.readlines()
    
    except FileNotFoundError as e:
        print("OpenPose first run, txt will be created.")
        with open(name_path, "w", encoding="utf-8") as f:
            folders = []
            pass
    
    if imgFolder+"\n" in folders:
        print("The folder has already been pose detect.")
        return save_folder
    
    command = [
        openpose_exe_path,
        "--image_dir", imgFolder,
        "--write_json", save_folder
    ]
    subprocess.run(command, cwd=OPENPOSE_ROOT)
    print(f"results have been saved in {results_folder}")
    
    with open(name_path, "a", encoding="utf-8") as f:
        print(imgFolder, file=f)
        
    return save_folder


def getKeyPoints(resultFolder):
    jsons = os.listdir(resultFolder)
    
    for jsonfile in jsons:
        name: str = jsonfile
        jsonfile = os.path.join(resultFolder, jsonfile)
        
        with open(jsonfile, "r", encoding="utf-8") as f:
            data = json.load(f)
            try: 
                print(jsonfile)
                points_conf = data["people"][0]["pose_keypoints_2d"]
            except (KeyError, IndexError) as e:
                print(f"Error occurs {e}")
                
        points = [points_conf[i:i+3] for i in range(0, len(points_conf) - 3, 3)]
        
        rsholder = points[2][:2]
        lsholder = points[5][:2]
        rhip = points[9][:2]
        lhip = points[12][:2]
        mt = points[1][:2]
        md = points[8][:2]
        
        name = name.strip("_keypoints.json")
        print(os.path.join("test_stage2/test/images/1211", f"{name}.jpg"))
        img = cv2.imread(os.path.join("test_stage2/test/images/1211", f"{name}.jpg"))
        het, wid, tun = img.shape
        
        point = [(int(_[0]), int(_[1])) for _ in (rsholder, lsholder, rhip, lhip, mt, md)]
        # drawPoints(point, img)    
        sliceNumberArea(img, point, .65)
        

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
        middleLine = Line(point0, point1)
        print(middleLine)
        
        if middleLine.k is None:
            weight = 1
        elif middleLine.k > 0:
            weight = 1 - weight
        elif middleLine.k <= 0:
            weight = weight
                
        x = min((point0[0], )) + abs(point0[0] - point1[0]) * weight
        y = middleLine(x = x)
        
        if y is None:
            y = (point0[1] + point1[1]) / 2
        return int(x), int(y)
    
    def genRectangle(point, line0: Line, wid, het):
        """生成一个矩形

        Args:
            point (tuple[int, int]): 中心点
            line (Line): 水平线
            wid (int): 宽度（半
            het (int): 高度（半
        """
        ...
        
          
    rsholder, lsholder, rhip, lhip, mt, md = points
    
    if (rsholder[0] - lsholder[0]) < 0 and (rhip[0] - lhip[0]) < 0:
        pass
        # return -1
    
    middlePoint = findMiddlePoint(mt, md, weight)
    hLine = getVerticalLine(middlePoint, Line(mt, md))
    
    wid1 = abs(rsholder[0] - lsholder[0])
    wid2 = abs(rhip[0] - lhip[0])
    wid = (wid1 + wid2) / 2 * weight2
    
    het1 = abs(rsholder[1] - lsholder[1])
    het2 = abs(rhip[1] - rhip[1])
    het = (het1 + het2) / 2 * weight3
    
    drawPoints((middlePoint, rsholder, lsholder, rhip, lhip, mt, md), img)
    drawLine(img, hLine, middlePoint)
    drawLine(img, Line(mt, md), middlePoint)
    

def drawPoints(points, image):

    i = 0
    colorList = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (100, 150, 200)]
    
    if points is not None:
        for point in points:
            if point is not None:
                cv2.circle(image, point, radius=1, color=colorList[i % 4], thickness=-1)
                i += 1

    cv2.imshow('Image with Points', image)
    cv2.waitKey(0)
    
    
def drawLine(img, line: Line, point, x: int=30):
    x, y = point
    x1 = x + 30
    x2 = x - 30
    y1 = line(x = x1)
    y2 = line(x = x2)
    print(x1, y1, x2, y2)
    img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness=1)
    cv2.imshow("line", img)
    cv2.waitKey(0)
    return 0
    

if __name__ == "__main__":
    getKeyPoints(writeKeyPoints(r"D:\CDR\test_stage2\test\images\1211"))