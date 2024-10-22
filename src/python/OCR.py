import cv2
import numpy as np
import pytesseract
import re
import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule
from timeout import *
# from multiprocessing import Process, Value, Queue


pytesseract.pytesseract.tesseract_cmd = r"D:\Software\OCR\tesseract.exe"

parseq = torch.hub.load('baudm/parseq', 'parseq_tiny', pretrained=True).eval()


def predict(img):
    if img:=clarity(img) is None:
        return ""
    # img = cv2.equalizeHist(img)
    
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    custom_config = r'--oem 3 --psm 7 outputbase digits'
    recognized_text = pytesseract.image_to_string(img, config=custom_config)
    num = extract_number_from_str(recognized_text)

    if num:
        print(f"预测数字为：{num}")

    return num

def predict1(img):
        
    if not is_color_image(img):
        return ""
    
    # 计算新尺寸
    global parseq
    scale_factor = 2
    new_width = int(img.shape[1] * scale_factor)
    new_height = int(img.shape[0] * scale_factor)

    # 调整图像大小
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # cv2.imshow("small", img)

    # Load model and image transforms
    
    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
    
    # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
    # img = torch.from_numpy(img)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 转换为 RGB 格式
    img = img_transform(img).unsqueeze(0)

    logits = parseq(img)
    logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

    # Greedy decoding
    pred = logits.softmax(-1)
    label, confidence = parseq.tokenizer.decode(pred)
    
    number = extract_number_from_str(label[0])
    
    return number
    
predict = predict1


def getRectangle(img,  rectangle: tuple[tuple[int, int]]):
    
    p1, p2 = rectangle
    x0, y0 = p1
    x1, y1 = p2
    x, y = x0, y0
    w = x1 - x0
    h = y1 - y0
    
    if x0 < 0:
        x0 = 0
    if x1 < 0:
        x1 = 0
    if y0 < 0:
        y0 = 0
    if y1 < 0:
        y1 = 0
    if x1 > img.shape[1]:
        x1 = img.shape[1]
    if y1 > img.shape[0]:
        y1 = img.shape[0]
        
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    cropped_img = img[y:y+h, x:x+w]
    return cropped_img
        
        
def is_color_image(img):
    if img is None:
        return False
    
    # 1. 检查是否为三维矩阵
    if img.ndim != 3:
        return False
    
    # 2. 检查是否为空
    if img.size == 0:
        return False
    
    # 3. 检查是否有任一维度的大小为 0
    if any(dim == 0 for dim in img.shape):
        return False
    
    # 4. 检查是否为 RGB 图像（第三个维度的大小应该为 3）
    if img.shape[2] != 3:
        return False

    return True

def clarity(img):
    if not is_color_image(img):
        return None
    
    # 将25x25的小图像放大到100x100
    ratio = 2
    het, wid, tun = img.shape
    img = cv2.resize(img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)


    # 2. 转换为灰度图像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 应用高斯模糊去除噪声
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # img = cv2.bilateralFilter(img, 9, 75, 75)
    

    # 4. 图像锐化（使用自定义卷积核）
    kernel = np.array([[0, -1, 0], 
                    [-1, 5,-1], 
                    [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    return img


def extract_number_from_str(string):

    # 使用正则表达式提取所有数字
    numbers: list = re.findall(r'\d+', string)

    # 输出结果
    result = ''.join(numbers)
    return result


if __name__ == "__main__":
    print(predict(cv2.imread("train_stage2/train/images/0/0_1.jpg")))
        