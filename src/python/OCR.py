import cv2
import numpy as np
import pytesseract
import re



pytesseract.pytesseract.tesseract_cmd = r"D:\Software\OCR\tesseract.exe"


def predict(img):
    if not is_color_image(img):
        return ""
    
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
    # img = cv2.equalizeHist(img)
    
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    custom_config = r'--oem 3 --psm 7 outputbase digits'
    recognized_text = pytesseract.image_to_string(img, config=custom_config)
    num = extract_number_from_str(recognized_text)

    if num:
        print(f"预测数字为：{num}")

    return num


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


def extract_number_from_str(string):

    # 使用正则表达式提取所有数字
    numbers: list = re.findall(r'\d+', string)

    # 输出结果
    result = ''.join(numbers)
    return result


if __name__ == "__main__":
    string_list = ["+213/" for i in range(100)]
    
    for string in string_list:
        number = extract_number_from_str(string)
        print(number)
        