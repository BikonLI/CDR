import numpy as np
import os

priors = np.ones(100)
priors = priors / 100

# 重置概率矩阵
def reset_priors():
    global priors
    priors = np.ones(100)
    priors = priors / 100
    

def gen_ocr_error_model_100():

    ocr_error_model_100 = np.ones((100, 100))
    for i in range(100):
        for j in range(100):       
            overlaping_rate = calculate_num_overlaping_rate(i, j)
            print(f"{i} {j} {overlaping_rate}")
            ocr_error_model_100[i, j] = overlaping_rate
            
    ocr_error_model_100 = ocr_error_model_100 / np.sum(ocr_error_model_100)
    print(ocr_error_model_100)
    return ocr_error_model_100
            

def calculate_num_overlaping_rate(i, j):
    """计算两个整数的重合度

    Args:
        i (int): 第一个两位数或一位数
        j (int): 第二个两位数或一位数
    """
    num1 = [int(char) for char in str(i)]
    num2 = [int(char) for char in str(j)]
    
    result = [0, 0]
    
    if len(num1) == 1 and len(num2) == 2:
        cur = num1
        num1 = num2
        num2 = cur
    
    bias = 0
    
    if len(num1) == 2 and len(num2) == 2:
        result[0] = 1 if num1[0] == num2[0] else 0
        result[1] = 1 if num1[1] == num2[1] else 0
        
        if num1[0] == num2[1] or num1[1] == num2[0]:
            bias = .255
        else:
            bias = 0.025
    elif len(num1) == 2 and len(num2) == 1:
        result[0] = 1 if num1[0] == num2[0] else 0
        result[1] = 1 if num1[1] == num2[0] else 0
        
        if (result[0] + result[1]) == 2:
            bias = -.333
        if (result[0] + result[1]) == 0:
            bias = .03
    else:
        result = [1, 1] if num1[0] == num2[0] else [0, 0]
        if (result[0] + result[1]) == 0:
            bias = .05
    
    rate = (result[0] + result[1]) / 2 + bias
    if rate >= 1:
        rate = 1
    
    return rate


def update_probabilities1(ocr_result, refresh_rate=0.5):
    """
    根据 OCR 结果更新概率。
    - ocr_result: OCR 识别的结果，可能为空或一个数字（0~99）
    - refresh_rate: 刷新速率，值越大，刷新速度越快，影响更新的显著性
    """
    global priors
    global ocr_error_model_100
    if ocr_result == '' or ocr_result is None:
        # 忽略空结果，不更新概率
        return
    
    # 转换成整数（0-99）范围的数字
    try:
        ocr_result = int(ocr_result)
    except ValueError:
        # 如果 OCR 结果不是合法的数字，直接忽略
        return
    
    if ocr_result >= 100:
        return

    # 4. 更新概率
    # 计算每个实际数字给定 OCR 结果的可能性（即 P(OCR = ocr_result | 实际为 X)）
    likelihoods = ocr_error_model_100[:, ocr_result]

    # 使用贝叶斯公式更新后验概率
    # priors = priors * likelihoods

    # 归一化 (使所有概率加起来为1)

    # 应用概率刷新率，防止单次更新导致波动过大
    # priors = priors * likelihoods 
    priors = priors * (1 - refresh_rate) + likelihoods * refresh_rate
    # print(likelihoods)
    priors /= np.sum(priors)
    # print(priors)
    

# def gradient_descent_update(priors, y_true, ocr_result, learning_rate=.01):
#     """
#     使用梯度下降更新 ocr_error_model_100 的参数
#     - ocr_error_model_100: OCR 错误概率矩阵 (100 x 100)
#     - priors: 当前贝叶斯模型输出的概率分布 (100,)
#     - y_true: 真实的标签 (one-hot 编码, 100,)
#     - ocr_result: 当前 OCR 识别出的结果 (整数)
#     - learning_rate: 学习率，控制每次参数更新的幅度
#     """
#     # 计算交叉熵损失
#     loss = compute_loss(priors, y_true)
    
#     # 计算损失函数相对于 ocr_error_model_100 的梯度
#     global ocr_error_model_100
#     grad = np.zeros_like(ocr_error_model_100)
    
#     # 计算针对每个实际值 X 的梯度
#     for actual_value in range(100):
#         # 计算该位置的梯度，使用交叉熵公式的导数
#         grad[actual_value, ocr_result] = -y_true[actual_value] / (priors[actual_value] + 1e-8)
    
#     # 使用梯度下降更新 OCR 错误概率矩阵
#     ocr_error_model_100 -= learning_rate * grad
#     ocr_error_model_100 = ocr_error_model_100 / np.sum(ocr_error_model_100)
    
#     return ocr_error_model_100, loss


# def compute_loss(priors, y_true):
#     """
#     计算交叉熵损失
#     - priors: 贝叶斯模型输出的概率分布 (100,)
#     - y_true: 真实的标签，one-hot编码 (100,)
#     """
#     # 为了防止log(0)的情况，加入一个小值epsilon
#     epsilon = 1e-8
#     priors = np.clip(priors, epsilon, 1.0 - epsilon)
    
#     # 计算交叉熵损失
#     loss = -np.sum(y_true * np.log(priors))
    # return loss


# def train(priors, actual_result):
#     y_true = np.zeros((100, ))
#     y_true[actual_result] = 1
    
#     global ocr_error_model_100
#     print("更新前=")
#     print(ocr_error_model_100)
#     ocr_error_model_100, loss = gradient_descent_update(priors, y_true, actual_result)
#     print("更新后=")
#     print(ocr_error_model_100)
#     print(f"loss={loss}")
#     np.save(bayes_model, ocr_error_model_100)
    
def getPriors():
    global priors
    return priors


def get_most_likely_number1(thresh: float=.833, predictThresh: float=.01125):
    global priors
    sorted_indices = np.argsort(priors)
    
    for i in range(100):
        print(f"{sorted_indices[i]}:{priors[sorted_indices[i]]}", end=" ")
    
    if np.sort(priors)[-1] < predictThresh:
        return -1
        
    return int(sorted_indices[-1])

reset_priors()

ocr_error_model_100 = gen_ocr_error_model_100()
get_most_likely_number1()