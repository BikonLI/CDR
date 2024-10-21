import numpy as np
import os

bayes_model = "./bayes.npy"
if os.path.exists(bayes_model):
    ocr_error_model_100 = np.load(bayes_model)
    
else:

    ocr_error_model_10 = np.zeros((10, 10))
    np.fill_diagonal(ocr_error_model_10, 0.9)

    for i in range(10):
        for j in range(10):
            if i != j and ocr_error_model_10[i, j] == 0:
                ocr_error_model_10[i, j] = 0.01
                
    ocr_error_model_10 = ocr_error_model_10 / ocr_error_model_10.sum(axis=1, keepdims=True)

    priors = None
    tens = [0, 0] # tens/total


# 重置概率矩阵
def reset_priors():
    global priors
    global tens
    
    priors = np.ones(100)
    priors = priors / 100
    tens = [0, 0]

        
reset_priors()
ocr_error_model_100 = np.random() ###


def update_probabilities(ocr_result, refresh_rate=0.5):
    """
    根据 OCR 结果更新概率。
    - ocr_result: OCR 识别的结果，可能为空或一个数字（0~99）
    - refresh_rate: 刷新速率，值越大，刷新速度越快，影响更新的显著性
    """
    global priors
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
    priors /= np.sum(priors)
    

def gradient_descent_update(priors, y_true, ocr_result, learning_rate=0.005):
    """
    使用梯度下降更新 ocr_error_model_100 的参数
    - ocr_error_model_100: OCR 错误概率矩阵 (100 x 100)
    - priors: 当前贝叶斯模型输出的概率分布 (100,)
    - y_true: 真实的标签 (one-hot 编码, 100,)
    - ocr_result: 当前 OCR 识别出的结果 (整数)
    - learning_rate: 学习率，控制每次参数更新的幅度
    """
    # 计算交叉熵损失
    loss = compute_loss(priors, y_true)
    
    # 计算损失函数相对于 ocr_error_model_100 的梯度
    grad = np.zeros_like(ocr_error_model_100)
    
    # 计算针对每个实际值 X 的梯度
    for actual_value in range(100):
        # 计算该位置的梯度，使用交叉熵公式的导数
        grad[actual_value, ocr_result] = -y_true[actual_value] / (priors[actual_value] + 1e-8)
    
    # 使用梯度下降更新 OCR 错误概率矩阵
    ocr_error_model_100 -= learning_rate * grad
    
    return ocr_error_model_100, loss


def compute_loss(priors, y_true):
    """
    计算交叉熵损失
    - priors: 贝叶斯模型输出的概率分布 (100,)
    - y_true: 真实的标签，one-hot编码 (100,)
    """
    # 为了防止log(0)的情况，加入一个小值epsilon
    epsilon = 1e-8
    priors = np.clip(priors, epsilon, 1.0 - epsilon)
    
    # 计算交叉熵损失
    loss = -np.sum(y_true * np.log(priors))
    return loss


def train(priors, actual_result):
    y_true = np.zeros((100, ))
    y_true[actual_result] = 1
    
    global ocr_error_model_100
    ocr_error_model_100, loss = gradient_descent_update(priors, y_true, actual_result)
    print(f"loss={loss}")
    np.save(bayes_model, ocr_error_model_100)
    
def getPriors():
    return priors
