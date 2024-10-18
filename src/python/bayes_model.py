import numpy as np

# 1. 定义 OCR 识别概率模型
# 每一行表示某个数字被识别为其它数字的概率分布
# 行是实际数字，列是OCR识别结果
ocr_error_model_10 = np.zeros((10, 10))
np.fill_diagonal(ocr_error_model_10, 0.9)

# 手动设置一些常见的识别错误
# ocr_error_model_10[0, 6] = 0.04 将0识别为6的概率（伪），之后会归一化
# ocr_error_model_10[0, 6] = 0.04
# ocr_error_model_10[6, 0] = 0.05
# ocr_error_model_10[8, 9] = 0.04
# ocr_error_model_10[9, 8] = 0.04
# ocr_error_model_10[1, 7] = 0.04
# ocr_error_model_10[7, 1] = 0.06
# ocr_error_model_10[3, 8] = 0.03
# ocr_error_model_10[8, 3] = 0.02
# ocr_error_model_10[8, 4] = 0.03
# ocr_error_model_10[9, 4] = 0.045
# ocr_error_model_10[8, 0] = 0.06
# ocr_error_model_10[0, 8] = 0.01
# ocr_error_model_10[8, 2] = 0.03
# ocr_error_model_10[8, 4] = 0.03
# ocr_error_model_10[6, 4] = 0.03
ocr_error_model_10[4, 7] = 0.06

for i in range(10):
    for j in range(10):
        if i != j and ocr_error_model_10[i, j] == 0:
            ocr_error_model_10[i, j] = 0.01
            
ocr_error_model_10 = ocr_error_model_10 / ocr_error_model_10.sum(axis=1, keepdims=True)

priors = None
tens = [0, 0] # tens/total
tens_statistic = {} # {"num": count}
singles_statistic = {}

def state_num(num):
    global tens_statistic
    
    if num <= 9:
        if singles_statistic.get(f"{num}") is None:
            singles_statistic[f"{num}"] = 1
        singles_statistic[f"{num}"] += 1
        return
    
    if tens_statistic.get(f"{num}") is None:
        tens_statistic[f"{num}"] = 1
    tens_statistic[f"{num}"] += 1

# 重置概率矩阵
def reset_priors():
    global priors
    global tens
    global tens_statistic
    global singles_statistic
    priors = np.ones(100)
    priors = priors / 100
    tens = [0, 0]
    tens_statistic = {}
    singles_statistic = {}
    
reset_priors()
    
ocr_error_model_100 = np.zeros((100, 100))

# 利用 10x10 矩阵推广到 100x100
for true_number in range(100):  # 真值 (0-99)
    true_tens = true_number // 10  # 真值的十位数
    true_ones = true_number % 10   # 真值的个位数
    
    for recognized_number in range(100):  # 被识别为的数字 (0-99)
        recognized_tens = recognized_number // 10  # 识别结果的十位数
        recognized_ones = recognized_number % 10   # 识别结果的个位数
        
        if recognized_tens != 0:
            weight = .35
        else: weight = 1
        
        # 误识别的概率 = 十位误识别概率 * 个位误识别概率
        ocr_error_model_100[true_number, recognized_number] = (
            ocr_error_model_10[true_tens, recognized_tens] *  # 十位误识别概率
            ocr_error_model_10[true_ones, recognized_ones] *  # 个位误识别概率
            weight
        )


# 归一化处理，确保每行的概率加起来为1
ocr_error_model_100 = ocr_error_model_100 / ocr_error_model_100.sum(axis=1, keepdims=True)

# 3. 定义函数处理 OCR 结果
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

    global tens
    tens[1] += 1
    state_num(ocr_result)
    
    if ocr_result >= 10:
        tens[0] += 1
    cur = refresh_rate
    
    if ocr_result >= 10 and tens[0] >= 2:
        if f"{ocr_result}" == max(tens_statistic, key=lambda x: tens_statistic[x]):
            nor = 1 / (tens_statistic[f"{ocr_result}"] ** 1.5)
            wt =  (1 - nor) * 1
        else:
            wt = .40
        
        refresh_rate = refresh_rate + (1 - refresh_rate) * wt
    else: refresh_rate = cur

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
    

def get_most_likely_number(thresh: float=.833, predictThresh: float=.3):
    """
    返回最有可能的球员背号
    阈值越高，越有可能是两位数。若thresh为0，则必然为两位数。           
    """ 
    # print(np.sort(priors))
    
    sorted_indices = np.argsort(priors)                     # 本段代码逻辑如下
    # print(sorted_indices)                                     # 计算有多少个两位数预测，若超过阈值，则认为是两位数
                                                              # 若不是小于阈值，则返回概率最大的（大概率是一个一位数）
    thresh = 1 - thresh                                       # 若大于阈值，若概率最大的为一个两位数，那么返回它
                                                              # 若不是，则检索概率最大的两个一位数。
    if tens[1] == 0:                                          # 寻找由这两位数字组成的10位数且概率最大的。
        return -1                                             # 例如，概率最大的为，1、2，那么会找12和21，
    
    try:
        max_num =  max(tens_statistic, key=lambda x: tens_statistic[x])
        if tens_statistic[max_num] >= 5 and (int(max_num) in sorted_indices[-4:]):
            return int(max_num)
        
    except ValueError:
        pass
    
    if np.sort(priors)[-1] < predictThresh:
        return -1
    
    singles_list = sorted(singles_statistic, key=lambda x: singles_statistic[x])
    try:
        first = singles_list[-1]
        second = singles_list[-2]
        summer = sum((singles_statistic[i] for i in singles_list[:-2]))
        if singles_statistic[second] > summer * 3.7:
            thresh = thresh * .55
        elif singles_statistic[second] > summer * 2:
            thresh = thresh * .7
        elif singles_statistic[second] > summer * 1.5:
            thresh = thresh * .8
    except Exception:
        pass
    
    
    if (tens[0] / tens[1]) > thresh:                          # 并判断这两个哪个概率更大，返回概率大的。
        
        if sorted_indices[-1] >= 10:
            return sorted_indices[-1]
        
        index = len(sorted_indices) - 1
        number_list = []
        j = 0
        for i in range(len(sorted_indices)):
            number = sorted_indices[index]
            index -= 1
            
            if j == 2:
                break
            
            if number <= 9:
                number_list.append(number)
                j += 1
        
        index = len(sorted_indices) - 1
        for i in range(len(sorted_indices)):
            number = sorted_indices[index]
            index -= 1
            if number >= 10:
                if number_list[0] == 0 or number_list[1] == 0:
                    return max(number_list) * 10
                if f"{number}" == f"{number_list[0]}{number_list[1]}" or \
                    f"{number}" == f"{number_list[1]}{number_list[0]}":
                    if number == 99:
                        number = -1
                    return number
    number = sorted_indices[-1]
    if number == 99:
        number = -1
    return number

if __name__ == "__main__":
    tests = [
        [7, 1, 7, 4, 7, 7, 0, 7, 7, 7, 7, 7, 4, 7, 3, 7, 7, 7, 1, 7],   # 7 15%
        [3, 0, 3, 3, 3, 8, 3, 0, 3, 9, 8, 3, 3, 3, 5, 3, 3, 3, 8, 0, 3, 3], # 3 25%
        [0, 0, 0, 0, 8, 8, 0, 0, 0, 8, 3, 0, 0, 9, 0, 6, 0, 0, 7, 0, 5],    # 0 20%
        [8, 8, 0, 8, 8, 0, 8, 8, 0, 3, 8, 8, 8, 0, 0, 8, 8, 0, 8, 8, 0],    # 8 20%
        [1, 1, 7, 1, 1, 1, 0, 4, 1, 1, 1, 9, 7, 1, 1, 1, 0, 1, 1, 1, 1],    # 1 20%
        [45, 45, 45, 4, 5, 45, 4, 4, 8, 45, 9, 45, 45, 0, 45, 4, 5, 45, 45, 45, 4, 9, 7, 5, 4], # 45 25%
        [92, 9, 9, 92, 2, 0, 92, 9, 92, 9, 9, 9, 92, 8, 4, 2, 92, 2, 92, 4, 92, 9, 2, 7, 92],   # 92 25%
        [67, 6, 67, 7, 67, 67, 6, 67, 9, 67, 0, 67, 8, 67, 67, 67, 6, 67, 7, 6, 7, 6, 5, 67],   # 67 20%
        [8, 83, 3, 83, 83, 0, 8, 83, 4, 83, 8, 3, 0, 83, 83, 8, 8, 83, 4, 83, 8, 3],    # 83 20%
        [1, 14, 14, 1, 4, 1, 14, 14, 4, 14, 14, 9, 14, 1, 7, 0, 14, 4, 14, 1, 14],  # 14 20%
        [25, 2, 5, 25, 5, 2, 5, 0, 25, 25, 5, 25, 25, 25, 25, 0, 2, 2, 25, 25, 5, 25, 25],  # 25 15%
        [5, 6, 56, 6, 56, 6, 5, 56, 6, 56, 56, 4, 5, 5, 56, 6, 56, 6, 5, 56, 0, 56, 56],    # 56 15%
        [78, 8, 78, 7, 8, 78, 0, 78, 9, 8, 78, 8, 7, 8, 78, 0, 78, 8, 7, 7, 7, 0],      # 78 20%
        [31, 1, 3, 1, 31, 31, 3, 3, 0, 31, 3, 3, 9, 3, 31, 1, 31, 0, 1, 3, 1, 0],       # 31 25%
        [5, 5, 50, 5, 0, 50, 5, 5, 50, 0, 50, 50, 5, 5, 5, 50, 50, 50, 0, 5, 50],       # 50 15%
        [9, 9, 94, 4, 9, 9, 9, 9, 94, 94, 94, 94, 9, 0, 94, 4, 94, 0, 94, 94, 94],      # 94 10%
        [2, 2, 22, 2, 22, 2, 2, 22, 22, 2, 4, 22, 22, 22, 0, 22, 2, 22, 2, 22, 2, 22],  # 22 15%
        [61, 1, 1, 1, 6, 61, 61, 61, 1, 6, 1, 1, 61, 1, 6, 61, 6, 1, 0, 6, 1, 1],       # 61 20%
        [11, 1, 1, 1, 11, 0, 1, 1, 1, 11, 1, 1, 1, 1, 0, 1, 1, 1, 9, 11, 0, 1, 0],      # 11 25%
        [2, 29, 29, 2, 2, 9, 9, 29, 29, 29, 9, 0, 29, 2, 29, 0, 2, 2, 29],              # 29 25%
    ]

    real = [7, 3, 0, 8, 1, 45, 92, 67, 83, 14, 25, 56, 78, 31, 50, 94, 22, 61, 11, 29]

    # 遍历 OCR 结果并更新概率
    trues = 0
    total = 0
    i = 0
    for test in tests:
        for result in test:
            update_probabilities(result, .05)
        most_likely_number = get_most_likely_number()
        if most_likely_number == real[i]:
            trues += 1
        total += 1
        print(f"最有可能的球员背号是: {most_likely_number}")
        i += 1
        reset_priors()

    print(f"{trues}/{total} accuracy: {trues / total * 100:.2f}%")
