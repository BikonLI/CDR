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
    ocr_error_model_100 = np.zeros((100, 100))