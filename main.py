'''
Created on Jan 25, 2022
@author: Xingchen Li
'''

from adaboost import *
from ROC_plot import *
from stumpClassify import *

if __name__ == "__main__":
    datArr,labelArr = loadDataSet('horseColicTraining.txt')   
    classifierArray, aggClassEst = adaBoostTrainDS(datArr,labelArr,1000)
    plotROC(aggClassEst.T,labelArr)
    # classify()