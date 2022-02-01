# -*- coding: utf-8 -*-
'''
Created on Jan 25, 2022
@author: Xingchen Li
'''


from numpy import *
from stumpClassify import *
#adaBoost algorithm
#@dataArr：Data matrix
#@classLabels: Label vector
#@numIt: The number of iterations   
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    '''
    @adaBoost algorithm
    @dataArr：Data matrix
    @classLabels: Label vector
    @numIt: The number of iterations     
    '''
    #Weak classifier information list
    weakClassArr=[]
    #Gets the number of rows of the dataset
    m=shape(dataArr)[0]
    #Initialize each term of the weight vector to be equal
    D=mat(ones((m,1))/m)
    #The cumulative estimate vector
    aggClassEst=mat((m,1))
    #Number of iterations
    for i in range(numIt):
        #The optimal single-layer decision tree is established according to the current data set, label and weight
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)
        #Print weight vector
        # print("D:",D.T)
        #Find the coefficient 'alpha' of the single-layer decision tree
        alpha=float(0.5*log((1.0-error)/(max(error,1e-16))))
        #Stores the decision tree's coefficient 'alpha' into the dictionary
        bestStump['alpha']=alpha
        #Store the decision tree to the list
        weakClassArr.append(bestStump)
        #Print the prediction results of the decision tree
        # print("classEst:",classEst.T)
        #Exp (-alpha) for the right prediction, exp(alpha) for the wrong prediction.
        #That is to increase the weight of incorrectly classified samples and reduce the weight of correctly classified data points
        expon=multiply(-1*alpha*mat(classLabels).T,classEst)
        #Update the weight vector
        D=multiply(D,exp(expon))
        D=D/D.sum()
        #Sum the weighted predicted value of the current single-layer decision tree
        aggClassEst = aggClassEst + alpha * classEst
        #aggClassEst = array(aggClassEst)
        # print("aggClassEst",aggClassEst.T)
        #Find the number of misclassified samples
        aggErrors=multiply(sign(aggClassEst)!=\
                    mat(classLabels).T,ones((m,1)))
        #Calculate the error rate
        errorRate=aggErrors.sum()/m
        print("total error:",errorRate,"\n")
        #Exit the loop with an error rate of 0.0
        if errorRate==0.0:break
    #Returns a combinatorial list of weak classifiers
    return weakClassArr, aggClassEst


#Test adaBoost，adaBoostClassification function
#@datToClass:Test data point
#@classifierArr：Build the final classifier
def adaClassify(datToClass,classifierArr):
    #Construct data vectors or matrices
    dataMatrix=mat(datToClass)
    #Gets the number of rows of the matrix
    m=shape(dataMatrix)[0]
    #Initialize the final classifier
    aggClassEst=mat(zeros((m,1)))
    #Iterate over each weak classifier in the classifier list
    for i in range(len(classifierArr)):
        #Each weak classifier classifies the test data in a predictive way
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                classifierArr[i]['thresh'],
                                classifierArr[i]['ineq'])
        #The prediction results of each classifier are weighted and accumulated
        aggClassEst+=classifierArr[i]['alpha']*classEst
        # print('aggClassEst',aggClassEst)
    #The sign function predicts +1 or -1 based on whether the result is greater than or less than 0
    return sign(aggClassEst)

#Adaptive data loading functions
def loadDataSet(filename):
    #Create data set matrix, label vector
    dataMat=[];labelMat=[]
    #Get the number of features (including tags of the last category)
    #readline():Read a line from a file
    #readlines:Read all lines of the entire file
    numFeat=len(open(filename).readline().split('\t'))
    #Open the file
    fr=open(filename)
    #Iterate over each line of text
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        #Data matrix
        dataMat.append(lineArr)
        #Label vector
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#Train and test classifiers
def classify():
    #Use training set to train classifier
    datArr,labelArr=loadDataSet('horseColicTraining.txt')
    #Get trained classifiers
    classifierArray = adaBoostTrainDS(datArr,labelArr,100)
    #The test set is used to test the classification effect of classifier
    testArr,testLabelArr=loadDataSet('horseColicTest.txt')
    prediction=adaClassify(testArr,classifierArray[0])
    #Output error rate
    error = mat(ones((67,1)))
    err_num = error[prediction != mat(testLabelArr ).T].sum() 
    error_rate = float(err_num)/float(67)
    print("the errorRate is: %.4f" % error_rate)
