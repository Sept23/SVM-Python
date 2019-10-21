import numpy
from sklearn import svm
import os

TRAINING_FILE='train.csv'
TEST_FILE='test.csv'

def get_data(filename):
    train_data = []  # Vector X
    train_lable = []  # Vector Y
    with open(filename,'r') as f:
        for line in f.readlines():
            line=line.strip('\n').strip(' ')
            line=list(map(float,line.split(',')))
            if(line[0]==float(0)):
                train_lable.append(float(-1))
            else:
                train_lable.append(line[0])
            train_data.append(line[1:])
    return train_data,train_lable

def predict(w,b,X_testData,Y_testLable):
    pointNum,featureNum=X_testData.shape
    print(pointNum)
    result=[]
    w=numpy.array(w)
    count=0
    for i in range(pointNum):
        # print(testData[i])
        predictNum=numpy.dot(w,testData[i])+b
        if predictNum>0:
            # print(numpy.dot(w.T, testData[i]) + b)
            # print(numpy.dot(w.T, testData[i]) + b > 1)
            result.append(float(1))
        elif predictNum <0:
            # print(numpy.dot(w.T, testData[i]) + b)
            result.append(float(-1))
    for j in range(pointNum):
        if float(result[j])==float(Y_testLable[j]):
            count=count+1
    print(count)
    accuate=count/pointNum
    return accuate



def findSV(alpha,C):
    for i in range(len(alpha)):
        if alpha[i]>0 and alpha[i]<C:
            return i

if __name__ == '__main__':

    curretnpath=os.path.abspath('.')+'/'
    print(curretnpath)
    trainData, trainLable = get_data(os.path.join(curretnpath,TRAINING_FILE))
    testData, testLable = get_data(os.path.join(curretnpath,TEST_FILE))
    train_dataMat = numpy.array(trainData)
    train_lableMat = numpy.array(trainLable)
    testDataMat = numpy.array(testData)
    testLableMat = numpy.array(testLable)
    # model = svm_train(train_lableMat, train_dataMat, '-t 2 -g 0.005 -c 0.25')
    # print(model.nSV, " ")
    # p_labs, p_acc, p_vals = svm_predict(testLableMat, testDataMat, model)
    # print(p_labs, "  ", p_acc, " ", p_vals)
    # c = predict(testLableMat, p_labs)

    clf = svm.SVC(C=0.25)
    clf.fit(train_dataMat, train_lableMat)
    print(clf.score(testDataMat,testLableMat))
    alphas=clf.dual_coef_
    w = numpy.matmul(clf.dual_coef_,clf.support_vectors_)
    b = clf.intercept_
    print("w_libsvm ",w.flatten()," ",b)
    # dual_accurate = predict(w, b, testDataMat, testLableMat)
    # print("Accurate of Dual Problem is: %f" % dual_accurate)
    print("norm of libsvm: ",numpy.linalg.norm(w), "mean of libsvm ",numpy.mean(w),"mean of alpha ",numpy.mean(alphas),"norm of alphsa",numpy.linalg.norm(alphas))
