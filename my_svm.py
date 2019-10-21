import cvxopt
import numpy
import os
from numpy import linalg

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

def gaussian_kernel(x1, x2, sigma=1/200):
    return numpy.exp(-linalg.norm(x1-x2)**2 / (2 * (sigma ** 2)))

def primal_soft(train_data,train_lable,C=0.25):
    '''
    n represent sample num
    m represent feature num
    '''
    # cvxopt.solvers.options['show_progress'] = False
    n,m=train_dataMat.shape
    #vector (w,b,r)
    p = numpy.zeros((m+n+1, m+n+1))
    for i in range(m):
            p[i,i]=1
    q = numpy.vstack([numpy.zeros((m+1,1)), C*numpy.ones((n,1))])
    a = []
    for i in range(n):
        tem0 = numpy.zeros((1, n))
        tem1=tem0.tolist()[0]
        tem1[i]=train_lable[i]
        tmp = train_data[i]+[1]+tem1
        for j in range(m +n+ 1):
            tmp[j] = -1 * train_lable[i] * tmp[j]
        a.append(tmp)
    A=numpy.array(a)
    A1=numpy.zeros((n,m+1))
    A2=numpy.eye(n)*-1
    A3=numpy.hstack((A1,A2))
    G=numpy.vstack((A,A3))
    # print(G.shape)
    p=cvxopt.matrix(p)
    q=cvxopt.matrix(q)
    g=cvxopt.matrix(G)
    h = numpy.zeros((2*n,1))
    h[:n]=-1
    h=cvxopt.matrix(h)
    sol = cvxopt.solvers.qp(p, q, g, h)
    print(sol)
    return sol
def dual_soft(C = 0.25):

    n_samples, n_features = train_dataMat.shape

    # #Gaussian
    # K = numpy.zeros((n_samples, n_samples))
    # for i in range(n_samples):
    #     for j in range(n_samples):
    #         K[i, j] = gaussian_kernel(train_dataMat[i],train_dataMat[j])
    # P = cvxopt.matrix(numpy.outer(train_lableMat,train_lableMat)*K)


    #no kernel
    P=numpy.outer(train_lableMat,train_lableMat)*numpy.dot(train_dataMat,train_dataMat.T)
    P = cvxopt.matrix(P)


    q = cvxopt.matrix(numpy.ones(n_samples) * -1)
    A = cvxopt.matrix(train_lableMat, (1, n_samples))
    b = cvxopt.matrix(0.0)


    tmp1 = numpy.diag(numpy.ones(n_samples) * -1)
    tmp2 = numpy.identity(n_samples)
    G = cvxopt.matrix(numpy.vstack((tmp1, tmp2)))
    tmp1 = numpy.zeros(n_samples)
    tmp2 = numpy.ones(n_samples) * C
    h = cvxopt.matrix(numpy.hstack((tmp1, tmp2)))

    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    alphas = numpy.array(solution['x'])
    print(solution)
    return alphas

def predict(w,b,X_testData,Y_testLable):
    pointNum,featureNum=X_testData.shape
    print(pointNum)
    result=[]
    w=numpy.array(w)
    count=0
    for i in range(pointNum):
        # print(testData[i])
        predictNum=numpy.dot(w.T,testData[i])+b
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

def caculateW_B(train_dataMat,train_lableMat,alphas):
    w = numpy.dot(train_dataMat.T, train_lableMat * alphas)
    index = findSV(alphas, C)
    print(alphas[index])
    b = train_lableMat[index] - numpy.dot(train_dataMat[index], w)
    return w,b

def getDualityGap(w_primal,b_primal,w_dual,b_dual):

    w_primal=numpy.array(w_primal).flatten()
    b_primal=numpy.array(b_primal).flatten()
    print("norm of w_primal: ",numpy.linalg.norm(w_primal), "mean of w_primal :", numpy.mean(w_primal))

    w_dual=numpy.array(w_dual).flatten()
    b_dual=numpy.array(b_dual).flatten()
    print("norm of w_dual: ",numpy.linalg.norm(w_dual),"mean of w_primal :",numpy.mean(w_dual))

    primal_sol=numpy.hstack((w_primal,b_primal))
    print()
    dual_sol=numpy.hstack((w_dual,b_dual))
    # norm_primal=numpy.linalg.norm(primal_sol)
    # norm_dual=numpy.linalg.norm(dual_sol)

    # norm_primal=numpy.linalg.norm(primal_sol)
    # norm_dual=numpy.linalg.norm(dual_sol)
    print("Difference between primal and dual \n",primal_sol-dual_sol)
    temp=0
    for i in range(len(primal_sol)):
        temp=primal_sol[i]-dual_sol[i]+temp
    return temp


if __name__ == '__main__':

    C=0.25
    curretnpath=os.path.abspath('.')+'/'
    trainData, trainLable = get_data(os.path.join(curretnpath, TRAINING_FILE))
    testData, testLable = get_data(os.path.join(curretnpath, TEST_FILE))
    train_dataMat=numpy.array(trainData)
    train_lableMat=numpy.array(trainLable)
    testDataMat=numpy.array(testData)
    testLableMat=numpy.array(testLable)
    n,m=train_dataMat.shape
    print(len(train_lableMat))

    #primal soft problem successful 1453 0.968667
    print("Solving Primal Problem")
    primal_sol=primal_soft(trainData,trainLable)
    w_primal=primal_sol['x'][:m]
    b_primal=primal_sol['x'][m]
    primal_accurate=predict(w_primal,b_primal,testDataMat,testLableMat)
    w_primal=numpy.array(w_primal).flatten()
    print("w and b of primal \n",w_primal," ",b_primal)
    print("Accurate of Primal Problem is: %f" % primal_accurate)
    print("Finish Primal Problem")

    # dual soft problem  gaussian 1369  0.9126666666666666 no kernel: 1500 1384 0.922667
    print("Solving Dual Problem")
    alphas = dual_soft()
    alphas=alphas.flatten()
    print('mean of alpha: ', numpy.mean(alphas),"norm of alphas ",numpy.linalg.norm(alphas))
    w_dual,b_dual=caculateW_B(train_dataMat,train_lableMat,alphas)
    print('w and b of dual \n', w_dual,"    ",b_dual)
    dual_accurate=predict(w_dual,b_dual,testDataMat,testLableMat)
    print("Accurate of Dual Problem is: %f" % dual_accurate)
    print("Finish Dual Problem")
    #
    dualityGap=getDualityGap(w_primal,b_primal,w_dual,b_dual)
    print("duality gap is :",dualityGap)