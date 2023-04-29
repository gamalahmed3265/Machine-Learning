import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.impute import SimpleImputer
import cv2
import os
import math
import random
import random as rnd
from operator import mod

# # #arrange=np.ones((4,3))*np.zeros((4,3))
# # #z=np.zeros(5)
# # # ls=[[2,3,4,5,6],[2,3,4,5,6],[2,3,4,5,6]]
# # # print(ls)
# # # arr=np.array(ls)
# # # print(arr)
# # # arrange=np.azero(3,12,3)
# # # print(arrange)
# # # rand=np.random.rand(3)
# # # rand=np.random.rand(2,4)
# # # rand2=np.random.randint(5,100,(5,2))
# # # rand=np.random.randint(5,100,10)
# # # print(rand)

# # # shape=rand.reshape(5,2)
# # # print(shape)
# # #ran=np.random.randint(5,100,(5,2))
# # # ran=np.arange(1,20,2)
# # # np.random.shuffle(ran)
# # # shape=ran.reshape(2,3)
# # # print(ran)
# # # max=ran.max()
# # # print(max)


# # # ls1=[[2,3,4,5,6],[2,3,4,5,6],[2,3,4,5,6]]
# # # ls2=[[2,3,4,5,6],[2,3,4,5,6],[2,3,4,5,6]]

# # # print(np.divide(ls1,ls2))


# # # ls=[[2,3,4,5,6],[2,3,4,5,6],[2,3,4,5,6]]
# # # ls=["gamal","ahmed","elsayed","hassen"]
# # # ls=[2,3,4,5,6]
# # # ser=pn.Series(ls)

# # # print(ser)

# # # print(ser.describe())
# # # print(ser.sum())




# # # ls=[2,3,4,5,6]
# # # ser=pn.DataFrame(ls,index=["a","b","c","x","e"],columns=["first"])

# # # print(ser)



# # # l1=["gamal",2000,1500,3948495]
# # # l2=["ahmed",3566,2000,3948495]
# # # l3=["tarak",7989,6000,3948495]
# # # l4=["ali",8975,7866,3948495]
# # # l5=["mohamed",2334,1200,3948495]


# # # fr=pn.DataFrame([l1,l2,l3,l4,l5],index=["A","B","C","E","S"],columns=["NAME","INCOME","WITHDRAW","PHONE"])


# # # print(fr)
# # # print("*"*40)
# # # print(fr.describe())
# # # print("*"*40)
# # # print(fr["NAME"])
# # # print(fr["NAME"].describe())
# # # print("*"*40)
# # # print(fr.loc["A":"C","INCOME":"PHONE"])#  return     row : column
# # # print("*"*40)
# # # print(fr.loc["A":"C","INCOME":"PHONE"].describe())
# # # print("*"*40)
# # # print(fr.iloc[2:4,3:5])



# # # ser=pn.Series(["a","b","a"])

# # # print(ser)









# # # l1=["gamal",2000,"A",1500,"gamal@gmail.com",3948495,"y"]
# # # l2=["ahmed",3566,"B",2000,"ahmed@gmail.com",3948495,"n"]
# # # l3=["tarak",7989,"A",6000,"tarak@gmail.com",3948495,"y"]
# # # l4=["ali",8975,"C",7866,"ali@gmail.com",3948495,"y"]
# # # l5=["mohamed",2334,"C",1200,"mohamed@gmail.com",3948495,"n"]




# # # da=pn.DataFrame([l1,l2,l3,l4,l5],index=["a","b","c","e","f"],columns=["NAME","INCOME","CLASS","WITHDRAW","email","PHONE","ACTIVE"])




# # # print(da)
# # # print("*"*40)
# # # print(da.describe())












# # path="C:\\Projects\\Collage\ML\\ML.csv"

# # data=pn.read_csv(path,header=None,names=["Population","Profit"])

# # # print('new data = \n' ,data.head(10))
# # # print("data describe \n",data.describe)
# # # print("*"*40)
# # # print("data describe \n",data.describe())

# # data.plot(kind="scatter",x="Population",y="Profit",figsize=(5,5))



# # data.insert(0,"ones",1)
# # # print("new data after insert ",data.head(10))


# # cols=data.shape[1]

# # # print("collumn",cols)

# # x=data.iloc[: , :cols-1]
# # y=data.iloc[: , cols-1:cols]

# # # print(x.head(10))
# # # print(y.head(10))

# # x=np.matrix(x.values)
# # y=np.matrix(y.values)       

# # theta=np.matrix(np.array([0,0])) #[0,0]  (1, 2)

# # # print("matrix data Population",x.shape)
# # # print("matrix data Profit",y)
# # # print(np.array([0,0]))
# # # print(np.matrix(np.array([0,0])).shape)





# # def computCost(x,y,theta):
    
# #     z=np.power((x*theta.T)-y,2)
   
# #     return np.sum(z)/(2*len(x))
    
# # # print(computCost(x, y, theta))




# # def gradientDescnet(x,y,theta,alpha,iters):
    
# #     temp=np.matrix(np.zeros(theta.shape))  #[0,0]     (1, 2)
# #     paramters=int(theta.ravel().shape[1]) #(1, 2)  [1]  2
# #     costs=np.zeros(iters)                #(1,1)        0 *1000
    
    
# #     for i in range(iters):
# #         error=(x*theta.T)-y
# #         for j in range(paramters):
# #             term=np.multiply(error,x[:,j])
# #             temp[0,j]=theta[0,j]-( (alpha/ len(x) ) * np.sum(term) ) 
# #         theta=temp
# #         costs[i]=computCost(x, y, theta)
# #     return theta,costs





# # alpha=0.01
# # iters=1000


# # # g,costs=gradientDescnet(x, y, theta, alpha, iters)
# # # print("gradientDescnet\n",costs[0:20])
# # # print("*"*40)
# # # print("gradientDescnet\n",g)




# # # x=np.linspace(data.Population.min(), data.Population.max(),100)

# # # print("*"*40)
# # # print("Line space\n",x)


# # # f=g[0,0]+(g[0,1]*x)


# # # print("*"*40)
# # # print("f\n",f)



# # # fig,ax=plt.subplots(figsize=(5,5))

# # # ax.plot(x,f,"g",label="Prediction")
# # # ax.scatter(data.Population,data.Profit,label="traning data")
# # # ax.legend(loc=2)
# # # ax.set_xlabel("Population")
# # # ax.set_ylabel("Profit")
# # # ax.set_title("Prediction Profit VS Population")



# # # fig2,ax2=plt.subplots(figsize=(5,5))
# # # ax2.plot(np.arange(iters),costs,"r")
# # # ax2.legend(loc=2)
# # # ax2.set_xlabel("itertion")
# # # ax2.set_ylabel("costs")
# # # ax2.set_title("costs VS Itertion")





# # sns.heatmap(data.corr(),annot=True)









# # path="C:\\Users\\gamal\\asmaafils\\datamulivarible.txt"

# # data=pd.read_csv(path,names=["bedrooms","size","price"])


# # # print(data.describe())
# # # print(data)

# # data=(data-data.mean())/data.std()

# # # print(data)
# # data.insert(0,"ones",1)
# # # print(data)


# # cols=data.shape[1]
# # # print(cols)


# # x=data.iloc[:,0:cols-1]
# # y=data.iloc[:,cols-1:cols]
# # # print(x.head(20))
# # # print("y\n",y.head(20))
# # # print("y\n",y.values)

# # x=np.matrix(x.values)
# # y=np.matrix(y.values)
# # theta=np.matrix(np.array([0,0,0]))
# # # print(x,theta)

# # alpha=0.1
# # iters=100

# # g,costs=gradientDescnet(x, y, theta, alpha, iters)

# # # print(g)
# # # print(costs)





# # # def drawSupPlpts(x,g):
    
# # # x=np.linspace(data.bedrooms.min(), data.bedrooms.max(),100)

# # #     # print(x)
# # #     f=g[0,0] +(g[0,1]*x)
# # #     # print(f)
    
# # #     fig,axis=plt.subplots(figsize=(5,5))
# # #     axis.plot(x, f, label="bedrooms")
# # #     axis.scatter(data.bedrooms,data.price,label="tring data")
# # #     axis.legend(loc=2)
# # #     axis.set_xlabel("bedrooms")
# # #     axis.set_ylabel("price")
# # #     axis.set_title("Price data")











# # x=np.linspace(data.bedrooms.min(), data.bedrooms.max(),100)

# # # print(x)
# # f=g[0,0] +(g[0,1]*x)
# # # print(f)

# # fig,axis=plt.subplots(figsize=(5,5))
# # axis.plot(x, f,"g", label="bedrooms")
# # axis.scatter(data.bedrooms,data.price,label="tring data")
# # axis.legend(loc=2)
# # axis.set_xlabel("bedrooms")
# # axis.set_ylabel("price")
# # axis.set_title("Price data")




# # fig,axis=plt.subplots(figsize=(5,5))
# # axis.plot(np.arange(iters),costs,"r")
# # axis.set_ylabel("costs")
# # axis.set_xlabel("iterstion")
# # axis.set_title("costs of table")











# path="C:\\Users\gamal\\asmaafils\\Hesham Asem\\dataclassfiction.txt"
# data=pd.read_csv(path,names=["Exam1","Exam2","Admitted"])


# postive=data[data["Admitted"].isin([1])]
# negitive=data[data["Admitted"].isin([0])]
# # print(postive)
# # print(negitive)


# fig,axis=plt.subplots(figsize=(8,5))
# axis.set_title("Exam1 VS Exam2 ")
# axis.set_xlabel("Exam1")
# axis.set_ylabel("Exam2")

# axis.scatter(postive.Exam1,postive["Exam2"],label="postive data",c="b",marker="x")
# axis.scatter(negitive.Exam1,negitive["Exam2"],label="negitive data",c="g",marker="o")
# axis.legend()


# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))
# nums = np.arange(-10, 10, step=1)
# fig, ax = plt.subplots(figsize=(5,5))

# ax.plot(nums, sigmoid(nums), 'r')

# def cost(theta, X, y):
#     theta = np.matrix(theta)
#     X = np.matrix(X)
#     y = np.matrix(y)
#     first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
#     second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
#     return np.sum(first - second) / (len(X))


# # add a ones column - this makes the matrix multiplication work out easier
# data.insert(0, 'Ones', 1)
# # set X (training data) and y (target variable)
# cols = data.shape[1]
# X = data.iloc[:,0:cols-1]
# y = data.iloc[:,cols-1:cols]
# #convart yo matrix
# # x=np.matrix(x)
# # y=np.matrix(y)
# # theta=np.matrix(np.array(np.zeros(3)))
# # or
# # theta=np.matrix(np.array([0,0,0]))
# #or

# # convert to numpy arrays and initalize the parameter array theta


# X = np.array(X.values)
# y = np.array(y.values)
# theta = np.zeros(3)

# # print()
# # print('X.shape = ' , X.shape)
# # print('theta.shape = ' , theta.shape)
# # print('y.shape = ' , y.shape)
# thiscost = cost(theta, X, y)
# # print()
# # print('cost = ' , thiscost)


# def gradient(theta, X, y):
#     theta = np.matrix(theta)
#     X = np.matrix(X)
#     y = np.matrix(y)
#     parameters = int(theta.ravel().shape[1])
#     grad = np.zeros(parameters)
#     error = sigmoid(X * theta.T) - y
#     for i in range(parameters):
#         term = np.multiply(error, X[:,i])
#         grad[i] = np.sum(term) / len(X)
#     return grad


# result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))



# print(result,"*"*40)
# costafteroptimize = cost(result[0], X, y)
# print()
# print('cost after optimize = ' , costafteroptimize)
# print()







# def predict(theta, X):
#     probability = sigmoid(X * theta.T)
#     return [1 if x >= 0.5 else 0 for x in probability]

# theta_min = np.matrix(result[0])

# predictions = predict(theta_min, X)

# correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
# accuracy = (sum(map(int, correct)) % len(correct))
# print ('accuracy = {0}%'.format(accuracy))



























# from datetime import datetime
# import numpy as np
# import pandas as pd
# from numpy.random import randn,seed
# import matplotlib.pyplot as plt
# from sklearn.preproessing import Imputer
def sparte():
    print("-"*40)



# l=[[1,3,5,6],[1,3,5,6],[1,3,5,6]]
# arr=np.array(l)
# print(arr)
# print(np.array(np.arange(0,12,8)))
# print(np.arange(0,26,5))
# print(np.arange(0,26,5))
# print(np.arange(0,26,5).reshape(3,2))
# print(np.arange(0,26,5).reshape(3,2).max())
# print(np.arange(0,26,5).reshape(3,2).min())
# print(np.arange(0,26,5).reshape(3,2).argmax())
# print(np.arange(0,26,5).reshape(3,2).argmin())
# print(np.arange(0,26,5).reshape(3,2).shape)
# print(np.arange(0,26,5).reshape(3,2)[1][1])

# print(np.zeros((5,5)))
# print(np.ones((5,5)))
# print(np.linspace(0,5,5))
# print(np.eye(5))

# print(np.random.rand(5,5))
# print(np.random.rand(5))
# print(np.random.randint(5,200))
# print(np.random.randint(5,200,(3,3)))
# print(np.random.randint(5,200,100))

# arr=np.arange(0,26)
# arr_copy=arr.copy()[5:9]=100
# print(arr_copy)

# print(np.arange(0,40).reshape(5,8))
# print(np.arange(0,40).reshape(5,8)>20)

# arr=np.arange(0,26)
# arr2=np.arange(9,35)
# sparte="*"*40
# print(f"{arr}\n{sparte}\n{arr2}\n{sparte}\n{arr2/arr}") # /
# print(f"{arr}\n{sparte}\n{arr2}\n{sparte}\n{arr2-arr}") # -
# print(f"{arr}\n{sparte}\n{arr2}\n{sparte}\n{arr2+arr}") # +
# print(f"{arr}\n{sparte}\n{arr2}\n{sparte}\n{arr2*arr}") # *

# print(np.sqrt(arr),2)
# print(np.sqrt(arr))
# print(np.max(arr))
# print(np.min(arr))
# print(np.sin(arr))
# print(np.cos(arr))
# print(np.log(arr))
# print(np.exp(arr))



# exerice

# print(np.zeros(10))
# print(np.ones(10))
# print(np.ones(10)*5)
# print(np.arange(10,51))
# print(np.arange(10,51,2))
# print(np.arange(0,9).reshape(3,3))
# print(np.eye(3,3))
# print(np.random.rand(1))
# print(np.random.randn(25))
# print(np.arange(1,101).reshape(10,10)/100)
# print(np.linspace(0,1,10))
# print(np.linspace(0,1,20))

# print(np.arange(1,26).reshape(5,5))
# print("*"*30)
# print(np.arange(1,26).reshape(5,5)[2:,1:])
# print(np.arange(1,26).reshape(5,5)[3,-1])
# print(np.arange(1,26).reshape(5,5)[:3,1].reshape(3,1))
# print(np.arange(1,26).reshape(5,5)[-1,:])
# print(np.arange(1,26).reshape(5,5)[-2:,:])
# print(np.arange(1,26).reshape(5,5).sum())
# print(np.arange(1,26).reshape(5,5).std())
# print(np.sum(np.arange(1,26).reshape(5,5),axis=0))

