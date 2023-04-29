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


# pandas

# li=[6,9,4,3,2]
# index=["a","b","c","e","u"]
# dc={"a":4,"b":5,"c":7,"e":9,"u":7}
# print(pd.Series(data=li,index=index))
# print(pd.Series(li,index))
# print(pd.Series(li))
# print(pd.Series(dc))
# print(pd.Series(dc)+pd.Series(data=li,index=index))

# data=np.arange(50).reshape(10,5)
# data=np.random.randint(0,50,(10,5))
# data by random use seed random
# seed(1)
# data=np.random.randn(10,5)

# print(data.shape)
# index="1 2 3 4 5 6 7 8 9 10".upper().split(" ")
# index=np.arange(10)
# columns="a b c d e".upper().split(" ")
# dataFram=pd.DataFrame(data=data,index=index,columns=columns)
# dataFram=pd.DataFrame(data=data,columns=columns)
# dataFram=pd.DataFrame(data=data)

# print(dataFram)
# print(dataFram["A"][3]) coumn=>A -- row =>3
# select=dataFram[["A","B"]]
# dataFram["new"]=dataFram["A"]+dataFram["B"]
# print(dataFram["new"])
# print(dataFram)
# dataFram.drop("new",axis=1,inplace=True)
# print(dataFram)
# print(dataFram.iloc[3]) # row by index
# print(dataFram.loc[1]) # row name 
# select=dataFram.loc[[1,2],["A","B"]] # index row by name column: 1=>A   2=>B
# print(select)


# outside=["G1","G1","G1","G2","G2","G2",]
# inside=[1,2,3,1,2,3]
# hiring_index=list(zip(outside,inside))
# hiring_index=pd.MultiIndex.from_tuples(hiring_index)
# dataframe=pd.DataFrame(np.random.randn(6,2),index=hiring_index,columns=["A","B"])
# print(dataframe)
# sparte()
# print(dataframe.loc["G1"])
# sparte()
# print(dataframe.loc["G1"].loc[1])
# sparte()
# print(dataframe.loc["G1"].loc[1].loc["B"])






# data={
#     "a": [1,2,3,np.nan],
#     "b": [9,7,np.nan,3],
#     "c": [4,3,np.nan,8],
# }
# dataFrame=pd.DataFrame(data=data)
# print(dataFrame)
# sparte()
# print(dataFrame.dropna())
# sparte()
# print(dataFrame.dropna(axis=1))
# sparte()
# print(dataFrame.fillna(value="gamal"))
# sparte()
# print(dataFrame.groupby("a").describe())



# data={
#     "a": [1,2,3,6,7],
#     "b": [9,7,9,3,3],
#     "c": [4,3,2,7,7],
# }
# data2={
#     "t": [1,2,3,6,7],
#     "u": [9,7,9,3,3],
#     "c": [4,3,2,7,8],
# }
# data3={
#     "p": [1,2,3,6,7],
#     "q": [9,7,9,3,3],
#     "c": [4,3,2,7,8],
# }

# df1=pd.DataFrame(data)
# df2=pd.DataFrame(data2)
# df3=pd.DataFrame(data3)
# concat=pd.concat([df1,df2,df3],axis=1)
# print(concat.groupby("a"))
# print(pd.merge(df1,df2,how="outer",on="c"))
# print(pd.merge(df1,df2,how="iner",on="c"))

# print(df1["c"].unique())
# print(df1["c"].nunique())# number of unique
# print(df1["c"].value_counts()) #detils number of unique


# def time2(x):
#     return x**2

# print(df1.apply)
# sparte()
# print(df1.apply(time2))
# sparte()
# print(df1.apply(lambda x:x**2)) #use lambda


# row axis 0
# column axis 1



# print(df1.index)
# print(df1.columns)

# print(df1.sort_values("a"))
# print(df1.isnull())

# print(df1.pivot_table(index=["a","b"],columns=["c"],values=["a"],))
# path="C:\Projects\Collage\ML\courseudmay\homeprices.csv"
# read=pd.read_csv(path)

# print(read)

# print(read.to_excel("sheet2.xlsx",sheet_name="NewSheet"))


# path="https://fileinfo.com/extension/html"
# print(pd.read_html(path))

# print("skks")





# import math

# result = math.e != math.pow(2, 4)

# print(int(result))

#! 

# # print (float("1", "3" ))

# path="C:\Projects\\traingPython\Machine learing\P14-Part1-Data-Preprocessing\Section 3 - Data Preprocessing in Python\Python\Data.csv"
# dataSet=pd.read_csv(path)

# # print(dataSet)
# x=dataSet.iloc[:,:-1]
# y=dataSet.iloc[:,-1]
# # print(x)
# # sparte()
# # print(y)

# imputer=SimpleImputer(strategy="mean",missing_values=np.nan,verbose=0)

# # print(np.nan)
# # x_with_out_string=x.iloc[:,1:]
# # print(x_with_out_string)
# # print(x[:,1:])
# imputer=imputer.fit(x.iloc[:,1:])
# # print(imputer)

# # sparte()

# x.iloc[:,1:]=imputer.fit_transform(x.iloc[:,1:])

# print(imputer)


# print(x.iloc[:,1:])


# # Dummy Encoding
# # transform first column string to numbeer
# from sklearn.preprocessing import LabelEncoder ,OneHotEncoder


# labelEncoder_x=LabelEncoder()

# labelEncoder_y=LabelEncoder()

# # x_with_string_transform=x.iloc[:,0]

# x.iloc[:,0]=labelEncoder_x.fit_transform(x.iloc[:,0])


# y=labelEncoder_y.fit_transform(y)
# sparte()
# print(y)

# print(x.iloc[:,0])


# oneHotEncoder=OneHotEncoder()


# print(x)
# x=oneHotEncoder.fit_transform(x).toarray()

# y=oneHotEncoder.fit_transform(y).toarray()

# print(y)

# print(x+y)
