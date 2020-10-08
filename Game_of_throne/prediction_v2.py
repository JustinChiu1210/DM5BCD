#!/usr/bin/python3.8.5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #繪圖，可視化
from sklearn.preprocessing import LabelEncoder #自動將文字轉換數值
from sklearn.ensemble import RandomForestClassifier#,  GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import export_graphviz
from subprocess import call
from sklearn import tree


def DC(row):
    if row['Death Chapter']>0:
        return 1
    else :
        return 0  
def alleg(row,alle_col):
    if row['Allegiances']==alle_col:
        return 1
    else :
        return 0    

ct_data= pd.read_csv('character-deaths.csv')
#print(ct_data.head(5))
ct_data.drop(['Death Year','Book of Death'],axis=1,inplace=True)#將death year, book of death丟棄 留death chapter
#ct_data.info()
size=ct_data.shape[0]
#print(ct_data.shape)#由shape得知總共資料有917筆，info得知Death chapter, book intro chapter資料有空直
ct_data['Book Intro Chapter'].fillna(value=0, inplace=True)#將空直以0代替
ct_data['Death Chapter']=ct_data.apply(lambda row : DC(row), axis=1)#依題目給予有值1無值0 1=dead

# alleg_list=ct_data['Allegiances'].unique()#儲存各種不同值得alleg欄位值
# for alle_col in alleg_list:
#     ct_data[alle_col]=ct_data.apply (lambda row: alleg(row,alle_col), axis=1) # 將20種植建立欄位 有對應值給1無則0
mod_data=pd.get_dummies(data=ct_data,columns=['Allegiances'])
print(mod_data.head())

mod_data.drop(['Name'],axis=1,inplace=True)#,'Allegiances'
#random.shuffle(ct_data)
x=mod_data.drop(['Death Chapter'], axis=1)#建立訓練資料集
z=pd.DataFrame(mod_data,columns=['Death Chapter'])#建立目標資料集
y=mod_data['Death Chapter']#建立目標資料集
X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=0.75)#分割資料集（0.75/0.25）
print(Ｘ_train.head(5))
data_features= x.columns.values#訓練集欄位，用於圖形
pre_features=z.columns.values#目標集欄位，用於圖形
print(pre_features)
rd=RandomForestClassifier(random_state=2,n_estimators=10,min_samples_split=20,oob_score=True)#,max_depth=8
rd=rd.fit(X_train,y_train)
finalMdR=rd.predict(X_test)
#print(rd.oob_score_)
print(accuracy_score(y_test, finalMdR))
# print(precision(y_test, finalMdR))
# print(reacll_score(y_test, finalMdR))

matrix1 = confusion_matrix(y_test, finalMdR, labels=[0,1])#,normalize='true'
print('Confusion matrix : \n',matrix1)

matrix2 = classification_report(y_test, finalMdR,labels=[0,1])
print('Classification report : \n',matrix2)

# # Extract single tree
# estimator = rd.estimators_[5]

# # Export as dot file
# export_graphviz(estimator, out_file='tree_v1.dot', feature_names = data_features,class_names = pre_features[0],rounded = True, proportion = False, precision = 2, filled = True)
# #若出現找不到dot則去安裝（brew install graphviz）
# #若出現index error可能為訓練欄位與這邊data_features數量不同導致
# #Convert to png using system command (requires Graphviz)
# from subprocess import call
# call(['dot', '-Tpng', 'tree_v1.dot', '-o', 'tree_v1.png', '-Gdpi=600'])

# # Display in jupyter notebook
# from IPython.display import Image
# Image(filename = 'tree_v1.png')

