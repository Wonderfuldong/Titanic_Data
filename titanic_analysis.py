#### 安装graphviz-2.38
##### 1.添加系统环境变量
.建立变量名GRAPHVIZ_DOT
值为安装的路径C:\Program Files (x86)\Graphviz2.38\bin\dot.exe
##### 2. 设置环境变量 在用户环境变量添加以下一个变量
.建立变量名 GRAPHVIZ_INSTALL_DIR, 值为如C:\Program Files (x86)\Graphviz2.38
##### 3 在系统环境变量 建立变量名PATH中添加Graphviz的bin目录路径，如C:\Program Files (x86)\Graphviz2.34\bin


import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
import os

# 数据加载
os.chdir(r'C:\Users\Happydogs\Desktop\Chenyang\Titanic_Data-master\Titanic_Data-master')
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')
# 数据探索
print(train_data.info())
print('-'*30)
print(train_data.describe())
print('-'*30)
print(train_data.describe(include=['O']))
print('-'*30)
print(train_data.head())
print('-'*30)
print(train_data.tail())
# 数据清洗
# 使用平均年龄来填充年龄中的 nan 值
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
# 使用票价的均值填充票价中的 nan 值
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)
print(train_data['Embarked'].value_counts())


# 使用登录最多的港口来填充登录港口的 nan 值
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S',inplace=True)

# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]

dvec=DictVectorizer(sparse=False)
train_features=dvec.fit_transform(train_features.to_dict(orient='record'))
print(dvec.feature_names_)

# 构造 ID3 决策树
clf = DecisionTreeClassifier(criterion='entropy')
# 决策树训练
clf.fit(train_features, train_labels)

test_features=dvec.transform(test_features.to_dict(orient='record'))
# 决策树预测
pred_labels = clf.predict(test_features)

# 得到决策树准确率 其实是不符合实际的
acc_decision_tree = round(clf.score(train_features, train_labels), 6)
print(u'score 准确率为 %.4lf' % acc_decision_tree)

import numpy as np
from sklearn.model_selection import cross_val_score
# 使用 K 折交叉验证 统计决策树准确率
print(u'cross_val_score 准确率为 %.4lf' % np.mean(cross_val_score(clf, train_features, train_labels, cv=10)))



from sklearn.tree import export_graphviz
export_graphviz(clf, "./my_decision_tree.dot", feature_names = feature_names)

eature_names = dvec.get_feature_names()

export_graphviz(clf, "./my_decision_tree.dot", feature_names = feature_names)
