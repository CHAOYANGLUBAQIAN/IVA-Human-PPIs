import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")
# 定义模型
LGB = lgb.LGBMClassifier()
AB = AdaBoostClassifier()
XGB = XGBClassifier()
svm = SVC()
LR = LogisticRegression()
knn = KNeighborsClassifier()
# 定义数据集列表
datasets = ['data/features/合/基于度的差异负采样/AAC.csv', 'data/features/合/基于度的差异负采样/APAAC.csv', 'data/features/合/基于度的差异负采样/CT.csv', 'data/features/合/基于度的差异负采样/CTDC.csv', 'data/features/合/基于度的差异负采样/CTDT.csv',
'data/features/合/基于度的差异负采样/CTDD.csv','data/features/合/基于度的差异负采样/Geary.csv','data/features/合/基于度的差异负采样/Moran.csv','data/features/合/基于度的差异负采样/PAAC.csv','data/features/合/基于度的差异负采样/QSO.csv','data/features/合/基于度的差异负采样/SOCN.csv'
            ,'data/features/合/基于度的差异负采样/5.csv','data/features/合/基于度的差异负采样/APAAC+QSO.csv','data/final/CT+Moran.csv','data/features/合/基于度的差异负采样/MoreauBroto.csv']
# datasets = ['data/features/合/基于度的差异负采样/5.csv', 'data/features/合/基于度的差异负采样/n_A.csv',
#             'data/features/合/基于度的差异负采样/n_C.csv', 'data/features/合/基于度的差异负采样/n_D.csv', 'data/features/合/基于度的差异负采样/n_G.csv', 'data/features/合/基于度的差异负采样/n_Q.csv']
# datasets = ['data/features/合/基于度的差异负采样/CT.csv','data/features/合/基于度的差异负采样/Moran.csv','data/out/merge/CT+Moran.csv']
# # 定义用于存储准确率的列表
results = []
mean_scores = []
# 对每个数据集进行测试
table = pd.DataFrame(columns=['Feature','ACC'])
for dataset in datasets:
    scores = []
    print(dataset)
    # 读取数据集
    df = pd.read_csv(dataset, header=0, encoding='utf-8')
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱顺序，固定随机因子
    # 提取特征和标签
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]  # 最后一列为标签
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, index in enumerate(kf.split(X)):
        train_index, test_index = index
        # 分割数据集
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 训练模型并预测
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        # 计算准确率并存储
        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)
        print('{}的{}准确率为{:.4f}'.format(dataset,i,acc))
    mean_score = np.mean(scores)
    mean_scores.append(mean_score)
    print('{}的平均准确率为{:.4f}'.format(dataset,mean_score))
    table = table.append({'Feature': dataset,
                              'ACC': np.mean(scores)},ignore_index=True
                              )

table.to_csv('data/out/2_27_features.csv')
# print(datasets)
# print(mean_scores)
print(mean_scores)
