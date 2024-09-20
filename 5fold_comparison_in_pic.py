import numpy as np
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score, matthews_corrcoef,f1_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

input_file1 = 'data/final/CT+Moran.csv'
df = pd.read_csv(input_file1, header=0, encoding='utf-8')
df = df.sample(frac=1, random_state=42).reset_index(drop=True) #打乱顺序，固定随机因子

# 提取特征和标签
# X = df.iloc[:, 1:-1] #取属性值 不包含第一列的ID和最后一列的标签
X = df.iloc[:, 1:-1]
y = df.iloc[:,-1]  #最后一列为标签

# 定义模型列表
models = [
    XGBClassifier(),
    RandomForestClassifier(),
    LGBMClassifier(),
    ExtraTreesClassifier(),
    AdaBoostClassifier(),
    SVC(),
    LogisticRegression(),
    KNeighborsClassifier()
    # MultinomialNB()
]

# 定义交叉验证器
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 定义结果列表
results = []

# 循环遍历模型列表
for model in models:
    # 定义模型名称
    name = type(model).__name__
    print(f'Training {name}...')

    # 定义评估指标
    scores = []

    # 循环进行交叉验证
    for train_index, test_index in kf.split(X):
        # 分割数据集
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 训练模型
        model.fit(X_train, y_train)

        # 预测测试集
        y_pred = model.predict(X_test)

        # 计算准确率
        score = accuracy_score(y_test, y_pred)

        # 将评估指标添加到列表中
        scores.append(score)

    # 将模型名称和评估指标添加到结果列表中
    results.append({'model': name, 'scores': scores})

# 绘制折线图
fig, ax = plt.subplots()

# 循环遍历结果列表
for result in results:
    # 提取模型名称和评估指标
    name = result['model']
    scores = result['scores']
    print(name+'的准确率为{}'.format(np.mean(scores)))
    # 绘制折线
    plt.plot(np.arange(1, 6), scores, label=name)

# 设置图像标题和标签
plt.legend(loc='lower right')
plt.xlabel('Cross-validation folds')
plt.ylabel('Accuracy')
plt.title('Comparison of ML models')
plt.xticks(np.arange(1, 6))
plt.show()
#############################################################################################
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)
# # 训练 XGBoost 模型
# xgb_model = XGBClassifier()
# xgb_model.fit(X_train, y_train)
# xgb_score = xgb_model.score(X_test, y_test)
#
# # 训练 Random Forest 模型
# rf_model = RandomForestClassifier()
# rf_model.fit(X_train, y_train)
# rf_score = rf_model.score(X_test, y_test)
#
# # 训练 LightGBM 模型
# lgb_model = lgb.LGBMClassifier()
# lgb_model.fit(X_train, y_train)
# lgb_score = lgb_model.score(X_test, y_test)
#
# # 训练 SVM 模型
# svm_model = SVC()
# svm_model.fit(X_train, y_train)
# svm_score = svm_model.score(X_test, y_test)
#
# # 将模型的准确率画在同一张图上进行比较
# fig, ax = plt.subplots()
# ax.bar(['XGBoost', 'Random Forest', 'LightGBM', 'SVM'], [xgb_score, rf_score, lgb_score, svm_score])
# ax.set_ylabel('Accuracy')
# ax.set_title('Model Comparison')
# plt.show()
#########################################################################################################################
# # 定义模型列表
# models = [
#     XGBClassifier(),
#     RandomForestClassifier(),
#     LGBMClassifier(),
#     SVC()
# ]
#
# # 定义交叉验证器
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
#
# # 定义结果列表
# results = []
#
# # 循环遍历模型列表
# for model in models:
#     # 定义模型名称
#     name = type(model).__name__
#     print(f'Training {name}...')
#
#     # 定义评估指标
#     scores = []
#
#     # 循环进行交叉验证
#     for train_index, test_index in kf.split(X):
#         # 分割数据集
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#         # 训练模型
#         model.fit(X_train, y_train)
#
#         # 预测测试集
#         y_pred = model.predict(X_test)
#
#         # 计算准确率
#         score = accuracy_score(y_test, y_pred)
#
#         # 将评估指标添加到列表中
#         scores.append(score)
#
#     # 计算评估指标的平均值
#     mean_score = np.mean(scores)
#
#     # 将模型名称和评估指标添加到结果列表中
#     results.append({'model': name, 'accuracy': mean_score})
#
# # 将结果转换成数据框
# results_df = pd.DataFrame(results)
#
# # 绘制折线图
# plt.plot(results_df['model'], results_df['accuracy'])
# plt.title('Comparison of Models')
# plt.xlabel('Model')
# plt.ylabel('Accuracy')
# plt.ylim([0, 1])
# plt.xticks(rotation=45)
#
# # 显示图像
# plt.show()
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score
# from xgboost import XGBClassifier
# from sklearn.ensemble import RandomForestClassifier
# from lightgbm import LGBMClassifier
# from sklearn.svm import SVC
#
# # 读取数据集
#
#
# # 准备数据集



