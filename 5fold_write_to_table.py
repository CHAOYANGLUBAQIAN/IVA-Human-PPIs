import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score, matthews_corrcoef,f1_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.filterwarnings("ignore")
# 读取数据集
input_file1 = 'data/features/合/基于度/CT.csv'
# input_file1 = 'protr_pair_train/group/Pseudo-amino acid composition.csv'
# input_file1 = 'protr_pair_train/combination/all.csv'
# input_file1 = 'protr_pair_train/selected/best_150_of_all.csv'
df = pd.read_csv(input_file1, header=0, encoding='utf-8')
df = df.sample(frac=1, random_state=42).reset_index(drop=True) #打乱顺序，固定随机因子

# 提取特征和标签
# X = df.iloc[:, 1:-1] #取属性值 不包含第一列的ID和最后一列的标签
X = df.iloc[:, 1:-1]
y = df.iloc[:,-1]  #最后一列为标签
models = {'XGBoost': XGBClassifier(),
          'Random Forest': RandomForestClassifier(),
          'LightGBM': lgb.LGBMClassifier(),
          'ExtraTrees' : ExtraTreesClassifier(),
          'AdaBoost' : AdaBoostClassifier(),
          'SVM' : SVC(),
          'LR' : LogisticRegression()
          }


# 定义指标列表
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
mcc_scores = []
# 创建五折交叉验证对象
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1','MCC'])
##使用循环对所有模型验证
#对于每个折叠
for model_name, model in models.items():
    for train_index, test_index in kf.split(X):
        # 分割数据
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 训练模型
        model.fit(X_train, y_train)

        # 对测试集进行预测
        y_pred = model.predict(X_test)

        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        # print(accuracy)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        # 将指标添加到列表中
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        mcc_scores.append(mcc)
    results = results.append({'Model': model_name,
                              'Accuracy': np.mean(accuracy_scores),
                              'Precision': np.mean(precision_scores),
                              'Recall': np.mean(recall_scores),
                              'F1': np.mean(f1_scores),
                              'MCC': np.mean(mcc_scores)}, ignore_index=True)
    # 输出指标的平均值和标准差
    print("{}的综合表现如下：".format(model_name))
    print("Accuracy: {:.2f} (+/- {:.2f})".format(np.mean(accuracy_scores), np.std(accuracy_scores)))
    print("Precision: {:.2f} (+/- {:.2f})".format(np.mean(precision_scores), np.std(precision_scores)))
    print("Recall: {:.2f} (+/- {:.2f})".format(np.mean(recall_scores), np.std(recall_scores)))
    print("F1 score: {:.2f} (+/- {:.2f})".format(np.mean(f1_scores), np.std(f1_scores)))
    print("MCC: {:.2f} (+/- {:.2f})".format(np.mean(mcc_scores), np.std(mcc_scores)))

results.to_csv('data/out/基于度/CT.csv', index=False)