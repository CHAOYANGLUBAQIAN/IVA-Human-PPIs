import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score, matthews_corrcoef,f1_score,confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")
# 加载数据集
# input_file1 = 'testdata/denovo/final/CT+Moran_train.csv'
input_file1 = 'testdata/zhou_ebola/final/CT+Moran_train.csv'
df = pd.read_csv(input_file1, header=0, encoding='utf-8')
df = df.sample(frac=1, random_state=42).reset_index(drop=True) #打乱顺序，固定随机因子

# 定义特征和标签
X_train = df.iloc[:, 2:-1]
y_train = df.iloc[:,-1]
XGB = XGBClassifier(learning_rate=0.1, max_depth=7, n_estimators=310)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
train_acc = []
# 训练模型
for i, (train_idx, valid_idx) in enumerate(cv.split(X_train)):
    print(f"Training fold {i+1}")
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
    XGB.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
    y_pred = XGB.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    train_acc.append(accuracy)
print('训练集的准确率为{}'.format(np.mean(train_acc)))

# 保存模型
joblib.dump(XGB, 'data/model/random_sort_zhou_ebola.kpl')

# # 加载模型
# # model = XGB.Booster()
# # model.load_model("xgb_model.model")
# XGB = joblib.load('xgb.pkl')
# # 在测试集上进行预测
# y_pred = XGB.predict(X_test)
# test_acc = accuracy_score(y_test, y_pred)
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# print('训练集的准确率为{}，测试集的准确率为{}'.format(np.mean(train_acc),test_acc))
# XGB.fit(X_train, y_train)
# print('Accuracy over train set: ', XGB.score(X_train, y_train))
# print('Accuracy over test set: ', XGB.score(X_test,y_test))
#
# XGB.fit(X_train, y_train)
# y_pred = XGB.predict(X_test)
# precision = precision_score(y_pred, y_test)
# print('蛋白对AAC在LGB上的综合表现为')
# print('precision:%f' % precision)
# acc = accuracy_score(y_test, y_pred)
# print('accuracy:%f' % acc)
# recall = recall_score(y_test, y_pred)
# print('recall:%f' % recall)
# mcc = matthews_corrcoef(y_test, y_pred)
# print('MCC:%f' % mcc)
# f1 = f1_score(y_test, y_pred)
# print('f1_score:%f' % f1)
# cv = KFold(n_splits=5, shuffle=True, random_state=0)
# scores = cross_val_score(XGB, X, y, cv=cv, scoring='precision', error_score='raise')
# prec = scores.mean()
# print("五折交叉验证{}kfold:{}".format(XGB, prec))
