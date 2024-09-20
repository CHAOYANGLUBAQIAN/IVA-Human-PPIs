# 导入所需的库
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# 加载数据集
input_file1 = 'data/features/合/基于度的差异负采样/CT.csv'
df = pd.read_csv(input_file1, header=0, encoding='utf-8')

# 提取特征和标签
X = df.iloc[:, 1:-1] #取属性值 不包含第一列的ID和最后一列的标签
y = df.iloc[:,-1]  #最后一列为标签
# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# # 实例化XGBoost分类器
# model = svm.SVC(kernel='rbf')
# 创建KNN分类器
# model = KNeighborsClassifier()
# # 创建逻辑回归模型
# model = LogisticRegression()
#创建lightgbm模型

#adaboost
## 创建基分类器
base_classifier = DecisionTreeClassifier()

model = AdaBoostClassifier(base_classifier)
# 设置搜索空间
param_grid = {
    'n_estimators': list(np.arange(50,500,10)),
    'learning_rate':list(np.arange(0.01,0.2,0.01))
}
# param_grid = {'C': list(np.arange(5,6,0.1)), 'penalty': ['l1', 'l2']}

# 使用网格搜索选择最佳参数
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf,scoring='precision')
grid_search.fit(X_train, y_train)

# 输出最佳参数和得分
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
#SVM的最佳参数
#Best parameters:  {'C': 100, 'gamma': 0.01}
#Best score:  0.9767479569993359
#KNN最佳参数
#Best parameters:  {'metric': 'manhattan', 'n_neighbors': 2}
#LR最佳参数
#Best parameters:  {'C': 6, 'penalty': 'l2'}
#Best score:  0.8734598172611001
#lightGBM最佳参数
#Best parameters:  {'learning_rate': 0.16, 'max_depth': 7, 'n_estimators': 490}
#AdaboostBest parameters:  {'learning_rate': 0.16, 'n_estimators': 250}
#Best score:  0.889495891616557
##ET
##'max_depth': 2, 'min_samples_split': 2, 'n_estimators': 190