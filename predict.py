import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier

# 加载模型
XGB = joblib.load('data/model/random_sort_XGB.kpl')

# 加载待预测的数据
data = pd.read_csv('data/predict/test.csv')

# 提取特征列
features = data.drop('Combined_ID', axis=1)
# 获取预测概率值
probabilities = XGB.predict_proba(features)[:, 1]  # 提取属于正类的概率值
# 将预测概率值与对应的ID保存到文件
results = pd.DataFrame({'ID': data['Combined_ID'], 'Probability': probabilities})
results.to_csv('data/predict/out/detail1.csv', index=False)

# 筛选出概率大于0.9的数据对应的ID
high_prob_ids = data.loc[probabilities > 0.9, 'Combined_ID']

# 将概率大于0.9的ID保存到文件
high_prob_ids.to_csv('data/predict/out/detail11.csv', index=False)

######################################################################
# # 进行预测
# predictions = XGB.predict(features)
#
# # 将预测结果与对应的ID组合起来
# results = pd.DataFrame({'ID': data['Combined_ID'], 'Prediction': predictions})
# results_with_data = pd.concat([results, data], axis=1)
# # 保存每条数据及其对应的预测得分到文件
# results_with_data.to_csv('data/predict/out/detail.csv', index=False)

# # 筛选出预测为正的数据对应的ID
# positive_predictions = results[results['Prediction'] >0.5 ]
#
# # 输出所有预测为正的数据对应的ID
# positive_ids = positive_predictions['ID'].tolist()
# print(positive_ids)
#
# # 将预测为正的ID存入文件中
# with open('data/predict/out/test.csv', 'w') as file:
#     for id in positive_ids:
#         file.write(str(id) + '\n')
