import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")

# 加载数据
input_file1 = 'data/features/CT+Moran.csv'
df = pd.read_csv(input_file1, header=0, encoding='utf-8')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# 划分独立测试集
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义模型
models = {
    'XGBoost': XGBClassifier(learning_rate=0.1, max_depth=7, n_estimators=310),
    'Random Forest': RandomForestClassifier(n_estimators=100, min_samples_split=3),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=310),
    'AdaBoost': AdaBoostClassifier(learning_rate=0.1, n_estimators=310),
    'SVM': SVC(C=100, gamma=0.01),
    'LR': LogisticRegression(C=6, penalty='l2')
}


kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = pd.DataFrame(columns=['Model', 'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_MCC'])
fold_accuracies = pd.DataFrame(columns=['Model', 'Fold', 'Accuracy'])


for model_name, model in models.items():
    print(f"Training {model_name} ...")
    fold_accuracies_list = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_train_full), start=1):
        X_fold_train, X_fold_val = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
        y_fold_train, y_fold_val = y_train_full.iloc[train_index], y_train_full.iloc[val_index]

        model.fit(X_fold_train, y_fold_train)

        y_pred_val = model.predict(X_fold_val)

        accuracy = accuracy_score(y_fold_val, y_pred_val)
        fold_accuracies_list.append({'Model': model_name, 'Fold': fold, 'Accuracy': accuracy})
        print(f"{model_name} Fold {fold}  Accuracy: {accuracy:.4f}")

    fold_accuracies = pd.concat([fold_accuracies, pd.DataFrame(fold_accuracies_list)], ignore_index=True)

    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    test_mcc = matthews_corrcoef(y_test, y_pred_test)

    print(f"{model_name} The performance on the independent test set is as follows：")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 score: {test_f1:.4f}")
    print(f"Test MCC: {test_mcc:.4f}")

    # 保存测试集结果
    results = results.append({'Model': model_name,
                              'Test_Accuracy': test_accuracy,
                              'Test_Precision': test_precision,
                              'Test_Recall': test_recall,
                              'Test_F1': test_f1,
                              'Test_MCC': test_mcc}, ignore_index=True)

# 保存结果
results.to_csv('data/results/results_of_model_comparison.csv', index=False)
fold_accuracies.to_csv('data/results/model_fold_accuracies.csv', index=False)
