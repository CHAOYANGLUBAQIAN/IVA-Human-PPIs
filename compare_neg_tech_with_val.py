import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.filterwarnings("ignore")

models = {
    
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Extra Trees': ExtraTreesClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'XGBoost': XGBClassifier(),
    'LightGBM': lgb.LGBMClassifier()
}

datasets = ['data/features/Random sampling.csv', 'data/features/subcellular localization.csv',
            'data/features/Dissimilarity-based sampling.csv','data/features/Degree-based sampling.csv','data/features/Degree and Dissimilarity-based.csv']
results_df = pd.DataFrame(columns=['Model', 'Dataset', 'CV_Accuracy', 'Test_Accuracy'])


table = pd.DataFrame(columns=['Feature', 'ACC'])

for dataset in datasets:
    scores = []
    print("Dataset:", dataset)

    df = pd.read_csv(dataset, header=0, encoding='utf-8')
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)


    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for model_name, model in models.items():
        print("Model:", model_name)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []

        for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

            model.fit(X_fold_train, y_fold_train)
            y_pred = model.predict(X_fold_val)
            cv_accuracy = accuracy_score(y_fold_val, y_pred)
            cv_scores.append(cv_accuracy)
            print("Fold {} CV Accuracy: {:.4f}".format(fold + 1, cv_accuracy))

        mean_cv_accuracy = np.mean(cv_scores)
        print("Mean CV Accuracy:", mean_cv_accuracy)


        y_pred_test = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        print("Test Accuracy:", test_accuracy)

        results_df = results_df.append({'Model': model_name, 'Dataset': dataset,
                                        'CV_Accuracy': mean_cv_accuracy, 'Test_Accuracy': test_accuracy},
                                       ignore_index=True)

results_df.to_csv('data/results/Comparative_results_of_negative_sampling_methods.csv', index=False)
