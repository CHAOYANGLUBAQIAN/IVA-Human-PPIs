# Prediction of influenza A virus-human protein-protein interactions using XGBoost with continuous and discontinuous amino acids information
   A sequence based prediction model for the interaction between influenza A virus and human proteins was constructed to validate and compare the choices made throughout the entire process. Propose a new method for constructing negative samples and compare it with four commonly used methods for constructing negative samples, confirming that our method is better than mainstream methods. Secondly, for feature extraction, the dataset was transformed into a digital vector using 11 commonly used features from sequence based features, and the most suitable combination of features was found to ultimately determine the CT+Moran combination. This process not only demonstrates the importance of feature selection, but also provides valuable references for subsequent research. In addition, by exploring the implementation principles of these two features and conducting ablation experiments, in terms of classifier selection, in addition to comparing some widely used PPI classifiers such as Random Forest (RF), Support Vector Machine (SVM), Logistic Regression (LR), K-Neighbor (KNN), other tree models such as Extreme Random Tree (ET) and Adaptive Boosting (Adaboost) were also compared, and ultimately XGBoost was chosen for prediction. Subsequently, grid search was used to optimize the parameters of XGBoost to obtain the optimal model. Then, five fold cross validation was performed on all datasets for sufficient training, and the trained model was predicted on an independent dataset prepared for prediction. Finally, 32855 pairs of interactions were predicted, from which 3269 potential human target proteins corresponding to 2995 target genes were identified.
##  Installation
'''
conda create -n ivahuman python=3.8
conda activate ivahuman
pip install scikit-learn==1.5.1
pip install xgboost==2.1.2
pip install pandas==2.0.3
pip install numpy==1.23.5
pip install numpy==1.23.5
pip install numpy==1.23.5
pip install numpy==1.23.5
pip install numpy==1.23.5
pip install numpy==1.23.5
'''
## 
