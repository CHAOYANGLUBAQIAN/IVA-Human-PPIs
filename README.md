# Prediction of influenza A virus-human protein-protein interactions using XGBoost with continuous and discontinuous amino acids information
   A sequence based prediction model for the interaction between influenza A virus and human proteins was constructed to validate and compare the choices made throughout the entire process.   
##  Install dependencies  
```
conda create -n ivahuman python=3.8  
conda activate ivahuman  
pip install scikit-learn==1.5.1  
pip install xgboost==2.1.2  
pip install pandas==2.0.3  
pip install numpy==1.23.5
``` 
##  How To Run
###  1.Construct negative samples  

- Input paths to the following resources (separated by space)  
1.'data/features/Random sampling.csv'  
2.'data/features/subcellular localization.csv'  
3.'data/features/Dissimilarity-based sampling.csv'  
4.'data/features/Degree-based sampling.csv'  
5.'data/features/Degree and Dissimilarity-based.csv'  
- Output paths to the following resources  
data/results/Comparative_results_of_negative_sampling_methods.csv  
```
python compare_neg_tech_with_val.py
```
###  2.Selection of features  
The dataset was transformed into a digital vector using 11 commonly used features from sequence based features,  
and the most suitable combination of features was found to ultimately determine the CT+Moran combination.  
```

python  compare_features.py
```
###  3.Selection of classifier 
To comparing some widely used PPI classifiers such as Random Forest (RF), Support Vector Machine (SVM), Logistic Regression (LR), K-Neighbor (KNN),
other tree models such as Extreme Random Tree (ET) and Adaptive Boosting (Adaboost) were also compared, and ultimately XGBoost was chosen for prediction.
```
python  compare_models.py  
- Input paths to the following resources (separated by space)

```
###  4.Ablation experiment   

```
python  ablation_study_of_CT+Moran.py

```
###  5.The trained model is prepared for prediction  
predict data:data/predict/test.csv  
```
python  predict.py

```
