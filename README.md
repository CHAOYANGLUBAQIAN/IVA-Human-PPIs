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
```
python compare_neg_tech_with_val.py

```
###  2.Ablation experiment   
```
python  ablation_study_of_CT+Moran.py

```
###  3.Selection of classifier 
```
python  compare_models.py

```
###  4.XGBoost uses grid search for parameters  
```
python  search_best_param.py

```
###  5.The trained model is prepared for prediction  
```
python  predict.py

```
