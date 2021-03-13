# [How to Configure XGBoost for Imbalanced Classification](https://machinelearningmastery.com/xgboost-for-imbalanced-classification/)

## Performance - Baseline model
```python
# fit xgboost on an imbalanced classification dataset
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=7)
# define model
model = XGBClassifier()
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.5f' % mean(scores))
```

Output:
```
Mean ROC AUC: 0.95724
```


## Weighted XGBoost for Class Imbalance
- XGBoost is trained to minimize a loss function and the *“gradient”* in gradient boosting refers to the steepness of this loss function, e.g. the amount of error.
- Gradients are used as the basis for fitting subsequent trees added to boost or correct errors made by the existing state of the ensemble of decision trees.
- The **`scale_pos_weight`** value is used to **scale the gradient for the positive class**.
- This has the effect of scaling errors made by the model during training on the positive class and encourages the model to over-correct them.
  - In turn, this can help the model achieve **better performance when making predictions on the positive class**.
  - Pushed too far, it may result in the model overfitting the positive class at the cost of worse performance on the negative class or both classes.
- As such, the `scale_pos_weight` can be used to **train a class-weighted or cost-sensitive version of XGBoost for imbalanced classification**.
- A sensible default value to set for the scale_pos_weight hyperparameter is the inverse of the class distribution.
  - The XGBoost documentation suggests a fast way to estimate this value using the training dataset as the total number of examples in the majority class divided by the total number of examples in the minority class.
  - `scale_pos_weight = total_negative_examples / total_positive_examples`

```python
# fit balanced xgboost on an imbalanced classification dataset
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=7)
# define model
model = XGBClassifier(scale_pos_weight=99)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.5f' % mean(scores))
```

Output:
```
Mean ROC AUC: 0.95990
```
