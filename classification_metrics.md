# [F1 Score vs ROC AUC vs Accuracy vs PR AUC: Which Evaluation Metric Should You Choose?](https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc)

## 1. Accuracy:

<p align="center">
  <img src="https://i2.wp.com/neptune.ai/wp-content/uploads/acc_eq.png?fit=422%2C84&ssl=1">
</p>

- You **shouldn’t use accuracy on imbalanced problems**
- Accuracy **depends on the threshold choice**
- When to use it?
  - When your problem is balanced using accuracy is usually a good start
  - An additional benefit is that it is really **easy to explain it to non-technical stakeholders** in your project
  - When **every class is equally important** to you

```python
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred_class = y_pred_pos > threshold
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
accuracy = (tp + tn) / (tp + fp + fn + tn)

# or simply

accuracy_score(y_true, y_pred_class)
```


## 2. F1 score:
<p align="center">
  <img src="https://i1.wp.com/neptune.ai/wp-content/uploads/fbeta_eq.png?fit=604%2C88&ssl=1">
</p>

- When choosing beta in your F-beta score **the more you care about recall** over precision **the higher beta**
- It is important to remember that F1 score is calculated from Precision and Recall which, in turn, are calculated on the predicted classes (not prediction scores)
- Remember that **F1 score** is balancing precision and recall on the **positive class** while **accuracy** looks at correctly classified observations **both positive and negative**
- When to use it?
  - Pretty much in every binary classification problem where you care more about the positive class. **It is my go-to metric** when working on those problems
  - It **can be easily explained to business stakeholders** which in many cases can be a deciding factor

```python
from sklearn.metrics import f1_score

y_pred_class = y_pred_pos > threshold
f1_score(y_true, y_pred_class)
```


## 3. ROC AUC:
<p align="center">
  <img src="https://i1.wp.com/neptune.ai/wp-content/uploads/roc_auc_curve.png?fit=1024%2C768&ssl=1">
</p>

- Tradeoff between true positive rate (TPR) and false positive rate (FPR)
- **For every threshold**, we calculate TPR and FPR and plot it on one chart
- ROC AUC score is equivalent to calculating the rank correlation between predictions and targets → **how good at ranking predictions your model is**
- When to use it?
  - You **should use it** when you ultimately **care about ranking predictions** and not necessarily about outputting well-calibrated probabilities
  - You **should not use it** when your **data is heavily imbalanced**
    - The intuition is the following: false positive rate for highly imbalanced datasets is pulled down due to a large number of true negatives
  - You **should use it when you care equally about positive and negative classes**
```python
from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_true, y_pred_pos)
```


## 4. PR AUC | Average Precision:
<p align="center">
  <img src="https://i1.wp.com/neptune.ai/wp-content/uploads/prec_rec_curve.png?fit=1024%2C768&ssl=1">
</p>

- It is a curve that combines precision (=PPV, Positive Predicted Value) and Recall (=TPR) in a single visualization
- The higher on y-axis your curve is the better your model performance
- Knowing **at which recall your precision starts to fall fast** can help you choose the threshold and deliver a better model
- **Think of PR AUC as the average of precision scores calculated for each recall threshold**
- When to use it?
  - When you want to **communicate precision/recall decision** to other stakeholders
  - When you want to **choose the threshold that fits the business problem**
  - When your **data is heavily imbalanced**:
    - The intuition is the following: since PR AUC focuses mainly on the positive class (PPV and TPR) it cares less about the frequent negative class
  - When **you care more about positive than negative class**
```python
from sklearn.metrics import average_precision_score

average_precision_score(y_true, y_pred_pos)
```

## Extra: ROC AUC vs PR AUC
- They both look at prediction scores of classification models and not thresholded class assignments
- **ROC AUC** looks at a true positive rate **TPR** and false positive rate **FPR**, while **PR AUC** looks at positive predictive value **PPV** and true positive rate **TPR**
  - Because of that **if you care more about the positive class, then using PR AUC**, which is more sensitive to the improvements for the positive class, **is a better choice**
- The improvements calculated in Average Precision (PR AUC) are larger and clearer
  - ROC AUC can give a **false sense of very high performance**
