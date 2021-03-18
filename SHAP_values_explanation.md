[SHAP values explained exactly how you wished someone explained to you](https://towardsdatascience.com/shap-explained-the-way-i-wish-someone-explained-it-to-me-ab81cc69ef30)

- As seen above, the original SHAP formula requires to train 2 ^ F models (F = Number of features)
- For a model with just 50 features, this would mean to train 1e15 models!
  - Indeed, as F increases, the formula seen above becomes inapplicable soon
