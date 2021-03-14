## 1. [Target encoding done the right way](https://maxhalford.github.io/blog/target-encoding/)
There are various ways to handle overfitting when target-encoding:
- A popular way is to use cross-validation and compute the means in each out-of-fold dataset (H2O)
- Another approach which I much prefer is to use **additive smoothing** (as IMDB supposedly does)
  - The trick is to “smooth” the average by **including the average rating over all movies**.

The idea is that the higher `m` is, the more you’re going to rely on the overall mean `w`. If `m` is equal to 0 then you’re simply going to compute the empirical mean.
In other words you’re not doing any smoothing whatsoever, just a regular Target Encoding.

```python
def calc_smooth_mean(df, by, on, m):
    # Compute the global mean
    mean = df[on].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + m * mean) / (counts + m)

    # Replace each value by the according smoothed mean
    return df[by].map(smooth)
```


## 2. [K-Fold Target Encoding](https://medium.com/@pouryaayria/k-fold-target-encoding-dfe9a594874b)
- k-fold target encoding can be applied to reduce the overfitting
- In this method:
  1. We divide the dataset into the k-folds
  2. We calculate mean-target for fold all folds but one
  3. We use the calculated values to estimate mean encoding for the remaining fold
      <p align="center">
      <img src="https://miro.medium.com/max/700/1*ZKD4eZXzd_FdN0SQDszFVQ.png">
      </p>
  4. We repeat the process for all folds
      <p align="center">
      <img src="https://miro.medium.com/max/700/1*VMVwPoOKWQ7Lcr5NCAXC8Q.png">
      </p>
  5. Now the remaining part is creating “Feature_Kfold_Target_Enc” column in the test dataset.This column values can be obtained from getting mean of “Feature_Kfold_mean_Enc” train column for the categorical variables “A” and “B”
  <p align="center">
  <img src="https://miro.medium.com/max/700/1*G-_64tzNnjhIriodzTXAIA.png">
  </p>
- Code:
- `KFoldTargetEncoderTrain` class:
  - Gets:
    - name of the feature column
    - name of the target column
    - number of folds
  - Returns:
    - DataFrame included “Feature_Kfold_mean_Enc”
    - *Note that, if a fold is not included, for example, “B” categorical variable, thus, it results of NAN which we fill NAN with the global mean of the target.*
    ```python
    class KFoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):

        def __init__(self, colnames,targetName,n_fold=5,verbosity=True,discardOriginal_col=False):
            self.colnames = colnames
            self.targetName = targetName
            self.n_fold = n_fold
            self.verbosity = verbosity
            self.discardOriginal_col = discardOriginal_col

        def fit(self, X, y=None):
            return self

        def transform(self,X):
            assert(type(self.targetName) == str)
            assert(type(self.colnames) == str)
            assert(self.colnames in X.columns)
            assert(self.targetName in X.columns)
            mean_of_target = X[self.targetName].mean()
            kf = KFold(n_splits = self.n_fold, shuffle = False, random_state=2019)
            col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
            X[col_mean_name] = np.nan
            for tr_ind, val_ind in kf.split(X):
                X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
    #             print(tr_ind,val_ind)
                X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)[self.targetName].mean())
            X[col_mean_name].fillna(mean_of_target, inplace = True)
            if self.verbosity:
                encoded_feature = X[col_mean_name].values
                print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,
                                                                                          self.targetName,
                                                                                          np.corrcoef(X[self.targetName].values, encoded_feature)[0][1]))
            if self.discardOriginal_col:
                X = X.drop(self.targetName, axis=1)
            return X
    ```
  - To run the code:
    ```python
    targetc = KFoldTargetEncoderTrain('Feature','Target',n_fold=5)
    new_train = targetc.fit_transform(train)
    ```
- `KFoldTargetEncoderTest` class:
  - Gets:
    - train dataset
    - name of the feature column
    - name of the encoded column 
  ```python
  class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):

      def __init__(self,train,colNames,encodedName):
          self.train = train
          self.colNames = colNames
          self.encodedName = encodedName

      def fit(self, X, y=None):
          return self

      def transform(self,X):
          mean = self.train[[self.colNames,self.encodedName]].groupby(self.colNames).mean().reset_index() 
          dd = {}
          for index, row in mean.iterrows():
              dd[row[self.colNames]] = row[self.encodedName]
          X[self.encodedName] = X[self.colNames]
          X = X.replace({self.encodedName: dd})
          return X
  ```
  - To run the code:
    ```python
    test_targetc = KFoldTargetEncoderTest(new_train,
                                          'Feature',
                                          'Feature_Kfold_Target_Enc')
    new_test = test_targetc.fit_transform(test)
    ```  
