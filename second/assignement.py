import xgboost as xgb
from sklearn.tree import export_text
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import random

# Counter is a dict subclass for counting hashable objects
from collections import Counter

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# To ignore warnings in the notebook
import warnings
warnings.filterwarnings("ignore")

# to display up to 500 rows in the output of the jupyter notebook cell
pd.set_option('display.max_rows', 500)


# standardization

# decision tree classifier


fraud_data = pd.read_csv('../../module_2/datasets/fraud_data.csv')

# I - Data preparation (data cleaning)

# I - 1 take a look at the data
print(fraud_data.shape)  # (59054, 434)
print(fraud_data.head())

"""
       TransactionID  isFraud  TransactionDT  TransactionAmt ProductCD  ...  id_36  id_37  id_38 DeviceType  DeviceInfo
0        2994681        0         242834          25.000         H  ...      F      T      T    desktop     rv:56.0
1        3557242        0       15123000         117.000         W  ...    NaN    NaN    NaN        NaN         NaN
2        3327470        0        8378575          73.773         C  ...    NaN    NaN    NaN        NaN         NaN
3        3118781        0        2607840         400.000         R  ...      F      T      F     mobile  iOS Device
4        3459772        0       12226544          31.950         W  ...    NaN    NaN    NaN        NaN         NaN

[5 rows x 434 columns]
"""

# The target variable is isFraud. le'ts take a look at it
# isFraud = 0 --> normal transaction
# isFraud = 1 --> fraudulent transaction
print(fraud_data.isFraud.value_counts())

""" 
isFraud
0    57049
1     2005
Name: count, dtype: int64
"""
print(fraud_data.isFraud.value_counts(normalize=True) * 100)

""" 
isFraud
0    96.604802
1     3.395198
"""

# visualize the target variable column


plt.title("Histogram")
plt.xlabel("Percentage of fraud")
plt.ylabel("Frequency")

plt.hist(fraud_data.isFraud)
plt.show()

""" 
The dataset is imbalanced with more than 96% of non fraudulent transactions. 
This can affect the results of the model.
"""

# I - 2 Missing values detection and treatment

# Missing values - To get percentage of missing data in each column
fraud_data.isnull().sum() / len(fraud_data) * 100   # print

""" 
TransactionID      0.000000
isFraud            0.000000
TransactionDT      0.000000
TransactionAmt     0.000000
ProductCD          0.000000
card1              0.000000
card2              1.549429
card3              0.267552
card4              0.274325
card5              0.751854
card6              0.269245
addr1             11.392962
addr2             11.392962
dist1             59.865547
dist2             93.443289
P_emaildomain     15.934568
R_emaildomain     76.572290
C1                 0.000000
C2                 0.000000
C3                 0.000000
C4                 0.000000
C5                 0.000000
C6                 0.000000
C7                 0.000000
C8                 0.000000
...
id_37             75.945745
id_38             75.945745
DeviceType        75.979612
DeviceInfo        79.813391
dtype: float64
"""

# getting all the numerical columns
num_cols = fraud_data.select_dtypes(include=np.number).columns

# filling missing values of numerical columns with mean value
fraud_data[num_cols] = fraud_data[num_cols].fillna(fraud_data[num_cols].mean())

# getting all the categorical columns
cat_cols = fraud_data.select_dtypes(include='object').columns

# fills the missing values with maximum occuring element in the column (the mode)
fraud_data[cat_cols] = fraud_data[cat_cols].fillna(
    fraud_data[cat_cols].mode().iloc[0])

# check for missing values at the end of the process
fraud_data.isnull().sum() / len(fraud_data) * 100  # print

""" 
TransactionID     0.0
isFraud           0.0
TransactionDT     0.0
TransactionAmt    0.0
ProductCD         0.0
card1             0.0
card2             0.0
card3             0.0
card4             0.0
card5             0.0
card6             0.0
addr1             0.0
addr2             0.0
dist1             0.0
dist2             0.0
P_emaildomain     0.0
R_emaildomain     0.0
C1                0.0
C2                0.0
C3                0.0
C4                0.0
C5                0.0
C6                0.0
C7                0.0
C8                0.0
...
id_37             0.0
id_38             0.0
DeviceType        0.0
DeviceInfo        0.0
dtype: float64
"""

# II - 3 One Hot encoding (manage categorical variables)
""" 
Machine learning models require all input and output variables to be numeric. 
Run the model with data as-is and then iterate for feature engineering. This
means that if your data contains categorical data, you must encode it to numbers
before you can fit and evaluate a model.

The one-hot encoding creates one binary variable for each category.
"""
# earlier we have collected all the categorical columns in cat_cols
print(fraud_data.shape)  # (59054, 434)
# get all categorical columns of the dataset
fraud_data[cat_cols] = fraud_data[cat_cols].fillna(
    fraud_data[cat_cols].mode().iloc[0])

# Convert categorical variable into dummy/indicator variables.
# Each variable is converted in as many 0/1 variables as there are different values.
fraud_data = pd.get_dummies(
    fraud_data, columns=cat_cols).replace({False: 0, True: 1})

# get the new shape
print(fraud_data.shape)  # (59054, 1667)

print(fraud_data.head())

""" 
   TransactionID  isFraud  TransactionDT  TransactionAmt  ...  DeviceInfo_rv:60.0  DeviceInfo_verykools4009  DeviceInfo_verykools5034  DeviceInfo_vivo
0        2994681        0         242834          25.000  ...                   0                         0                         0                0
1        3557242        0       15123000         117.000  ...                   0                         0                         0                0
2        3327470        0        8378575          73.773  ...                   0                         0                         0                0
3        3118781        0        2607840         400.000  ...                   0                         0                         0                0
4        3459772        0       12226544          31.950  ...                   0                         0                         0                0

[5 rows x 1667 columns]
"""


# II - Features Engineering/ Transformation

# II - 1  Scaling

""" 
In most cases, the numerical features of the dataset do not have a certain 
range and they differ from each other. In real life, it is nonsense to expect 
age and income columns to have the same range. But from the machine learning 
point of view, how these two columns can be compared?
Scaling solves this problem.
"""

# Separate input features and output feature
# input features
X = fraud_data.drop(columns=['isFraud'])

# output feature
Y = fraud_data.isFraud

#  Normalization (or min-max normalization)
# scale all values in a fixed range between 0 and 1 to reduce the effects of aoutliers


# Standardization (or z-score normalization)


scaled_features = StandardScaler().fit_transform(X)
scaled_features = pd.DataFrame(data=scaled_features)
scaled_features.columns = X.columns

# Let's see how the data looks after scaling
print(scaled_features.head())

""" 
   TransactionID  TransactionDT  TransactionAmt     card1  ...  DeviceInfo_rv:60.0  DeviceInfo_verykools4009  DeviceInfo_verykools5034  DeviceInfo_vivo
0      -1.688548      -1.544958       -0.468203 -0.021940  ...           -0.010888                 -0.004115                 -0.004115        -0.004115
1       1.615662       1.681426       -0.073540 -0.406928  ...           -0.010888                 -0.004115                 -0.004115        -0.004115
2       0.266093       0.219070       -0.258976  0.585989  ...           -0.010888                 -0.004115                 -0.004115        -0.004115
3      -0.959645      -1.032167        1.140478  0.491581  ...           -0.010888                 -0.004115                 -0.004115        -0.004115
4       1.043171       1.053404       -0.438389 -0.185621  ...           -0.010888                 -0.004115                 -0.004115        -0.004115

[5 rows x 1666 columns]
"""


# III - Decision tree

# Splitting the data
""" 
rain - what we use to train the model

Validation - what we use to evaluate the model

Test - data that is unexposed to the model 
"""

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42)

# Handle Imbalance

""" 
An imbalanced classification problem is an example of a classification problem where
the distribution of examples across the known classes is biased or skewed 

Over sampling minority class
"""

# 'resample' is located under sklearn.utils

# concatenate training data back together
train_data = pd.concat([X_train, Y_train], axis=1)

# separate minority and majority class
not_fraud = train_data[train_data.isFraud == 0]
fraud = train_data[train_data.isFraud == 1]

# Unsample minority; we are oversampling the minority class to match the number of majority classs
fraud_upsampled = resample(fraud,
                           replace=True,  # Sample with replacement
                           # Match number in majority class
                           n_samples=len(not_fraud),
                           random_state=27)

# combine majority and upsampled minority
upsampled = pd.concat([not_fraud, fraud_upsampled])

# Now let's check the classes count
print(upsampled.isFraud.value_counts())

""" 
isFraud
0    39942
1    39942
Name: count, dtype: int64
"""

""" 
SMOTE - Synthetic Minority Oversampling Technique

Here we will use imblearn's SMOTE or Synthetic Minority Oversampling Technique.
SMOTE uses a nearest neighbors' algorithm to generate new and synthetic data we 
can use for training our model.
"""

# import SMOTE

sm = SMOTE(random_state=25, sampling_strategy=1.0)


# fit the sampling
X_train, Y_train = sm.fit_resample(X_train, Y_train)
# distribution of target class after sythetic sampling
print(Y_train.value_counts())

""" 
isFraud
0    39942
1    39942
Name: count, dtype: int64

"""
# X_train, X_val = train_test_split(X_train, test_size=0.25, random_state=11)

# print(len(X_train)) # 59913
# print(len(X_val)) # 19971
# print(len(X_test)) # 17717

# training the model

dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)

# To check if the result is good, we need to evaluate the predictive performance of the
# model on the validation set. Let’s use AUC (area under the ROC curve) for that.


y_pred = dt.predict_proba(X_train)[:, 1]
print(roc_auc_score(Y_train, y_pred))  # 1.0

y_pred = dt.predict_proba(X_test)[:, 1]
print(roc_auc_score(Y_test, y_pred))  # 0.6909205032548271

# After running, we see that AUC on validation is only 69%.

""" 
We just observed a case of overfitting. The tree learned the training data so well that
it simply memorized the outcome for each customer. However, when we applied it to
the Test set, the model failed. The rules it extracted from the data turned out to
be too specific to the training set, so it worked poorly for customers it didn't see
during training. In such cases, we say that the model cannot generalize.
"""

# Tunning

""" 
We have multiple ways to control the complexity of a tree. One option is to restrict
its size: we can specify the max_depth parameter, which controls the maximum number 
of levels. The more levels a tree has, the more complex rules it can learn

The default value for the max_depth parameter is None, which means that the tree can
grow as large as possible. We can try a smaller value and compare the results.
"""

dt = DecisionTreeClassifier(max_depth=2)
dt.fit(X_train, Y_train)

# visualize the tree


tree_text = export_text(dt, feature_names=X_train.columns)
print(tree_text)

""" 
|--- card3 <= 150.00
|   |--- V294 <= 0.00
|   |   |--- class: 0
|   |--- V294 >  0.00
|   |   |--- class: 1
|--- card3 >  150.00
|   |--- V258 <= 1.00
|   |   |--- class: 0
|   |--- V258 >  1.00
|   |   |--- class: 1
"""

print(dt.feature_importances_)
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, Y_train)

# visualize the tree
tree_text = export_text(dt, feature_names=X_train.columns)
print(tree_text)

""" 
|--- card3 <= 150.00
|   |--- V317 <= 0.00
|   |   |--- card6_debit <= 0.50
|   |   |   |--- card6_credit <= 0.50
|   |   |   |   |--- card6_debit or credit <= 0.50
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- card6_debit or credit >  0.50
|   |   |   |   |   |--- class: 0
|   |   |   |--- card6_credit >  0.50
|   |   |   |   |--- V189 <= 1.03
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- V189 >  1.03
|   |   |   |   |   |--- class: 1
|   |   |--- card6_debit >  0.50
|   |   |   |--- V38 <= 1.00
|   |   |   |   |--- V284 <= 0.01
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- V284 >  0.01
|   |   |   |   |   |--- class: 0
|   |   |   |--- V38 >  1.00
|   |   |   |   |--- V38 <= 1.17
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- V38 >  1.17
|   |   |   |   |   |--- class: 0
|   |--- V317 >  0.00
|   |   |--- V282 <= 0.00
|   |   |   |--- V294 <= 1.00
|   |   |   |   |--- class: 1
|   |   |   |--- V294 >  1.00
|   |   |   |   |--- M6_T <= 0.50
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- M6_T >  0.50
|   |   |   |   |   |--- class: 0
|   |   |--- V282 >  0.00
|   |   |   |--- D3 <= 0.00
|   |   |   |   |--- V281 <= 0.00
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- V281 >  0.00
|   |   |   |   |   |--- class: 1
|   |   |   |--- D3 >  0.00
|   |   |   |   |--- V29 <= 1.00
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- V29 >  1.00
|   |   |   |   |   |--- class: 0
|--- card3 >  150.00
|   |--- V258 <= 1.00
|   |   |--- V54 <= 0.00
|   |   |   |--- V39 <= 1.01
|   |   |   |   |--- C14 <= 0.50
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- C14 >  0.50
|   |   |   |   |   |--- class: 0
|   |   |   |--- V39 >  1.01
|   |   |   |   |--- V39 <= 1.99
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- V39 >  1.99
|   |   |   |   |   |--- class: 0
|   |   |--- V54 >  0.00
|   |   |   |--- ProductCD_H <= 0.50
|   |   |   |   |--- C1 <= 0.50
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- C1 >  0.50
|   |   |   |   |   |--- class: 1
|   |   |   |--- ProductCD_H >  0.50
|   |   |   |   |--- class: 0
|   |--- V258 >  1.00
|   |   |--- V241 <= 1.00
|   |   |   |--- V173 <= 1.00
|   |   |   |   |--- V232 <= 0.00
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- V232 >  0.00
|   |   |   |   |   |--- class: 1
|   |   |   |--- V173 >  1.00
|   |   |   |   |--- D7 <= 3.63
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- D7 >  3.63
|   |   |   |   |   |--- class: 1
|   |   |--- V241 >  1.00
|   |   |   |--- V50 <= 1.00
|   |   |   |   |--- V29 <= 1.00
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- V29 >  1.00
|   |   |   |   |   |--- class: 0
|   |   |   |--- V50 >  1.00
|   |   |   |   |--- C13 <= 0.50
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- C13 >  0.50
|   |   |   |   |   |--- class: 0
"""

""" 
GradientBoostingClassifier from Scikit-learn, XGBoost, LightGBM and CatBoost. In this
part, we use XGBoost (short for “Extreme Gradient Boosting”), which is the most
popular implementation.
"""


dtrain = xgb.DMatrix(X_train, label=Y_train,
                     feature_names=X_train.columns.tolist())

# Let’s do the same for the test dataset
dtest = xgb.DMatrix(X_test, label=Y_test,
                    feature_names=X_test.columns.tolist())

# specifying the parameters for training
xgb_params = {
    'eta': 0.3,
    'max_depth': 10,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'nthread': 8,
    'seed': 1,
    'silent': 1
}

""" 
For us, the most important parameter now is objective: it specifies the learning task.
We're solving a binary classification problem — that's why we need to choose 
binary:logistic. We cover the rest of these parameters later in this section.

For training an XGBoost model, we use the train function. Let's start with a tree 
of deepth 10:
"""

model = xgb.train(xgb_params, dtrain, num_boost_round=1)

y_pred = model.predict(dtest)

print(y_pred[:10])

""" 
[0.35908747 0.3603783  0.42723784 0.39878592 0.36274105 0.35908747
 0.35908747 0.38943648 0.36274105 0.35908747]
"""
print(roc_auc_score(Y_test, y_pred))  # 0.8125452911136942 (81%)

# add the evaluation metric
xgb_params = {
    'eta': 0.3,
    'max_depth': 10,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',  # this line
    'nthread': 8,
    'seed': 1,
    'silent': 1
}

watchlist = [(dtrain, 'train'), (dtest, 'test')]

model = xgb.train(xgb_params, dtrain,
                  num_boost_round=100,
                  evals=watchlist, verbose_eval=10)

""" 
[0]     train-auc:0.98661       test-auc:0.81255
[10]    train-auc:0.99816       test-auc:0.88477
[20]    train-auc:0.99959       test-auc:0.90582
[30]    train-auc:0.99983       test-auc:0.90797
[40]    train-auc:0.99991       test-auc:0.91169
[50]    train-auc:0.99995       test-auc:0.91394
[60]    train-auc:0.99998       test-auc:0.91326
[70]    train-auc:0.99999       test-auc:0.91311
[80]    train-auc:1.00000       test-auc:0.91354
[90]    train-auc:1.00000       test-auc:0.91324
[99]    train-auc:1.00000       test-auc:0.91180
"""

# change the learning rate with the parameter eta

xgb_params = {
    'eta': 0.1,  # here
    'max_depth': 10,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'silent': 1
}

model = xgb.train(xgb_params, dtrain,
                  num_boost_round=500, verbose_eval=10,
                  evals=watchlist)

""" 
[0]     train-auc:0.98661       test-auc:0.81255
[10]    train-auc:0.99581       test-auc:0.85697
[20]    train-auc:0.99718       test-auc:0.87536
[30]    train-auc:0.99806       test-auc:0.88790
[40]    train-auc:0.99886       test-auc:0.89561
[50]    train-auc:0.99938       test-auc:0.90451
[60]    train-auc:0.99967       test-auc:0.90935
[70]    train-auc:0.99981       test-auc:0.91158
[80]    train-auc:0.99986       test-auc:0.91232
[90]    train-auc:0.99990       test-auc:0.91394
[100]   train-auc:0.99992       test-auc:0.91453
[110]   train-auc:0.99993       test-auc:0.91415
[120]   train-auc:0.99995       test-auc:0.91421
[130]   train-auc:0.99996       test-auc:0.91434
[140]   train-auc:0.99997       test-auc:0.91407
[150]   train-auc:0.99997       test-auc:0.91439
[160]   train-auc:0.99998       test-auc:0.91408
[170]   train-auc:0.99998       test-auc:0.91399
[180]   train-auc:0.99999       test-auc:0.91470
[190]   train-auc:0.99999       test-auc:0.91459
[200]   train-auc:0.99999       test-auc:0.91453
[210]   train-auc:0.99999       test-auc:0.91455
[220]   train-auc:0.99999       test-auc:0.91504
[230]   train-auc:1.00000       test-auc:0.91449
[240]   train-auc:1.00000       test-auc:0.91457
[250]   train-auc:1.00000       test-auc:0.91489
[260]   train-auc:1.00000       test-auc:0.91435
[270]   train-auc:1.00000       test-auc:0.91489
[280]   train-auc:1.00000       test-auc:0.91493
[290]   train-auc:1.00000       test-auc:0.91509
[300]   train-auc:1.00000       test-auc:0.91511
[310]   train-auc:1.00000       test-auc:0.91483
[320]   train-auc:1.00000       test-auc:0.91538
[330]   train-auc:1.00000       test-auc:0.91556
[340]   train-auc:1.00000       test-auc:0.91572
[350]   train-auc:1.00000       test-auc:0.91561
[360]   train-auc:1.00000       test-auc:0.91527
[370]   train-auc:1.00000       test-auc:0.91557
[380]   train-auc:1.00000       test-auc:0.91609
[390]   train-auc:1.00000       test-auc:0.91588
[400]   train-auc:1.00000       test-auc:0.91576
[410]   train-auc:1.00000       test-auc:0.91560
[420]   train-auc:1.00000       test-auc:0.91591
[430]   train-auc:1.00000       test-auc:0.91613
[440]   train-auc:1.00000       test-auc:0.91607
[450]   train-auc:1.00000       test-auc:0.91591
[460]   train-auc:1.00000       test-auc:0.91582
[470]   train-auc:1.00000       test-auc:0.91580
[480]   train-auc:1.00000       test-auc:0.91586
[490]   train-auc:1.00000       test-auc:0.91568
[499]   train-auc:1.00000       test-auc:0.91551
"""
