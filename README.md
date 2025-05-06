# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi (1).csv")
df.head(10)
```
![Screenshot 2025-05-06 135149](https://github.com/user-attachments/assets/d343b857-c829-4a3b-aac6-97bdef6541e4)

```
df.dropna()
```
![Screenshot 2025-05-06 135155](https://github.com/user-attachments/assets/35a3c089-ca23-4dda-9163-ae7ee416ff56)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![Screenshot 2025-05-06 135303](https://github.com/user-attachments/assets/3b13de20-c5e7-454d-bc3d-a76cce10649f)

```
df1=pd.read_csv("/content/bmi (1).csv")
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![Screenshot 2025-05-06 135202](https://github.com/user-attachments/assets/96183672-eded-4374-9810-e1c96a1a6197)

```
df2=pd.read_csv("/content/bmi (1).csv")
from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
df2[['Height','Weight']]=mm.fit_transform(df2[['Height','Weight']])
df2
```
![Screenshot 2025-05-06 135209](https://github.com/user-attachments/assets/35bef1b5-d4fc-4b1d-98f7-cb68d707278b)

```
df22=pd.read_csv("/content/bmi (1).csv")
from sklearn.preprocessing import Normalizer
nm=Normalizer()
df22[['Height','Weight']]=nm.fit_transform(df22[['Height','Weight']])
df22
```
![Screenshot 2025-05-06 135216](https://github.com/user-attachments/assets/9bee9774-7cc0-4f45-8469-63f7b643553c)

```
df3=pd.read_csv("/content/bmi (1).csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![Screenshot 2025-05-06 135225](https://github.com/user-attachments/assets/7e5b5652-6a3c-4792-ad8b-40eba4be2a4e)

```
df4=pd.read_csv("/content/bmi (1).csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head(10)
```
![Screenshot 2025-05-06 135230](https://github.com/user-attachments/assets/9eaca988-c7e3-460a-9e6d-8a6307fd43b3)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load the 'tips' dataset from seaborn
import seaborn as sns
tips = sns.load_dataset('tips')

# Display the first few rows of the dataset
tips.head()
```
![Screenshot 2025-05-06 135236](https://github.com/user-attachments/assets/8161696a-3d26-4833-8d0c-886e6b8542e6)

```
contingency_table = pd.crosstab(tips['sex'], tips['time'])
print(contingency_table)
```
![Screenshot 2025-05-06 135241](https://github.com/user-attachments/assets/61bead95-e91b-47b6-acea-f3e9ae6e1184)

```
chi2, p, _, _ = chi2_contingency(contingency_table)

# Display the results
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```
![Screenshot 2025-05-06 135246](https://github.com/user-attachments/assets/08f57f37-d590-4cb2-89fe-52f7c682b9d5)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

# Create a sample dataset
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': ['A', 'B', 'C', 'A', 'B'],
    'Feature3': [0, 1, 1, 0, 1],
    'Target': [0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Separate features and target
X = df[['Feature1', 'Feature3']]
y = df['Target']

# SelectKBest with mutual_info_classif for feature selection
selector = SelectKBest(score_func=mutual_info_classif, k=1)
X_new = selector.fit_transform(X, y)

# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)

# Print the selected features
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![Screenshot 2025-05-06 135252](https://github.com/user-attachments/assets/e56fb3b8-8e10-4e09-93f8-e7f41b2f0455)

# RESULT:
   Thus to read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file is created.
