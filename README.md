# Employee_Burnout_Analysis_and_Predictions
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from google.colab import drive
drive.mount('/content/drive')

pd.set_option('display.max_columns', None)
burnoutDF=pd.read_csv('/content/drive/MyDrive/employee_burnout_analysis.csv')
burnoutDF

# converting into dateTime datatype
burnoutDF["Date of Joining"]= pd.to_datetime(burnoutDF["Date of Joining"])

# giving no.of rows and columns
burnoutDF.shape

#general information
burnoutDF.info()

# show top 5 rows
burnoutDF.head()

#extract all columns of the dataset
burnoutDF.columns

#check for null values
burnoutDF.isna().sum()

# check for duplicate values
burnoutDF.duplicated().sum()

# calculate the mean, std, max, min, count of every attributes
burnoutDF.describe()

# shows the unique values
for i, col in enumerate(burnoutDF.columns):
  print(f"\n\n{burnoutDF[col].unique()}")
  print(f"\n{burnoutDF[col].value_counts()}\n\n")

  #dorpout the irrelevent column
burnoutDF=burnoutDF.drop(['Employee ID'],axis=1)

#check the skewness of the attributes
intFloatburnoutDF=burnoutDF.select_dtypes([np.int,np.float])
for i, col in enumerate(intFloatburnoutDF.columns):
 if (intFloatburnoutDF[col].skew() >= 0.1):
  print("\n",col, "feature is Positively Skewed and value is: ", intFloatburnoutDF[col].skew())
 elif (intFloatburnoutDF[col].skew() <= -0.1):
  print("\n",col, "feature is Negtively skewed and value is: ", intFloatburnoutDF[col].skew())
 else:
  print("\n",col, "feature is Normally Distributed and value is: ", intFloatburnoutDF[col].skew())

  # replace null value with mean
burnoutDF['Resource Allocation'].fillna(burnoutDF['Resource Allocation'].mean(), inplace=True)
burnoutDF["Mental Fatigue Score"].fillna(burnoutDF['Mental Fatigue Score'].mean(),inplace=True)
burnoutDF['Burn Rate'].fillna(burnoutDF['Burn Rate'].mean(), inplace=True)

#check for null values
burnoutDF.isna().sum()

# shows the correlation
burnoutDF.corr()

#Plotting heat map to check the correlation
Corr=burnoutDF.corr()
sns.set(rc={'figure.figsize':(14,12)})
fig = px.imshow(Corr, text_auto=True, aspect="auto")
fig.show()

# count plot distribution of "Gender"
plt.figure(figsize=(10,8))
sns.countplot(x="Gender", data=burnoutDF, palette="magma")
plt.title("plot Distribution of Gender")
plt.show()

# count plot distribution of "Company Type"
plt.figure(figsize=(10,8))
sns.countplot(x="Company Type", data=burnoutDF, palette="Spectral")
plt.title("Plot Distribution of Company Type")
plt.show()

# count plot distribution of "WFH Setup Available"
plt.figure(figsize=(10,8))
sns.countplot(x="WFH Setup Available", data=burnoutDF, palette="dark:salmon_r")
plt.title("Plot Distribution of WFH Setup Available")
plt.show()

# count plot distribution of attributes with help of the histogram
burn_st=burnoutDF.loc[:,'Date of Joining':'Burn Rate']
burn_st=burn_st.select_dtypes([int,float])
for i, col in enumerate(burn_st.columns):
  fig=px.histogram(burn_st, x=col, title="PLOt Distribution of "+col, color_discrete_sequence=['indianred'])
  fig.update_layout(bargap=0.2)
  fig.show()

  # plot distribution of burn rate on the basis of designation
fig=px.line(burnoutDF, y="Burn Rate", color="Designation", title="Burn rate on the basis of Designation", color_discrete_sequence=px.colors.qualitative.Pastel1)
fig.update_layout(bargap=0.1)
fig.show()

# plot distribution of burn rate on the basis of Gender
fig=px.line(burnoutDF, y="Burn Rate", color="Gender", title="Burn rate on the basis of Gender", color_discrete_sequence=px.colors.qualitative.Pastel1)
fig.update_layout(bargap=0.2)
fig.show()

# plot distribution of Mental fatigue score on the basis of Designation
fig=px.line(burnoutDF, y="Mental Fatigue Score", color="Designation", title="Mentel fatigue vs Designation", color_discrete_sequence=px.colors.qualitative.Pastel1)
fig.update_layout(bargap=0.2)
fig.show()

# plot distribution of "Designation vs mental fatigue score" as per company type, burn rate and gender
sns.relplot(
    data=burnoutDF, x="Designation", y="Mental Fatigue Score", col="Company Type",
    hue="Company Type", size="Burn Rate", style="Gender",
    palette=["g","r"], sizes=(50,200)
)

# label encoding and assign in new variable
from sklearn import preprocessing
Label_encode = preprocessing.LabelEncoder()

# Assign in new vaiable
burnoutDF['GenderLabel']=Label_encode.fit_transform(burnoutDF['Gender'].values)
burnoutDF['Company_TypeLabel']=Label_encode.fit_transform(burnoutDF['Company Type'].values)
burnoutDF['WFH_Setup_AvailableLabel'] = Label_encode.fit_transform(burnoutDF['WFH Setup Available'].values)

#check assigned values
gn = burnoutDF.groupby('Gender')
gn = gn['GenderLabel']
gn.first()


#check assigned values
ct = burnoutDF.groupby('Company Type')
ct = ct['Company_TypeLabel']
ct.first()

#check assigned values
wsa = burnoutDF.groupby('WFH Setup Available')
wsa = wsa['WFH_Setup_AvailableLabel']
wsa.first()

# show last 10 rows
burnoutDF.tail(10)

# Feature selection
Columns=['Designation','Resource Allocation','Mental Fatigue Score','GenderLabel','Company_TypeLabel','WFH_Setup_AvailableLabel']
X=burnoutDF[Columns]
y=burnoutDF['Burn Rate']

print(X)

print(y)

# Principle component Analysis
from sklearn.decomposition import PCA
pca=PCA(0.95)
X_pca=pca.fit_transform(X)
print("PCA shape of X is:",X_pca.shape,"and original shape is:",X.shape)
print("% of importance of selected features is:", pca.explained_variance_ratio_)
print("The number of features selected through PCA is:", pca.n_components_)

# Data splitting in train and test
from sklearn.model_selection import train_test_split
X_train_pca, X_test, Y_train, Y_test = train_test_split(X_pca,y, test_size = 0.25, random_state=10)

#print the shape of splitted data
print(X_train_pca.shape, X_test.shape,Y_train.shape, Y_test.shape)

from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor()
rf_model.fit(X_train_pca, Y_train)

train_pred_rf=rf_model.predict(X_train_pca)
train_r2=r2_score(Y_train,train_pred_rf)
test_pred_rf=rf_model.predict(X_test)
test_r2=r2_score(Y_test,test_pred_rf)
# Accuracy score
print("Accuracy score of train data:"+str(round(100*train_r2,4))+"%")
print("Accuracy score of test data:"+str(round(100*test_r2,4))+"%")

from sklearn.ensemble import AdaBoostRegressor

abr_model = AdaBoostRegressor()
abr_model.fit(X_train_pca, Y_train)

train_pred_adboost=abr_model.predict(X_train_pca)
train_r2=r2_score(Y_train,train_pred_adboost)
test_pred_adaboost=abr_model.predict(X_test)
test_r2=r2_score(Y_test,test_pred_adaboost)

# Accuracy score
print("Accuracy score of train data:"+str(round(100*train_r2,4))+"%")
print("Accuracy score of test data:"+str(round(100*test_r2,4))+"%")
