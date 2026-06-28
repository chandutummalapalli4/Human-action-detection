import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%

from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# %%

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm

# %%

from sklearn.metrics import classification_report, r2_score, accuracy_score, recall_score,\
precision_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# %%

df=pd.read_csv("mhealth_raw_data.csv")
df

# %%

df.info()

# %%

df.describe().T

# %%

df.isnull().sum()

# %%

df.duplicated().sum()

# %%

plt.figure(figsize=(10,8))
df['Activity'].value_counts().plot.bar()

# %%

data_activity_0= df[df["Activity"] == 0]
data_activity_else = df[df["Activity"] != 0]

# %%

data_activity_0=data_activity_0.sample(n=40000) 
df=pd.concat([data_activity_0, data_activity_else])

# %%

plt.figure(figsize=(10,8))
df['Activity'].value_counts().plot.bar()

# %%

len(df)

# %%

activity_label= {
0: "None",
1: "Standing still (1 min)",
2: "Sitting and relaxing (1 min)",
3: "Lying down (1 min)",
4: "Walking (1 min)",
5: "Climbing stairs (1 min)",
6: "Waist bends forward (20x)",
7: "Frontal elevation of arms (20x)",
8: "Knees bending (crouching) (20x)",
9: "Cycling (1 min)",
10: "Jogging (1 min)",
11: "Running (1 min)",
12: "Jump front & back (20x)"
}

# %%

subject1 = df[df['subject'] == 'subject1']
readings = ['a', 'g']
for i in range(1,13):
    for r in readings:
        print(f"======================={activity_label[i]} - {r}==========================")
        plt.figure(figsize=(14,4))
        plt.subplot(1,2,1)
        plt.plot(subject1[subject1['Activity'] == i].reset_index(drop=True) [r+ "lx"], color='blue', alpha=0.7, label=r+ "1x")
        plt.plot(subject1 [subject1['Activity'] == i].reset_index(drop=True) [r + "ly"], color='red', alpha=0.7, label=r+ "ly")
        plt.plot(subject1[subject1['Activity'] == i].reset_index(drop=True) [r + "lz"], color='orange', alpha =  0.7, label = r + "lz")
        plt.title("Left ankle sensor")
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(subject1 [subject1['Activity'] == i].reset_index(drop=True) [r + "rx"], color='blue', alpha=0.7, label=r+ "rx")
        plt.plot(subject1[subject1['Activity'] == i].reset_index(drop=True) [r + "ry"], color='red', alpha=0.7, label= r+ "ry")
        plt.plot(subject1 [subject1['Activity'] == i].reset_index(drop=True) [r+ "rz"], color='orange', alpha=0.7, label = r + "rz")
        plt.title("Right wrist sensor")
        plt.legend() 
        plt.show()

# %%

for i in range(1,13):
    for r in readings:
        print(f"======================={activity_label[i]} - {r}==========================")
        plt.figure(figsize=(14,4))
        plt.subplot(1,2,1)
        plt.hist(subject1[subject1['Activity'] == i].reset_index(drop=True) [r+ "lx"], color='blue', alpha=0.7, label=r+ "1x")
        plt.hist(subject1 [subject1['Activity'] == i].reset_index(drop=True) [r + "ly"], color='red', alpha=0.7, label=r+ "ly")
        plt.hist(subject1[subject1['Activity'] == i].reset_index(drop=True) [r + "lz"], color='orange', alpha =  0.7, label = r + "lz")
        plt.title("Left ankle sensor")
        plt.legend()
        plt.subplot(1,2,2)
        plt.hist(subject1 [subject1['Activity'] == i].reset_index(drop=True) [r + "rx"], color='blue', alpha=0.7, label=r+ "rx")
        plt.hist(subject1[subject1['Activity'] == i].reset_index(drop=True) [r + "ry"], color='red', alpha=0.7, label= r+ "ry")
        plt.hist(subject1 [subject1['Activity'] == i].reset_index(drop=True) [r+ "rz"], color='orange', alpha=0.7, label = r + "rz")
        plt.title("Right wrist sensor")
        plt.legend() 
        plt.show()

# %%

df['Activity']=df['Activity'].replace([0,1,2,3,4,5,6,7,8,9,10,11,12],['None',
                                                                      'Standing still (1 min)',
                                                                      'Sitting and relaxing (1 min)',
                                                                      'Lying down (1 min)',
                                                                      'Walking (1 min)',
                                                                       'Climbing stairs (1 min)', 
                                                                      'Waist bends forward (20x)',
                                                                      'Frontal elevation of arms (20x)',
                                                                       'Knees bending (crouching) (20x)',
                                                                       'Cycling (1 min)', 'Jogging (1 min)',
                                                                       'Running (1 min)', 'Jump front & back (20x)'])

# %%

df["Activity"]

# %%

df.Activity.value_counts()

# %%

plt.figure(figsize=(12,8))
round(df["Activity"].value_counts()/df.shape[0]*100,2).plot.pie(autopct= '%2.1f%%')

# %%

df1 = df.copy()

for feature in df1.columns[:-2]:
    lower_range = np.quantile(df[feature], 0.01)
    upper_range = np.quantile(df[feature], 0.99)
    print(feature, "range:", lower_range, 'to', upper_range)
    df1 = df1.drop(df1[(df1[feature] >upper_range) | (df1[feature] < lower_range)].index, axis = 0)
    print('shape', df1.shape)

# %%

df1

# %%

le = LabelEncoder()
df['subject'] = le.fit_transform(df['subject'])

# %%

df['Activity'] = le.fit_transform(df['Activity'])

# %%

df.plot(kind='box', subplots=True, layout = (5,5), figsize=(20,15))
plt.show()

# %%

X = df.drop(["Activity", "subject"], axis = 1).values
y= df['Activity'].values

# %%

X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.25)

# %%

ro_scaler = RobustScaler().fit(X_train)
X_train_scaled = ro_scaler.transform(X_train)
X_test_scaled=ro_scaler.transform(X_test)

# %%

def resultsSummarizer(y_true, y_pred, cm_en=True):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    if cm_en:
        plt.figure(figsize=(15,15))
        sns.heatmap(cm, annot=True, cmap="Blues",
                    xticklabels=activity_label.values(),
                    yticklabels=activity_label.values())
        plt.title('Confusion Matrix') 
        plt.show()

    print(f'Accuracy Score: {acc:.4%}')
    print(f'Precision Score: {prec:.4%}')
    print(f'Recall Score: {rec:.4%}')
    print(f'F1 Score: {f1:.4%}')

# %%

lr = LogisticRegression()
lr.fit(X_train, y_train)

# %%

lr.score(X_train, y_train)

# %%

lr.score(X_test, y_test)

# %%

lr2 = LogisticRegression()
lr2.fit(X_train_scaled,y_train)

# %%

lr2.score(X_train_scaled, y_train)

# %%

lr2.score(X_test_scaled, y_test)

# %%

y_pred_lr = lr2.predict(X_test_scaled)

# %%

resultsSummarizer(y_test, y_pred_lr)

# %%

knn1 = KNeighborsClassifier(n_neighbors=5)
knn1.fit(X_train, y_train)

# %%

y_pred_knn = knn1.predict(X_test)

# %%

resultsSummarizer(y_test, y_pred_knn)

# %%

knn2 = KNeighborsClassifier(n_neighbors=5)
knn2.fit(X_train_scaled, y_train)
y_pred_knn2 = knn2.predict(X_test_scaled)


# %%

resultsSummarizer(y_test, y_pred_knn2, cm_en=False)

# %%

for n in range(1,11):
    knn  = KNeighborsClassifier(n_neighbors=n) 
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled) 
    print(f"\n==============No of Neighbors: {n}=====================\n")
    resultsSummarizer(y_test, y_pred, cm_en=False)

# %%

dt = DecisionTreeClassifier(max_depth=14)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
resultsSummarizer(y_test, y_pred_dt)

# %%

