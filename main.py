import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

df1 = pd.read_csv("dataset.csv")
df1.head()

df1.info()

df1.isnull().sum()

df1["Disease"].unique()
df1["Disease"].nunique()

df2 = df1[
    (df1["Disease"]=="Fungal infection") |
    (df1["Disease"]=="Allergy") |
    (df1["Disease"]=="Diabetes ") |
    (df1["Disease"]=="Hypertension ") |
    (df1["Disease"]=="Malaria") |
    (df1["Disease"]=="Dengue") |
    (df1["Disease"]=="Typhoid") 
        ]

df2.info()

df2.isnull().sum()

df2 = df2.drop(["Symptom_5","Symptom_6","Symptom_7","Symptom_8","Symptom_9","Symptom_10","Symptom_11","Symptom_12","Symptom_13",
                "Symptom_14","Symptom_15","Symptom_16","Symptom_17"],axis=1)

df2 = df2.dropna()

df_sev1 = pd.read_csv("Symptom-severity.csv")

df2["Disease"].value_counts()

df2["Symptom_4"].unique()

df_sev1[df_sev1["Symptom"]=="dischromic_patches"]

df2["Symptom_1"] = df2["Symptom_1"].map({"itching":1," continuous_sneezing":4," fatigue":4," weight_loss":3,
                                         " headache":3," chest_pain":7," chills":3," vomiting":5," skin_rash":3
                                        })
df2["Symptom_2"] = df2["Symptom_2"].map({" skin_rash":3," shivering":5," weight_loss":3," restlessness":5,
                                         " chest_pain":7," dizziness":4," vomiting":5," high_fever":7," chills":3,
                                         " joint_pain":3," fatigue":4
                                        })
df2["Symptom_3"] = df2["Symptom_3"].map({" nodal_skin_eruptions":4," chills":3," restlessness":5," lethargy":2,
                                         " dizziness":4," loss_of_balance":4," high_fever":7," sweating":3,
                                         " joint_pain":3," vomiting":5," fatigue":4,
                                        })
df2["Symptom_4"] = df2["Symptom_4"].map({" dischromic _patches":6," watering_from_eyes":4," lethargy":2," irregular_sugar_level":5,
                                         " loss_of_balance":4," lack_of_concentration":3," sweating":3,
                                         " headache":3," vomiting":5," fatigue":4," high_fever":7
                                        })


from sklearn.model_selection import train_test_split
X = df2.drop(["Disease"],axis=1).values

y = df2["Disease"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

rfc = RandomForestClassifier()
dtc = DecisionTreeClassifier()
lr = LogisticRegression(max_iter=1000)

rfc.fit(X_train,y_train)
dtc.fit(X_train,y_train)
lr.fit(X_train,y_train)

print(rfc.score(X_test,y_test))
print(dtc.score(X_test,y_test))
print(lr.score(X_test,y_test))

y_pred = dtc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,y_pred))

filename = 'model.pkl'
pickle.dump(rfc, open(filename, 'wb'))
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
result



