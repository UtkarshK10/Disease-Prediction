import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

df1 = pd.read_csv("data.csv")
df1.head()

df1.info()

df1.isnull().sum()

df1["Disease"].unique()
df1["Disease"].nunique()
'''
df2 = df1[
    (df1["Disease"]=="Fungal infection") |
    (df1["Disease"]=="Allergy") |
    (df1["Disease"]=="Diabetes ") |
    (df1["Disease"]=="Hypertension ") |
    (df1["Disease"]=="Malaria") |
    (df1["Disease"]=="Dengue") |
    (df1["Disease"]=="Typhoid") 
        ]
'''
df2=df1
df2.info()

df2.isnull().sum()

df2 = df2.drop(["Symptom_5","Symptom_6","Symptom_7","Symptom_8","Symptom_9","Symptom_10","Symptom_11","Symptom_12","Symptom_13",
                "Symptom_14","Symptom_15","Symptom_16","Symptom_17"],axis=1)

df2 = df2.dropna()

df_sev1 = pd.read_csv("Symptom-severity.csv")

df2["Disease"].value_counts()

df2["Symptom_4"].unique()

df_sev1[df_sev1["Symptom"]=="dischromic_patches"]
'''

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
'''

df2["Symptom_1"] = df2["Symptom_1"].map({'itching':1, ' continuous_sneezing':4, ' stomach_pain':5, ' acidity':3,
       ' vomiting':5, ' skin_rash':3, ' indigestion':5, ' muscle_wasting':2,
       ' fatigue':4, ' weight_loss':3, ' cough':4, ' headache':3, ' chest_pain':7,
       ' back_pain':3, ' weakness_in_limbs':7, ' chills':3, ' joint_pain':3,
       ' yellowish_skin':3, ' constipation':4, ' pain_during_bowel_movements':5,
       ' cramps':4, ' weight_gain':3, ' mood_swings':3, ' neck_pain':5,
       ' muscle_weakness':2, ' stiff_neck':4, ' burning_micturition':6,
       ' high_fever':7})
df2["Symptom_2"] = df2["Symptom_2"].map({' skin_rash':3, ' shivering':5, ' acidity':3, ' ulcers_on_tongue':4,
       ' vomiting':5, ' yellowish_skin':3, ' stomach_pain':5,
       ' loss_of_appetite':4, ' indigestion':5, ' patches_in_throat':6,
       ' weight_loss':3, ' restlessness':5, ' sunken_eyes':3, ' cough':4,
       ' high_fever':7, ' chest_pain':7, ' dizziness':4, ' headache':3,
       ' weakness_in_limbs':7, ' neck_pain':5, ' fatigue':4, ' chills':3,
       ' joint_pain':3, ' lethargy':2, ' nausea':5, ' abdominal_pain':4,
       ' pain_during_bowel_movements':5, ' pain_in_anal_region':6,
       ' breathlessness':4, ' cramps':4, ' bruising':4, ' weight_gain':3,
       ' cold_hands_and_feets':5, ' mood_swings':3, ' anxiety':4, ' knee_pain':3,
       ' stiff_neck':4, ' swelling_joints':5, ' pus_filled_pimples':2,
       ' bladder_discomfort':4, ' skin_peeling':3, ' blister':4
                                        })
df2["Symptom_3"] = df2["Symptom_3"].map({' nodal_skin_eruptions':4, ' chills':3, ' ulcers_on_tongue':4,
       ' vomiting':5, ' yellowish_skin':3, ' nausea':5, ' stomach_pain':5,
       ' burning_micturition':6, ' abdominal_pain':4, ' loss_of_appetite':4,
       ' high_fever':7, ' restlessness':5, ' lethargy':2, ' dehydration':4,
       ' breathlessness':4, ' dizziness':4, ' loss_of_balance':4, ' headache':3,
       ' blurred_and_distorted_vision':5, ' neck_pain':5,
       ' weakness_of_one_body_side':4, ' fatigue':4, ' weight_loss':3,
       ' sweating':3, ' joint_pain':3, ' dark_urine':4, ' swelling_of_stomach':7,
       ' cough':4, ' pain_in_anal_region':6, ' bloody_stool':5, ' bruising':4,
       ' obesity':4, ' cold_hands_and_feets':5, ' mood_swings':3, ' anxiety':4,
       ' knee_pain':3, ' hip_joint_pain':2, ' swelling_joints':5,
       ' movement_stiffness':5, ' spinning_movements':6, ' blackheads':2,
       ' foul_smell_of urine':5, ' skin_peeling':3, ' silver_like_dusting':2,
       ' blister':4, ' red_sore_around_nose':2
                                        })
df2["Symptom_4"] = df2["Symptom_4"].map({' dischromic _patches':6, ' watering_from_eyes':4, ' vomiting':5,
       ' cough':4, ' nausea':5, ' loss_of_appetite':4, ' burning_micturition':6,
       ' spotting_ urination':6, ' passage_of_gases':5, ' abdominal_pain':4,
       ' extra_marital_contacts':5, ' lethargy':2, ' irregular_sugar_level':5,
       ' diarrhoea':6, ' breathlessness':4, ' family_history':5,
       ' loss_of_balance':4, ' lack_of_concentration':3,
       ' blurred_and_distorted_vision':5, ' excessive_hunger':4, ' dizziness':4,
       ' altered_sensorium':2, ' weight_loss':3, ' high_fever':7, ' sweating':3,
       ' headache':3, ' fatigue':4, ' dark_urine':4, ' yellowish_skin':3,
       ' yellowing_of_eyes':4, ' swelling_of_stomach':7,
       ' distention_of_abdomen':4, ' bloody_stool':5, ' irritation_in_anus':6,
       ' chest_pain':7, ' obesity':4, ' swollen_legs':5, ' mood_swings':3,
       ' restlessness':5, ' hip_joint_pain':2, ' swelling_joints':5,
       ' movement_stiffness':5, ' painful_walking':2, ' spinning_movements':6,
       ' scurring':2, ' continuous_feel_of_urine':6, ' silver_like_dusting':2,
       ' small_dents_in_nails':2, ' red_sore_around_nose':2,
       ' yellow_crust_ooze':3       })


from sklearn.model_selection import train_test_split
X = df2.drop(["Disease"],axis=1).values

y = df2["Disease"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

rfc = RandomForestClassifier()
dtc = DecisionTreeClassifier()
lr = LogisticRegression(max_iter=3000)

rfc.fit(X_train,y_train)
dtc.fit(X_train,y_train)
lr.fit(X_train,y_train)

print(rfc.score(X_test,y_test))
print(dtc.score(X_test,y_test))
print(lr.score(X_test,y_test))

y_pred = rfc.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)



y_pred = dtc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,y_pred))


filename = 'model.pkl'
pickle.dump(rfc, open(filename, 'wb'))
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
result



