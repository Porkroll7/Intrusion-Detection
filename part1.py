#Code Created by: Kyle Ketterer
#Date: 04/07/2025

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import time #wanted to time the training of the models

#Helper functions ====================================================================
def write_to_part1(text):
    f.write(f"{text}:\n")
    f.write(classification_report(y_test, y_pred))
    f.write(f"Training time: {end - start:.5f} seconds\n")
    f.write("\n\n")

def print_part1(text):
    print(text)
    print(classification_report(y_test, y_pred))
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print(f"LR Training time: {end - start:.5f} seconds")
    print("\n\n")

#Preprocessing Data Section ==========================================================

#import datasets
train_ks = pd.read_csv('train_kdd_small.csv')
test_ks = pd.read_csv('test_kdd_small.csv')

#create encoders
protocol_encoder = LabelEncoder()
service_encoder = LabelEncoder()
flag_encoder = LabelEncoder()

#fit and transform training data
train_ks['protocol_type'] = protocol_encoder.fit_transform(train_ks['protocol_type'])
train_ks['service'] = service_encoder.fit_transform(train_ks['service'])
train_ks['flag'] = flag_encoder.fit_transform(train_ks['flag'])

#transorm test data
test_ks['protocol_type'] = protocol_encoder.transform(test_ks['protocol_type'])
test_ks['service'] = service_encoder.transform(test_ks['service'])
test_ks['flag'] = flag_encoder.transform(test_ks['flag'])

#turn label strings into integers, 0 = normal, 1 = attack
def label_to_int(x):
    if x == 'normal':
        return 0
    else:
        return 1

#identify which rows are normal and which are attacks
train_ks['label'] = train_ks['label'].apply(label_to_int)
test_ks['label'] = test_ks['label'].apply(label_to_int)

#get all features
X_train = train_ks.drop(columns=['label'])
X_test = test_ks.drop(columns=['label'])

#get the label
y_train = train_ks['label']
y_test = test_ks['label']

#scale features so that features evenly contribute to the model
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


#Creating and Testing Model Section ==================================================


#create the logistic regression model-------------------------------------------------
model = LogisticRegression() #default iterations is 100

#train the model
start = time.time()
model.fit(X_train, y_train)
end = time.time()


f = open("part1.txt", "w")

#make predictions
y_pred = model.predict(X_test)

#classification report
print_part1("Logistic Regression Classification Report")
write_to_part1("Logistic Regression Classification Report") #writes to part1.txt



#create the SVM model --------------------------------------------------------------
model = SVC()

#train the SVM model
start = time.time()
model.fit(X_train, y_train)
end = time.time()

#make predictions
y_pred = model.predict(X_test)

#classification report
print_part1("SVM Classification Report")
write_to_part1("SVM Classification Report") #writes to part1.txt



#create the Random Forest model ---------------------------------------------------
model = RandomForestClassifier()

#train the Random Forest model
start = time.time()
model.fit(X_train, y_train)
end = time.time()

#make predictions
y_pred = model.predict(X_test)

#classification report
print_part1("Random Forest Regression Classification Report")
write_to_part1("Random Forest Regression Classification Report") #writes to part1.txt