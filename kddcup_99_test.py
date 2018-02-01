#import all the required packages
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

#create a list with all the variable names - 42 of them
names = [
"duration",
"protocol_type",
"service",
"flag",
"src_bytes",
"dst_bytes",
"land",
"wrong_fragment",
"urgent",
"hot",
"num_failed_logins",
"logged_in",
"num_compromised",
"root_shell",
"su_attempted",
"num_root",
"num_file_creations",
"num_shells",
"num_access_files",
"num_outbound_cmds",
"is_host_login",
"is_guest_login",
"count",
"srv_count",
"serror_rate",
"srv_serror_rate",
"rerror_rate",
"srv_rerror_rate",
"same_srv_rate",
"diff_srv_rate",
"srv_diff_host_rate",
"dst_host_count",
"dst_host_srv_count",
"dst_host_same_srv_rate",
"dst_host_diff_srv_rate",
"dst_host_same_src_port_rate",
"dst_host_srv_diff_host_rate",
"dst_host_serror_rate",
"dst_host_srv_serror_rate",
"dst_host_rerror_rate",
"dst_host_srv_rerror_rate",
"connection_type"
]

#load the kddcup.data_10_percent_corrected dataset to build the model
#this dataset has 10 percent of the overall data - approximately 500000 records of the 5 million

kddcup_99 = pd.read_csv('E:\PGDM\Assignment\BA03 -1 Assignments\Intrusion Detection\kddcup.data_10_percent_corrected', sep=",", names = names)

#check the first 50 records to ensure that the dataset has been imported correctly
kddcup_99.head(50)
kddcup_99.describe()

#check for missing values
null_data = kddcup_99[kddcup_99.isnull().any(axis=1)]

#check the datatypes of all the variables
kddcup_99.dtypes

#the 3 variables, protocol_type, service and flag are strings
#these variables can either be deleted (using the command in the next line) or convereted to numeric data type by encoding numbers for each of the values
##kddcup_99 = kddcup_99.drop(kddcup_99.columns[[1, 2, 3]], axis=1)

#check the count of the values for each of the 3 variables
kddcup_99['protocol_type'].value_counts()
kddcup_99['service'].value_counts()
kddcup_99['flag'].value_counts()

#encode the values as numbers - for example, the 3 protocol types are numbered as 0, 1 and 2
label_E = LabelEncoder()
kddcup_99['protocol_type'] = label_E.fit_transform(kddcup_99['protocol_type'])
kddcup_99['service'] = label_E.fit_transform(kddcup_99['service'])
kddcup_99['flag'] = label_E.fit_transform(kddcup_99['flag'])

#check the description and datatypes for the dataset and also check the value count for the 3 values after encoding
kddcup_99.describe()
kddcup_99.dtypes
kddcup_99['protocol_type'].value_counts()
kddcup_99['service'].value_counts()
kddcup_99['flag'].value_counts()

#feature selection
#to reduce the number of features without impacting the model, 
#first check the variance of all the features, if there is minimal variance, then such features can be ignored
kdd_var = np.var(kddcup_99)

#delete the features which have minimal variance (<0.05)
kddcup_99 = kddcup_99.drop(kddcup_99.columns[[7, 10, 13, 16, 17, 18, 19, 20, 21, 29, 30, 34]], axis=1)

#next, calculate the correlation between the features
#if two features have high correlation, then one of those features can be retained instead of both
kdd_corr = kddcup_99.corr()
kddcup_99 = kddcup_99.drop(kddcup_99.columns[[12, 19]], axis=1)

#the target variable contains details of the connection type,
#i.e., if the connection type is normal, it is a good connection, 
#the rest of the connection types are bad connections or attacks
kddcup_99['connection_type'].value_counts()

#rename the connection type as either good or bad
kddcup_99.loc[kddcup_99.connection_type != 'normal.', 'connection_type'] = 'bad'
kddcup_99.loc[kddcup_99.connection_type == 'normal.', 'connection_type'] = 'good'

#Build predictive models on this final dataset

#Naive Bayes Model

#define the predictor and target variables
X = kddcup_99.ix[:,:27]
y = kddcup_99.ix[:,27:]

#split the data into train and test datasets (70 - 30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#fit the model and predict the target variable for the test dataset
gnb = GaussianNB()
y_pred_gnb = gnb.fit(X_train, y_train).predict(X_test)

#create the confusion matrix for the actual target value in the test data against the predicted value
cnf_matrix_gnb = confusion_matrix(y_test, y_pred_gnb)

#check the accuracy score of the model
accuracy_score(y_test, y_pred_gnb)



#Decision Tree Model

#split the data into train and test datasets (70 - 30) and define the predictor and target variables
train_DT, test_DT = train_test_split(kddcup_99, test_size = 0.3)
X = train_DT.ix[:,:27]
y = train_DT.ix[:,27:]
x_test = test_DT.ix[:,:27]
y_test = test_DT.ix[:,27:]

#fit the model and predict the target variable for the test dataset
model = tree.DecisionTreeClassifier(criterion='gini')
model.fit(X, y)

#Predict Output
predicted = model.predict(x_test)

#check the model score
model.score(X, y)
accuracy_score(y_test, predicted)
