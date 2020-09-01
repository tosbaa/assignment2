import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv("abalone.csv")

#Getting the age column alone
ages = data.iloc[:,-1:].values

# ##1.1 and 1.2 100 and 1000 samples for training

# #Split the samples into train and validation
# x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,1:-1],ages,train_size=0.2396,random_state=0)

# print(len(x_train))

# #Create the instance and run the learning algorithm
# gnb = BernoulliNB()
# gnb.fit(x_train, y_train.ravel())

# #Getting the confusion matrix and results
# result = gnb.predict(x_test)
# cm = confusion_matrix(y_test,result)
# print(cm)

# print(len(y_test))

# #Getting the accuracy
# accuracy = accuracy_score(y_test, result)
# print(accuracy)
### End q1

## Question 2

x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,1:-1],ages,train_size=0.2396,random_state=0)

## 2.1

# sfs = SequentialFeatureSelector(RandomForestClassifier(), 
#             k_features=3, 
#             forward=False, 
#             floating=False,
#             scoring='accuracy',
#             cv=2)

# sfs = sfs.fit(x_train, y_train.ravel())
# selected_features = x_train.columns[list(sfs.k_feature_idx_)]
# print(selected_features)

## End of 2.1

## 2.2 and 2.3

# #Split the samples into train and validation
# x_train, x_test, y_train, y_test = train_test_split(data[['Length', 'Height', 'Diameter']],ages,train_size=0.024,random_state=0)

# print(len(x_train))

# #Create the instance and run the learning algorithm
# gnb = MultinomialNB()
# gnb.fit(x_train, y_train.ravel())

# #Getting the confusion matrix and results
# result = gnb.predict(x_test)
# cm = confusion_matrix(y_test,result)
# print(cm)

# print(len(y_test))

# #Getting the accuracy
# accuracy = accuracy_score(y_test, result)
# print(accuracy)


## End of 2.2 and 2.3

## End q2





 

