import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


data = pd.read_csv("abalone.csv")

#Getting the age column alone
ages = data.iloc[:,-1:].values

##1.1 and 1.2 100 and 1000 samples for training

#Split the samples into train and validation
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,1:-1],ages,train_size=0.2396,random_state=0)

print(len(x_train))

#Create the instance and run the learning algorithm
gnb = BernoulliNB()
gnb.fit(x_train, y_train.ravel())

#Getting the confusion matrix and results
result = gnb.predict(x_test)
cm = confusion_matrix(y_test,result)
print(cm)

print(len(y_test))

#Getting the accuracy
accuracy = accuracy_score(y_test, result)
print(accuracy)
##





 

