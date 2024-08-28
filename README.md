# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load Data: Read the dataset using pd.read_csv() and display first and last few rows.

2. Prepare Data: Separate features (hours) and target variable (scores) for training and testing.

3. Split Data: Use train_test_split() to divide the dataset into training and testing sets.

4. Train Model: Fit a linear regression model using the training data.

5. Evaluate and Plot: Predict scores on the test set, and visualize results with scatter and line plots.

## Program and Output:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SETHUKKARASI C
RegisterNumber:  212223230201
*/
```

```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
```
```
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```


![output1](/o1.png)


```
dataset.info()
```


![output2](/o2.png)


```
#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)
```


![output3](/o3.png)


```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/5,random_state=11)
X_train.shape
```


![output4](/o4.png)


```
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
```
```
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
```


![output5](/o5.png)


```
plt.scatter(X_train,Y_train,color="green")
plt.plot(X_train,reg.predict(X_train),color="red")
plt.title("Training set(H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_train,reg.predict(X_train),color="gold")
plt.title("Testing set(H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```


![output6](/o6.png)
![output7](/o7.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
