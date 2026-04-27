# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NAVEEN M
RegisterNumber:  212225230197
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("student_scores.csv")
df.head(10)

plt.scatter(df['Hours'], df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
x = df.iloc[:,0:1]
y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)

y_pred = lr.predict(X_test)

plt.scatter(df['Hours'],df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.plot(X_train, lr.predict(X_train), color='red')

lr.coef_
lr.intercept_

mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
<img width="751" height="626" alt="Screenshot 2026-04-27 215940" src="https://github.com/user-attachments/assets/ec1c040a-9548-462b-8a9f-26c4df62274c" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
