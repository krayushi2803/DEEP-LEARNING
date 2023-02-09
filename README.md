# Logistic Regression
**This code works on titanic.csv file where on X axis we have Age and Fare and Y axis is Survived. We find the accuracy of the data using Logistic Regression.**

# Imports
**In order to find the accuracy and logistic regression model we need to import the following**

```imports pandas as pd```

```from sklearn.model_selection import train_test_split```

```from sklearn.linear_model import LogisticRegression```

```from sklearn.metrics import accuracy_score```

# Import dataset
**To read the dataset**

```pd.read_csv("titanic.csv")```

# Logistic Regression
**to create the model**

```df=df.dropna()```

```X=df[["Age","Fare"]]```

```Y=df["Survived"]```

```from sklearn.model_selection import train_test_split```

```X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)```

```from sklearn.linear_model import LogisticRegression```

```model= LogisticRegression()```

```model.fit(X_train,Y_train)```

# Accuracy
**To calculate the accuracy**

```from sklearn.metrics import accuracy_score```

```Y_predict= model.predict(X_test)```

```print(accuracy_score(Y_predict,Y_test))```

# Ploting 3D Model for the same

```import plotly.express as px```

```fig=px.scatter_3d(df,x='Age',y='Fare',z='Survived')```

```fig.show()```








