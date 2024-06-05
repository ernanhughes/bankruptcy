# Using Machine Learning to predict  likely bankruptcy

In this notebook we will analyze som data to determine if we can predict a bankruptcy 

The data comes from Kaggle   
[bankruptcy-prediction](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction)

You will need to install python and then create an environment to work in.

```
python -m venv .venv
.\.venv\Scripts\activate
```

Next we install the required libraries

```
pip install -r .\requirements.txt
```

The [bankruptcy notebook](bankruptcy.ipynb) then analyzes the data. It is based upon this [kaggle](https://www.kaggle.com/code/ahmedtronic/company-bankruptcy-prediction) 


I had to create a copy of the imblearn project and include it for the code to run.

## Step by step

1. Load the data form the csv file.
2. We then view the contents using the head function.
```
df = pd.read_csv("data/data.csv")
df.head()
```
3. The headers contain spaces so we replace these with an _.
```
df.columns = [c.replace(' ', '_') for c in df.columns]
df.head()
```
4. We then display info on all the columns data.
```
df.info()
```
5. We then do a comparison on the _Net_Income_to_Total_Assets between the bankrupt and non bankrupt. This feature is identified later as being the most important in this dataset for prediction of likely bankruptcy.
```
df['Bankrupt?'].value_counts(normalize= True).plot(kind= 'bar')
plt.xlabel("Bankrupt classes")
plt.ylabel("Frequency")
plt.title("Class balance")
```
6. Then describe the data in this column. Also create histogram to see if the distribution is skewed.
```
df['_Net_Income_to_Total_Assets'].describe()

df["_Net_Income_to_Total_Assets"].hist()
plt.xlabel("Net Income to Total Assets")
plt.ylabel("count")
plt.title("Distrbution of Net Income to Total Assets Ratio")

q1 , q9 = df['_Net_Income_to_Total_Assets'].quantile([0.1,0.9])
mask = df["_Net_Income_to_Total_Assets"].between(q1 , q9)
sns.boxplot(x='Bankrupt?' , y='_Net_Income_to_Total_Assets', data= df[mask])
plt.xlabel("Bankrupt")
plt.ylabel(" Net Income to Total Assets")
plt.title("Distribution of Net Income to Total Assets Ratio, by Bankruptcy Status")
```
7. Create a heatmap on all the data in relation to the bankrupt column.
```
corr = df.drop(columns=['Bankrupt?']).corr()
sns.heatmap(corr)
```

8. We spit the data into two groups testing and training

```
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)
```

9. Create an [RandomOverSampler](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html)

```
over_sampler = RandomOverSampler(random_state=42)
X_train_over , y_train_over = over_sampler.fit_resample(X_train , y_train)
print(X_train_over.shape)
X_train_over.head()
```

10. Calculate the baseline accuracy

```
acc_baseline = y_train.value_counts(normalize=True).max()
print("Baseline Accuracy:", round(acc_baseline, 4))
```

11. Create a [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier)

```
clf = RandomForestClassifier(random_state=42)
params= {
    
    "n_estimators":range(25 , 100 , 25),
    "max_depth": range(10 , 70 , 10)
    
}
params
```

12. Create a [GridSearchSV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) model using the classifier and params.

```
model = GridSearchCV(

    clf,
    param_grid= params,
    cv=5,
    n_jobs=-1,
    verbose= 1

)
model
```

13. Fit the model and display the results

```
model.fit(X_train_over , y_train_over)
cv_results = pd.DataFrame(model.cv_results_)
cv_results.sort_values('rank_test_score').head(10)
```

14. We then use a classification report to determine the most important features for prediction.

```
print(classification_report(

    y_test,
    model.predict(X_test)

))

features = X_test.columns
importances = model.best_estimator_.feature_importances_
feat_imp = pd.Series(importances , index=features).sort_values()
feat_imp.tail().plot(kind= 'barh')
plt.xlabel("Gini Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")

```

15. Finally we export the model

```
with open("model-1" , "wb") as f:
    pickle.dump(model ,f)
```