import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np

from sklearn.model_selection import train_test_split 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score) 
from sklearn.ensemble import RandomForestRegressor 

df = pd.read_csv("Housing.csv") 
print("First 5 rows")
print(df.head())
print(df.shape) 
print(df.info) 

plt.figure(figsize =(8,5)) 
sns.histplot(df["price"], bins=30, kde=True)
plt.title("House Price Distribution")
plt.show()
plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=np.number).corr(),annot=True,cmap="coolwarm")
plt.title("correlation matrix")
plt.show()
X = df.drop("price",axis = 1) 
y = df["price"]
numeric_features = X.select_dtypes( include=["int64", "float64"]).columns 
categorical_features = X.select_dtypes(include=["object"]).columns
numeric_features = Pipeline(steps = [("imputer",SimpleImputer(strategy="median"))])
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),("encoder", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer(transformers=[("num", numeric_features, numeric_features),("cat", categorical_transformer, categorical_features)])
model = RandomForestRegressor(n_estimators = 300,max_depth = 12,random_state=42) 
pipeline = Pipeline(steps = [ ("preprocessor", preprocessor),("model", model)])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42) 
pipeline.fit(X_train,y_train) 
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test,y_pred) 
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

print("\n==========================")
print("MODEL PERFORMANCE")
print("==========================")
print(f"MAE  : {mae:,.2f}")
print(f"MSE  : {mse:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R2   : {r2:.4f}")
plt.figure(figsize = (8,6))
plt.scatter()
sample_house = pd.DataFrame({
    "area":[5000],
    "bedrooms":[3],
    "bathrooms":[2],
    "stories":[2],
    "mainroad":["yes"],
    "guestroom":["no"],
    "basement":["yes"],
    "hotwaterheating":["no"],
    "airconditioning":["yes"],
    "parking":[2],
    "prefarea":["yes"],
    "furnishingstatus":["semi-furnished"]
    })
predicted_price = pipeline.predict(sample_house) 
print("\nPredicted House Price:")
print(f"₹ {predicted_price[0]:,.2f}")
encoded_features = pipeline.named_steps[ "preprocessor"].get_feature_names_out()
importance = pipeline.named_steps["preprocessor"].get_feature_names_out() 
feature_importance = pd.DataFrame({"Feature": encoded_features,"Importance": importance})
feature_importance = feature_importance.sort_values(by="Importance",ascending=False)
print("\nTop 10 Important Features")
print(feature_importance.head(10))
plt.figure(figsize = (10,6))
sns.barplot( data=feature_importance.head(10),x="Importance",y="Feature")
plt.title("top 10 important features")
plt.show() 
