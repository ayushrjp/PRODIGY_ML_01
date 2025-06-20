# linear_model.py


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



df = pd.read_csv('../data/train.csv')
print("✅ Dataset Loaded:", df.shape)
df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]

# Step 4: Visualize
sns.pairplot(df)
plt.suptitle("Feature Relationships", y=1.02)
plt.tight_layout()
plt.show()


X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Model Trained")

# Step 7: Predict
y_pred = model.predict(X_test)

# Step 8: Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 MSE: {mse:.2f}")
print(f"✅ R2 Score: {r2:.2f}")


import joblib
joblib.dump(model, '../models/linear_model.pkl')
print("💾 Model saved to models/linear_model.pkl")
