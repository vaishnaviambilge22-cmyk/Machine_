# Example FastAPI Code
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/rul_model.joblib")

FEATURES = ["cycle"] + [f"sensor_{i}" for i in range(1, 22)]

@app.post("/predict")
def predict(data: dict):
    row = pd.DataFrame([data])[FEATURES]
   )
   prediction = model.predict(row)[0]
    return {"predicted_RUL_cycles": float(prediction)}

Complete Baseline Implementation
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. Load data
n_sensors = 21
columns = (
    ["unit", "cycle"] +
    [f"setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, n_sensors + 1)]
)

df = pd.read_csv("data/train_FD001.txt",
                 sep=r"\s+", header=None, names=columns)

# 2. Create RUL
max_cycle = df.groupby("unit")["cycle"].transform("max")
df["RUL"] = max_cycle - df["cycle"]

# 3. Basic validation
df = df.dropna().drop_duplicates()

# 4. Select features
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
feature_cols = ["cycle"] + sensor_cols

X = df[feature_cols]
y = df["RUL"]

# 5. Unit-aware holdout: last 20% of engine IDs
units = sorted(df["unit"].unique())
cut = int(len(units) * 0.8)
train_units = units[:cut]
test_units = units[cut:]

train = df[df["unit"].isin(train_units)]
test = df[df["unit"].isin(test_units)]

X_train = train[feature_cols]
y_train = train["RUL"]
X_test = test[feature_cols]
y_test = test["RUL"]

# 6. Train model
model = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 7. Predict
pred = model.predict(X_test)

# 8. Evaluate
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# 9. Save model
joblib.dump(model, "models/rul_model.joblib")

# Example EDA Code
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print(df.shape)
print(df.isna().sum().sum())
print(df.describe())

plt.figure(figsize=(8, 4))
sns.histplot(df["RUL"], bins=40, kde=True)
plt.title("RUL Distribution")
plt.xlabel("Remaining Useful Life")
plt.show()

corr = df.select_dtypes("number").corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.show()
