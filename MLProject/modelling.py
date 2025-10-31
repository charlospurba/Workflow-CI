import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# 1. Set URI MLflow lokal
mlflow.set_tracking_uri("file:./mlruns")

# 2. Muat dataset
df = pd.read_csv("heart_preprocessed.csv")

# 3. Pisahkan fitur dan target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Aktifkan autolog
mlflow.autolog()

# 6. Jalankan eksperimen MLflow
with mlflow.start_run() as run:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model: Logistic Regression")
    print(f"Accuracy: {accuracy}")

    # Simpan model secara eksplisit ke artifacts
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"\nModel disimpan ke: mlruns/0/{run.info.run_id}/artifacts/model")

print("\n✅ Eksperimen selesai.")
