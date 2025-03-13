import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore

# Define Paths
save_path = r"D:\Online Assigenment\MSC_DataAnalytic\Simulating-a-Big-Data-Scenario-using-TensorFlow"
os.makedirs(save_path, exist_ok=True)

#  Generate Synthetic Data
num_rows = 1_000_000
num_features = 10
data = np.random.rand(num_rows, num_features)
columns = [f"feature_{i}" for i in range(num_features)]
df = pd.DataFrame(data, columns=columns)

#  Split Data (80% Train, 20% Test)
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

#  Normalize Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#  Save Data to Parquet
train_file = os.path.join(save_path, "train_data.parquet")
test_file = os.path.join(save_path, "test_data.parquet")
pd.DataFrame(X_train, columns=columns).to_parquet(train_file)
pd.DataFrame(X_test, columns=columns).to_parquet(test_file)

#  Define Neural Network
model = Sequential([
    Dense(64, activation="relu", input_shape=(num_features,)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(num_features, activation="linear")  # Output same as input (autoencoder approach)
])

#  Compile Model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

#  Define Callbacks
model_path = os.path.join(save_path, "best_model.keras")  # New Keras format
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(filepath=model_path, save_best_only=True)
]

#  Train the Model
history = model.fit(
    X_train, X_train,
    epochs=50,
    batch_size=1024,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

#  Save Training History
history_dict = history.history
np.save(os.path.join(save_path, "training_history.npy"), history_dict)

#  Evaluate Model
test_loss, test_mae = model.evaluate(X_test, X_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

#  Plot Training & Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(history_dict["loss"], label="Train Loss")
plt.plot(history_dict["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid()
plt.show()

#  Plot Training & Validation MAE
plt.figure(figsize=(10, 5))
plt.plot(history_dict["mae"], label="Train MAE")
plt.plot(history_dict["val_mae"], label="Validation MAE")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.title("Training & Validation MAE")
plt.legend()
plt.grid()
plt.show()
