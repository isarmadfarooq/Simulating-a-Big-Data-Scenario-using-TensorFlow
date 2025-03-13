# Simulating a Big Data Scenario using TensorFlow

This project generates a synthetic dataset with 1,000,000 rows and multiple features, preprocesses it, and trains a neural network using TensorFlow. The dataset is stored in **Parquet** format, and the model is saved in the **Keras format (`.keras`)**.

## 📌 Project Structure
```
D:\Online Assigenment\MSC_DataAnalytic\Simulating-a-Big-Data-Scenario-using-TensorFlow
│── train_data.parquet
│── test_data.parquet
│── best_model.keras
│── training_history.npy
│── simulate_big_data.py
│── README.md
```

## 🚀 Steps Involved
### 1️⃣ Generate Synthetic Data
- 1,000,000 rows with 10 random features.
- Stored in a Pandas DataFrame.
- Saved as a Parquet file for efficient storage.

### 2️⃣ Preprocess the Data
- Split into **80% training** and **20% testing**.
- Standardized using `StandardScaler`.

### 3️⃣ Build and Compile a Neural Network
- A simple feed-forward **autoencoder** with dropout layers.
- Compiled using the **Adam optimizer** and `mse` loss function.

### 4️⃣ Train the Model
- Uses **early stopping** to prevent overfitting.
- Best model checkpoint saved as **best_model.keras**.

### 5️⃣ Evaluate and Visualize Results
- Computes test **loss** and **mean absolute error (MAE)**.
- Plots **Training vs. Validation Loss**.
- Plots **Training vs. Validation MAE**.

## 📦 Dependencies
Install required packages:
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib pyarrow dask
```

## 🏃‍♂️ Run the Project
Execute the Python script:
```bash
python simulate_big_data.py
```

## 📊 Results
After training, visualizations will show:
- Loss curves over epochs.
- MAE curves over epochs.

## 🔹 Model Saving
The trained model is saved in the **Keras format (`best_model.keras`)** for future use:
```python
model.save("best_model.keras")
```

## 📝 Notes
- This project demonstrates how to handle **big data simulation** and **deep learning workflows**.
- **Dask** can be used for parallel processing when scaling to larger datasets.

