# Enabling Long Path Support in Windows

# Overview

  If you are encountering file path length issues while installing TensorFlow or working with long file names, you need to enable long path support in Windows. This guide provides step-by-step instructions to do so using the Windows Registry Editor and PowerShell.
  
  Enable Long Paths via Registry Editor
  
  1. Open Registry Editor
  
  Press Win + R, type regedit, and press Enter.
  
  Click "Yes" if prompted by User Account Control (UAC).
  
  2. Navigate to the Correct Path
  
  Go to:
  
  HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
  
  3. Modify or Create the LongPathsEnabled Key
  
  Look for a DWORD (32-bit) value named LongPathsEnabled.
  
  If it does not exist:
  
  Right-click FileSystem, select New > DWORD (32-bit) Value.
  
  Name it LongPathsEnabled.
  
  Double-click LongPathsEnabled, set the Value data to 1, and click OK.
  
  4. Restart Your Computer
  
  The changes will only take effect after a system restart.
  
  Enable Long Paths via PowerShell
  
  Alternatively, you can enable long path support using PowerShell.
  
  1. Run PowerShell as Administrator
  
  Press Win + S, type PowerShell.
  
  Right-click PowerShell and select "Run as administrator".
  
  Click "Yes" if prompted by UAC.
  
  2. Execute the Following Command
  
  New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
  
  3. Restart Your Computer
  
  Restart your PC for changes to take effect.
  
  Troubleshooting
  
  If you receive a "Requested registry access is not allowed" error, ensure you are running PowerShell as an administrator.
  
  If the FileSystem key is missing in Registry Editor, manually create it under HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control.







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

