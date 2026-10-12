# PEMS-BAY-Multivariate-Traffic-Speed-Forecasting-with-Simple_RNN-LSTM-GRU
Multivariate traffic speed forecasting on the PEMS-BAY dataset using SimpleRNN, LSTM, and GRU. Compares deep learning models for multi-step prediction of highway speeds across 325 sensors. Includes data preprocessing, sequence generation, model training, evaluation (MAE/RMSE), visualization, and model saving for deployment.
Data Source: PEMS-BAY dataset from California Department of Transportation (Caltrans).  
Download all (.py, .ipynb, .pkl, .h5) files in a folder In pems_bay.py file, go to terminal and run streamlit To run streamlit, go to terminal of the .py file and enter       py -m streamlit run "file_location\pems_bay.py" 
For example, in my device I enter: 
py -m streamlit run "L:\Softwares\VS Code\Files\RNN\PEMS-BAY\pems_bay.py" 
A localhost appears in your default web browser and finally you are about to predict the traffic speed of 3 timesteps (i.e. 15 minutes) from the past 24 timesteps (i.e. 2 hours) on the basis of sensor number in which data is given in 15 minutes intervals.

**SimpleRNN vs LSTM vs GRU | Multi-step Traffic Prediction with Streamlit UI**

A complete deep learning pipeline for multivariate traffic speed forecasting using the PEMS-BAY dataset. This project compares three recurrent neural network architectures (SimpleRNN, LSTM, and GRU), selects the best performing model, and provides an interactive Streamlit web application for real-time predictions.


##  Project Overview

| Item                    | Details                                      |
|-------------------------|----------------------------------------------|
| **Dataset**             | PEMS-BAY (325 traffic sensors)               |
| **Task**                | Multivariate Multi-step Traffic Speed Forecasting |
| **Models Compared**     | SimpleRNN, LSTM, GRU                         |
| **Best Model**          | GRU                                          |
| **Input Horizon**       | Past 24 timesteps (2 hours)                  |
| **Prediction Horizon**  | Next 3 timesteps (15 minutes)                |
| **Interface**           | Streamlit Web App                            |

---

##  Dataset

- **Name**: PEMS-BAY
- **Source**: California Performance Measurement System (PeMS)
- **Sensors**: 325
- **Time Resolution**: 5 minutes
- **Total Timesteps**: ~52,116 (January – June 2017)
- **Target Variable**: Traffic Speed (mph)

---

##  Model Architecture
All three models follow a similar structure:
Input (24 timesteps × 325 sensors)
↓
Hidden / Recurrent Layer (SimpleRNN / LSTM / GRU)
↓
Dropout
↓
Dense Layers
↓
Output (3 timesteps × 325 sensors)



### Key Training Settings
- Loss: Mean Squared Error (MSE)
- Metric: Mean Absolute Error (MAE)
- Optimizer: Adam
- Callbacks: EarlyStopping + ReduceLROnPlateau
- Normalization: StandardScaler


##  Project Structure
PEMS-BAY-Traffic-Forecasting/
│
├── PEMS_BAY.ipynb                  # Full training & evaluation notebook
├── pems_bay.py                     # Streamlit application
├── pems_bay_gru_forecast.h5        # Best trained model (GRU)
├── pems_bay_scaler.pkl             # Fitted StandardScaler
├── PEMS-BAY.csv                    # Dataset (not included due to size)
└── README.md



##  Getting Started

### 1. Installation

# Clone the repository
git clone https://github.com/your-username/PEMS-BAY-Traffic-Forecasting.git
cd PEMS-BAY-Traffic-Forecasting

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install tensorflow streamlit plotly pandas numpy scikit-learn matplotlib seaborn


The notebook performs the following steps:
Load and preprocess the PEMS-BAY dataset
Create sliding window sequences (N_PAST=24, N_FUTURE=3)
Train SimpleRNN, LSTM, and GRU models
Evaluate using MAE and RMSE
Visualize training curves and predictions
Save the best model (pems_bay_gru_forecast.h5) and scaler


Streamlit User Interface Features
Select from Real Data: Choose any time window from the original PEMS-BAY dataset using a slider
Generate Random Data: Create synthetic traffic speed data for testing
Upload CSV: Upload your own past speed data
Sensor Selection: Choose which sensors to visualize (up to 8)
Interactive Charts: Plotly-based visualization of past + predicted speeds
Prediction Summary: Table showing last known speed, predicted values, and change


Results Summary
Model               MAE (mph)                 RMSE (mph)                  Remarks
SimpleRNN           Higher                    Higher                      Baseline
LSTM                Good                      Good                        Strong long-term memory
GRU                 Best                      Best                        Selected as final model

The GRU model achieved the lowest error and was saved for deployment.


How the Prediction Works:
User provides past 24 timesteps of speed data (325 sensors).
Data is scaled using the saved StandardScaler.
The GRU model predicts the next 3 timesteps.
Predictions are inverse-transformed back to original scale (mph).
Results are displayed in tables and interactive charts.


License
This project is intended for educational and research purposes.


Acknowledgement to PEMS-BAY dataset from California Department of Transportation (Caltrans).

All three models follow a similar structure:
