# Stock Price Prediction using ARIMA and LSTM

## Overview
This repository explores and implements time series forecasting models, specifically ARIMA (Autoregressive Integrated Moving Average) and LSTM (Long Short-Term Memory) neural networks, for predicting future stock prices. The project focuses on analyzing historical stock data, preprocessing it, training these models, and evaluating their performance to generate future price predictions for multiple stocks.

## Tech Stack & Data
**Languages:**
*   Python

**Libraries:**
*   `numpy`
*   `pandas`
*   `matplotlib`
*   `pmdarima` (for `auto_arima`)
*   `scikit-learn` (for `MinMaxScaler`, `mean_squared_error`)
*   `statsmodels` (for `SARIMAX`)
*   `Keras` (for `Sequential`, `LSTM`, `Dense`, `Dropout`)
*   `tensorflow` (as Keras backend)

**Data Sources:**
CSV files located in the `data/stocks_train/` and `data/stocks_test/` directories. These include historical data for several stocks, with features such as Open, Close, High, Low, VWAP (Volume Weighted Average Price), Turnover, Volume, and derived features like Open-Close spread (OC) and High-Low spread (HL). Additional files like `new_sample_submission.csv` and `new_test.csv` are used for defining prediction targets and submission formats.

## Key Findings/Features
*   Implementation of ARIMA and SARIMAX models for univariate time series forecasting.
*   Implementation of LSTM neural networks for multivariate time series forecasting, incorporating additional stock features.
*   Robust data preprocessing pipeline, including date parsing, feature extraction, and Min-Max scaling for neural network inputs.
*   Clear methodology for splitting data into training, validation, and test sets to ensure rigorous model evaluation.
*   Comprehensive evaluation of model performance using Root Mean Squared Error (RMSE) to quantify prediction accuracy.
*   Generation of stock price predictions for multiple individual stocks, demonstrating the applicability of both classical and deep learning methods.

## Methodology/Pipeline
The project follows a standard machine learning pipeline for time series forecasting:

1.  **Data Loading:** Historical stock data for various stocks is loaded from individual CSV files.
2.  **Feature Engineering & Preprocessing:**
    *   Dates are parsed and set as index for time series analysis.
    *   Relevant features (e.g., 'Close' prices, 'Open', 'High', 'Low', 'Volume', 'VWAP', 'Turnover', 'OC', 'HL') are extracted.
    *   For LSTM models, numerical features are scaled using `MinMaxScaler` to normalize input ranges.
3.  **Data Splitting:** The dataset for each stock is meticulously divided into training, validation, and test sets. This sequential splitting ensures that models are evaluated on unseen future data.
4.  **Model Selection & Training:**
    *   **ARIMA/SARIMAX:** The `autoarima.ipynb` notebook employs `pmdarima`'s `auto_arima` to identify optimal parameters for ARIMA models. SARIMAX models are then iteratively trained on individual stock time series.
    *   **LSTM:** The `LSTM.ipynb` notebook constructs sequential LSTM neural networks. These models are trained using historical sequences of preprocessed features to learn complex temporal dependencies and predict future closing prices.
5.  **Prediction & Evaluation:** Trained models are used to generate predictions on the validation and test sets. Prediction accuracy is quantified using the Root Mean Squared Error (RMSE) metric.
6.  **Submission Generation:** Final predictions for the test data are compiled into a structured submission file.

## Project Structure
```
StockPricePrediction/
├───autoarima.ipynb         # Jupyter notebook for ARIMA/SARIMAX model implementation
├───LSTM.ipynb              # Jupyter notebook for LSTM model implementation
├───data/                   # Directory containing all raw and processed data
│   ├───new_sample_submission.csv  # Sample submission format
│   ├───new_test.csv               # General test data (metadata/dates)
│   ├───new_train.csv              # General training data (metadata/dates)
│   ├───stocks_test/               # Folder containing individual stock data files for testing
│   │   └───stock_1.csv
│   │   └───stock_2.csv
│   │   └───stock_3.csv
│   │   └───stock_4.csv
│   │   └───stock_5.csv
│   └───stocks_train/              # Folder containing individual stock data files for training
│       └───stock_1.csv
│       └───stock_2.csv
│       └───stock_3.csv
│       └───stock_4.csv
│       └───stock_5.csv
└───README.md               # Project README file
```

## Reproduction Guide
To reproduce the analysis and predictions presented in this repository, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/P1YU5H-50N1/StockPricePrediction
    cd StockPricePrediction
    ```
2.  **Install dependencies:** It is highly recommended to use a virtual environment. The following Python libraries are required:
    ```bash
    pip install numpy pandas matplotlib pmdarima scikit-learn statsmodels keras tensorflow
    ```
    *(Note: TensorFlow serves as the backend for Keras. Ensure it's installed correctly for Keras to function.)*
3.  **Run the notebooks:**
    *   Open and execute the cells in `autoarima.ipynb` sequentially to train the ARIMA/SARIMAX models and generate their respective predictions.
    *   Open and execute the cells in `LSTM.ipynb` sequentially to train the LSTM models and generate their respective predictions.
    *   Ensure that the data files in the `data/` directory are correctly structured as expected by the notebooks.

## Additional Info
This project provides a comparative study of two distinct yet powerful time series forecasting methodologies. The notebooks are designed to be self-contained and offer detailed steps for understanding and implementing each model. The primary evaluation metric, RMSE, directly indicates the average magnitude of the prediction errors, offering a clear measure of model accuracy. These models were prepared as a part of Learn ML 2021 challenge and these solutions obtained were ranked 181 out of 812 participants. 
