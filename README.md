# Raspberry Price Prediction using Time Series Models

## Overview

This project focuses on the **price prediction of raspberries** using various **time series models**. Time series analysis helps us comprehend systemic patterns or trends over time and provides insights into future trends through forecasting. In this project, several time series models are implemented and compared, including both traditional and deep learning methods.

The models used for forecasting include:
- **SARIMA** (Seasonal Autoregressive Integrated Moving Average)
- **MLP** (Multilayer Perceptron)
- **CNN** (Convolutional Neural Networks)
- **LSTM** (Long Short-Term Memory Networks)

The objective of this project is to evaluate the performance of these models on raspberry price data and identify the most effective model for price prediction.

## Data Source

The dataset used in this project is the **weekly price data of raspberries** from **Santa Maria, California**. The data spans from **November 2010 to February 2022**. The raw dataset is subjected to multiple preprocessing steps to make it suitable for time series analysis.

### Preprocessing Steps:
- **Data Cleaning**: Identification and correction of errors, removal of columns with no variance, and duplicate rows.
- **Handling Missing Values**: Missing data points are marked and filled using the average value of the data.
- **Outlier Detection**: Identification of normal data and outliers.
- **Feature Selection**: Selecting the most relevant input variables for analysis.
- **Dimensionality Reduction**: Reducing the complexity of the dataset.

The preprocessed dataset is then used for building the models.

## Models Used

### 1. **SARIMA (Seasonal Autoregressive Integrated Moving Average)**
   - **SARIMA** is a widely used statistical model for time series forecasting that captures both seasonal and non-seasonal components of the data.
   
### 2. **Multilayer Perceptron (MLP)**
   - An MLP is a type of feedforward artificial neural network. It can automatically learn temporal dependencies in time series data but is generally more suited for simpler problems compared to other deep learning models.
   
### 3. **Convolutional Neural Networks (CNN)**
   - **CNNs**, although primarily used for image data, can also be adapted for time series data by identifying important temporal patterns, making them effective for short-term forecasts.
   
### 4. **Long Short-Term Memory (LSTM) Networks**
   - **LSTMs** are a special kind of recurrent neural network (RNN) capable of learning long-term dependencies, making them ideal for time series forecasting, especially when dealing with sequences.

## Methodology

- **Data Preprocessing**: All models are trained on the preprocessed raspberry price dataset.
- **Model Training**: Each model is trained using the weekly price data, and the parameters are fine-tuned to achieve optimal performance.
- **Evaluation**: The models are evaluated based on performance metrics such as **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)**.

## Results

After running the models, the results will be presented in terms of evaluation metrics. You can compare the performance of each model to determine which one offers the best price prediction accuracy.

- **SARIMA Performance**: 
10% Mean absolute percent error, 34 cents price variation from the actual and predicted value as per RMSE and 0.981 R2 value.
- **MLP Performance**: 
Performance statistics of the model shows 7.7% MAE, 27 cents price variation from actual and predicted values per RMSE and 0.985 R2 value. 
- **CNN Performance**: 
RMSE shows that there is 36 cents price variation, 12% MAE and 0.88 R2 value.
- **LSTM Performance**: 
RMSE shows that there is only 26 cents price variation from the actual and predicted value of LSTM model and 0.986 R2 value.

## Conclusion

This project evaluates the performance of different time series forecasting models on the raspberry price dataset. 

Analysis of the time series forecasting on the 12-year Raspberry dataset extracted from November 2010 to February 2022 using SARIMA and multiple deep learning models like MLP, CNN, and LSTM shows the performance statistics below, it is evident that the **LSTM model** performed better compared to all other analyzed models for the provided dataset, whereas the **CNN model** performed the least. In terms of computational cost, the **SARIMA model** is the most efficient amongst all the models.

| Model  | RMSE (cents) | MSE (%) | R²   |
|--------|--------------|---------|------|
| SARIMA | 34           | 10      | 0.981|
| MLP    | 27           | 7.7     | 0.985|
| CNN    | 36           | 12      | 0.88 |
| LSTM   | 26           | 7.0     | 0.986|
