# Ensemble Machine Learning with Time Series Forecasting for Optimized Electric Vehicle Charging

## Overview
This repository contains the source code and report as part of my MEng Final Year Project, titled "Ensemble Machine Learning with Time Series Forecasting for Optimized Electric Vehicle Charging". The project was supervised by Dr. Mardavij Roozbehani at the Laboratory for Information and Decision Systems (LIDS) at Massachusetts Institute of Technology (MIT).

## Key Features
- **Prediction of EV User Behavior**: Uses cross-correlations between users and an ensemble machine learning algorithm to predict the two key charging parameters for EV scheduling, namely stay duration and energy consumption of users at charging stations (see Chapter 3 of the [report](https://github.com/giacovides/FYP/blob/main/MEng%20Final%20Year%20Project%20Report.pdf)).
- **Optimized EV Charging Scheduling**: Integrates the predictions into an optimal scheduling algorithm that utilizes Model Predictive Control (MPC) to minimize peak load and reduce charging costs to EV owners (see Chapter 4 of the [report](https://github.com/giacovides/FYP/blob/main/MEng%20Final%20Year%20Project%20Report.pdf)).
- **Time Series Forecasting**: Implements a hybrid time series model that combines low-rank projection with LSTM networks and noise reduction technique to enhance the accuracy of arrival time predictions for EV users and other forecasting tasks (see Chapter 5 of the [report](https://github.com/giacovides/FYP/blob/main/MEng%20Final%20Year%20Project%20Report.pdf)).

## Key Results
- Reduced prediction errors for stay duration and energy consumption by 24.2% and 24.5% respectively, compared to the previous state-of-the-art (SOTA) implementation.
- Achieved a daily average reduction of 76.3% in peak load demand across one week of testing, resulting in a coressponding decrease in charging costs to EV owners of 16.4%.
- Accomplished SOTA results in both time series forecasting tasks tested, outperforming leading time series models including Prophet, DeepAR, TFT and iTransformer.

