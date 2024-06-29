# Algorithmic-Trading
This repository contains the implementation and evaluation of various algorithmic trading strategies, developed as part of the Data-Driven Optimization course at ISCTE. The project aims to compare reinforcement learning-based trading strategies with traditional machine learning and statistical models.

## Project Overview and Structure
Algorithmic trading involves using computational models to automate the buying and selling of financial assets. This project explores the history of algorithmic trading, its objectives, data sources, and the implementation of several trading strategies.
- Data: from Yahoo Finance
- Notebook 1 -> AXP_RandomForest
- Notebook 2 -> AXP_QLearning

## Data
The data used in this project is sourced from Yahoo Finance, focusing on the American Express Company (ticker: AXP) between 2019-01-01 and 2023-09-30. 

## Technical Analysis 
After loading there is 1195 rows and 7 columns. Each record corresponds to the daily performance of the company. The records are indexed by date. The description of the variables is registered in the following table. All insights related to this section can be found on the notebook.

| Variable      | Type  | Description                                                     |
|---------------|-------|-----------------------------------------------------------------|
| Open          | Float | Opening price of the stock at the beginning of the period       |
| High          | Float | Maximum price reached by the stock during the period            |
| Low           | Float | Minimum price reached by the stock during the period            |
| Close         | Float | Closing price of the stock at the end of the period             |
| Volume        | Int   | Total number of shares traded during the period                 |
| Dividends     | Float | Dividends – Amount distributed to shareholders as company profit|
| Stock Splits  | Float | Number of times the stocks were split into smaller units        |

## Data Preparation
The data preparation involves checking missing values, creation of trading indicators Moving Average and Moving Average Convergence/Divergence (MACD), creation of new features like Daily Returns, Cumulative Returns and Signal (as target). It was also included lag features.

## Auxiliary Functions
There were created auxiliary functions for efficiency, the explanation of each are present in the notebook.

## Random Forest Model 
The Random Forest Classifier, a supervised learning algorithm, was implemented to aggregate predictions from multiple decision trees. The data was split into training (70%) and testing (30%) sets, covering the periods from 03-01-2019 to 29-04-2022 and 02-05-2022 to 29-09-2023, respectively. A random seed (random_state = 42) was used to ensure reproducibility, though slight variations in results were observed due to the algorithm's inherent randomness.

### Random Forest with Moving Average (MA)
- Model 1: Used a window size of 10, as determined by the optimize_ma_parameters function. All data variables were used as features.
- Model 2: Included only the most important variables identified by the top_features function, with "Signal_MA" and "Daily Returns" being the most significant.
- Model 3: Optimized combination of features ("Short_MA", "Long_MA", "Volume", "Dividends", "Stock Splits", and "Open") and a moving average window of 40 periods.

### Random Forest with Moving Average Convergence/Divergence (MACD)
- Model 4: Implemented with parameters short_window = 5, long_window = 10, and signal_periods = 9. All data variables were used as features.
- Model 5: Included the most relevant features identified by the top_features function.
- Model 6: Optimized combination of features ("Long_MA", "Volume", "Close", "Open", "Dividends", and "Cumulative Returns") and MACD parameters short_window = 5, long_window = 10, and signal_periods = 7.

## Results of Random Forest
Model 3 (MA) showed a cumulative return of 1.9068x with an expected mean return of 0.0021. Model 6 (MACD) achieved a cumulative return of 2.5037x with an expected mean return of 0.0035.
The MACD strategy outperformed the MA strategy in terms of investment return, though it had lower precision, indicating higher risk. Both strategies exhibited similar volatility levels.

## Reinforcement Learning 
In Reinforcement Learning there is an agent that learns by interacting with an environment in order to maximize the reward signal. Q-Learning is an RL value-based method that is used to supply information to an agent for an impending action. It is an off-policy algorithm allowing the agent to learn outside a policy by exploration.
In this case, there is a trading agent that will maximize the reward signal in this case ROI (Return of Investment). 

To be continued...



