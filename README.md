# Algorithmic-Trading
This repository contains the implementation and evaluation of various algorithmic trading strategies, developed as part of the Data-Driven Optimization course at ISCTE. The project aims to compare reinforcement learning-based trading strategies with traditional machine learning and statistical models.

## Project Overview and Structure
Algorithmic trading involves using computational models to automate the buying and selling of financial assets. This project explores the history of algorithmic trading, its objectives, data sources, and the implementation of several trading strategies.
- Data: from Yahoo Finance
- Notebook 1 -> AXP_RandomForest.ipynb
- Notebook 2 -> AXP_QLearning.ipynb

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

## Q learning configuration
The Q-learning is structured using object-oriented programming principles and modular functions to handle data processing, reinforcement learning logic, and visualization of trading signals. This approach also takes in human intervention by definition of exit and entry points. 
For this approach we need to define the following:
- Environment: AXP market (historical price data and indicators);
- State: Current market condition of the environment Close price;
- Action: 3 actions - Buy, Sell or Hold;
- Policy: Maximizing returns;
- Reward: Gains in the market;
- Agent: Learns to make trading decisions based on the environment state;
- Epsilon-greedy policy: Decision to choose action that has the highest expected reward;
- Q-Learning update rule: Q-value * learning rate + neagative decay term.

#### Note: Due to the many values that variable Close can take the values were further discretized by grouping into 10 bins, so that the state = 10.

### Data Preparation
The data preparation phase involves fetching historical stock data, computing technical indicators, and preparing the dataset for training the Q-learning agent. It was added in the data the following columns:
- Daily Returns: Calculated as the percentage change in closing prices.
- Cumulative Returns: Computed from daily returns to track the overall performance.
- Lagged Features: Lagged closing prices are added to capture historical trends (10 lags).

As for the technical indicators for signal generating, it was calculated the following:
- RSI (Relative Strength Index): Computes the RSI based on price changes.
- Bollinger Bands: Calculates upper and lower bands using rolling mean and standard deviation.

### Trading signals (BUY, SELL, HOLD)
The generate_signals function calculates trading signals based on technical indicators to guide buy and sell decisions. It first computes the short-term (50-day) and long-term (200-day) moving averages, the Relative Strength Index (RSI), and Bollinger Bands from the closing prices. The default signal is set to 'HOLD'. Buy signals are generated when the short-term moving average is above the long-term moving average, the RSI is below 40, and the closing price is below the lower Bollinger Band. Conversely, sell signals are generated when the short-term moving average is below the long-term moving average, the RSI is above 60, and the closing price is above the upper Bollinger Band. The function updates the signals in the dataset and creates lists of entry (buy) and exit (sell) points. Finally, it returns dictionaries mapping these points to 'BUY' and 'SELL' actions, respectively. This process helps identify optimal trading times based on market conditions.

## Defining Q-Learning Classes
- Q-Learning Agent: Implements the Q-learning algorithm with methods for updating the Q-table, choosing actions based on epsilon-greedy policy, and saving/loading the Q-table.

### First Run of the algorithm
The trading function takes 3 actions and 10 states, it also takes entry points and exit points generated throught the generate_signals function. The initial innvestment is 100€ and it was given the following parameter values: num_actions=3, alpha=0.1, gamma=0.95, epsilon=0.2. After defining the agent and implementing the trading strategy, the results were plotted and returned along with the Total Reward and Final Portfolio value, which for these parameters was:
- Total Reward: -0.17005016571561457
- Final Portfolio Value: 72.87760785694825
- Return on Investment (ROI): -27.12%

### Second run of the algorithm
Due to the not so good results, its was used a Grid Search in order to hypertune the algorithm. Hyperparameter tuning aims to optimize the performance of the Q-learning agent by finding the best combination of parameters (alpha, gamma, epsilon) that maximize Return on Investment (ROI). The best parameters returned were -> Best Params: {'alpha': 0.1, 'epsilon': 0.1, 'gamma': 0.9} returning the Best ROI: 157.98%

It was returned with the new parameters, a ROI of 429.84% with an initial investment of 100€ we have 429€, which consists of 329€ of profit. At last the Q-table that returned these values was saved as "q_table.pkl" also found in this repository.

All graphics and outputs related to this part of the project are available in the respective script (AXP_Qlearning.ipynb) 
