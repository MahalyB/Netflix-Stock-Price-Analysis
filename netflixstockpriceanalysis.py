import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

start = "2023-01-01"
end = "2024-01-01"

# download data for Netflix from yfinance
netflix_data = yf.download("NFLX", start, end)

# percent daily returns
netflix_daily_returns = netflix_data['Adj Close'].pct_change()

# Calculate Statistical Measures
mean_daily_return = netflix_daily_returns.mean()
std_dev_daily_return = netflix_daily_returns.std()

print(f'Mean Daily Return: {mean_daily_return}')
print(f'Standard Deviation of Daily Return: {std_dev_daily_return}')

# Risk Assessment
sharpe_ratio = (mean_daily_return / std_dev_daily_return) * (250 ** 0.5)
print(f'Sharpe Ratio: {sharpe_ratio}')

# Simple Moving Average with 5 day window
window_size = 5
netflix_data['SMA'] = netflix_data['Adj Close'].rolling(window=window_size).mean()

# Visualize trends with adjusted closing price and date
plt.figure(figsize=(10,6))
plt.plot(netflix_data['Adj Close'], label='Adjusted Closing Price')
plt.plot(netflix_data['SMA'], label='Moving Average')
plt.title('Netflix Stock Price Trends')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Monte Carlo simulation for risk analysis
num_simulations = 1000
num_days = 252
np.random.seed(42)

# create an empty array to store simulated price paths
simulated_price_paths = np.empty((num_simulations, num_days))

# loop through each simulation
for i in range(num_simulations):
    # generate random daily returns for each day
    daily_returns = np.random.normal(netflix_daily_returns.mean(), netflix_daily_returns.std(), num_days)
   
    # calculate simulated prices based on cumulative product of (1 + daily returns).
    simulated_prices = netflix_data['Adj Close'].iloc[-1] * (1 + np.cumprod(1 + daily_returns))

    # store simulated prices in stimulated_price_paths array
    simulated_price_paths[i] = simulated_prices

# calculate mean and median across simulations
mean_simulated_prices = np.mean(simulated_price_paths, axis=0)
median_simulated_prices = np.median(simulated_price_paths, axis=0)

# visualize Monte Carlo Simulation
plt.figure(figsize=(10,6)) 

# plots simulations
for i in range(num_simulations):
    plt.plot(simulated_price_paths[i, :])

# plot mean and median
plt.plot(mean_simulated_prices, label='Mean', linestyle='--', color='black', linewidth=2)
plt.plot(median_simulated_prices, label='Median', linestyle='--', color='red', linewidth=2)
plt.title('Monte Carlo Simulation for Netflix Stock Price')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

