import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn import metrics
from utils import *
from math import floor
from scipy import stats
import random
from datetime import datetime, timedelta
import bitfinex
import time

import yfinance as yf

num_repeats = 2
fromdate_randombase = '2015-01-01'  # 1 day, 15 m 
todate_randombase = '2023-10-13'   # 1 day, 15 m
# fromdate_randombase = '2016-05-07'  # 1m
# todate_randombase = '2023-03-01'   # 1m
#fromdate_randombase = '2019-05-07'  # 1 hour
#todate_randombase = '2022-03-01'   # 1 hour
timeframe = 1
skip_first_steps = 4


interval = '1d'
# stock = 'ETH-USD'
stock = 'BTC-USD'
benchmarkstock = 'BTIC.SW'
# benchmarkstock = 'ETHE.SW'

# fromdate = '2022-10-13'
fromdate = '2023-01-01'
# fromdate = '2023-10-06'
# fromdate = '2023-10-12'
todate = '2023-10-13'
todate_benchmark = '2023-10-13'

# Set thresholds for buying and selling
buy_threshold = 0.005 
sell_threshold = -0.005 

profit_percentages = []
sharpe_ratios = []
t_statistics = []
profits = []
buy_and_hold_profits = []
p_values = []
trade_counts = []
dates_choosen = []

##1 minute
#dfgg = pd.read_csv('C:/Users/Master/Documents/Sources/crypto_comparison/btcusd.csv')
#print("start",dfgg.head())

# #1 hour
# dfgg = pd.read_csv('C:/Users/Master/Documents/Sources/crypto_comparison/btcusd_hourly.csv')
# # print(dfgg.head())

# #1 day
# dfgg = pd.read_csv('C:/Users/Master/Documents/Sources/crypto_comparison/btcusd_daily.csv')
# print(dfgg.head())

firstbalance = 100000 
balance = firstbalance

def get_historical_data(symbol):
    if symbol != benchmarkstock:
        btic = yf.download(symbol, start=fromdate, end=todate, interval=interval)
    else:
        # btic = yf.download(stock, start='2014-09-17', end='2022-10-12') #BTC
        btic = yf.download(stock, start='2017-11-09', end='2022-10-12') #ETH
    # btic = yf.Ticker(symbol)
    # btic = btic.history(period='1d', interval="1m")
    df = btic.rename(columns = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adj close', 'Volume': 'volume'})
    df = df.drop(['adj close', 'volume', 'high', 'low'], axis=1)
    df.index.names = ['trade_date']
    # display(btic)
    return df

def cut_date_range(long_start_unix, long_end_unix, short_start_unix, short_end_unix):
    long_start = datetime.utcfromtimestamp(long_start_unix)
    long_end = datetime.utcfromtimestamp(long_end_unix)
    short_start = datetime.utcfromtimestamp(short_start_unix)
    short_end = datetime.utcfromtimestamp(short_end_unix)

    if short_start <= long_start or short_end >= long_end:
        return None  # Invalid input, short_range should be within long_range

    range1 = (long_start, short_start)
    range2 = (short_end, long_end)

    # Convert date ranges back to Unix timestamps
    range1_unix = (int(range1[0].timestamp()*1000), int(range1[1].timestamp()*1000))
    range2_unix = (int(range2[0].timestamp()*1000), int(range2[1].timestamp()*1000))

    return range1_unix, range2_unix

# data = pd.read_csv('./601988.SH.csv')
# test_set2 = data.loc[3501:, :] 
# data.index = pd.to_datetime(data['trade_date'], format='%Y%m%d') 
# data = data.drop(['ts_code', 'trade_date'], axis=1)
# data = pd.DataFrame(data, dtype=np.float64)
# training_set = data.loc['2007-01-04':'2021-06-21', :]  # 3501
# test_set = data.loc['2021-06-22':, :]  # 180
# print(data.head(10))
# print(training_set.head(10))
# print(test_set.head(10))

date_ranges = [
['2021-01-30', '2022-01-30'], ['2018-12-13', '2019-12-13'], ['2018-03-31', '2019-03-31'], ['2016-10-22', '2017-10-22'], ['2016-06-02', '2017-06-02'], ['2018-12-07', '2019-12-07'], ['2021-03-15', '2022-03-15'], ['2017-06-01', '2018-06-01'], ['2019-09-01', '2020-08-31'], ['2021-08-04', '2022-08-04']
]

# for _ in range(num_repeats):
#     fromdate_datetime = datetime.strptime(fromdate_randombase, "%Y-%m-%d")
#     todate_datetime = datetime.strptime(todate_randombase, "%Y-%m-%d")

#     # Calculate the maximum possible fromdate to ensure a 7-day range
#     max_fromdate = todate_datetime - timedelta(days=timeframe)

#     # Generate a random fromdate within the 7-day range
#     fromdate = random.choice(pd.date_range(fromdate_datetime, max_fromdate).strftime('%Y-%m-%d'))

#     # Calculate the corresponding todate
#     # todate = (datetime.strptime(fromdate, "%Y-%m-%d") + timedelta(days=timeframe)).strftime('%Y-%m-%d')

for date_range in date_ranges:
    fromdate = date_range[0]
    todate = date_range[1]

    # Calculate the corresponding todate
    #fromdate_datetime = datetime.strptime(fromdate, "%Y-%m-%d")
    fromdate_datetime = datetime.strptime(fromdate, "%Y-%m-%d %H:%M:%S") # For hours when dates are known
    todate_datetime = fromdate_datetime + timedelta(days=timeframe)

    # Use timestamp() method to convert datetime objects to Unix timestamps
    fromdate = int(fromdate_datetime.timestamp() * 1000)
    todate = int(todate_datetime.timestamp() * 1000)

    fromdate_randombase_unix = int(datetime.strptime(fromdate_randombase, "%Y-%m-%d").timestamp() * 1000)
    todate_randombase_unix = int(datetime.strptime(todate_randombase, "%Y-%m-%d").timestamp() * 1000)
    # print(fromdate_randombase_unix, todate_randombase_unix)
    print("Sample: ",fromdate_datetime, todate_datetime)

    timeframe_cut_result = cut_date_range(fromdate_randombase_unix/1000, todate_randombase_unix/1000, fromdate/1000, todate/1000)
    googl_training0 = dfgg[(dfgg['datetime'] >= timeframe_cut_result[0][0]) & (dfgg['datetime'] <= timeframe_cut_result[0][1])].copy()
    googl_training1 = dfgg[(dfgg['datetime'] >= timeframe_cut_result[1][0]) & (dfgg['datetime'] <= timeframe_cut_result[1][1])].copy()
    googl_training = pd.concat([googl_training0, googl_training1], axis=0)
    googl_testing = dfgg[(dfgg['datetime'] >= fromdate) & (dfgg['datetime'] <= todate)].copy()
    
    googl_testing['datetime'] = pd.to_datetime(googl_testing['datetime'], unit='ms')
    # googl_testing['datetime'] = googl_testing['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
    googl_training['datetime'] = pd.to_datetime(googl_training['datetime'], unit='ms')
    # googl_training['datetime'] = googl_training['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
    # print(googl_training.head(), googl_testing.head())

    # googl_testing['datetime'] = pd.to_datetime(dfgg['datetime'], unit='ms')
    googl_testing.set_index(googl_testing['datetime'], inplace=True)
    # googl_training['datetime'] = pd.to_datetime(dfgg['datetime'], unit='ms')
    googl_training.set_index(googl_training['datetime'], inplace=True)
    #-----------------------
    # data = get_historical_data(benchmarkstock)
    data = googl_training
    data['dummy1'] = 0
    data['dummy2'] = 0
    data['dummy3'] = 0
    data['change'] = data['close'].diff()
    first95percent = floor(len(data.index)*.95)
    print(first95percent)
    # datatesting = get_historical_data(stock)
    datatesting = googl_testing
    test_set2 = datatesting
    training_set = data
    test_set = datatesting
    # print(len(data.tail(floor((len(data.index)*.05)))))
    # print(data.head(5))
    print(training_set.head(5))
    print(test_set.head(5))
    #------------------------
    #WAZNE
    # plt.figure(figsize=(10, 6))
    # plt.plot(training_set['close'], label='training_set')
    # plt.plot(test_set['close'], label='test_set')
    # plt.title('Close price')
    # plt.xlabel('time', fontsize=12, verticalalignment='top')
    # plt.ylabel('close', fontsize=14, horizontalalignment='center')
    # plt.legend()
    # plt.show()

    temp = np.array(training_set['close'])
    training_set['diff_1'] = training_set['close'].diff(1)
    training_set['diff_2'] = training_set['diff_1'].diff(1)

    temp1 = np.diff(training_set['close'], n=1)

    # white noise test
    training_data1 = training_set['close'].diff(1)
    # training_data1_nona = training_data1.dropna()
    temp2 = np.diff(training_set['close'], n=1)
    # print(acorr_ljungbox(training_data1_nona, lags=2, boxpierce=True, return_df=True))
    print(acorr_ljungbox(temp2, lags=2, boxpierce=True))
    # p-value=1.53291527e-08, non-white noise time-seriess

    # acf_pacf_plot(training_set['close'],acf_lags=160)

    price = list(temp2)
    data2 = {
        'trade_date': training_set['diff_1'].index[1:], 
        'close': price
    }

    df = pd.DataFrame(data2)
    # print('mordo',training_set.head())
    # print('siema',df.head())
    # df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y-%m-%d %H:%M:%S+00:00')

    training_data_diff = df.set_index(['trade_date'], drop=True)
    # print('&', training_data_diff)

    # acf_pacf_plot(training_data_diff) #tutaj

    # order=(p,d,q)
    model = sm.tsa.ARIMA(endog=training_set['close'], order=(2, 1, 0)).fit()
    #print(model.summary())

    history = [x for x in training_set['close']]
    # print('history', type(history), history)
    predictions = list()
    # print('test_set.shape', test_set.shape[0])

    # Initialize lists to store buy and sell data
    buy_signals = []
    sell_signals = []

    # Initialize variables for position and holding status
    position = 0  # 0: no position, 1: long position
    num_shares = 0
    holding = False

    # Initialize an empty list to store the balance values
    balance_values = []

    # Initialize an empty list to store the daily returns
    returns = []

    # Trading strategy
    time_step = 0
    for t in range(test_set.shape[0]):
        time_step += 1  # Increment the time step

        model1 = sm.tsa.ARIMA(history, order=(2, 1, 0))
        model_fit = model1.fit()
        yhat = model_fit.forecast()
        yhat = np.float(yhat[0])
        predictions.append(yhat)
        obs = test_set2.iloc[t, 1]
        obs = np.float(obs)
        history.append(obs)

        # Check if the price change exceeds the thresholds
        price_change = (obs - history[-2]) / history[-2]
        if price_change > buy_threshold and not holding and time_step >= skip_first_steps:
            # Buy as many shares as possible
            num_shares = floor(balance / obs)
            balance -= num_shares * obs
            position = 1
            holding = True
            print("Buying ", num_shares, " shares at $", obs)
            # Append the date and price to the buy_signals list
            buy_signals.append((test_set.index[t], obs))
        elif price_change < sell_threshold and time_step >= skip_first_steps:
            if holding:
                # Sell all shares
                balance += num_shares * obs
                position = 0
                holding = False
                num_shares = 0
                print("Selling all shares at $", obs)
                # Append the date and price to the sell_signals list
                sell_signals.append((test_set.index[t], obs))
            else:
                # Sell without owning a share
                balance += balance * abs(price_change)
                print("Selling without owning a share at $", obs)
                # Append the date and price to the sell_signals list
                sell_signals.append((test_set.index[t], obs))

        # Update the balance value
        balance_values.append(balance)

        # Calculate the daily return
        if len(balance_values) > 1:
            daily_return = (balance_values[-1] - balance_values[-2]) / balance_values[-2]
            returns.append(daily_return)

    # print('predictions', predictions)

    # If we own the stock at the end of the test period, sell it
    if holding:
        balance += num_shares * obs
        print("Selling ", num_shares, " shares at $", obs)

    print("Final balance: $", balance)

    # Calculate the Sharpe ratio
    returns = np.array(returns)
    sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns) #where 252 represents the number of trading days in a year.
    print("Sharpe Ratio:", sharpe_ratio)

    #Buy and hold
    buy_price0 = datatesting['close'].iloc[0]
    num_shares_hold0 = floor(firstbalance / buy_price0)
    balance_after_buy0 = firstbalance - num_shares_hold0 * buy_price0
    # Hold until the end of the test period
    sell_price0 = datatesting['close'].iloc[-1]
    final_balance_hold0 = num_shares_hold0 * sell_price0 + balance_after_buy0
    print("Final balance of buy and hold strategy: $", final_balance_hold0)

    #WAZNE
    # # Plot the balance values
    # plt.figure(figsize=(10, 6))
    # plt.plot(balance_values, label='Balance')
    # plt.title('Balance')
    # plt.xlabel('Time', fontsize=12, verticalalignment='top')
    # plt.ylabel('Balance', fontsize=14, horizontalalignment='center')
    # plt.legend()
    # plt.show()

    # Convert the buy and sell signals to a DataFrame for easier plotting
    buy_signals = pd.DataFrame(buy_signals, columns=['Date', 'Price']).set_index('Date')
    sell_signals = pd.DataFrame(sell_signals, columns=['Date', 'Price']).set_index('Date')

    predictions1 = {
        'trade_date': test_set.index[:],
        'close': predictions
    }
    predictions1 = pd.DataFrame(predictions1)
    predictions1 = predictions1.set_index(['trade_date'], drop=True)
    #WAZNE
    # predictions1.to_csv('./ARIMA.csv')
    # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.plot(test_set['close'], label='Stock Price')
    # plt.plot(predictions1, label='Predicted Stock Price')
    # plt.scatter(buy_signals.index, buy_signals['Price'], color='green', label='Buy Signal', marker='^', alpha=1)
    # plt.scatter(sell_signals.index, sell_signals['Price'], color='red', label='Sell Signal', marker='v', alpha=1)
    # plt.title('ARIMA: Stock Price Prediction with Buy/Sell Signals')
    # plt.xlabel('Time', fontsize=12, verticalalignment='top')
    # plt.ylabel('Close', fontsize=14, horizontalalignment='center')
    # plt.legend()
    # plt.show()

    model2 = sm.tsa.ARIMA(endog=data['close'], order=(2, 1, 0)).fit()
    residuals = pd.DataFrame(model2.resid)
    # tutaj
    #  fig, ax = plt.subplots(1, 2)
    # residuals.plot(title="Residuals", ax=ax[0])
    # residuals.plot(kind='kde', title='Density', ax=ax[1])
    # plt.show()
    residuals.to_csv('./ARIMA_residuals1.csv')
    evaluation_metric(test_set['close'],predictions)
    adf_test(temp)
    adf_test(temp1)

    #tutaj
    # predictions_ARIMA_diff = pd.Series(model.fittedvalues, copy=True)
    # predictions_ARIMA_diff = predictions_ARIMA_diff[first95percent:]
    # print('#', predictions_ARIMA_diff)
    # plt.figure(figsize=(10, 6))
    # plt.plot(training_data_diff, label="diff_1")
    # plt.plot(predictions_ARIMA_diff, label="prediction_diff_1")
    # plt.xlabel('time', fontsize=12, verticalalignment='top')
    # plt.ylabel('diff_1', fontsize=14, horizontalalignment='center')
    # plt.title('DiffFit')
    # plt.legend()
    # plt.show()

    # Initialize variables to count the number of trades
    num_buy_trades = len(buy_signals)
    num_sell_trades = len(sell_signals)

    print("Number of Trades:", num_buy_trades+num_sell_trades)

    # Calculate the daily returns of the buy and hold strategy
    buy_and_hold_returns = (datatesting['close'] - buy_price0) / buy_price0
    buy_and_hold_returns = buy_and_hold_returns[1:]  # Remove the first row since it's NaN


    
    # Calculate the profit percentage and store it in the list
    # profit_percentages.append(profit_percentage)
    profits.append(balance)
    buy_and_hold_profits.append(sum(buy_and_hold_returns))
    
    # Perform a t-test on the daily returns
    # t_stat, p_value = stats.ttest_ind(returns, buy_and_hold_returns)
    t_stat, p_value = stats.ttest_1samp(profits, popmean=0)
    print("T-Test Results:")
    print("T-Statistic:", t_stat)
    print("P-Value:", p_value)

    # Calculate the Sharpe Ratio and store it in the list
    sharpe_ratios.append(sharpe_ratio)
    
    # Calculate the t-statistic and store it in the list
    t_statistics.append(t_stat)
    p_values.append(p_value)
    trade_counts.append(num_buy_trades+num_sell_trades)

    fromdatedt = datetime.utcfromtimestamp(fromdate/1000)
    todatedt = datetime.utcfromtimestamp(todate/1000)
    dates_choosen.append([fromdatedt.strftime("%Y-%m-%d %H:%M:%S"), todatedt.strftime("%Y-%m-%d %H:%M:%S")])

    # Calculate descriptive statistics for your data
    mean_close = np.mean(datatesting['close'])
    median_close = np.median(datatesting['close'])
    std_close = np.std(datatesting['close'])
    min_close = np.min(datatesting['close'])
    max_close = np.max(datatesting['close'])
    q1 = np.percentile(datatesting['close'], 25)
    q3 = np.percentile(datatesting['close'], 75)

    # Print the descriptive statistics
    print("Descriptive Statistics for 'close' column:")
    print(f"Mean: {mean_close}")
    print(f"Median: {median_close}")
    print(f"Standard Deviation: {std_close}")
    print(f"Minimum: {min_close}")
    print(f"Maximum: {max_close}")
    print(f"1st Quartile (Q1): {q1}")
    print(f"3rd Quartile (Q3): {q3}")

# Calculate the mean values from all repeats
mean_profit = np.mean(profits)
mean_buy_and_hold_profits = np.mean(buy_and_hold_profits)
mean_ttest_fromzero, mean_pvalue_fromzero = stats.ttest_1samp(profits, popmean=0)
mean_profit_percentage = np.mean(profit_percentages)
mean_sharpe_ratio = np.mean(sharpe_ratios)
mean_trade_count = np.mean(trade_counts)
mean_t_statistic = np.mean(t_statistics)
mean_p_value = np.mean(p_values)

# Print the mean values
print("--------------------------------")
print('Choosen dates: ', dates_choosen)
print("Profits: ", profits)
print("Buy and hold profits: ", buy_and_hold_profits)
print('Dates count: ', len(dates_choosen))
print("Mean Profit:", mean_profit)
print("Mean Profit Buy and hold:", mean_buy_and_hold_profits)
print("Mean Sharpe Ratio:", mean_sharpe_ratio)
print("Mean Trade Count:", mean_trade_count)
print("Mean T-statistic:", mean_t_statistic)
print("Mean P-value:", mean_p_value)
print("Mean T-statistic fromzero:", mean_ttest_fromzero)
print("Mean P-value fromzero:", mean_pvalue_fromzero)
