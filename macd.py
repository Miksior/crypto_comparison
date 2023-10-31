import requests
import pandas as pd
import numpy as np
from math import floor
from termcolor import colored as cl
import matplotlib.pyplot as plt
import yfinance as yf
from IPython.display import display
from scipy import stats
import random
from datetime import datetime, timedelta
import bitfinex
import time

plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('fivethirtyeight')

num_repeats = 1
fromdate_randombase = '2015-01-01'  # 1 day, 15 m 
todate_randombase = '2023-10-13'   # 1 day, 15 m
# fromdate_randombase = '2016-05-07'  # 1m
# todate_randombase = '2023-03-01'   # 1m
#fromdate_randombase = '2019-05-07'  # 1 hour
#todate_randombase = '2022-03-01'   # 1 hour
timeframe = 7


interval = '1m'
stock = 'BTC-USD'
# stock = 'ETH-USD'
benchmarkstock = 'BTIC.SW'
# benchmarkstock = 'ETHE.SW'
# fromdate = '2022-10-13'
# fromdate = '2023-01-01'
# fromdate = '2023-10-06'
fromdate = '2023-10-12'
todate = '2023-10-13'
todate_benchmark = '2023-10-13'

fromdate = '2023-10-21'
todate = '2023-10-28'

profit_percentages = []
sharpe_ratios = []
t_statistics = []
profits = []
buy_and_hold_profits = []
p_values = []
trade_counts = []
dates_choosen = []

algo = 'MACD'

##1 minute
#dfgg = pd.read_csv('C:/Users/Master/Documents/Sources/crypto_comparison/btcusd.csv')
#print("start",dfgg.head())

# #1 hour
# dfgg = pd.read_csv('C:/Users/Master/Documents/Sources/crypto_comparison/btcusd_hourly.csv')
# # print(dfgg.head())

# #1 day
# dfgg = pd.read_csv('C:/Users/Master/Documents/Sources/crypto_comparison/btcusd_daily.csv')
# print(dfgg.head())


# dfgg['datetime'] = pd.to_datetime(dfgg['datetime'], unit='ms')
# dfgg['datetime'] = dfgg['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S+00:00')

def get_historical_data_yahoo(symbol, start_date = None):
    if symbol != benchmarkstock:
        btic = yf.download(symbol, start=fromdate, end=todate, interval=interval)
    else:
        btic = yf.download(symbol, start=fromdate, end=todate_benchmark)
    # btic = yf.Ticker(symbol)
    # btic = btic.history(period='1d', interval="1m")
    df = btic.rename(columns = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj close': 'adj close', 'Volume': 'volume'})
    # if start_date:
    #     df = df[df.index >= start_date]
    display(btic)
    return df
# print(get_historical_data_yahoo(stock).head())
# print('--------------------------------')

# def get_macd(price, slow, fast, smooth):
#     exp1 = price.ewm(span = fast, adjust = False).mean()
#     exp2 = price.ewm(span = slow, adjust = False).mean()
#     macd = pd.DataFrame(exp1 - exp2).rename(columns = {'close':'macd'})
#     signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'})
#     hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
#     frames =  [macd, signal, hist]
#     df = pd.concat(frames, join = 'inner', axis = 1)
#     return df
def get_macd(price, slow, fast, smooth):
    exp1 = price.ewm(span=fast, adjust=False).mean()
    exp2 = price.ewm(span=slow, adjust=False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns={'close': 'macd'})
    signal = pd.DataFrame(macd.ewm(span=smooth, adjust=False).mean()).rename(columns={'macd': 'signal'})
    hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns={0: 'hist'})

    frames = [macd.reset_index(drop=True), signal.reset_index(drop=True), hist.reset_index(drop=True)]
    df = pd.concat(frames, axis=1)

    return df


def plot_macd(prices, macd, signal, hist):
    ax1 = plt.subplot2grid((8,1), (0,0), rowspan = 5, colspan = 1)
    ax2 = plt.subplot2grid((8,1), (5,0), rowspan = 3, colspan = 1)

    ax1.plot(prices)
    ax2.plot(macd, color = 'grey', linewidth = 1.5, label = 'MACD')
    ax2.plot(signal, color = 'skyblue', linewidth = 1.5, label = 'SIGNAL')

    for i in range(len(prices)):
        if str(hist[i])[0] == '-':
            ax2.bar(prices.index[i], hist[i], color = '#ef5350')
        else:
            ax2.bar(prices.index[i], hist[i], color = '#26a69a')

    plt.legend(loc = 'lower right')

def implement_macd_strategy(prices, data):    
    buy_price = []
    sell_price = []
    macd_signal = []
    signal = 0
    trade_count = 0  # Initialize the trade count variable

    for i in range(len(data)):
        if data['macd'][i] > data['signal'][i]:
            if signal != 1:
                buy_price.append(prices.iloc[i])
                sell_price.append(np.nan)
                signal = 1
                macd_signal.append(signal)
                trade_count += 1  # Increment trade count for buy signal
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)
        elif data['macd'][i] < data['signal'][i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices.iloc[i])
                signal = -1
                macd_signal.append(signal)
                trade_count += 1  # Increment trade count for sell signal
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            macd_signal.append(0)
            
    return buy_price, sell_price, macd_signal, trade_count  # Return trade count

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
    # fromdate_datetime = datetime.strptime(fromdate, "%Y-%m-%d")
    fromdate_datetime = datetime.strptime(fromdate, "%Y-%m-%d %H:%M:%S") # For hours when dates are known
    todate_datetime = fromdate_datetime + timedelta(days=timeframe)

    # Use timestamp() method to convert datetime objects to Unix timestamps
    fromdate = int(fromdate_datetime.timestamp() * 1000)
    todate = int(todate_datetime.timestamp() * 1000)

    # googl = get_historical_data(stock, fromdate)
    # fromdate = datetime.strptime(fromdate + " 00:00:00", "%Y-%m-%d %H:%M:%S")
    # todate = datetime.strptime(todate + " 00:00:00", "%Y-%m-%d %H:%M:%S")
    print(fromdate_datetime, todate_datetime)
    # fromdate = time.mktime(fromdate)
    # todate = time.mktime(todate)
    
    # googl = get_historical_data(stock, fromdate, todate, interval, 1000, 60000000)
    # googl = get_historical_data(stock, fromdate, todate, interval, 1000, 60000000)
    # print(dfgg.head())
    # googl = get_historical_data_yahoo(stock, fromdate)
    googl = dfgg[(dfgg['datetime'] >= fromdate) & (dfgg['datetime'] <= todate)].copy()
    googl['datetime'] = pd.to_datetime(googl['datetime'], unit='ms')
    googl['datetime'] = googl['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
    # googl.set_index('datetime', inplace=True)
    print('googl',googl.head())


    googl_macd = get_macd(googl['close'], 26, 12, 9)
    googl_macd

    # plot_macd(googl['close'], googl_macd['macd'], googl_macd['signal'], googl_macd['hist'])

    buy_price, sell_price, macd_signal, trade_count = implement_macd_strategy(googl['close'], googl_macd)

    # ax1 = plt.subplot2grid((8,1), (0,0), rowspan = 5, colspan = 1)
    # ax2 = plt.subplot2grid((8,1), (5,0), rowspan = 3, colspan = 1)

    # ax1.plot(googl['close'], color = 'skyblue', linewidth = 2, label = stock)
    # ax1.plot(googl.index, buy_price, marker = '^', color = 'green', markersize = 10, label = 'BUY SIGNAL', linewidth = 0)
    # ax1.plot(googl.index, sell_price, marker = 'v', color = 'r', markersize = 10, label = 'SELL SIGNAL', linewidth = 0)
    # ax1.legend()
    # ax1.set_title(f'{stock} {algo} SIGNALS')
    # ax2.plot(googl_macd['macd'], color = 'grey', linewidth = 1.5, label = algo)
    # ax2.plot(googl_macd['signal'], color = 'skyblue', linewidth = 1.5, label = 'SIGNAL')

    # for i in range(len(googl_macd)):
    #     if str(googl_macd['hist'][i])[0] == '-':
    #         ax2.bar(googl_macd.index[i], googl_macd['hist'][i], color = '#ef5350')
    #     else:
    #         ax2.bar(googl_macd.index[i], googl_macd['hist'][i], color = '#26a69a')
            
    # plt.legend(loc = 'lower right')
    # plt.show()

    print(f'Total number of trades made: {trade_count}')

    position = []
    for i in range(len(macd_signal)):
        if macd_signal[i] > 1:
            position.append(0)
        else:
            position.append(1)
            
    for i in range(len(googl['close'])):
        if macd_signal[i] == 1:
            position[i] = 1
        elif macd_signal[i] == -1:
            position[i] = 0
        else:
            position[i] = position[i-1]
            
    macd = googl_macd['macd']
    signal = googl_macd['signal']
    close_price = googl['close']
    macd_signal = pd.DataFrame(macd_signal).rename(columns = {0:'macd_signal'}).set_index(googl.index)
    position = pd.DataFrame(position).rename(columns = {0:'macd_position'}).set_index(googl.index)

    frames = [close_price, macd, signal, macd_signal, position]
    strategy = pd.concat(frames, join = 'inner', axis = 1)

    strategy

    googl_ret = pd.DataFrame(np.diff(googl['close'])).rename(columns = {0:'returns'})
    macd_strategy_ret = []

    for i in range(len(googl_ret)):
        try:
            returns = googl_ret['returns'][i]*strategy['macd_position'][i]
            macd_strategy_ret.append(returns)
        except:
            pass
        
    macd_strategy_ret_df = pd.DataFrame(macd_strategy_ret).rename(columns = {0:'macd_returns'})

    investment_value = 100000
    number_of_stocks = floor(investment_value/googl['close'].iloc[0])
    macd_investment_ret = []

    # print(macd_strategy_ret_df.tail())
    for i in range(len(macd_strategy_ret_df['macd_returns'])):
        returns = number_of_stocks*macd_strategy_ret_df['macd_returns'].iloc[i]
        macd_investment_ret.append(returns)

    macd_investment_ret_df = pd.DataFrame(macd_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(macd_investment_ret_df['investment_returns']), 2)
    profit_percentage = round((total_investment_ret/investment_value)*100,2)
    print(cl(f'Profit gained from the {algo} strategy by investing $100k in {stock} : {total_investment_ret}', attrs = ['bold']))
    print(cl(f'Profit percentage of the {algo} strategy : {profit_percentage}%', attrs = ['bold']))

    investment_value = 100000
    # total_benchmark_investment_ret = round(sum(benchmark['investment_returns']), 2)
    # benchmark_profit_percentage = round((total_benchmark_investment_ret/investment_value)*100,2)

    number_of_stocks = investment_value / googl['open'].iloc[0]
    final_investment_value = number_of_stocks * googl['close'].iloc[-1]
    buy_and_hold_profit = final_investment_value - investment_value
    buy_and_hold_profit_percentage = (buy_and_hold_profit / investment_value) * 100

    # print(cl('Benchmark profit by investing $100k : {}'.format(total_benchmark_investment_ret), attrs = ['bold']))
    # print(cl('Benchmark Profit percentage : {}%'.format(benchmark_profit_percentage), attrs = ['bold']))
    print('Profit gained from the Buy and Hold strategy: $', buy_and_hold_profit)
    print('Profit percentage from the Buy and Hold strategy: {}%'.format(buy_and_hold_profit_percentage))
    # print(cl(f'{algo} Strategy profit is {round(profit_percentage - benchmark_profit_percentage,2)}% higher than the Benchmark Profit', attrs = ['bold']))
    print(cl(f'{algo} Strategy profit is {round(profit_percentage - buy_and_hold_profit_percentage,2)}% higher than the Buy and hold Profit', attrs = ['bold']))

    # Calculating daily returns
    macd_strategy_daily_returns = macd_strategy_ret_df['macd_returns'].replace([np.inf, -np.inf], np.nan).dropna()

    # Calculate the mean and standard deviation of daily returns
    mean_returns = macd_strategy_daily_returns.mean()
    std_returns = macd_strategy_daily_returns.std()

    # Calculate the excess returns by subtracting the risk-free rate. 
    # Assuming the risk-free rate is 0.01 (or 1% per annum), we divide it by 252 because there are 252 trading days in a year.
    excess_returns_macd = macd_strategy_daily_returns - 0.01/252

    # Calculate the daily returns of the buy and hold strategy
    buy_and_hold_returns = googl_ret['returns'].replace([np.inf, -np.inf], np.nan).dropna()

    # Calculate the mean and standard deviation of daily returns for the buy and hold strategy
    mean_buy_and_hold_returns = buy_and_hold_returns.mean()
    std_buy_and_hold_returns = buy_and_hold_returns.std()

    # Calculate the excess returns for the buy and hold strategy
    excess_returns_buy_and_hold = buy_and_hold_returns - 0.01/252

    # Calculate the mean and standard deviation of excess returns
    mean_excess_returns_macd = excess_returns_macd.mean()
    std_excess_returns_macd = excess_returns_macd.std()

    # Calculating the Sharpe Ratio
    sharpe_ratio = mean_returns / std_returns
    print('Sharpe Ratio of the MACD trading strategy: ', sharpe_ratio)

    # Calculate the t-statistic and p-value
    t_statistic, p_value = stats.ttest_ind(excess_returns_macd, excess_returns_buy_and_hold)

    print(f'T-statistic: {t_statistic}')
    print(f'P-value: {p_value}')

    # Define a significance level (e.g., 0.05)
    alpha = 0.05

    if p_value < alpha:
        print(f'The p-value ({p_value}) is less than the significance level ({alpha}). The MACD strategy returns are statistically significant.')
    else:
        print(f'The p-value ({p_value}) is greater than or equal to the significance level ({alpha}). The MACD strategy returns are not statistically significant.')

    # Calculate the profit percentage and store it in the list
    profit_percentages.append(profit_percentage)
    profits.append(total_investment_ret)
    buy_and_hold_profits.append(buy_and_hold_profit)
    
    # Calculate the Sharpe Ratio and store it in the list
    sharpe_ratios.append(sharpe_ratio)
    
    # Calculate the t-statistic and store it in the list
    t_statistics.append(t_statistic)
    p_values.append(p_value)
    trade_counts.append(trade_count)

    fromdatedt = datetime.utcfromtimestamp(fromdate/1000)
    todatedt = datetime.utcfromtimestamp(todate/1000)
    dates_choosen.append([fromdatedt.strftime("%Y-%m-%d %H:%M:%S"), todatedt.strftime("%Y-%m-%d %H:%M:%S")])

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
print('Dates count: ', len(dates_choosen))
print("Mean Profit:", mean_profit)
print("Mean Profit Buy and hold:", mean_buy_and_hold_profits)
print("Mean Sharpe Ratio:", mean_sharpe_ratio)
print("Mean Trade Count:", mean_trade_count)
print("Mean T-statistic:", mean_t_statistic)
print("Mean P-value:", mean_p_value)
print("Mean T-statistic fromzero:", mean_ttest_fromzero)
print("Mean P-value fromzero:", mean_pvalue_fromzero)