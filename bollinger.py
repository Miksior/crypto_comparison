# # IMPORTING PACKAGES

import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
from math import floor
from termcolor import colored as cl
import yfinance as yf
from IPython.display import display
import scipy.stats as stats
import random
from datetime import datetime, timedelta
import bitfinex
import time

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

num_repeats = 10
fromdate_randombase = '2015-01-01'  # 1 day, 15 m 
todate_randombase = '2023-10-13'   # 1 day, 15 m
# fromdate_randombase = '2016-05-07'  # 1m
# todate_randombase = '2023-03-01'   # 1m
#fromdate_randombase = '2019-05-07'  # 1 hour
#todate_randombase = '2022-03-01'   # 1 hour
timeframe = 7


interval = '1d'
# stock = 'ETH-USD'
stock = 'BTC-USD'
benchmarkstock = 'BTIC.SW'
# benchmarkstock = 'ETHE.SW'
fromdate = '2022-10-13'
# fromdate = '2023-01-01'
# fromdate = '2023-10-06'
# fromdate = '2023-10-12'
todate = '2023-10-13'
todate_benchmark = '2023-10-13'

profit_percentages = []
sharpe_ratios = []
t_statistics = []
profits = []
buy_and_hold_profits = []
p_values = []
trade_counts = []
dates_choosen = []

algo = 'BB'

##1 minute
#dfgg = pd.read_csv('C:/Users/Master/Documents/Sources/crypto_comparison/btcusd.csv')
#print("start",dfgg.head())

# #1 hour
# dfgg = pd.read_csv('C:/Users/Master/Documents/Sources/crypto_comparison/btcusd_hourly.csv')
# # print(dfgg.head())

# #1 day
# dfgg = pd.read_csv('C:/Users/Master/Documents/Sources/crypto_comparison/btcusd_daily.csv')
# print(dfgg.head())


def get_historical_data(symbol, start_date = None):
    if symbol != benchmarkstock:
        btic = yf.download(symbol, start=fromdate, end=todate, interval=interval)
    else:
        btic = yf.download(symbol, start=fromdate, end=todate_benchmark)
    # btic = yf.Ticker(symbol)
    # btic = btic.history(period='1d', interval="1m")
    df = btic.rename(columns = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj close': 'adj close', 'Volume': 'volume'})
    if start_date:
        df = df[df.index >= start_date]
    # display(btic)
    return df

def sma(data, lookback):
    sma = data.rolling(lookback).mean()
    return sma

def get_bb(data, lookback):
    std = data.rolling(lookback).std()
    upper_bb = sma(data, lookback) + std * 2
    lower_bb = sma(data, lookback) - std * 2
    middle_bb = sma(data, lookback)
    return upper_bb, middle_bb, lower_bb

def get_kc(high, low, close, kc_lookback, multiplier, atr_lookback):
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift()))
    tr3 = pd.DataFrame(abs(low - close.shift()))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.ewm(alpha = 1/atr_lookback).mean()
    
    kc_middle = close.ewm(kc_lookback).mean()
    kc_upper = close.ewm(kc_lookback).mean() + multiplier * atr
    kc_lower = close.ewm(kc_lookback).mean() - multiplier * atr
    
    return kc_middle, kc_upper, kc_lower

def get_rsi(close, lookback):
    ret = close.diff()
    up = []
    down = []
    for i in range(len(ret)):
        if ret[i] < 0:
            up.append(0)
            down.append(ret[i])
        else:
            up.append(ret[i])
            down.append(0)
    up_series = pd.Series(up)
    down_series = pd.Series(down).abs()
    up_ewm = up_series.ewm(com = lookback - 1, adjust = False).mean()
    down_ewm = down_series.ewm(com = lookback - 1, adjust = False).mean()
    rs = up_ewm/down_ewm
    rsi = 100 - (100 / (1 + rs))
    rsi_df = pd.DataFrame(rsi).rename(columns = {0:'rsi'}).set_index(close.index)
    rsi_df = rsi_df.dropna()
    return rsi_df[3:]

def bb_kc_rsi_strategy(prices, upper_bb, lower_bb, kc_upper, kc_lower, rsi):
    buy_price = []
    sell_price = []
    bb_kc_rsi_signal = []
    signal = 0
    buy_count = 0
    sell_count = 0
    
    for i in range(len(prices)):
        if lower_bb[i] < kc_lower[i] and upper_bb[i] > kc_upper[i] and rsi[i] < 30:
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                bb_kc_rsi_signal.append(signal)
                buy_count += 1  # Increment the buy signal counter
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_kc_rsi_signal.append(0)
                
        elif lower_bb[i] < kc_lower[i] and upper_bb[i] > kc_upper[i] and rsi[i] > 70:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                bb_kc_rsi_signal.append(signal)
                sell_count += 1  # Increment the sell signal counter
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_kc_rsi_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            bb_kc_rsi_signal.append(0)
                        
    return buy_price, sell_price, bb_kc_rsi_signal, buy_count, sell_count

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
    print("Sample: ",fromdate_datetime, todate_datetime)
    # fromdate = time.mktime(fromdate)
    # todate = time.mktime(todate)
    
    # googl = get_historical_data(stock, fromdate, todate, interval, 1000, 60000000)
    # googl = get_historical_data(stock, fromdate, todate, interval, 1000, 60000000)
    # print(dfgg.head())
    googl = dfgg[(dfgg['datetime'] >= fromdate) & (dfgg['datetime'] <= todate)].copy()
    googl['datetime'] = pd.to_datetime(googl['datetime'], unit='ms')
    googl['datetime'] = googl['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
    googl.set_index('datetime', inplace=True)
    # print(googl.head())

    # aapl = get_historical_data(stock, fromdate)
    aapl = googl.copy()
    aapl.tail()

    investment_value = 100000

    bandh_buy_price = aapl['close'][0]
    print(bandh_buy_price)
    bandh_final_price = aapl['close'][-1]
    print(bandh_final_price)
    bandh_number_of_stocks = floor(investment_value / bandh_buy_price)
    bandh_final_value = bandh_number_of_stocks * bandh_final_price
    buy_and_hold_profit = bandh_final_price - bandh_number_of_stocks * bandh_buy_price
    buy_and_hold_profit_percentage = round((buy_and_hold_profit / investment_value) * 100,2)

    # BOLLINGER BANDS CALCULATION
    aapl['upper_bb'], aapl['middle_bb'], aapl['lower_bb'] = get_bb(aapl['close'], 20)
    aapl.tail()

    # BOLLINGER BANDS PLOT

    # plot_data = aapl[aapl.index >= fromdate]

    # plt.plot(plot_data['close'], linewidth = 2.5)
    # plt.plot(plot_data['upper_bb'], label = 'UPPER BB 20', linewidth = 2, color = 'violet')
    # plt.plot(plot_data['middle_bb'], label = 'MIDDLE BB 20', linewidth = 1.5, color = 'grey')
    # plt.plot(plot_data['lower_bb'], label = 'LOWER BB 20', linewidth = 2, color = 'violet')
    # plt.title(f'{stock} BB 20')
    # plt.legend(fontsize = 15)
    # plt.show()


    aapl['kc_middle'], aapl['kc_upper'], aapl['kc_lower'] = get_kc(aapl['high'], aapl['low'], aapl['close'], 20, 2, 10)
    aapl.tail()

    aapl['rsi_14'] = get_rsi(aapl['close'], 14)
    aapl = aapl.dropna()
    aapl.tail()

    buy_price, sell_price, bb_kc_rsi_signal, buy_count, sell_count = bb_kc_rsi_strategy(aapl['close'], aapl['upper_bb'], aapl['lower_bb'],
                                                            aapl['kc_upper'], aapl['kc_lower'], aapl['rsi_14'])

    # ax1 = plt.subplot2grid((11,1), (0,0), rowspan = 5, colspan = 1)
    # ax2 = plt.subplot2grid((11,1), (6,0), rowspan = 5, colspan = 1)
    # ax1.plot(aapl['close'])
    # ax1.plot(aapl.index, buy_price, marker = '^', markersize = 10, linewidth = 0, color = 'green', label = 'BUY SIGNAL')
    # ax1.plot(aapl.index, sell_price, marker = 'v', markersize = 10, linewidth = 0, color = 'r', label = 'SELL SIGNAL')
    # ax1.set_title(f'{stock} STOCK PRICE')
    # ax2.plot(aapl['rsi_14'], color = 'purple', linewidth = 2)
    # ax2.axhline(30, color = 'grey', linestyle = '--', linewidth = 1.5)
    # ax2.axhline(70, color = 'grey', linestyle = '--', linewidth = 1.5)
    # ax2.set_title(f'{stock} RSI 10')
    # plt.show()

    # POSITION

    position = []
    for i in range(len(bb_kc_rsi_signal)):
        if bb_kc_rsi_signal[i] > 1:
            position.append(0)
        else:
            position.append(1)
            
    for i in range(len(aapl['close'])):
        if bb_kc_rsi_signal[i] == 1:
            position[i] = 1
        elif bb_kc_rsi_signal[i] == -1:
            position[i] = 0
        else:
            position[i] = position[i-1]
            
    kc_upper = aapl['kc_upper']
    kc_lower = aapl['kc_lower']
    upper_bb = aapl['upper_bb'] 
    lower_bb = aapl['lower_bb']
    rsi = aapl['rsi_14']
    close_price = aapl['close']
    bb_kc_rsi_signal = pd.DataFrame(bb_kc_rsi_signal).rename(columns = {0:'bb_kc_rsi_signal'}).set_index(aapl.index)
    position = pd.DataFrame(position).rename(columns = {0:'bb_kc_rsi_position'}).set_index(aapl.index)

    frames = [close_price, kc_upper, kc_lower, upper_bb, lower_bb, rsi, bb_kc_rsi_signal, position]
    print(frames)
    strategy = pd.concat(frames, join = 'inner', axis = 1)

    strategy.tail()

    # BACKTESTING

    aapl_ret = pd.DataFrame(np.diff(aapl['close'])).rename(columns = {0:'returns'})
    bb_kc_rsi_strategy_ret = []

    for i in range(len(aapl_ret)):
        returns = aapl_ret['returns'][i]*strategy['bb_kc_rsi_position'][i]
        bb_kc_rsi_strategy_ret.append(returns)
        
    bb_kc_rsi_strategy_ret_df = pd.DataFrame(bb_kc_rsi_strategy_ret).rename(columns = {0:'bb_kc_rsi_returns'})

    number_of_stocks = floor(investment_value/aapl['open'][0])
    bb_kc_rsi_investment_ret = []

    for i in range(len(bb_kc_rsi_strategy_ret_df['bb_kc_rsi_returns'])):
        returns = number_of_stocks*bb_kc_rsi_strategy_ret_df['bb_kc_rsi_returns'][i]
        bb_kc_rsi_investment_ret.append(returns)

    bb_kc_rsi_investment_ret_df = pd.DataFrame(bb_kc_rsi_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(bb_kc_rsi_investment_ret_df['investment_returns']), 2)
    profit_percentage = round((total_investment_ret/investment_value)*100,2)
    print(cl(f'Profit gained from the BB KC RSI strategy by investing $100k in {stock} : {total_investment_ret}', attrs = ['bold']))
    print(cl('Profit percentage of the BB KC RSI strategy : {}%'.format(profit_percentage), attrs = ['bold']))

    # SPY ETF COMPARISON



    # bandh_buy_price = aapl['open'][0]
    # bandh_final_price = aapl['close'][-1]
    # number_of_stocks = floor(investment_value / bandh_buy_price)
    # final_investment_value = number_of_stocks * bandh_buy_price
    # buy_and_hold_profit = final_investment_value - investment_value * bandh_buy_price
    # buy_and_hold_profit_percentage = round((buy_and_hold_profit / investment_value) * 100,2)

    print('Profit gained from the Buy and Hold strategy: $', buy_and_hold_profit)
    print('Profit percentage from the Buy and Hold Strategy: {}%'.format(buy_and_hold_profit_percentage))
    print(cl('BB KC RSI Strategy profit is {}% higher than the Buy and hold Profit'.format(round(profit_percentage - buy_and_hold_profit_percentage, 2)), attrs = ['bold']))

    # Calculate the returns of the strategy
    strategy['returns'] = strategy['close'].pct_change()

    # Calculate the returns of the buy and hold strategy
    buy_and_hold_returns = aapl['close'].pct_change()

    # Calculate the excess returns by subtracting the risk-free rate. 
    # Assuming the risk-free rate is 0.01 (or 1% per annum), we divide it by 252 because there are 252 trading days in a year.
    strategy['excess_returns'] = strategy['returns'] - 0.01/252

    # Calculate the Sharpe Ratio
    sharpe_ratio = np.sqrt(252) * (strategy['excess_returns'].mean() / strategy['excess_returns'].std())

    print('The Sharpe Ratio is:', sharpe_ratio)

    strategy['cumulative_returns'] = (1 + strategy['returns']).cumprod()
    excess_returns = strategy['excess_returns'].dropna()

    # Perform a t-test
    t_statistic, p_value = stats.ttest_ind(strategy['excess_returns'].dropna(), buy_and_hold_returns.dropna())

    print("T-Statistic:", t_statistic)
    print("P-Value:", p_value)

    # Define a significance level (e.g., 0.05)
    alpha = 0.05

    if p_value < alpha:
        print(f'The p-value ({p_value}) is less than the significance level ({alpha}). The MACD strategy returns are statistically significant.')
    else:
        print(f'The p-value ({p_value}) is greater than or equal to the significance level ({alpha}). The MACD strategy returns are not statistically significant.')

    print("Number of Trades:", sell_count+buy_count)

    # Calculate the profit percentage and store it in the list
    profit_percentages.append(profit_percentage)
    profits.append(total_investment_ret)
    buy_and_hold_profits.append(buy_and_hold_profit)
    
    # Calculate the Sharpe Ratio and store it in the list
    sharpe_ratios.append(sharpe_ratio)
    
    # Calculate the t-statistic and store it in the list
    t_statistics.append(t_statistic)
    p_values.append(p_value)
    trade_counts.append(sell_count+buy_count)

    fromdatedt = datetime.utcfromtimestamp(fromdate/1000)
    todatedt = datetime.utcfromtimestamp(todate/1000)
    dates_choosen.append([fromdatedt.strftime("%Y-%m-%d %H:%M:%S"), todatedt.strftime("%Y-%m-%d %H:%M:%S")])

    descriptivestats = pd.DataFrame({
    'Profit': profits,
    'Buy and Hold Profit': buy_and_hold_profits,
    'Sharpe Ratio': sharpe_ratios,
    'Trade Count': trade_counts,
    'T-statistic': t_statistics,
    'P-value': p_values
    })

    print("Descriptive Statistics:", descriptivestats.describe())

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
print("Profits: ", profits)
print("Buy and hold profits: ", buy_and_hold_profits)
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