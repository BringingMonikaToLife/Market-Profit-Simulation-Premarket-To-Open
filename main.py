import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for ticker '{ticker}' between {start_date} and {end_date}.")
    print(f"Downloaded columns for {ticker}:", data.columns)
    if isinstance(data.columns, pd.MultiIndex):
        if 'Ticker' in data.columns.names:
            data.columns = data.columns.droplevel('Ticker')
        else:
            data.columns = data.columns.droplevel(0)
    data.columns = [col.title() for col in data.columns]
    for col in ['Open', 'Close']:
        if col not in data.columns:
            raise KeyError(f"Expected column '{col}' not found in data columns: {data.columns.tolist()}")
    data = data[['Open', 'Close']].copy()
    data['PrevClose'] = data['Close'].shift(1)
    data.dropna(inplace=True)
    return data

def simulate_trades(data, sample_size, premarket_threshold=0.0):
    sample_data = data.sample(n=sample_size, random_state=42).reset_index(drop=True)
    sample_data['PreMkt_Positive'] = sample_data['Open'] > (sample_data['PrevClose'] + premarket_threshold)
    trades = sample_data[sample_data['PreMkt_Positive']].copy()
    if trades.empty:
        print("No trades signaled with the given criteria.")
        return trades, None
    trades['Price_10'] = trades['Open'] + 0.5 * (trades['Close'] - trades['Open'])
    trades['Return'] = trades['Price_10'] / trades['Open'] - 1
    trades['Profit'] = trades['Return'] * 100
    summary = {
        'Total Trades': trades.shape[0],
        'Total Profit (%)': trades['Profit'].sum(),
        'Mean Return (%)': trades['Return'].mean() * 100,
        'Std Return (%)': trades['Return'].std() * 100,
        'Min Return (%)': trades['Return'].min() * 100,
        'Max Return (%)': trades['Return'].max() * 100
    }
    return trades, summary

def plot_trade_returns(trades, ticker):
    plt.figure(figsize=(8, 5))
    plt.hist(trades['Return'] * 100, bins=20, edgecolor='black', alpha=0.7)
    plt.title(f"Distribution of Trade Returns for {ticker}")
    plt.xlabel("Return (%)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def main():
    print("Welcome to the Stock Strategy Simulator!")
    ticker1 = input("Enter the primary ticker symbol (e.g., AAPL): ").upper().strip()
    ticker2 = input("Enter a second ticker symbol to compare (or press Enter to skip): ").upper().strip()
    start_date = input("Enter the start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter the end date (YYYY-MM-DD): ").strip()
    try:
        sample_size = int(input("Enter the number of trading days to simulate (e.g., 1000): ").strip())
    except ValueError:
        print("Invalid input for sample size. Using default of 1000 days.")
        sample_size = 1000
    try:
        threshold = float(input("Enter the pre-market threshold (e.g., 0 for no extra threshold, or 0.5 for a $0.5 gap): ").strip())
    except ValueError:
        print("Invalid input for threshold. Using default of 0.")
        threshold = 0.0
    try:
        data1 = get_stock_data(ticker1, start_date, end_date)
    except Exception as e:
        print(e)
        return
    trades1, summary1 = simulate_trades(data1, sample_size, premarket_threshold=threshold)
    print(f"\n--- Simulation Results for {ticker1} ---")
    print(f"Total number of sampled days: {sample_size}")
    if summary1:
        print(f"Number of trades signaled: {summary1['Total Trades']}")
        print(f"Total net profit: {summary1['Total Profit (%)']:.2f}%")
        print("Return statistics (in %):")
        for key in ['Mean Return (%)', 'Std Return (%)', 'Min Return (%)', 'Max Return (%)']:
            print(f"  {key}: {summary1[key]:.2f}")
    else:
        print("No trades were executed based on the strategy criteria.")
    if not trades1.empty:
        plot_trade_returns(trades1, ticker1)
    if ticker2:
        try:
            data2 = get_stock_data(ticker2, start_date, end_date)
        except Exception as e:
            print(e)
            return
        trades2, summary2 = simulate_trades(data2, sample_size, premarket_threshold=threshold)
        print(f"\n--- Simulation Results for {ticker2} ---")
        print(f"Total number of sampled days: {sample_size}")
        if summary2:
            print(f"Number of trades signaled: {summary2['Total Trades']}")
            print(f"Total net profit: {summary2['Total Profit (%)']:.2f}%")
            print("Return statistics (in %):")
            for key in ['Mean Return (%)', 'Std Return (%)', 'Min Return (%)', 'Max Return (%)']:
                print(f"  {key}: {summary2[key]:.2f}")
        else:
            print("No trades were executed based on the strategy criteria.")
        if not trades2.empty:
            plot_trade_returns(trades2, ticker2)
        if summary1 and summary2:
            labels = ['Total Profit (%)', 'Mean Return (%)']
            ticker1_stats = [summary1['Total Profit (%)'], summary1['Mean Return (%)']]
            ticker2_stats = [summary2['Total Profit (%)'], summary2['Mean Return (%)']]
            x = np.arange(len(labels))
            width = 0.35
            fig, ax = plt.subplots(figsize=(8, 5))
            rects1 = ax.bar(x - width/2, ticker1_stats, width, label=ticker1)
            rects2 = ax.bar(x + width/2, ticker2_stats, width, label=ticker2)
            ax.set_ylabel('Percentage (%)')
            ax.set_title('Comparison of Strategy Performance')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.show()

if __name__ == "__main__":
    main()
