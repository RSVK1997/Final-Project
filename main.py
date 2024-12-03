import requests
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Function to fetch data from CoinGecko API
def fetch_data():
    api_url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30&interval=daily'
    response = requests.get(api_url)
    data = response.json()
    return data


# Function to automate a process (e.g., data cleaning)
def clean_data(data):
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


# Function to visualize data
def create_dashboard(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['price'], marker='o', linestyle='-')
    plt.title('Bitcoin Price Over the Last 30 Days')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.show()


# Function to perform regression analysis
def perform_regression(df):
    df['timestamp_ordinal'] = df['timestamp'].map(pd.Timestamp.toordinal)
    X = sm.add_constant(df['timestamp_ordinal'])
    y = df['price']

    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)

    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['price'], marker='o', linestyle='-', label='Actual Price')
    plt.plot(df['timestamp'], predictions, color='red', label='Regression Line')
    plt.title('Bitcoin Price Regression Analysis')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model.summary()


# Main function to execute the project
def main():
    data = fetch_data()

    cleaned_data = clean_data(data)

    create_dashboard(cleaned_data)

    regression_summary = perform_regression(cleaned_data)
    print(regression_summary)


if __name__ == "__main__":
    main()