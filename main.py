import requests
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

# Function to fetch stock data from Alpha Vantage API
def fetch_stock_data(api_key: str, symbol: str) -> dict:
    """
    Fetch stock data from Alpha Vantage API.

    Parameters:
    api_key (str): API key for Alpha Vantage.
    symbol (str): Stock symbol to fetch data for.

    Returns:
    dict: JSON response from the API containing stock data.
    """
    api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'
    response = requests.get(api_url)
    data = response.json()
    return data

# Function to clean and process the data
def clean_data(data: dict) -> pd.DataFrame:
    """
    Clean and process the stock data.

    Parameters:
    data (dict): Raw stock data from the API.

    Returns:
    pd.DataFrame: Processed data as a DataFrame.
    """
    time_series = data['Time Series (Daily)']
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df = df.rename(columns={
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '5. volume': 'volume'
    })
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    return df

# Function to save data to a file
def save_data_to_file(df: pd.DataFrame, filename: str) -> None:
    """
    Save DataFrame to a CSV file.

    Parameters:
    df (pd.DataFrame): DataFrame to save.
    filename (str): Name of the file to save the data to.
    """
    df.to_csv(filename, index=False)

# Function to load data from a file
def load_data_from_file(filename: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters:
    filename (str): Name of the file to load data from.

    Returns:
    pd.DataFrame: Loaded data as a DataFrame, or None if file does not exist.
    """
    if os.path.exists(filename):
        return pd.read_csv(filename, parse_dates=True)
    return None

# Function to get stock data and save to local file
def get_stock_data(api_key: str, symbol: str) -> pd.DataFrame:
    """
    Get stock data and save to a local file.

    Parameters:
    api_key (str): API key for Alpha Vantage.
    symbol (str): Stock symbol to fetch data for.

    Returns:
    pd.DataFrame: Stock data as a DataFrame.
    """
    filename = f'{symbol}_data.csv'
    df = load_data_from_file(filename)
    if df is None:
        stock_data = fetch_stock_data(api_key, symbol)
        df = clean_data(stock_data)
        save_data_to_file(df, filename)
    return df

# Function to perform regression analysis and add regression line to the graph
def perform_regression(df: pd.DataFrame) -> tuple:
    """
    Perform regression analysis on the stock data.

    Parameters:
    df (pd.DataFrame): DataFrame containing stock data.

    Returns:
    tuple: Regression model and predictions.
    """
    df['timestamp_ordinal'] = df.index.map(pd.Timestamp.toordinal)
    X = sm.add_constant(df['timestamp_ordinal'])
    y = df['close']

    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)

    return model, predictions

# Function to perform forecast using Exponential Smoothing
def perform_forecast(df: pd.DataFrame) -> pd.Series:
    """
    Perform forecast using Exponential Smoothing.

    Parameters:
    df (pd.DataFrame): DataFrame containing stock data.

    Returns:
    pd.Series: Forecasted values.
    """
    model = ExponentialSmoothing(df['close'], trend='add', seasonal=None).fit()
    forecast = model.forecast(steps=30)
    return forecast

# Fetch and clean data for multiple stocks
api_key = 'your_api_key_here'  # Replace with your Alpha Vantage API key
symbols = ['IBM', 'AAPL', 'GOOGL']  # List of stock symbols to analyze
data_frames = {symbol: get_stock_data(api_key, symbol) for symbol in symbols}

# Create Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Stock Market Dashboard"),

    dcc.Dropdown(
        id='stock-dropdown',
        options=[{'label': symbol, 'value': symbol} for symbol in symbols],
        value='IBM'
    ),

    dcc.Graph(id='stock-graph'),

    dcc.Graph(id='volume-graph')
])

# Callback to update graphs based on selected stock
@app.callback(
    [Output('stock-graph', 'figure'),
     Output('volume-graph', 'figure')],
    [Input('stock-dropdown', 'value')]
)
def update_graphs(selected_stock: str) -> tuple:
    """
    Update graphs based on selected stock.

    Parameters:
    selected_stock (str): Selected stock symbol.

    Returns:
    tuple: Figures for stock price and volume graphs.
    """
    df = data_frames[selected_stock]

    # Perform regression analysis
    model, predictions = perform_regression(df)

    # Perform forecast
    forecast = perform_forecast(df)
    forecast_dates = pd.date_range(start=df.index[-1], periods=len(forecast), freq='D')

    # Line graph for stock prices with regression line
    fig_price = px.line(df, x=df.index, y='close', title=f'{selected_stock} Daily Closing Prices')
    fig_price.add_scatter(x=df.index, y=predictions, mode='lines', name='Regression Line')
    fig_price.add_scatter(x=forecast_dates, y=forecast, mode='lines', name='Forecast', line=dict(dash='dash'))

    # Bar graph for trading volume
    fig_volume = px.bar(df, x=df.index, y='volume', title=f'{selected_stock} Daily Trading Volume')

    return fig_price, fig_volume

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)