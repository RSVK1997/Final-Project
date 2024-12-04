import requests
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import os


# Function to fetch stock data from Alpha Vantage API
def fetch_stock_data(api_key, symbol):
    """
    Fetch stock data from Alpha Vantage API.

    Parameters:
    api_key (str): Alpha Vantage API key.
    symbol (str): Stock symbol.

    Returns:
    dict: JSON response from the API containing stock data.
    """
    api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'
    response = requests.get(api_url)
    data = response.json()
    return data


# Function to clean and process the data
def clean_data(data):
    """
    Clean and process the stock data.

    Parameters:
    data (dict): JSON response from the API containing stock data.

    Returns:
    DataFrame: Cleaned and processed stock data.
    """
    time_series = data.get('Time Series (Daily)')
    if not time_series:
        raise ValueError("Invalid data format received from API")

    df = pd.DataFrame.from_dict(time_series, orient='index')
    df = df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    })
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    return df


# Function to save data to a file
def save_data_to_file(df, filename):
    """
    Save stock data to a CSV file.

    Parameters:
    df (DataFrame): Stock data.
    filename (str): Name of the file to save the data.
    """
    df.to_csv(filename, index=True)


# Function to load data from a file
def load_data_from_file(filename):
    """
    Load stock data from a CSV file.

    Parameters:
    filename (str): Name of the file to load the data from.

    Returns:
    DataFrame: Loaded stock data.
    """
    if os.path.exists(filename):
        return pd.read_csv(filename, parse_dates=True, index_col=0)
    return None


# Function to get stock data and save to local file
def get_stock_data(api_key, symbol):
    """
    Get stock data and save it to a local file.

    Parameters:
    api_key (str): Alpha Vantage API key.
    symbol (str): Stock symbol.

    Returns:
    DataFrame: Cleaned and processed stock data.
    """
    filename = f'{symbol}_data.csv'

    # Try fetching data from API
    try:
        # stock_data = load_data_from_file(filename)
        stock_data = fetch_stock_data(api_key, symbol)
        df = clean_data(stock_data)
        if not df.empty:
            save_data_to_file(df, filename)
            return df
        else:
            raise ValueError("Empty DataFrame")

    # If API fails or returns empty DataFrame, load from file
    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"Error fetching data from API: {e}. Loading from file.")
        return load_data_from_file(filename)


# Function to perform regression analysis and add regression line to the graph
def perform_regression(df, column):
    """
    Perform regression analysis and add regression line to the graph.

    Parameters:
    df (DataFrame): Stock data.
    column (str): Column name for regression analysis.

    Returns:
    model: Regression model.
    predictions: Predicted values from the regression model.
    """
    # Convert date index to ordinal for regression analysis
    df['timestamp_ordinal'] = df.index.map(pd.Timestamp.toordinal)
    X = sm.add_constant(df['timestamp_ordinal'])
    y = df[column]

    # Fit regression model
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)

    return model, predictions


# Fetch and clean data for multiple stocks
api_key = '2ZYMISYN89KDZ6CX'  # Replace with your Alpha Vantage API key
symbols = ['IBM', 'AAPL', 'GOOGL', 'NVDA', 'MSFT']  # List of stock symbols to analyze
data_frames = {symbol: get_stock_data(api_key, symbol) for symbol in symbols}

# Create Dash app
app = dash.Dash(__name__)

# Layout of the dashboard with 2x2 view
app.layout = html.Div([
    html.H1("Stock Market Dashboard"),

    dcc.Dropdown(
        id='stock-dropdown',
        options=[{'label': symbol, 'value': symbol} for symbol in symbols],
        value='IBM'
    ),

    html.Div([
        dcc.Graph(id='open-close-graph'),
        dcc.Graph(id='volume-graph')
    ], style={'display': 'flex', 'flex-direction': 'row'}),

    html.Div([
        dcc.Graph(id='high-graph'),
        dcc.Graph(id='low-graph')
    ], style={'display': 'flex', 'flex-direction': 'row'})
])


# Callback to update graphs based on selected stock
@app.callback(
    [Output('open-close-graph', 'figure'),
     Output('volume-graph', 'figure'),
     Output('high-graph', 'figure'),
     Output('low-graph', 'figure')],
    [Input('stock-dropdown', 'value')]
)
def update_graphs(selected_stock):
    """
    Update graphs based on selected stock.

    Parameters:
    selected_stock (str): Selected stock symbol.

    Returns:
    fig_open_close: Bar graph for open and close prices.
    fig_volume: Bar graph for trading volume.
    fig_high: Scatter plot for daily high prices with regression line.
    fig_low: Scatter plot for daily low prices with regression line.
    """
    # Get the dataframe for the selected stock
    df = data_frames[selected_stock]

    # Bar graph for open and close prices
    fig_open_close = go.Figure(data=[
        go.Bar(name='Open', x=df.index, y=df['Open']),
        go.Bar(name='Close', x=df.index, y=df['Close'])
    ])
    fig_open_close.update_layout(
        title=f'{selected_stock} Daily Open and Close Prices',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        barmode='group',
        legend_title_text='Legend'
    )

    # Bar graph for trading volume
    fig_volume = px.bar(df, x=df.index, y='Volume', title=f'{selected_stock} Daily Trading Volume')
    fig_volume.update_layout(
        xaxis_title='Date',
        yaxis_title='Volume',
        legend_title_text='Legend'
    )

    # Scatter plot for daily high prices with regression line
    model_high, predictions_high = perform_regression(df, 'High')
    fig_high = px.scatter(df, x=df.index, y='High', title=f'{selected_stock} Daily High Prices')
    fig_high.add_scatter(x=df.index, y=predictions_high, mode='lines', name='Regression Line')
    fig_high.update_layout(
        xaxis_title='Date',
        yaxis_title='High Price (USD)',
        legend_title_text='Legend'
    )

    # Scatter plot for daily low prices with regression line
    model_low, predictions_low = perform_regression(df, 'Low')
    fig_low = px.scatter(df, x=df.index, y='Low', title=f'{selected_stock} Daily Low Prices')
    fig_low.add_scatter(x=df.index, y=predictions_low, mode='lines', name='Regression Line')
    fig_low.update_layout(
        xaxis_title='Date',
        yaxis_title='Low Price (USD)',
        legend_title_text='Legend'
    )

    return fig_open_close, fig_volume, fig_high, fig_low


# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)