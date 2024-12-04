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
    api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'
    response = requests.get(api_url)
    data = response.json()
    return data


# Function to clean and process the data
def clean_data(data):
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
    df.to_csv(filename, index=True)


# Function to load data from a file
def load_data_from_file(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename, parse_dates=True, index_col=0)
    return None


# Function to get stock data and save to local file
def get_stock_data(api_key, symbol):
    filename = f'{symbol}_data.csv'

    try:
        stock_data = fetch_stock_data(api_key, symbol)
        df = clean_data(stock_data)
        if not df.empty:
            save_data_to_file(df, filename)
            return df
        else:
            raise ValueError("Empty DataFrame")

    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"Error fetching data from API: {e}. Loading from file.")
        return load_data_from_file(filename)


# Function to perform multiple linear regression and add regression line to the graph
def perform_multiple_regression(df):
    X = df[['Open', 'High', 'Low']]
    X = sm.add_constant(X)
    y = df['Close']

    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)

    return model, predictions


# Function to perform linear regression for a single column
def perform_linear_regression(df, column):
    df['timestamp_ordinal'] = df.index.map(pd.Timestamp.toordinal)
    X = sm.add_constant(df['timestamp_ordinal'])
    y = df[column]

    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)

    return model, predictions


# Fetch and clean data for multiple stocks
api_key = 'OXR9HT66GVV909Z3'  # Replace with your Alpha Vantage API key
symbols = ['IBM', 'AAPL', 'GOOGL', 'NVDA', 'MSFT']  # List of stock symbols to analyze
data_frames = {symbol: get_stock_data(api_key, symbol) for symbol in symbols}

# Create Dash app
app = dash.Dash(__name__)

# Layout of the dashboard with 2x2 view and regression graph
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
    ], style={'display': 'flex', 'flex-direction': 'row'}),

    dcc.Graph(id='regression-graph')
])


# Callback to update graphs based on selected stock
@app.callback(
    [Output('open-close-graph', 'figure'),
     Output('volume-graph', 'figure'),
     Output('high-graph', 'figure'),
     Output('low-graph', 'figure'),
     Output('regression-graph', 'figure')],
    [Input('stock-dropdown', 'value')]
)
def update_graphs(selected_stock):
    df = data_frames[selected_stock]

    # Line graph for open and close prices
    fig_open_close = go.Figure()
    fig_open_close.add_trace(go.Scatter(x=df.index, y=df['Open'], mode='lines', name='Open'))
    fig_open_close.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    fig_open_close.update_layout(
        title=f'{selected_stock} Daily Open and Close Prices',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
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
    model_high, predictions_high = perform_linear_regression(df, 'High')
    fig_high = px.scatter(df, x=df.index, y='High', title=f'{selected_stock} Daily High Prices')
    fig_high.add_scatter(x=df.index, y=predictions_high, mode='lines', name='Regression Line')
    fig_high.update_layout(
        xaxis_title='Date',
        yaxis_title='High Price (USD)',
        legend_title_text='Legend'
    )

    # Scatter plot for daily low prices with regression line
    model_low, predictions_low = perform_linear_regression(df, 'Low')
    fig_low = px.scatter(df, x=df.index, y='Low', title=f'{selected_stock} Daily Low Prices')
    fig_low.add_scatter(x=df.index, y=predictions_low, mode='lines', name='Regression Line')
    fig_low.update_layout(
        xaxis_title='Date',
        yaxis_title='Low Price (USD)',
        legend_title_text='Legend'
    )

    # Perform multiple linear regression
    model_multiple, predictions_multiple = perform_multiple_regression(df)

    # Regression graph for open, high, low, and close prices
    fig_regression = go.Figure()
    fig_regression.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='markers', name='Actual Close'))
    fig_regression.add_trace(go.Scatter(x=df.index, y=predictions_multiple, mode='lines', name='Predicted Close'))
    fig_regression.update_layout(
        title=f'{selected_stock} Multiple Linear Regression (Open, High, Low -> Close)',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        legend_title_text='Legend'
    )

    return fig_open_close, fig_volume, fig_high, fig_low, fig_regression


# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)