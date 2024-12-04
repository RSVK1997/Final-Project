# Stock Market Dashboard

This project is a Stock Market Dashboard built using Dash and Plotly. The dashboard allows users to visualize stock data for multiple companies, including IBM, Apple, Google, Nvidia, and Microsoft. The dashboard displays various graphs such as open and close prices, trading volume, daily high prices, and daily low prices.

## Features

- Bar graph for daily open and close prices
- Bar graph for daily trading volume
- Scatter plot for daily high prices with regression line
- Scatter plot for daily low prices with regression line
- Dropdown menu to select different stocks

## Setup

Follow these steps to set up the project on your local machine:

1. *Clone the repository:*

    bash
    git clone https://github.com/RSVK1997/Final-Project.git
    cd Final-Project
    
2. *Install the required dependencies:*

3. *Run the application:*

4. *Open your web browser and navigate to:*

    
    http://127.0.0.1:8050/
    

## Project Structure

- main.py: The main application file that contains the Dash app and callbacks.
- *.csv: The CSV files with stock data.

## Data Source

The stock data is fetched from the Alpha Vantage API. If the API call fails or returns invalid data, the data is loaded from the saved CSV files.

## Acknowledgements

- PyCharm - The Python IDE for Professional Developers.
- Dash - A Python framework for building analytical web applications.
- Plotly - A graphing library for making interactive, publication-quality graphs.
- Alpha Vantage - A provider of free APIs for real-time and historical data on stocks.
