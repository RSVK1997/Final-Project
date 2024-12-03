'''import requests
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html

# Step 2: Fetch Data from OpenFDA Drug API

# Fetch data from OpenFDA Drug Adverse Events API
adverse_events_url = 'https://api.fda.gov/drug/event.json?limit=100'
response = requests.get(adverse_events_url)
adverse_events_data = response.json()

# Convert to DataFrame
adverse_events_df = pd.json_normalize(adverse_events_data['results'])

# Inspect the DataFrame columns
print(adverse_events_df.columns)

# Fetch data from OpenFDA Drug Enforcement Reports API
enforcement_url = 'https://api.fda.gov/drug/enforcement.json?limit=100'
response = requests.get(enforcement_url)
enforcement_data = response.json()

# Convert to DataFrame
enforcement_df = pd.json_normalize(enforcement_data['results'])

# Step 3: Data Cleaning and Preprocessing

# Clean Adverse Events Data
adverse_events_df['receivedate'] = pd.to_datetime(adverse_events_df['receivedate'])

# Adjust column names based on actual data structure
adverse_events_df = adverse_events_df[['receivedate', 'patient.drug', 'patient.reaction']]

# Flatten nested lists
adverse_events_df = adverse_events_df.explode('patient.drug')
adverse_events_df = adverse_events_df.explode('patient.reaction')

# Extract relevant fields from nested dictionaries
adverse_events_df['drug_name'] = adverse_events_df['patient.drug'].apply(lambda x: x.get('medicinalproduct') if isinstance(x, dict) else None)
adverse_events_df['reaction'] = adverse_events_df['patient.reaction'].apply(lambda x: x.get('reactionmeddrapt') if isinstance(x, dict) else None)

# Drop the original nested columns
adverse_events_df = adverse_events_df.drop(columns=['patient.drug', 'patient.reaction'])

# Display the cleaned DataFrame
print(adverse_events_df.head())

# Clean Enforcement Reports Data
enforcement_df['report_date'] = pd.to_datetime(enforcement_df['report_date'])
enforcement_df = enforcement_df[['report_date', 'product_description', 'reason_for_recall', 'classification']]

# Step 4: Calculate KPIs

# KPIs for Adverse Events
kpi_drug = adverse_events_df['drug_name'].value_counts().head(10)
kpi_reaction = adverse_events_df['reaction'].value_counts().head(10)

# KPIs for Enforcement Reports
kpi_product = enforcement_df['product_description'].value_counts().head(10)
kpi_reason = enforcement_df['reason_for_recall'].value_counts().head(10)

# Step 5: Create Visualizations

# Visualizations for Adverse Events
fig_drug = px.bar(kpi_drug, x=kpi_drug.index, y=kpi_drug.values, labels={'x': 'Drug', 'y': 'Number of Adverse Events'}, title='Top 10 Drugs by Adverse Event Reports')
fig_reaction = px.bar(kpi_reaction, x=kpi_reaction.index, y=kpi_reaction.values, labels={'x': 'Reaction', 'y': 'Number of Adverse Events'}, title='Top 10 Reactions by Adverse Event Reports')

# Visualizations for Enforcement Reports
fig_product = px.bar(kpi_product, x=kpi_product.index, y=kpi_product.values, labels={'x': 'Product', 'y': 'Number of Recalls'}, title='Top 10 Products by Recall Events')
fig_reason = px.bar(kpi_reason, x=kpi_reason.index, y=kpi_reason.values, labels={'x': 'Reason', 'y': 'Number of Recalls'}, title='Top 10 Reasons for Recalls')

# Step 6: Build the Dashboard

# Create a Dash app
app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='OpenFDA Drug Dashboard'),
    dcc.Graph(
        id='drug-graph',
        figure=fig_drug
    ),
    dcc.Graph(
        id='reaction-graph',
        figure=fig_reaction
    ),
    dcc.Graph(
        id='product-graph',
        figure=fig_product
    ),
    dcc.Graph(
        id='reason-graph',
        figure=fig_reason
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)'''

import requests
import pandas as pd

# Fetch data from OpenFDA Drug Adverse Events API
adverse_events_url = 'https://api.fda.gov/drug/event.json?limit=100'
response = requests.get(adverse_events_url)
adverse_events_data = response.json()

# Convert to DataFrame
adverse_events_df = pd.json_normalize(adverse_events_data['results'])

# Fetch data from OpenFDA Drug Enforcement Reports API
enforcement_url = 'https://api.fda.gov/drug/enforcement.json?limit=100'
response = requests.get(enforcement_url)
enforcement_data = response.json()

# Convert to DataFrame
enforcement_df = pd.json_normalize(enforcement_data['results'])

# Convert date fields to datetime
adverse_events_df['receivedate'] = pd.to_datetime(adverse_events_df['receivedate'])

# Extract relevant columns
adverse_events_df = adverse_events_df[['receivedate', 'patient.drug', 'patient.reaction']]
adverse_events_df = adverse_events_df.explode('patient.drug')
adverse_events_df = adverse_events_df.explode('patient.reaction')
adverse_events_df['drug_name'] = adverse_events_df['patient.drug'].apply(lambda x: x.get('medicinalproduct') if isinstance(x, dict) else None)
adverse_events_df['reaction'] = adverse_events_df['patient.reaction'].apply(lambda x: x.get('reactionmeddrapt') if isinstance(x, dict) else None)
adverse_events_df = adverse_events_df.drop(columns=['patient.drug', 'patient.reaction'])

# Convert date fields to datetime
enforcement_df['report_date'] = pd.to_datetime(enforcement_df['report_date'])
enforcement_df = enforcement_df[['report_date', 'product_description', 'reason_for_recall', 'classification']]

# KPI: Count of adverse events by drug
kpi_drug = adverse_events_df['drug_name'].value_counts().head(10)

# KPI: Count of adverse events by reaction
kpi_reaction = adverse_events_df['reaction'].value_counts().head(10)

# KPI: Count of recalls by product description
kpi_product = enforcement_df['product_description'].value_counts().head(10)

# KPI: Count of recalls by reason
kpi_reason = enforcement_df['reason_for_recall'].value_counts().head(10)

import plotly.express as px

# Visualization: Top 10 Drugs by Adverse Events
fig_drug = px.bar(kpi_drug, x=kpi_drug.index, y=kpi_drug.values, labels={'x': 'Drug', 'y': 'Number of Adverse Events'}, title='Top 10 Drugs by Adverse Event Reports')

# Visualization: Top 10 Reactions by Adverse Events
fig_reaction = px.bar(kpi_reaction, x=kpi_reaction.index, y=kpi_reaction.values, labels={'x': 'Reaction', 'y': 'Number of Adverse Events'}, title='Top 10 Reactions by Adverse Event Reports')

# Visualization: Top 10 Products by Recall Events
fig_product = px.bar(kpi_product, x=kpi_product.index, y=kpi_product.values, labels={'x': 'Product', 'y': 'Number of Recalls'}, title='Top 10 Products by Recall Events')

# Visualization: Top 10 Reasons for Recalls
fig_reason = px.bar(kpi_reason, x=kpi_reason.index, y=kpi_reason.values, labels={'x': 'Reason', 'y': 'Number of Recalls'}, title='Top 10 Reasons for Recalls')

import statsmodels.api as sm

adverse_events_time_series = adverse_events_df.groupby('receivedate').size().reset_index(name='count')
X = sm.add_constant(adverse_events_time_series.index)
model = sm.OLS(adverse_events_time_series['count'], X).fit()
adverse_events_time_series['trend'] = model.predict(X)

fig_trend = px.line(adverse_events_time_series, x='receivedate', y='count', title='Number of Adverse Events Over Time')
fig_trend.add_scatter(x=adverse_events_time_series['receivedate'], y=adverse_events_time_series['trend'], mode='lines', name='Trend')

from statsmodels.tsa.holtwinters import ExponentialSmoothing

model_es = ExponentialSmoothing(adverse_events_time_series['count'], trend='add', seasonal=None).fit()
forecast_es = model_es.forecast(steps=12)
forecast_dates = pd.date_range(start=adverse_events_time_series['receivedate'].max(), periods=12, freq='M')

fig_forecast = px.line(adverse_events_time_series, x='receivedate', y='count', title='Adverse Events Forecast')
fig_forecast.add_scatter(x=forecast_dates, y=forecast_es, mode='lines', name='Forecast')

import dash
from dash import dcc, html

# Create a Dash app
app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='OpenFDA Drug Dashboard'),
    dcc.Graph(
        id='drug-graph',
        figure=fig_drug
    ),
    dcc.Graph(
        id='reaction-graph',
        figure=fig_reaction
    ),
    dcc.Graph(
        id='product-graph',
        figure=fig_product
    ),
    dcc.Graph(
        id='reason-graph',
        figure=fig_reason
    ),
    dcc.Graph(
        id='trend-graph',
        figure=fig_trend
    ),
    dcc.Graph(
        id='forecast-graph',
        figure=fig_forecast
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)