#!/usr/bin/env python
# coding: utf-8

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load the data using pandas
URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/d51iMGfp_t0QpO30Lym-dw/automobile-sales.csv'
data = pd.read_csv(URL)

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = 'Automobile Statistics Dashboard'

# Create the dropdown menu options
dropdown_options = [
    {'label': 'Yearly Statistics', 'value': 'Yearly Statistics'},
    {'label': 'Recession Period Statistics', 'value': 'Recession Period Statistics'}
]

year_list = [i for i in range(1980, 2024, 1)]

# Create the layout of the app
app.layout = html.Div([
    html.H1('Automobile Statistics Dashboard', style={'textAlign': 'center', 'color': '#2c3e50'}),
    html.Div([
        html.Label('Select Statistics:'),
        dcc.Dropdown(
            id='select-statistics',
            options=dropdown_options,
            value='Yearly Statistics',
            placeholder='Choose statistics'
        )
    ], style={'width': '48%', 'margin': '20px auto'}),
    html.Div([
        html.Label('Select Year:'),
        dcc.Dropdown(
            id='select-year',
            options=[{'label': i, 'value': i} for i in year_list],
            value=1980,
            disabled=False
        )
    ], style={'width': '48%', 'margin': '20px auto'}),
    html.Div(id='output-container', style={'padding': '20px'})
], style={'fontFamily': 'Arial'})


@app.callback(
    Output(component_id='select-year', component_property='disabled'),
    Input(component_id='select-statistics', component_property='value')
)
def update_input_container(selected_statistics):
    if selected_statistics == 'Recession Period Statistics':
        return True
    return False


@app.callback(
    Output(component_id='output-container', component_property='children'),
    [Input(component_id='select-statistics', component_property='value'),
     Input(component_id='select-year', component_property='value')]
)
def update_output_container(selected_statistics, input_year):
    if selected_statistics == 'Recession Period Statistics':
        recession_data = data[data['Recession'] == 1]

        yearly_rec = recession_data.groupby('Year', as_index=False)['Automobile_Sales'].mean()
        R_chart1 = dcc.Graph(
            figure=px.line(
                yearly_rec,
                x='Year',
                y='Automobile_Sales',
                title='Average Automobile Sales fluctuation over Recession Period'
            )
        )

        average_sales = recession_data.groupby('Vehicle_Type', as_index=False)['Automobile_Sales'].mean()
        R_chart2 = dcc.Graph(
            figure=px.bar(
                average_sales,
                x='Vehicle_Type',
                y='Automobile_Sales',
                title='Average Automobile Sales by Vehicle Type during Recession'
            )
        )

        exp_rec = recession_data.groupby('Vehicle_Type', as_index=False)['Advertising_Expenditure'].sum()
        R_chart3 = dcc.Graph(
            figure=px.pie(
                exp_rec,
                values='Advertising_Expenditure',
                names='Vehicle_Type',
                title='Total Expenditure Share by Vehicle Type during Recessions'
            )
        )

        unemp_data = recession_data.groupby(['unemployment_rate', 'Vehicle_Type'], as_index=False)['Automobile_Sales'].mean()
        R_chart4 = dcc.Graph(
            figure=px.bar(
                unemp_data,
                x='unemployment_rate',
                y='Automobile_Sales',
                color='Vehicle_Type',
                labels={'unemployment_rate': 'Unemployment Rate', 'Automobile_Sales': 'Average Automobile Sales'},
                title='Effect of Unemployment Rate on Vehicle Type and Sales'
            )
        )

        return [
            html.Div([
                html.Div(R_chart1, style={'width': '48%', 'display': 'inline-block'}),
                html.Div(R_chart2, style={'width': '48%', 'display': 'inline-block'})
            ], style={'display': 'flex', 'flexWrap': 'wrap'}),
            html.Div([
                html.Div(R_chart3, style={'width': '48%', 'display': 'inline-block'}),
                html.Div(R_chart4, style={'width': '48%', 'display': 'inline-block'})
            ], style={'display': 'flex', 'flexWrap': 'wrap'})
        ]

    elif selected_statistics == 'Yearly Statistics' and input_year:
        yearly_data = data[data['Year'] == input_year]

        yas = data.groupby('Year', as_index=False)['Automobile_Sales'].mean()
        Y_chart1 = dcc.Graph(
            figure=px.line(
                yas,
                x='Year',
                y='Automobile_Sales',
                title='Yearly Automobile Sales'
            )
        )

        mas = yearly_data.groupby('Month', as_index=False)['Automobile_Sales'].sum()
        Y_chart2 = dcc.Graph(
            figure=px.line(
                mas,
                x='Month',
                y='Automobile_Sales',
                title='Total Monthly Automobile Sales'
            )
        )

        avr_vdata = yearly_data.groupby('Vehicle_Type', as_index=False)['Automobile_Sales'].mean()
        Y_chart3 = dcc.Graph(
            figure=px.bar(
                avr_vdata,
                x='Vehicle_Type',
                y='Automobile_Sales',
                title=f'Average Vehicles Sold by Vehicle Type in the year {input_year}'
            )
        )

        exp_data = yearly_data.groupby('Vehicle_Type', as_index=False)['Advertising_Expenditure'].sum()
        Y_chart4 = dcc.Graph(
            figure=px.pie(
                exp_data,
                values='Advertising_Expenditure',
                names='Vehicle_Type',
                title=f'Advertisement Expenditure by Vehicle Type in {input_year}'
            )
        )

        return [
            html.Div([
                html.Div(Y_chart1, style={'width': '48%', 'display': 'inline-block'}),
                html.Div(Y_chart2, style={'width': '48%', 'display': 'inline-block'})
            ], style={'display': 'flex', 'flexWrap': 'wrap'}),
            html.Div([
                html.Div(Y_chart3, style={'width': '48%', 'display': 'inline-block'}),
                html.Div(Y_chart4, style={'width': '48%', 'display': 'inline-block'})
            ], style={'display': 'flex', 'flexWrap': 'wrap'})
        ]

    return None


# Run the Dash app
if __name__ == '__main__':
    app.run(debug=True)

