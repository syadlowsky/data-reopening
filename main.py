import numpy as np
import scipy.stats
import csv
import sys

from utils import GrowthRateConversion, LoSFilter
from hosp_bed_demand_projection import HospBedDemandProjection
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

days_to_avg = 3

app.layout = html.Div([html.H1('COVID-19 Shelter-in-Place Alarm Threshold Calculator'), html.P(['Based on the methodology proposed ', html.A('here', href='tinyurl.com/reshelter-alarm')]), 'ICU Capacity for COVID-19 Patients (beds):',
    dcc.Input(id='icu-capacity', value=146, type='number', debounce=True),
    html.Br(),
    'Acute Care Capacity for COVID-19 Patients (beds):',
    dcc.Input(id='hosp-capacity', value=1200, type='number', debounce=True),
    html.Br(),
    'False Negative Probability (%):',
    dcc.Input(id='alpha', value=5, type='text', debounce=True),
    html.Br(),
    ' Time to React (days):',
    dcc.Input(id='time-to-react', value=2, type='number', debounce=True),
    html.Br(),
    ' Assumed Maximum Effective Reproduction Number:',
    dcc.Input(id='rt', value=1.4538, type='text', debounce=True),
    dcc.Loading(
        id="loading-1",
        type="default",
        children=[html.Div(id='thresh-div'), html.Div(id='fp-div')]
    ),
    html.Div([html.P(), html.P(['False positive calculations are based on assuming that the average rate of hospitalizations is 60% of what ', html.I('should'), ' trigger the alarm. It represents the probability that enough hospitalizations are observed to trigger the alarm due to randomly observing more than the average number of hospitalizations in a 3 day window.'])])
])


@app.callback(
    [Output(component_id='thresh-div', component_property='children'),
    Output(component_id='fp-div', component_property='children')],
    [Input(component_id='icu-capacity', component_property='value'),
     Input(component_id='hosp-capacity', component_property='value'),
     Input(component_id='alpha', component_property='value'),
     Input(component_id='rt', component_property='value'),
     Input(component_id='time-to-react', component_property='value')]
)
def update_output_div(icu_capacity, ac_capacity, alpha, Rt, time_to_react):
    if icu_capacity is None or ac_capacity is None or alpha is None or Rt is None or time_to_react is None:
        return ['Alarm Threshold: -- new COVID-19 admissions average over 3 days', 'False Positive Probability: --%']
    alpha = float(alpha) / 100
    Rt = float(Rt)
    time_to_react = int(time_to_react)
    hosp_capacity = ac_capacity + icu_capacity
    projection = HospBedDemandProjection(transmission_rate = Rt, icu=True, detection_to_change_lag = time_to_react + 1)
    icu_thresholds = projection.estimate_runway_threshold(icu_capacity, alpha=alpha / 2)
    threshold_for_icu = icu_thresholds['hospital_admissions_threshold']

    projection = HospBedDemandProjection(transmission_rate = Rt, icu=False, detection_to_change_lag = time_to_react + 1)
    hosp_thresholds = projection.estimate_runway_threshold(hosp_capacity, alpha=alpha / 2)
    threshold_for_hosp = hosp_thresholds['hospital_admissions_threshold']

    if threshold_for_icu < threshold_for_hosp:
        threshold = threshold_for_icu
        false_positive_hypoth = icu_thresholds['hospital_admissions_rate'] * 0.67
        fpr = 1-scipy.stats.poisson.cdf(threshold_for_icu * days_to_avg, false_positive_hypoth * days_to_avg)
    else:
        threshold = threshold_for_hosp
        false_positive_hypoth = hosp_thresholds['hospital_admissions_rate'] * 0.67
        fpr = 1-scipy.stats.poisson.cdf(threshold_for_hosp * days_to_avg, false_positive_hypoth * days_to_avg)

    return ['Alarm Threshold: {:.1f} new COVID-19 admissions average over 3 days'.format(threshold), 'False Positive Probability: {:.1f}%'.format(100 * fpr)]

if __name__ == '__main__':
    app.run_server(debug=True)
