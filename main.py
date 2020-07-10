import csv
import sys
from functools import lru_cache

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import scipy.stats

from utils import GrowthRateConversion, LoSFilter
from hosp_bed_demand_projection import HospBedDemandProjection

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

days_to_avg = 3

app.title = "Defining a Re-shelter in Place Alarm to Guide Reopening"

app.layout = html.Div([html.H1('Defining a Re-shelter in Place Alarm to Guide Reopening'), html.P(['A calculator implementing Yadlowsky et al’s methodology proposed ', html.A('here', href='http://tinyurl.com/reshelter-alarm'), '.']),
    'Number of ICU beds available for COVID-19 patients:',
    dcc.Input(id='icu-capacity', value=146, type='number', debounce=True),
    html.Br(),
    'Number of acute care beds available for COVID-19 patients:',
    dcc.Input(id='hosp-capacity', value=1200, type='number', debounce=True),
    html.Br(),
    'Acceptable chance of failing to trigger the alarm (%):',
    dcc.Input(id='alpha', value=5, type='text', debounce=True),
    html.Br(),
    ' Number of days to respond after alarm is triggered:',
    dcc.Input(id='time-to-react', value=2, type='number', debounce=True),
    html.Br(),
    ' Your estimate of the maximum effective reproduction number possible under current re-opening policies:',
    dcc.Input(id='rt', value=1.4538, type='text', debounce=True),
    dcc.Loading(
        id="loading-1",
        type="default",
        children=[html.Div(id='thresh-div'), html.Div(id='fp-div')]
    ),
    html.Div([html.P(), html.P(['False alarm occurs when enough hospitalizations to trigger the alarm occur due to random chance in a 3 day window, in spite of only being at 60% of the concerning level.'])])
])

@lru_cache(maxsize=1024)
def get_hosp_bed_projection_model(transmission_rate, icu, detection_to_change_lag):
    print("Cache miss")
    return HospBedDemandProjection(transmission_rate=transmission_rate,
                                   icu=icu, detection_to_change_lag=detection_to_change_lag)

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
    projection = get_hosp_bed_projection_model(transmission_rate = Rt, icu=True, detection_to_change_lag = time_to_react + 1)
    icu_thresholds = projection.estimate_runway_threshold(icu_capacity, alpha=alpha / 2)
    threshold_for_icu = icu_thresholds['hospital_admissions_threshold']

    projection = get_hosp_bed_projection_model(transmission_rate = Rt, icu=False, detection_to_change_lag = time_to_react + 1)
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

    return ['Alarm should be triggered when {:.1f} new COVID-19 admissions/day are seen as an average over 3 days.'.format(threshold),
            'Chance of a false alarm: {:.1f}%'.format(100 * fpr)]

server = app.server

if __name__ == '__main__':
    app.server.run(port=8080, debug=True)
