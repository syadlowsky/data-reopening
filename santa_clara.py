import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import csv
import sys

from utils import GrowthRateConversion, LoSFilter
from hosp_bed_demand_projection import HospBedDemandProjection

grc = GrowthRateConversion()
Rt = grc.R0_ez(0.08)
print(Rt)

alpha = 0.05

results = []

non_covid_icu_needs = int(366 * 0.6)

for time_to_react in range(1, 11, 1):
    # Non-surge
    icu_capacity = 366 - non_covid_icu_needs
    hosp_capacity = 2047
    projection = HospBedDemandProjection(transmission_rate = Rt, icu=True, detection_to_change_lag = time_to_react + 1)
    icu_thresholds = projection.estimate_runway_threshold(icu_capacity, alpha=alpha / 2)
    threshold_for_icu = icu_thresholds['hospital_admissions_threshold']

    results.append({'Type': 'Non-surge', 'Time to React (days)': time_to_react, 'ICU Capacity': icu_capacity, 'Hospital Capacity': hosp_capacity, 'Threshold': threshold_for_icu})

    # Surge
    icu_capacity = 899 - non_covid_icu_needs
    hosp_capacity = 3278
    projection = HospBedDemandProjection(transmission_rate = Rt, icu=True, detection_to_change_lag = time_to_react + 1)
    icu_thresholds = projection.estimate_runway_threshold(icu_capacity, alpha=alpha / 2)
    threshold_for_icu = icu_thresholds['hospital_admissions_threshold']

    results.append({'Type': 'Surge', 'Time to React (days)': time_to_react, 'ICU Capacity': icu_capacity, 'Hospital Capacity': hosp_capacity, 'Threshold': threshold_for_icu})

results = pd.DataFrame(results)
results = results.set_index(['Time to React (days)', 'Type'])

results.to_csv(sys.stdout, sep='\t', quoting=csv.QUOTE_NONE, na_rep='-')
