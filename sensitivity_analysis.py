import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import csv

from utils import GrowthRateConversion, LoSFilter
from hosp_bed_demand_projection import HospBedDemandProjection


alpha = 0.05
z_alpha_div_2 = scipy.stats.norm.ppf(1 - alpha / 2)

counties = ['FRESNO', 'LOS ANGELES', 'SANTA CLARA', 'SHASTA']

def main():
    county_thresholds = pd.read_csv('county_thresholds.csv')

    grc = GrowthRateConversion()
    county_max_utilization = []
    for growth in [0.06, 0.08, 0.10, 0.15, 0.20]:
        Rt = grc.R0_ez(growth)
        icu_projection = HospBedDemandProjection(transmission_rate = Rt, icu=True)
        infections, icu_intensity, lockdown_point = icu_projection._simulate()
        hosp_projection = HospBedDemandProjection(transmission_rate = Rt, icu=False)
        infections, hospital_intensity, lockdown_point = hosp_projection._simulate()
        for county in counties:
            county_data = county_thresholds.loc[county_thresholds['county_name']==county]
            trigger = county_data['recommended_threshold'].tolist()[0]
            trigger = 0.5 * (2*trigger + z_alpha_div_2**2 + np.sqrt(4 * trigger * z_alpha_div_2**2 + z_alpha_div_2 ** 4))
            hosp_capacity = county_data['hosp_capacity'].tolist()[0]
            icu_capacity = county_data['icu_capacity'].tolist()[0]
            scale = trigger / np.mean(icu_intensity[lockdown_point - 3:lockdown_point])
            icu_intensity_cty = icu_intensity * scale
            icu_max_cty = [scipy.stats.poisson.ppf(1-alpha/2, intensity) for intensity in icu_intensity_cty]

            hospital_intensity_cty = hospital_intensity * scale
            hospital_max_cty = [scipy.stats.poisson.ppf(1-alpha/2, intensity) for intensity in hospital_intensity_cty]

            county_max_utilization.append({'County Name':county, 'Max Hospital Census':np.max(hospital_max_cty), 'Hospital Capacity':hosp_capacity, 'Max ICU Census':np.max(icu_max_cty), 'ICU Capacity':icu_capacity, 'Daily Growth Rate (%)':100*growth, 'Rt':Rt})

    county_max_utilization = pd.DataFrame(county_max_utilization).set_index(['County Name', 'Daily Growth Rate (%)']).sort_index()
    pd.set_option('display.max_rows', county_max_utilization.shape[0] + 1)
    print(county_max_utilization)
    county_max_utilization.to_csv('sensitivity_analysis.csv')
    county_max_utilization.to_csv('sensitivity_analysis.tsv', sep='\t', quoting=csv.QUOTE_NONE, na_rep='-')

if __name__=='__main__':
    main()
