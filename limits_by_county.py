import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import csv

from utils import GrowthRateConversion, LoSFilter
from hosp_bed_demand_projection import HospBedDemandProjection

grc = GrowthRateConversion()
Rt = grc.R0_ez(0.08)
hospital_capacity_file = "hospital_capacity.pkl"

alpha = 0.05
z_alpha_div_2 = scipy.stats.norm.ppf(1 - alpha / 2)

def main():
    hospital_capacity = pd.read_pickle(hospital_capacity_file)

    county_thresholds = []
    for county in hospital_capacity.index.get_level_values('COUNTY_NAME').unique():
        print(county)
        icu = hospital_capacity.index.isin([(county, 'INTENSIVE CARE')]).any()
        if icu:
            # The 0.25 avoids weird rounding issues at 0.5
            icu_capacity = np.around(hospital_capacity.at[(county, 'INTENSIVE CARE'), 'BED_CAPACITY'] / 2 + 0.25)
            projection = HospBedDemandProjection(transmission_rate = Rt, icu=True)
            icu_thresholds = projection.estimate_runway_threshold(icu_capacity, alpha=alpha / 2)
            threshold_for_icu = icu_thresholds['hospital_admissions_threshold']
        else:
            icu_capacity = np.nan
            threshold_for_icu = np.nan

        if not hospital_capacity.index.isin([(county, 'UNSPECIFIED GENERAL ACUTE CARE')]).any():
            print("issue with county {}: only bed capacity types {}".format(county, hospital_capacity.loc[(county,)].index.get_level_values('BED_CAPACITY_TYPE').tolist()))
            hosp_capacity = np.nan
            threshold_for_hosp = np.nan
        else:
            # The 0.25 avoids weird rounding issues at 0.5
            hosp_capacity = np.around(hospital_capacity.at[(county, 'UNSPECIFIED GENERAL ACUTE CARE'), 'BED_CAPACITY'] / 2 + 0.25)
            projection = HospBedDemandProjection(transmission_rate = Rt, icu=False)
            hosp_thresholds=projection.estimate_runway_threshold(hosp_capacity, alpha=alpha / 2)
            threshold_for_hosp = hosp_thresholds['hospital_admissions_threshold']

        if icu:
            # Use lower threshold unless that threshold has high risk of false positives
            # Otherwise, make sure risk of false positives is low until switching
            # entirely to the upper threshold.
            thresholds = [threshold_for_icu, threshold_for_hosp]
            if np.min(thresholds) >= 23:
                rec_threshold = np.min(thresholds)
            elif np.max(thresholds) >= 23 and np.min(thresholds) < 23:
                rec_threshold = 23
            else:
                rec_threshold = np.max(thresholds)
        else:
            rec_threshold = threshold_for_hosp

        county_thresholds.append({'county_name':county, 'icu_capacity':icu_capacity, 'threshold_for_icu':threshold_for_icu, 'hosp_capacity':hosp_capacity, 'threshold_for_hosp':threshold_for_hosp, 'recommended_threshold':rec_threshold})

    county_thresholds = pd.DataFrame(county_thresholds).round(0).replace(0.0, np.nan)
    pd.set_option('display.max_rows', county_thresholds.shape[0] + 1)
    print(county_thresholds)
    county_thresholds.to_csv('county_thresholds.csv')
    county_thresholds.to_csv('county_thresholds.tsv', sep='\t', quoting=csv.QUOTE_NONE, na_rep='-')

if __name__=='__main__':
    main()
