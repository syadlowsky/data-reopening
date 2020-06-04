import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

from utils import GrowthRateConversion, LoSFilter
from hosp_bed_demand_projection import HospBedDemandProjection

grc = GrowthRateConversion()
Rt = grc.R0_ez(0.08)
hospital_capacity_file = "hospital_capacity.pkl"

def main():
    hospital_capacity = pd.read_pickle(hospital_capacity_file)

    county_thresholds = []
    for county in hospital_capacity.index.get_level_values('COUNTY_NAME').unique():
        icu = hospital_capacity.index.isin([(county, 'INTENSIVE CARE')]).any()
        if icu:
            icu_capacity = hospital_capacity.at[(county, 'INTENSIVE CARE'), 'BED_CAPACITY'] / 2
            projection = HospBedDemandProjection(transmission_rate = Rt, icu=True)
            icu_thresholds = projection.estimate_runway_threshold(icu_capacity)
            threshold_for_icu = icu_thresholds['hospital_admissions_threshold']
        else:
            icu_capacity = np.nan
            threshold_for_icu = np.nan

        if not hospital_capacity.index.isin([(county, 'UNSPECIFIED GENERAL ACUTE CARE')]).any():
            print("issue with county {}: only bed capacity types {}".format(county, hospital_capacity.loc[(county,)].index.get_level_values('BED_CAPACITY_TYPE').tolist()))
            hosp_capacity = np.nan
            threshold_for_hosp = np.nan
        else:
            hosp_capacity = hospital_capacity.at[(county, 'UNSPECIFIED GENERAL ACUTE CARE'), 'BED_CAPACITY'] / 2
            projection = HospBedDemandProjection(transmission_rate = Rt, icu=False)
            threshold_for_hosp = projection.estimate_runway_threshold(hosp_capacity)['hospital_admissions_threshold']


        if icu and threshold_for_icu >= 23:
            rec_threshold = threshold_for_icu
        else:
            rec_threshold = threshold_for_hosp
        county_thresholds.append({'county_name':county, 'icu_capacity':icu_capacity, 'threshold_for_icu':threshold_for_icu, 'hosp_capacity':hosp_capacity, 'threshold_for_hosp':threshold_for_hosp, 'recommended_threshold':rec_threshold})

    county_thresholds = pd.DataFrame(county_thresholds).round(0)
    pd.set_option('display.max_rows', county_thresholds.shape[0] + 1)
    print(county_thresholds)
    county_thresholds.to_csv('county_thresholds.csv')
    county_thresholds.to_csv('county_thresholds.tsv', sep='\t', quoting=csv.QUOTE_NONE, na_rep='-')

if __name__=='__main__':
    main()
