import numpy as np
import warnings
import scipy.stats

from utils import GrowthRateConversion, LoSFilter

class HospBedDemandProjection(object):
    def __init__(self,
                 initial_daily_infection_rate = 1.0,
                 infection_generation_filter = None,
                 transmission_rate=1.5,
                 lockdown_transmission_rate=0.85,
                 detection_to_change_lag = 2,
                 infection_icu_lag_filter = None,
                 admission_probability = 0.05,
                 los_filter = None,
                 icu = True,
                 icu_hospitalization_fraction = 0.15):
        if lockdown_transmission_rate > 1:
            raise Exception("Lockdown transmission rate is above one--healthcare demand will continue to rise")
        if lockdown_transmission_rate < 1.0 / transmission_rate:
            warnings.warn("This may not work very well under the assumption that things decay faster " + \
                  "after lockdown than before they start, because it depends on initial conditions.")

        self.initial_daily_infection_rate = initial_daily_infection_rate
        self._set_infection_generation_filter_with_default(infection_generation_filter)

        self.transmission_rate = transmission_rate
        self.lockdown_transmission_rate = lockdown_transmission_rate
        self.detection_to_change_lag = detection_to_change_lag

        self._set_infection_icu_lag_filter_with_default(infection_icu_lag_filter)

        # Assume limitinig factor is ICU (alternative is hospitaliizations)
        self.icu = icu
        self._set_los_filter_with_default(los_filter)

        if icu:
            self.admission_probability = icu_hospitalization_fraction * admission_probability
        else:
            self.admission_probability = admission_probability
        self.icu_hospitalization_fraction = icu_hospitalization_fraction

    def _set_los_filter_with_default(self, los_filter=None):
        if los_filter is None:
            if self.icu:
                los_filter = LoSFilter("los_icu.csv")
            else:
                los_filter = LoSFilter()
        self.los_filter = los_filter

    def _set_infection_generation_filter_with_default(self, infection_generation_filter=None):
        if infection_generation_filter is None:
            infection_generation_filter = self._default_infection_generation_filter()
        if np.abs(infection_generation_filter.sum() - 1) > 1e-4:
            warnings.warn("infection_generation_filter should sum to 1, instead it sums to {:4f}".format(infection_generation_filter.sum()))
        self.infection_generation_filter = infection_generation_filter

    def _default_infection_generation_filter(self):
        grc = GrowthRateConversion()
        return grc.filter()

    def _set_infection_icu_lag_filter_with_default(self, infection_icu_lag_filter=None):
        if infection_icu_lag_filter is None:
            infection_icu_lag_filter = self._default_infection_icu_lag_filter()
        if np.abs(infection_icu_lag_filter.sum() - 1) > 1e-4:
            warnings.warn("infection_icu_lag_filter should sum to 1, instead it sums to {:4f}".format(infection_icu_lag_filter.sum()))
        self.infection_icu_lag_filter = infection_icu_lag_filter

    def _default_infection_icu_lag_filter(self):
        return np.concatenate((np.zeros(5), np.ones(19) / 19.0))

    def _simulate(self):
        gen_filter_len = self.infection_generation_filter.shape[0]
        icu_filter_len = self.infection_icu_lag_filter.shape[0]
        filter_length = gen_filter_len + icu_filter_len
        flipped_infection_generation_filter = self.infection_generation_filter[::-1]
        infections = np.ones(filter_length)
        for i in range(gen_filter_len + self.detection_to_change_lag):
            infections = np.append(infections, self.transmission_rate * infections[-gen_filter_len:].dot(flipped_infection_generation_filter))
        lockdown_point = infections.shape[0] - self.detection_to_change_lag
        for i in range(icu_filter_len + 29):
            infections = np.append(infections, self.lockdown_transmission_rate * infections[-gen_filter_len:].dot(flipped_infection_generation_filter))
        icu_intensity = self.admission_probability * np.ones(icu_filter_len)
        flipped_icu_filter = self.infection_icu_lag_filter[::-1]
        for i in range(infections.shape[0] - icu_filter_len):
           icu_intensity = np.append(icu_intensity, self.admission_probability * infections[i:i+icu_filter_len].dot(flipped_icu_filter))
        return infections, icu_intensity, lockdown_point

    def _estimate_max_capacity_multiplier(self):
        _, new_icu_intensity_at_day, lockdown_point = self._simulate()
        icu_demand = self.los_filter.apply_filter(new_icu_intensity_at_day)
        return np.max(icu_demand) / np.mean(new_icu_intensity_at_day[lockdown_point - 3:lockdown_point])

    def estimate_runway_threshold(self, capacity_max, alpha=0.05, days_to_avg = 3.0):
        # computes intensity such that, if we stay below this intensity, with probability 1-alpha we will not exceed hospital capacity
        max_intensity = None
        for intensity in np.linspace(0.1, capacity_max, 40000)[::-1]:
            if scipy.stats.poisson.ppf(1-alpha, intensity) <= capacity_max:
                max_intensity = intensity
                break


        # backs out admission rate that yields this max_intensity, based on forward simulation of dynamics model
        if self.icu:
            icu_adm_rate = float(max_intensity) / self._estimate_max_capacity_multiplier()
            hosp_adm_rate = icu_adm_rate / self.icu_hospitalization_fraction
        else:
            hosp_adm_rate = float(max_intensity) / self._estimate_max_capacity_multiplier()
            icu_adm_rate = hosp_adm_rate * self.icu_hospitalization_fraction

        # need to pick threshold such that we will exceed the threshold w.h.p. assuming the rate is too high
        icu_adm_threshold = scipy.stats.poisson.ppf(alpha, icu_adm_rate * days_to_avg) / days_to_avg
        hosp_adm_threshold = scipy.stats.poisson.ppf(alpha, hosp_adm_rate * days_to_avg) / days_to_avg
        return {
            'icu_admissions_rate' : icu_adm_rate,
            'hospital_admissions_rate' : hosp_adm_rate,
            'icu_admissions_threshold' : icu_adm_threshold,
            'hospital_admissions_threshold' : hosp_adm_threshold
        }

def test():
    projection = HospBedDemandProjection()
    print(projection._simulate())

if __name__=='__main__':
    test()
