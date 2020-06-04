import numpy as np
import warnings
import sys

from utils import GrowthRateConversion, LoSFilter

class ICUProjection(object):
    def __init__(self,
                 initial_daily_infection_rate = 1.0,
                 infection_generation_filter = None,
                 transmission_rate=1.5,
                 lockdown_transmission_rate=0.85,
                 detection_to_change_lag = 2,
                 infection_icu_lag_filter = None,
                 admission_probability = 0.02,
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
        self.admission_probability = admission_probability
        self.icu_hospitalization_fraction = icu_hospitalization_fraction

        # Assume limitinig factor is ICU (alternative is hospitaliizations)
        self.icu = icu
        self._set_los_filter_with_default(los_filter)

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
        print(new_icu_intensity_at_day)
        icu_demand = self.los_filter.apply_filter(new_icu_intensity_at_day)
        print(np.argmax(icu_demand) - lockdown_point)
        return np.max(icu_demand) / np.mean(new_icu_intensity_at_day[lockdown_point - 3:lockdown_point])

    def estimate_runway_threshold(self, capacity_max, z_alpha = 1.64, days_to_avg = 3.0):
        capacity = 0.5 * (2*capacity_max + z_alpha**2 - np.sqrt(4*capacity_max*(z_alpha**2) + z_alpha**4))
        if self.icu:
            icu_adm_rate = float(capacity) / self._estimate_max_capacity_multiplier()
            hosp_adm_rate = icu_adm_rate / self.icu_hospitalization_fraction
        else:
            hosp_adm_rate = float(capacity) / self._estimate_max_capacity_multiplier()
            icu_adm_rate = hosp_adm_rate * self.icu_hospitalization_fraction
        icu_adm_threshold = icu_adm_rate - z_alpha * np.sqrt(icu_adm_rate / days_to_avg)
        hosp_adm_threshold = hosp_adm_rate - z_alpha * np.sqrt(hosp_adm_rate / days_to_avg)
        return {
            'icu_admissions_rate' : icu_adm_rate,
            'hospital_admissions_rate' : hosp_adm_rate,
            'icu_admissions_threshold' : icu_adm_threshold,
            'hospital_admissions_threshold' : hosp_adm_threshold
        }

def test():
    import matplotlib.pyplot as plt
    projection = ICUProjection()
    #print(projection._simulate())

    max_growth_list = np.arange(0.03, 0.15, 0.01)
    grc = GrowthRateConversion()
    print(grc.R0_ez(0.08))
    max_R_t_list = np.array([grc.R0_ez(g) for g in max_growth_list])

    #max_R_t_list = np.arange(1.2, 2.11, 0.1)
    #grc = GrowthRateConversion()
    #max_growth_list = np.array([grc.growth_rate(R0) for R0 in max_R_t_list])

    hospital_admissions_threshold = []
    for max_R_t in max_R_t_list:
        projection = ICUProjection(transmission_rate = max_R_t, icu=False)
        hospital_admissions_threshold.append(projection.estimate_runway_threshold(250)['hospital_admissions_threshold'])

    plt.plot(max_growth_list, hospital_admissions_threshold)
    plt.xlabel("Worst Case Daily Growth Rate")
    plt.ylabel("3-day Avg Hospital Admissions / day (per 1000 beds)")
    plt.title("Worst Case Growth Rate vs Allowable Hospital Admissions per Day")
    plt.plot([0.04], [hospital_admissions_threshold[1]], 'sk')
    plt.text(0.04 + 0.005, hospital_admissions_threshold[1], "{:.1f}".format(hospital_admissions_threshold[1]), verticalalignment='bottom', horizontalalignment='left')
    plt.plot([0.06], [hospital_admissions_threshold[3]], 'sk')
    plt.text(0.06 + 0.005, hospital_admissions_threshold[3], "{:.1f}".format(hospital_admissions_threshold[3]), verticalalignment='bottom', horizontalalignment='left')
    plt.plot([0.08], [hospital_admissions_threshold[5]], 'sk')
    plt.text(0.08 + 0.005, hospital_admissions_threshold[5], "{:.1f}".format(hospital_admissions_threshold[5]), verticalalignment='bottom', horizontalalignment='left')
    plt.plot([0.1], [hospital_admissions_threshold[7]], 'sk')
    plt.text(0.1 + 0.005, hospital_admissions_threshold[7], "{:.1f}".format(hospital_admissions_threshold[7]), verticalalignment='bottom', horizontalalignment='left')
    plt.plot([0.12], [hospital_admissions_threshold[9]], 'sk')
    plt.text(0.12 + 0.005, hospital_admissions_threshold[9], "{:.1f}".format(hospital_admissions_threshold[9]), verticalalignment='bottom', horizontalalignment='left')
    #plt.xlim(1.0, 2.1)
    plt.savefig("trigger_hosp.png")
    plt.show()

    max_R_t_list = np.arange(1.2, 2.11, 0.1)

    hospital_admissions_threshold = []
    for max_R_t in max_R_t_list:
        projection = ICUProjection(transmission_rate = max_R_t)
        hospital_admissions_threshold.append(projection.estimate_runway_threshold(250)['hospital_admissions_threshold'])

    plt.plot(max_R_t_list, hospital_admissions_threshold)
    plt.xlabel("Worst Case R(t)")
    plt.ylabel("3-day Avg Hospital Admissions / day (per 1000 beds)")
    plt.title("Worst Case R(t) vs Allowable Hospital Admissions per Day")
    plt.plot([1.3], [hospital_admissions_threshold[1]], 'sk')
    plt.text(1.3 + 0.02, hospital_admissions_threshold[1], "{:.1f}".format(hospital_admissions_threshold[1]), verticalalignment='bottom', horizontalalignment='left')
    plt.plot([1.4], [hospital_admissions_threshold[2]], 'sk')
    plt.text(1.4 + 0.02, hospital_admissions_threshold[2], "{:.1f}".format(hospital_admissions_threshold[2]), verticalalignment='bottom', horizontalalignment='left')
    plt.plot([1.5], [hospital_admissions_threshold[3]], 'sk')
    plt.text(1.5 + 0.02, hospital_admissions_threshold[3], "{:.1f}".format(hospital_admissions_threshold[3]), verticalalignment='bottom', horizontalalignment='left')
    plt.plot([1.6], [hospital_admissions_threshold[4]], 'sk')
    plt.text(1.6 + 0.02, hospital_admissions_threshold[4], "{:.1f}".format(hospital_admissions_threshold[4]), verticalalignment='bottom', horizontalalignment='left')
    plt.plot([1.7], [hospital_admissions_threshold[5]], 'sk')
    plt.text(1.7 + 0.02, hospital_admissions_threshold[5], "{:.1f}".format(hospital_admissions_threshold[5]), verticalalignment='bottom', horizontalalignment='left')
    plt.plot([1.8], [hospital_admissions_threshold[6]], 'sk')
    plt.text(1.8 + 0.02, hospital_admissions_threshold[6], "{:.1f}".format(hospital_admissions_threshold[6]), verticalalignment='bottom', horizontalalignment='left')
    plt.plot([1.9], [hospital_admissions_threshold[7]], 'sk')
    plt.text(1.9 + 0.02, hospital_admissions_threshold[7], "{:.1f}".format(hospital_admissions_threshold[7]), verticalalignment='bottom', horizontalalignment='left')
    plt.plot([2.0], [hospital_admissions_threshold[-2]], 'sk')
    plt.text(2.0 + 0.02, hospital_admissions_threshold[-2], "{:.1f}".format(hospital_admissions_threshold[-2]), verticalalignment='bottom', horizontalalignment='left')
    #plt.xlim(1.0, 2.1)
    plt.savefig("trigger_hosp_Rt.png")
    plt.show()

if __name__=='__main__':
    test()
