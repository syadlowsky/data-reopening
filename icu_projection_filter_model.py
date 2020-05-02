import numpy as np
import warnings

class ICUProjection(object):
    def __init__(self,
                 initial_daily_infection_rate = 1.0,
                 infection_generation_filter = None,
                 transmission_rate=1.5,
                 lockdown_transmission_rate=0.85,
                 detection_to_change_lag = 2,
                 infection_icu_lag_filter = None,
                 icu_admission_probability = 0.02):
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
        self.icu_admission_probability = icu_admission_probability

    def _set_infection_generation_filter_with_default(self, infection_generation_filter=None):
        if infection_generation_filter is None:
            infection_generation_filter = self._default_infection_generation_filter()
        if np.abs(infection_generation_filter.sum() - 1) > 1e-4:
            warnings.warn("infection_generation_filter should sum to 1, instead it sums to {:4f}".format(infection_generation_filter.sum()))
        self.infection_generation_filter = infection_generation_filter

    def _default_infection_generation_filter(self):
        return np.concatenate((np.zeros(1), np.ones(9) / 9.0))

    def _set_infection_icu_lag_filter_with_default(self, infection_icu_lag_filter=None):
        if infection_icu_lag_filter is None:
            infection_icu_lag_filter = self._default_infection_icu_lag_filter()
        if np.abs(infection_icu_lag_filter.sum() - 1) > 1e-4:
            warnings.warn("infection_icu_lag_filter should sum to 1, instead it sums to {:4f}".format(infection_icu_lag_filter.sum()))
        self.infection_icu_lag_filter = infection_icu_lag_filter

    def _default_infection_icu_lag_filter(self):
        return np.array([0, 0, 0, 0, 0, 0, 0.1, 0.13, 0.15, 0.11, 0.1, 0.1, 0.09, 0.07, 0.06, 0.04, 0.03, 0.02])

    def _simulate(self):
        gen_filter_len = self.infection_generation_filter.shape[0]
        icu_filter_len = self.infection_icu_lag_filter.shape[0]
        filter_length = gen_filter_len + icu_filter_len
        flipped_infection_generation_filter = self.infection_generation_filter[::-1]
        infections = np.ones(filter_length + self.detection_to_change_lag)
        for i in range(gen_filter_len):
            infections = np.append(infections, self.transmission_rate * infections[-gen_filter_len:].dot(flipped_infection_generation_filter))
        lockdown_point = infections.shape[0] - self.detection_to_change_lag
        for i in range(icu_filter_len):
            infections = np.append(infections, self.lockdown_transmission_rate * infections[-gen_filter_len:].dot(flipped_infection_generation_filter))
        icu_intensity = self.icu_admission_probability * np.ones(icu_filter_len)
        flipped_icu_filter = self.infection_icu_lag_filter[::-1]
        for i in range(infections.shape[0] - icu_filter_len):
            icu_intensity = np.append(icu_intensity, self.icu_admission_probability * infections[i:i+icu_filter_len].dot(flipped_icu_filter))
        return infections, icu_intensity, lockdown_point

    def _estimate_max_capacity_multiplier(self, integration_time):
        _, new_icu_intensity_at_day, lockdown_point = self._simulate()
        diffs = np.cumsum(new_icu_intensity_at_day)
        diffs[integration_time:] = diffs[integration_time:] - diffs[:-integration_time]
        diffs = diffs[integration_time - 1:]
        return np.max(diffs) / np.mean(new_icu_intensity_at_day[lockdown_point - 3:lockdown_point])

    def estimate_runway_threshold(self, icu_capacity):
        return {
            'icu_admissions_per_day' : float(icu_capacity) / self._estimate_max_capacity_multiplier(14),
        }

def test():
    import matplotlib.pyplot as plt
    projection = ICUProjection()
    print(projection._simulate())

    max_R_t_list = np.arange(1.2, 2.11, 0.1)
    icu_admissions_per_day = []
    for max_R_t in max_R_t_list:
        projection = ICUProjection(transmission_rate = max_R_t)
        icu_admissions_per_day.append(projection.estimate_runway_threshold(250)['icu_admissions_per_day'])
    plt.plot(max_R_t_list, icu_admissions_per_day)
    plt.xlabel("Worst Case R(t)")
    plt.ylabel("ICU Admissions / day (per 250 ICU beds)")
    plt.title("Worst Case R(t) vs Allowable ICU Admissions per Day")
    plt.plot([1.6], [icu_admissions_per_day[4]], 'sk')
    plt.text(1.6 + 0.02, icu_admissions_per_day[4], "{:.1f}".format(icu_admissions_per_day[4]), verticalalignment='bottom', horizontalalignment='left')
    plt.plot([1.8], [icu_admissions_per_day[6]], 'sk')
    plt.text(1.8 + 0.02, icu_admissions_per_day[6], "{:.1f}".format(icu_admissions_per_day[6]), verticalalignment='bottom', horizontalalignment='left')
    plt.plot([2.0], [icu_admissions_per_day[-2]], 'sk')
    plt.text(2.0 + 0.02, icu_admissions_per_day[-2], "{:.1f}".format(icu_admissions_per_day[-2]), verticalalignment='bottom', horizontalalignment='left')
    #plt.xlim(1.0, 2.1)
    plt.savefig("trigger_2.png")
    plt.show()

if __name__=='__main__':
    test()
