import numpy as np
import matplotlib.pyplot as plt

from utils import GrowthRateConversion, LoSFilter
from hosp_bed_demand_projection import HospBedDemandProjection

def main():
    max_growth_list = np.arange(0.03, 0.15, 0.01)
    grc = GrowthRateConversion()
    print(grc.R0_ez(0.08))
    max_R_t_list = np.array([grc.R0_ez(g) for g in max_growth_list])

    #max_R_t_list = np.arange(1.2, 2.11, 0.1)
    #grc = GrowthRateConversion()
    #max_growth_list = np.array([grc.growth_rate(R0) for R0 in max_R_t_list])

    hospital_admissions_threshold = []
    for max_R_t in max_R_t_list:
        projection = HospBedDemandProjection(transmission_rate = max_R_t, icu=True)
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
        projection = HospBedDemandProjection(transmission_rate = max_R_t)
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
    main()
