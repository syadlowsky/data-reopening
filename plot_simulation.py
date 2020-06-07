import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from utils import GrowthRateConversion
from hosp_bed_demand_projection import HospBedDemandProjection

alpha = 0.05

grc = GrowthRateConversion()
Rt = grc.R0_ez(0.08)

projection = HospBedDemandProjection(transmission_rate = Rt, icu=True)
projection.plot_simulation(alpha = alpha / 2)

plt.savefig('simulation.png')
plt.show()
