import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from utils import LoSFilter

los_hospitals = np.concatenate(((1,), LoSFilter().los_cdf[:, 1]))
los_icu = np.concatenate(((1,), LoSFilter('los_icu.csv').los_cdf[:, 1]))
day_hospitals = list(range(0, len(los_hospitals)))
day_icu = list(range(0, len(los_icu)))

fig, axs = plt.subplots(2)
axs[0].set_xlim(0, 31)
axs[1].set_xlim(0, 31)
axs[0].set_ylim(0, 1)
axs[1].set_ylim(0, 1)
axs[0].step(day_hospitals, los_hospitals, where='post')
axs[1].step(day_icu, los_icu, where='post')

axs[0].set_xlabel("t (days)")
axs[1].set_xlabel("t (days)")
axs[0].set_ylabel("P(Length of Stay <= t)")
axs[1].set_ylabel("P(Length of Stay <= t)")

axs[0].set_title("ICU + Acute Care Length of Stay CDF")
axs[1].set_title("ICU Length of Stay CDF")

fig.tight_layout()

plt.savefig('length_of_stay.png')
plt.show()
