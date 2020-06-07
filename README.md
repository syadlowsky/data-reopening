## Running things

### To get hospital bed data for California
Run `python get_hospital_counts.py`

### To get thresholds per county
Make sure you have up to date hospital bed data.
Run `python limits_by_county.py`

### To get figures
Simulation example is from `plot_simulation.py`
Length of stay plots are from `plot_los.py`

### Main simulation
The main code for the simulation is in `hosp_bed_demand_projection.py`.
Specifically, the forward simulation is in the `_simulate` method. This
code also has the logic for creating the confidence intervals to deal with
the statistical noise. `utils.py` contains some of the filters needed for
 running the simulation.

### Sensitivity analysis
The sensitivity analyses with running forward simulations under different
growth rates is in `sensitivity_analysis.py`.
