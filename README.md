# THz visibility prediction and AP assignment simulator

## Description

Simulator for predicting the visibility probability in THz communication networks and assigning access points (APs) to users in order to maximize network availability and reduce the reconfiguration overhead (i.e., number of changes in the serving access points).

The simulator is composed of two Python modules: THz visibility probability prediction (thz-visibility-prediction.py) and THz AP to user assignment (thz-ap-assignment.py). The input of the first module is the dataset generated by the BloTHz THz network blockage simulator ([https://doi.org/10.5281/zenodo.10681711](https://doi.org/10.5281/zenodo.10681711)).

The simulator is used in the publication: [https://doi.org/10.1016/j.comcom.2024.107956](https://doi.org/10.1016/j.comcom.2024.107956).

## Execution

	The modules must be executed sequentially, using as first input the dataset provided by the BloTHz simulator, placing the generated CSV in the "thz_datasets/OK/" path of this simulator with the name "SimData-NUM-users.csv" where NUM must be sustituted with the number of users in the simulation.
	
	Correspondingly, the values of the variables NUM_BS and NUM_USERS_ARRAY of both modules must be set accordingly to the available datasets under study, located in the "thz_datasets/OK" path.
	
	The values of the PROFILE_ARRAY and N_STEPS_ARRAY variables can also be configured in both modules, but they must have the same values for proper operation. Variable MAX_TS_ARRAY can also be configured in the THz visibility prediction module.
	
	After configuring the modules, the thz-visibility-prediction.py must be first executed. After the execution of the first module ends, the thz-ap-assignment.py must be executed.


## Results

The results are printed through the standard output stream of the THz AP to user assignment module (thz-ap-assignment.py) with the following format:

    PROFILE NUM_USERS NUM_BS N_STEPS ALGORITHM PROB_HYSTERESIS_THRESHOLD AVAILABILITY_AVG AVAILABILITY_CI95_LOW AVAILABILITY_CI95_HIGH RECONF_OVERHEAD_AVG RECONF_OVERHEAD_CI95_LOW RECONF_OVERHEAD_CI95_HIGH EXPERIMENTAL_LATENCY_AVG EXPERIMENTAL_LATENCY_CI95_LOW EXPERIMENTAL_LATENCY_CI95_HIGH

Status messages are printed through the standard error stream in order to provide information about the current status of the simulation.

## Copyright

Copyright ⓒ 2024 Pablo Fondo Ferreiro <pfondo@gti.uvigo.es>

This simulator is licensed under the GNU General Public License, version 3 (GPL-3.0). For more information see LICENSE.txt
