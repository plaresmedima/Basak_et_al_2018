# Basak_et_al_2018
This folder contains the data analysis code (fitAll.py), code libraries (function_fitting.py) and the codes for plotting figures for the  paper 'Analytical validation of single-kidney glomerular filtration rate and split renal function as measured with magnetic
resonance renography' by S. Basak et al. Magnetic Resonance Imaging 59, 53-60 (2019). 

Input: 
The signal versus time curves for artery, right and left kidneys are used as input to the code and they are stored in the subfolder ROIdata. A runtime version of the software used for analyzing MRI images is provided in the subfolder PMI-0.4-Runtime-Demo.

Results: 

The codes plot_SKGFR.py and plot_SRF_GFR.py read the output of fitAll.py and plot the results given in Figure 2 and 3, respectively. They also calculate the parameters shown in Table 3. 

The codes plot_3T_SKGFR.py and plot_3T_SRF_GFR.py read the output of fitAll.py and calculates the parameters given in Table 4.
