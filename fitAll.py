# This code reads the signal (S) versus time(t) curve for arterial input function (aif), 
#right (RK) and left kidney (LK) ROIs and fits them using a two compartment filtration model (2CFM).
#Theree different models are used for signal to concentration conversion. 
# The 'model' variable refers to the model chosen for signal to concentration conversion and it could be 
#chosen as 'Linear' or 'Nonlinear' or 'Delay'.
#'Linear': tracer concentration (C) is proportional to S-S0 where S0 is
#the precontrast signal. 
#'Nonlinear': tracer concentration (C) is calculated incorporating the noninearity
#between S and C by inverting the steady-state spoiled gradient echo signal model. 
#This is coded in the function Concentration_SPGRESS.
#'Delay': tracer concentration (C) is  proportional to S-S0 where S0 is
#the precontrast signal and the delay in the arrival of bolus from artery to the tissue of interest 
#has been incorporated using the function 'add_delay'.

#Input data: Three files per patient (aif, LK, RK) for S-t curve, One file per study containing 
#functional volume of kidneys, One file per study containing reference GFR values. 
#These files are stored in the folder 'ROIdata'.

#Output data: One file per study containing all the fitting parameters such as
#blood flow, blood mean transit time (MTT), blood volume, tubular MTT, exraction franction,
#tubular flow, GFR and isotope GFR.

import os
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import glob
import re
from function_fitting import model_2CFM, fun_2CFM, model_uptake, fun_uptake,Concentration_SPGRESS,add_delay

## Parameters ##

P = [0.3, 125.0/6000, 2.0/3, 12/120.] #[VP+VE, FP, VE/(VP+VE), PS/FP] Tissue parameters for initilizing model fitting
Hct = 0.45 #Haematocrit factor
bL = 9 # Baseline, default=9
r = 3.6 # Relaxivity in Hz/mM, used in function Concentration_SPGRESS
s = [-10,10,0.5] # Delay values: minimum, maximum and step size, used in function add_delay

study = [ 'MRH008_wellcome','PANDA','NKRF','1Tdata'] # Study groups #
path_data = 'C:/Users/medsbasa/Work/Data_Analysis/Data_Source'
model =  'Linear'#'Nonlinear'# 'Delay'#

## Loop over study groups ##
for d1 in study:
    P_fit =[]
    params =[]
    study_id =[]
    kidney = []
    name=[]
    
    ## Parameter values for 'Nonlinear' model, used in function Concentration_SPGRESS  ####
    if d1=='1Tdata':
        FA = 20.00 #Flip angle in degrees
        TR = 5.40 #Repetition time in msec
        R10_kid = 1/906 # Precontrast relaxation rate in kidney (de Bazelaire JMRI 2004)
        R10_aif = 1/1574.42# Precontrast relaxation rate in artery (Reference in Lu MRM 2004)
    else:    
        FA = 17.00
        TR = 5.05 
        R10_kid =  1/1142   # Precontrast relaxation rate in kidney (de Bazelaire JMRI 2004)
        R10_aif = (0.52 * Hct + 0.38)/1000 # Precontrast relaxation rate in artery (Lu MRM 2004)
    T10_kid = 1/R10_kid
    T10_aif = 1/R10_aif   
    
    ## Reading input files for functional volumes of kidneys ##
    Z = np.genfromtxt('%s_Vol.csv' %d1, dtype=None, delimiter=',',names=True, unpack = True) 
    patient = np.array([x.decode() for x in Z['Patient']])
    
    ## Reading input files for reference GFR values ##
    Ziso = np.genfromtxt('%s_IsoGFR.csv' %d1, dtype=None, delimiter=',',names=True, unpack = True) 
    patient_iso = np.array([x.decode() for x in Ziso['Patient']])
    
    os.chdir(path_data)
    
    ## Reading filenames for S-t curve
    for filename in glob.glob('%s_*' %d1):    
        if filename in glob.glob('*_aif.txt'):
            m = re.match(r"%s_(.*)\_aif" %d1,filename)
            name.append(m.group(1))  
            
    ## Loop over patients within study group ##            
    for n in name:
        n = n.rstrip()
        file_aif = "%s_%s_aif.txt" % (d1, n)
        file_LK = "%s_%s_LK.txt"  % (d1, n)
        file_RK = "%s_%s_RK.txt"  % (d1, n) 
        file_id = [file_LK, file_RK]
        kidney_id = ['LK','RK']
        print(n)
        
        ## Fixing patient-specific parameters ##
        if n == 'EA 14a':
            bL = 5
        elif n == 'MRH008-11':
            bL = 10
        elif n == 'KA 58':
            if model == 'Linear':
                P[3] = 1.2/120
            else:
                P[3] = 12/120 
            bL = 21 
        else:
            P[3] = 12/120
            bL = 9
        
        ## Functional volume of kidneys corresponding to patient n ##                    
        ind = np.where(patient == '%s' %n)               
        if len(ind[0]) == 2:
            Vol_kid = [Z['Volume'][ind[0][0]], Z['Volume'][ind[0][1]] ]
        else:
            Vol_kid = [Z['Volume'][ind[0][0]], Z['Volume'][ind[0][0]]]
        
        ## Reference GFR values corresponding to patient n ## 
        ind_iso = np.where(patient_iso == '%s' %n)                
        if len(ind_iso[0]) == 2:
            isoGFR_kid = [Ziso['IsoGFR'][ind_iso[0][0]], Ziso['IsoGFR'][ind_iso[0][1]]]
        else:
            isoGFR_kid = [Ziso['IsoGFR'][ind_iso[0][0]], Ziso['IsoGFR'][ind_iso[0][0]]]
        
        ## Reading aif file ##
        with open(file_aif,'r') as f:
            textfile_temp = f.read()
            a1 = textfile_temp.split('Y-values')[1]   
            a0 = a1.split()
            aif = np.asarray(a0,dtype=np.float64)  
        
        ## Loop for two kidneys ##
        for ii in range(0,2):        
                
            if Path(file_id[ii]).exists():
                with open(file_id[ii], 'r') as f:
                    textfile_temp = f.read()
                    t1 = textfile_temp.split('X-values')[1].split('Y-values')[0]
                    r1 = textfile_temp.split('Y-values')[1]
                Time0 = t1.split()
                Time = np.asarray(Time0,dtype=np.float64)
                r0 = r1.split()
                roi = np.asarray(r0,dtype=np.float64)
                
                roi0 = np.mean(roi[0:bL])
                aif0 =  np.mean(aif[0:bL])
                roi_new = roi-roi0  # Signal for kidney after baseline correction
                aif_new = (aif-aif0)/(1-Hct) # AIF after baseline and haematocrit correction
                
                ## Fitting concentration with chosen model ##
                if model == 'Linear':
                    if n=='AC 51a' and kidney_id[ii]== 'RK': # 2Cuptake model is used for one patient 
                        res = least_squares(fun_uptake, P, method='lm',  args=(Time, roi_new,aif_new), max_nfev=20000, verbose=1)       
                        fit = model_uptake(res.x,Time, aif_new)
                    else:  
                        res = least_squares(fun_2CFM, P, method='lm',  args=(Time, roi_new,aif_new), max_nfev=20000, verbose=1)       
                        fit = model_2CFM(res.x,Time, aif_new)
                    
                elif model == 'Nonlinear':
                    aif_nonlinear = Concentration_SPGRESS(aif, aif0, T10_aif, FA, TR, r)/(1-Hct)
                    roi_nonlinear = Concentration_SPGRESS(roi, roi0, T10_kid, FA, TR, r)
                    if n=='AC 51a' and kidney_id[ii]== 'RK': # 2Cuptake model is used for one patient
                        res = least_squares(fun_uptake, P, method='lm',  args=(Time, roi_nonlinear,aif_nonlinear), max_nfev=20000,verbose=1) 
                        fit = model_uptake(res.x,Time, aif_nonlinear)
                    else:
                        res = least_squares(fun_2CFM, P, method='lm',  args=(Time, roi_nonlinear,aif_nonlinear),max_nfev=20000, verbose=1) 
                        fit = model_2CFM(res.x,Time, aif_nonlinear)
                    
                elif model == 'Delay':
                    res, fit = add_delay(P,Time, aif_new,roi_new,s,n,kidney_id[ii])
                    
                ## Calculate parameters from fitting ##
                BF = 6000*res.x[1]/(1 -Hct)	#Blood Flow
                BMTT = 1*res.x[0]*(1-res.x[2])/res.x[1] #Blood MTT
                BV = 100*res.x[0]*(1-res.x[2])/(1 -Hct) #Blood Volume
                TMTT = 1*res.x[0]*res.x[2]/(res.x[1]*res.x[3])/60 #Tubular MTT
                EF = 100*res.x[3] #Extraction Fraction
                TF = 6000*res.x[1]*res.x[3]	#Tubular Flow
                GFR = Vol_kid[ii]*60*res.x[1]*res.x[3] 
                fVol = Vol_kid[ii] # Region Volume
                chiSq = 100*np.sum((fit -roi_new)**2)/np.sum(roi_new**2)
                isoGFR = isoGFR_kid[ii]
                
                P_fit.append([BF,BMTT,BV,TMTT,EF,TF,isoGFR,GFR,fVol,chiSq])
                params.append(["BF","BMTT","BV","TMTT","E","TF","Iso-SK-GFR","SK-GFR","fVol","Chi-square"])
                kidney.append(np.repeat('%s' %kidney_id[ii],10))
                study_id.append(np.repeat('%s' %n,10))              
   
#                plt.figure()
#                plt.plot(Time,roi_new,'o')#,markerfacecolor='None')
#                plt.plot(Time,fit,'r')
                #plt.text(150, 0.01,'Linear_%s %s' % (n,kidney_id))
                       
    os.chdir('..')                        
       
    P_fit = np.ravel(P_fit)
    params = np.ravel(params)
    study_id = np.ravel(study_id)        
    kidney = np.ravel(kidney)   
    access = np.ones(len(P_fit))
    accessSF = np.ones(len(P_fit))
    accessSF[np.where((study_id == 'MRH008-15'))[0]] = 0.0 #Patient with one kidney, not included in SRF calculation
    accessSF[np.where((study_id == 'Ha21'))[0]] = 0.0  #Patient with one kidney, not included in SRF calculation
    accessSF[np.where((study_id == 'Ra40'))[0]] = 0.0  #Patient with one kidney, not included in SRF calculation
    ab = np.column_stack((study_id,kidney,params,P_fit,access,accessSF))
    header = np.array(['Patient', 'Kidney', 'Parameter','Value_%s' %model,'Access','AccessSF'])
    ab1 = np.vstack((header,ab))
    ## Save parameters (one file per study group) ## 
    np.savetxt('%s_%s.csv' %(d1,model), ab1, fmt="%s",delimiter = ",")

   
        
        