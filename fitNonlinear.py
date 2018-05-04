# This code reads the signal (S) versus time(t) curve for arterial input function (aif), 
#right (RK) and left kidney (LK) ROIs and fits them using a two compartment filtration model (2CFM).
#Here the tracer concentration (C) is calculated incorporating the noninearity
#between S and C. 
#Input data: Three files (aif, LK, RK) for S-t curve per patient, One file per study containing 
#functional volume of kidneys, One file per study containing reference GFR values. 
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
from function_fitting import model_2CFM, fun_2CFM, model_uptake, fun_uptake, Concentration_SPGRESS

study = ['MRH008_wellcome','PANDA','NKRF']
P = [0.3, 125.0/6000, 2.0/3, 12/120.] #[VP+VE, FP, VE/(VP+VE), PS/FP], Tissue parameters
Hct = 0.45 # Haematocrit factor 
r = 3.6 # Relaxivity in Hz/mM
bL = 9 # Baseline, default=9
path_data = 'C:/Users/medsbasa/Work/Data_Analysis/Data_Source'

for d1 in study:
    P_fit =[]
    params =[]
    study_id =[]
    kidney = []
    name=[]
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
       
    Z = np.genfromtxt('%s_Vol.csv' %d1, dtype=None, delimiter=',',names=True, unpack = True) 
    patient = np.array([x.decode() for x in Z['Patient']])
    
    Ziso = np.genfromtxt('%s_IsoGFR.csv' %d1, dtype=None, delimiter=',',names=True, unpack = True) 
    patient_iso = np.array([x.decode() for x in Ziso['Patient']])
    
    os.chdir(path_data)
    for filename in glob.glob('%s_*' %d1):    
        if filename in glob.glob('*_aif.txt'):
            m = re.match(r"%s_(.*)\_aif" %d1,filename)
            name.append(m.group(1))
    
    for n in name:
        n = n.rstrip()
        file_aif = "%s_%s_aif.txt" % (d1, n)
        file_LK = "%s_%s_LK.txt"  % (d1, n)
        file_RK = "%s_%s_RK.txt"  % (d1, n)  
        print(n)
        
        if n == 'EA 14a':
            bL = 5
        elif n == 'MRH008-11':
            bL = 10
        elif n == 'KA 58':
            #P[3] = 1.2/120
            bL = 21 
        else:
            P[3] = 12/120
            bL = 9
             
        if Path(file_aif).exists(): 
            ind = np.where(patient == '%s' %n)
                        
            if len(ind[0]) == 2:
                Vol_LK =  Z['Volume'][ind[0][0]]   
                Vol_RK =  Z['Volume'][ind[0][1]] 
            else:
                Vol_LK =  Z['Volume'][ind[0][0]]   
                Vol_RK =  Z['Volume'][ind[0][0]] 
                
            ind_iso = np.where(patient_iso == '%s' %n)
                    
            if len(ind_iso[0]) == 2:
                isoGFR_LK =  Ziso['IsoGFR'][ind_iso[0][0]]   
                isoGFR_RK =  Ziso['IsoGFR'][ind_iso[0][1]] 
            else:
                isoGFR_LK =  Ziso['IsoGFR'][ind_iso[0][0]]   
                isoGFR_RK =  Ziso['IsoGFR'][ind_iso[0][0]]      
                
            with open(file_aif,'r') as f:
                textfile_temp = f.read()
                a1 = textfile_temp.split('Y-values')[1]   
                a0 = a1.split()
                aif = np.asarray(a0,dtype=np.float64)  
                
            for kidney_id in ['LK','RK']: 
                if kidney_id == 'LK':
                    if Path(file_LK).exists():
                        with open(file_LK, 'r') as f:
                            textfile_temp = f.read()
                            t1 = textfile_temp.split('X-values')[1].split('Y-values')[0]
                            r1 = textfile_temp.split('Y-values')[1]
                        Time0 = t1.split()
                        Time = np.asarray(Time0,dtype=np.float64)
                        r0 = r1.split()
                        roi = np.asarray(r0,dtype=np.float64)
                        
                        roi0 = np.mean(roi[0:bL])
                        aif0 =  np.mean(aif[0:bL])
                        aif_new = (aif) #(1-Hct)
                        
                        aif_nonlinear = Concentration_SPGRESS(aif_new, aif0, T10_aif, FA, TR, r)/(1-Hct)
                        roi_nonlinear = Concentration_SPGRESS(roi, roi0, T10_kid, FA, TR, r)
                        res = least_squares(fun_2CFM, P, method='lm',  args=(Time, roi_nonlinear,aif_nonlinear), max_nfev=20000,verbose=1) 
                        fit_LK = model_2CFM(res.x,Time, aif_nonlinear)
                        
                        BF = 6000*res.x[1]/(1 -Hct)	#Blood Flow
                        BMTT = 1*res.x[0]*(1-res.x[2])/res.x[1] #Blood MTT
                        BV = 100*res.x[0]*(1-res.x[2])/(1 -Hct) #Blood Volume
                        TMTT = 1*res.x[0]*res.x[2]/(res.x[1]*res.x[3])/60 #Tubular MTT
                        EF = 100*res.x[3] #Extraction Fraction
                        TF = 6000*res.x[1]*res.x[3]	#Tubular Flow
                        GFR = Vol_LK*60*res.x[1]*res.x[3] 
                        fVol = Vol_LK # Region Volume
                        chiSq = 100*np.sum((fit_LK-roi_nonlinear)**2)/np.sum(roi_nonlinear**2)
                        isoGFR = isoGFR_LK
                        
                        P_fit.append([BF,BMTT,BV,TMTT,EF,TF,isoGFR,GFR,fVol,chiSq])
                        params.append(["BF","BMTT","BV","TMTT","E","TF","Iso-SK-GFR","SK-GFR","fVol","Chi-square"])
                        kidney.append(np.repeat('%s' %kidney_id,10))
                        study_id.append(np.repeat('%s' %n,10))
                        
#                        plt.figure()
#                        plt.plot(Time,roi_nonlinear,'o')
#                        plt.plot(Time,fit_LK)
#                        plt.text(150, 0.01,'Non-linear_%s %s' % (n,kidney_id))
                else:
                    if Path(file_RK).exists():
                        with open(file_RK, 'r') as f:
                            textfile_temp = f.read()
                            t1 = textfile_temp.split('X-values')[1].split('Y-values')[0]
                            r1 = textfile_temp.split('Y-values')[1]
                        Time0 = t1.split()
                        Time = np.asarray(Time0,dtype=np.float64)
                        r0 = r1.split()
                        roi = np.asarray(r0,dtype=np.float64)
                        roi0 = np.mean(roi[0:bL])
                        aif0 =  np.mean(aif[0:bL])
                        aif_new = (aif) #(1-Hct)
                        aif_nonlinear = Concentration_SPGRESS(aif_new, aif0, T10_aif, FA, TR, r)/(1-Hct)
                        roi_nonlinear = Concentration_SPGRESS(roi, roi0, T10_kid, FA, TR, r)
                        
                        if n=='AC 51a':
                            res = least_squares(fun_uptake, P, method='lm',  args=(Time, roi_nonlinear,aif_nonlinear), max_nfev=20000,verbose=1) 
                            fit_RK = model_uptake(res.x,Time, aif_nonlinear)
                        else:
                            res = least_squares(fun_2CFM, P, method='lm',  args=(Time, roi_nonlinear,aif_nonlinear),max_nfev=20000, verbose=1) 
                            fit_RK = model_2CFM(res.x,Time, aif_nonlinear)
                        
                        BF = 6000*res.x[1]/(1 -Hct)	#Blood Flow
                        BMTT = 1*res.x[0]*(1-res.x[2])/res.x[1] #Blood MTT
                        BV = 100*res.x[0]*(1-res.x[2])/(1 -Hct) #Blood Volume
                        TMTT = 1*res.x[0]*res.x[2]/(res.x[1]*res.x[3])/60 #Tubular MTT
                        EF = 100*res.x[3] #Extraction Fraction
                        TF = 6000*res.x[1]*res.x[3]	#Tubular Flow
                        GFR = Vol_RK*60*res.x[1]*res.x[3] 
                        fVol = Vol_RK # Region Volume
                        chiSq = 100*np.sum((fit_RK-roi_nonlinear)**2)/np.sum(roi_nonlinear**2)   
                        isoGFR = isoGFR_RK
                        
                        P_fit.append([BF,BMTT,BV,TMTT,EF,TF,isoGFR,GFR,fVol,chiSq])
                        params.append(["BF","BMTT","BV","TMTT","E","TF","Iso-SK-GFR","SK-GFR","fVol","Chi-square"])
                        kidney.append(np.repeat('%s' %kidney_id,10))
                        study_id.append(np.repeat('%s' %n,10))
                        
#                        plt.figure()
#                        plt.plot(Time,roi_nonlinear,'o')
#                        plt.plot(Time,fit_RK)
#                        plt.text(150, 0.01,'Nonlinear_%s %s' % (n,kidney_id))
    #                
    os.chdir('..')        
    P_fit = np.ravel(P_fit)
    params = np.ravel(params)
    study_id = np.ravel(study_id)        
    kidney = np.ravel(kidney)        
    access = np.ones(len(P_fit))
    accessSF = np.ones(len(P_fit))
    accessSF[np.where((study_id == 'MRH008-15'))[0]] = 0.0
    accessSF[np.where((study_id == 'Ha21'))[0]] = 0.0  
    accessSF[np.where((study_id == 'Ra40'))[0]] = 0.0
    ab = np.column_stack((study_id,kidney,params,P_fit,access,accessSF))
    header = np.array(['Patient', 'Kidney', 'Parameter','Value_Nonlinear','Access','AccessSF'])
    ab1 = np.vstack((header,ab))
    #print(ab1)
    np.savetxt('%s_Nonlinear.csv' %d1, ab1, fmt="%s",delimiter = ",") 
          
        
        
        