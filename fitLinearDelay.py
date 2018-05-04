# This code reads the signal (S) versus time(t) curve for arterial input function (aif), 
#right (RK) and left kidney (LK) ROIs and fits them using a two compartment filtration model (2CFM).
#Here tracer concentration (C) is assumed to be proportional to S-S0 where S0 is
#the precontrast signal and the delay in the arrival of bolus from artery to the tissue of interest 
#has been incorporated using the function 'add_delay'.
#Input data: Three files (aif, LK, RK) for S-t curve per patient, One file per study containing 
#functional volume of kidneys, One file per study containing reference GFR values. 
#Output data: One file per study containing all the fitting parameters such as
#blood flow, blood mean transit time (MTT), blood volume, tubular MTT, exraction franction,
#tubular flow, GFR and isotope GFR.

import os#, glob
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import glob
import re
from function_fitting import model_2CFM, fun_2CFM, add_delay

P = [0.3, 125.0/6000, 2.0/3, 12/120.] #[VP+VTapp, FP, VTapp/(VP+VTapp), FT/FP]
FA = 17.00
TR = 5.05 
Hct = 0.45
r = 3.6
s = [-10,10,0.5] # Delay values: min, max and step size
delay = True
study = [ 'MRH008_wellcome','PANDA','NKRF']
bL = 9 # Baseline, default=9
path_data = 'C:/Users/medsbasa/Work/Data_Analysis/Data_Source'
#path_data = '/Users/susmitabasak/Documents/WorkLeeds/Data_Analysis/Data_Source'
for d1 in study:
    P_fit =[]
    params =[]
    study_id =[]
    kidney = []
    name=[]
    
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
                        roi_new = roi-roi0
                        aif_new = (aif-aif0)/(1-Hct)
                        
                        if delay == True:
                            res, fit_LK = add_delay(P,Time, aif_new,roi_new,s,n,kidney_id)
        
                        res2 = least_squares(fun_2CFM, P, method='lm',  args=(Time, roi_new,aif_new), max_nfev=20000, verbose=1)       
                        fit_LK2 = model_2CFM(res2.x,Time,aif_new)
                            
                        
                        BF = 6000*res.x[1]/(1 -Hct)	#Blood Flow
                        BMTT = 1*res.x[0]*(1-res.x[2])/res.x[1] #Blood MTT
                        BV = 100*res.x[0]*(1-res.x[2])/(1 -Hct) #Blood Volume
                        TMTT = 1*res.x[0]*res.x[2]/(res.x[1]*res.x[3])/60 #Tubular MTT
                        EF = 100*res.x[3] #Extraction Fraction
                        TF = 6000*res.x[1]*res.x[3]	#Tubular Flow
                        GFR = Vol_LK*60*res.x[1]*res.x[3] 
                        fVol = Vol_LK # Region Volume
                        chiSq = 100*np.sum((fit_LK-roi_new)**2)/np.sum(roi_new**2)
                        isoGFR = isoGFR_LK
                        
                        P_fit.append([BF,BMTT,BV,TMTT,EF,TF,isoGFR,GFR,fVol,chiSq])
                        params.append(["BF","BMTT","BV","TMTT","E","TF","Iso-SK-GFR","SK-GFR","fVol","Chi-square"])
                        kidney.append(np.repeat('%s' %kidney_id,10))
                        study_id.append(np.repeat('%s' %n,10))
                        
        #                plt.figure()
        #                plt.plot(Time,roi_new,'o')
        #                plt.plot(Time,fit_LK,label='delay')
        #                plt.plot(Time,fit_LK2,label='no delay')
        #                plt.legend()
        #                plt.text(150, 0.01,'Linear_%s %s' % (n,kidney_id))
                    
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
                        roi_new = roi-roi0
                        aif_new = (aif-aif0)/(1-Hct)
                        if delay == True:
                            res, fit_RK = add_delay(P,Time, aif_new,roi_new,s,n,kidney_id)

                        res2 = least_squares(fun_2CFM, P, method='lm',  args=(Time, roi_new,aif_new),max_nfev=20000, verbose=1)       
                        fit_RK2 = model_2CFM(res2.x,Time,aif_new)
                        
                        BF = 6000*res.x[1]/(1 -Hct)	#Blood Flow
                        BMTT = 1*res.x[0]*(1-res.x[2])/res.x[1] #Blood MTT
                        BV = 100*res.x[0]*(1-res.x[2])/(1 -Hct) #Blood Volume
                        TMTT = 1*res.x[0]*res.x[2]/(res.x[1]*res.x[3])/60 #Tubular MTT
                        EF = 100*res.x[3] #Extraction Fraction
                        TF = 6000*res.x[1]*res.x[3]	#Tubular Flow
                        GFR = Vol_RK*60*res.x[1]*res.x[3] 
                        fVol = Vol_RK # Region Volume
                        chiSq = 100*np.sum((fit_RK-roi_new)**2)/np.sum(roi_new**2)   
                        isoGFR = isoGFR_RK
                        
                        P_fit.append([BF,BMTT,BV,TMTT,EF,TF,isoGFR,GFR,fVol,chiSq])
                        params.append(["BF","BMTT","BV","TMTT","E","TF","Iso-SK-GFR","SK-GFR","fVol","Chi-square"])
                        kidney.append(np.repeat('%s' %kidney_id,10))
                        study_id.append(np.repeat('%s' %n,10))
                        
        #                plt.figure()
        #                plt.plot(Time,roi_new,'o')
        #                plt.plot(Time,fit_RK,label='delay')
        #                plt.plot(Time,fit_RK2,label='no delay')
        #                plt.legend()
        #                plt.text(150, 0.01,'Linear_%s %s' % (n,kidney_id))
                    
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
    header = np.array(['Patient', 'Kidney', 'Parameter','Value_Delay','Access','AccessSF'])
    ab1 = np.vstack((header,ab))
    #print(ab1)
    np.savetxt('%s_Delay.csv' %d1, ab1, fmt="%s",delimiter = ",") 
    
        
        

        
        