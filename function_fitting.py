


import numpy as np
from scipy.optimize import least_squares

# INT_TRAP(t, c) integrates vector c with respect to t using trapezoidal rule. 

def INT_TRAP(t, c):
    n = len(t)
    dt = t[1:n]-t[0:n-1]
    trapz_int = dt*(c[0:n-1]+c[1:n])/2.0
    trapz_int_2 = np.insert(trapz_int,0,0.0) 
    trapz_int_final = np.cumsum(trapz_int_2)
    return trapz_int_final

# EXPCONV(T, time, ca) evaluates convolution of ca with exp(-time/T)/T. More details can be found 
#in the appendix of Flouri et al., Magn Reson Med, 76 (2016), pp. 998-1006

def EXPCONV(T, time, ca):
    n = len(time)
    f = np.zeros(n)
    x = (time[1:n] - time[0:n-1])/T
    da = (ca[1:n] - ca[0:n-1])/x
    E = np.exp(-x)
    E0 = 1-E
    E1 = x-E0
    add = ca[0:n-1]*E0 + da*E1
    for i in range(0,n-1):
        f[i+1] = E[i]*f[i] + add[i]      
    return f

# model_2CFM(P, time, ca) evaluates concentration-time curve for a two-compartment filtration model (2CFM) 
#following the formula in Eq. 7 in Flouri et al., Magn Reson Med, 76 (2016), pp. 998-1006. 
#Where P = [VP+VE, FP, VE/(VP+VE), PS/FP], tissue parameters, ca = arterial concentration.

def model_2CFM(P, time, ca):
    Fp = P[1]
    PS = P[3]*Fp
    VE = P[2]*P[0]
    VP = P[0]-VE
    TP = VP/Fp
    TE = VE/PS
    TT = (VP+VE)/Fp
    Tpos = TE
    Tneg = TP
    if Tpos == Tneg:
        conc=Fp*Tpos*EXPCONV(Tpos, time, ca)
    else:
        Epos = (TT-Tneg)/(Tpos-Tneg)
        Eneg = 1-Epos
        conc = Fp*Epos*Tpos*EXPCONV(Tpos, time, ca) + Fp*Eneg*Tneg*EXPCONV(Tneg, time, ca)
    return conc  

#fun_2CFM(P, time, ct, ca) calculates the difference between concentration evaluated by model_2CFM
#and measurec concentration (ct).

def fun_2CFM(P, time, ct, ca):
    return model_2CFM(P, time, ca) - ct

#model_uptake(P, time, ca) evaluates concentration-time curve for a two-compartment uptake filtration model
#following the formula ct(t) = (1-E) FP exp(-t/TP) (convolution) ca(t) + E FP (convolution) ca(t)
#where P = [VP+VE, FP, VE/(VP+VE), PS/FP], E = PS/FP, TP = VP/FP. 

def model_uptake(P, time, ca):
    Fp = P[1]
    PS = P[3]*Fp
    VE = P[2]*P[0]
    VP = P[0]-VE
    TP = VP/Fp
    E = PS/Fp
    conc = Fp*(1-E)*TP*EXPCONV(TP, time, ca) + Fp*E*INT_TRAP( time, ca)
    return conc  

#fun_uptake(P, t,ct, ca) calculates the difference between concentration evaluated by model_uptake
#and measurec concentration (ct).
    
def fun_uptake(P, time,ct, ca):
    return model_uptake(P, time, ca) - ct

#Concentration_SPGRESS(S, S0, T10, FA, TR, r) calculates the tracer concentration ct from a 
#signal S and a precontrast signal S0 measured in the steady state of a T1-weighted 
#spoiled gradient-echo sequence
#ARGUMENTS
#S = Signal S(ct) at concentration ct
#S0 = Precontrast signal S(ct=0)
#FA = Flip angle in degrees
#TR = Repetition time in msec (=time between two pulses)
#Precontrast T10 in msec
#r= Relaxivity in Hz/mM

def Concentration_SPGRESS(S, S0, T10, FA, TR, r):
    E = np.exp(-TR/T10)
    ct = np.cos(FA*np.pi/180)
    Sn = (S/S0)*(1-E)/(1-ct*E)	#normalized signal
    R1 = -np.log((1-Sn)/(1-ct*Sn))/TR	#relaxation rate in 1/msec
    return 1000*(R1 - 1/T10)/r

#ShiftAif(ca, time, Delay) shifts the arterial concentration (ca) curve by a delay time using
# the formula ca(shifted) = ca(Time - Delay).


def ShiftAif(ca, time, Delay):
    ShAif = np.interp(time-Delay, time, ca)
    nneg = np.sum(time <= Delay)
    if nneg > 0 :
        ShAif[0:nneg-1] = 0
    return ShAif

#delay_error(roi,Delay,dt,nT,fit_shft) calculates the error of
#the concentration vs time curve shifted with a delay time against the original curve.
#Delay = delay time
#roi = tracer concentration in the tissue
#dt = time interval
#nT = number of time points
#fit_shft = shifted curve

def delay_error(roi,Delay,dt,nT,fit_shft):
    nd = abs(int(round(Delay/dt)))
    Error =  100*np.sum((roi[0:nT-nd-1]-fit_shft[0:nT-nd-1])**2)/(np.sum(roi[0:nT-nd-1]**2))
    return Error

#add_delay(P,time, ca,roi_new,shift,nid,fit_model) determines the delay value by fitting the 
#concentration vs time curve for a range 
#of delay values and seleting the one for which the least-square error is minimum.  It returns
#the fitted tissue parameters and tissue curve for the chosen delay value. 
#Arguments
#roi = tracer concentration in the tissue
#shift = array of three values- minimum value of delay, maxmum value of delay and step size.
#nid = patient id. 
#fit_model = Model used to fit the C-t curve, either 'uptake' or '2CFM'
    
def add_delay(P,time, ca,roi,shift,nid,fit_model):
    n = 1+(shift[1]-shift[0])/shift[2]
    Delay = shift[0] + shift[2]*np.arange(n)
    Error = Delay*0
    dt = time[1]-time[0]
    nT = len(time)
   
    for i in range(int(n)):
        if Delay[i] > 0 :
            aif_del = ShiftAif(ca,time,Delay[i]) 
        else:
            aif_del = ca            
        if Delay[i] < 0 :
            roi_del = ShiftAif(roi,time,-Delay[i]) 
        else:
            roi_del = roi           
        if fit_model == 'uptake':
            res = least_squares(fun_uptake, P, method='lm',  args=(time, roi_del,aif_del),max_nfev=20000, verbose=0)       
            fit = model_uptake(res.x,time, aif_del)
        elif fit_model == '2CFM':    
            res = least_squares(fun_2CFM, P, method='lm',  args=(time, roi_del,aif_del),max_nfev=20000, verbose=0) 
            fit = model_2CFM(res.x,time,aif_del)           
        if Delay[i] < 0 :
            fit_shft = ShiftAif(fit,time,Delay[i]) 
        else:
            fit_shft = fit
        Error[i] =  delay_error(roi,Delay[i],dt,nT,fit_shft)
           
    imin = np.where(Error==Error.min())
    Delay = Delay[imin]
    print('Delay',Delay)
    if Delay > 0 :
        aif_del = ShiftAif(ca,time,Delay) 
    else:
        aif_del = ca         
    if Delay < 0 :
        roi_del = ShiftAif(roi,time,-Delay) 
    else:
        roi_del = roi        
    if fit_model == 'uptake':
        res = least_squares(fun_uptake, P, method='lm',  args=(time, roi_del,aif_del),max_nfev=20000, verbose=0)       
        fit = model_uptake(res.x,time, aif_del)
    elif fit_model == '2CFM':
        res = least_squares(fun_2CFM, P, method='lm',  args=(time, roi_del,aif_del),max_nfev=20000, verbose=1) 
        fit = model_2CFM(res.x,time, aif_del)            
    if Delay < 0 :
        fit = np.interp(time-Delay, time, fit)    
    return  res, fit
