# INT_TRAP(t, c) integrates vector c with respect to t using trapezoidal rule. 

# EXPCONV(T, time, a) evaluates convolution of a with exp(-time/T)/T. More details can be found 
#in the appendix of Flouri et al., Magn Reson Med, 76 (2016), pp. 998-1006

# model_2CFM(P, t, ca) evaluates concentration-time curve for a two-compartment filtration model (2CFM) 
#following the formula in Eq. 7 in Flouri et al., Magn Reson Med, 76 (2016), pp. 998-1006. 
#Where P = [VP+VE, FP, VE/(VP+VE), PS/FP], tissue parameters, t =  time, ca = arterial concentration.

#fun_2CFM(P, t, ct, ca) calculates the difference between concentration evaluated by model_2CFM
#and measurec concentration (ct). 

#model_uptake(P, t, ca) evaluates concentration-time curve for a two-compartment uptake filtration model
#following the formula C(t) = (1-E) FP exp(-t/TP) (convolution) ca(t) + E FP (convolution) ca(t)
#where P = [VP+VE, FP, VE/(VP+VE), PS/FP], E = PS/FP, TP = VP/FP. 

#fun_uptake(P, t,ct, ca) calculates the difference between concentration evaluated by model_uptake
#and measurec concentration (ct).

#Concentration_SPGRESS(S, S0, T10, FA, TR, r) Calculates the tracer concentration C from a 
#signal S and a precontrast signal S0 measured in the steady state of a T1-weighted 
#spoiled gradient-echo sequence
#ARGUMENTS
#S = Signal S(C) at concentration C
#S0 = Precontrast signal S(C=0)
#FA = Flip angle in degrees
#TR = Repetition time in msec (=time between two pulses)
#Precontrast T10 in msec
#r= Relaxivity in Hz/mM

#ShiftAif(aif, Time, Delay) shifts the aif curve by a delay time using
# the formula aif(shifted) = aif(Time - Delay).

#add_delay(P,Time, aif,roi_new,s,nid,fit_model) determines the delay value by fitting the C-t curve for a range 
#of delay values and seleting the one for which the least-square error is minimum.  Finally, it returns
#the fitted tissue parameters and tissue curve for the chosen delay value. 
#Argument
#s = array of three values- minimum value of delay, maxmum value of delay and step size.
#nid = patient id. 
#fit_model = Model used to fit the C-t curve, either 'uptake' or '2CFM'


import numpy as np
from scipy.optimize import least_squares

def INT_TRAP(t, c):
    n = len(t)
    dt = t[1:n]-t[0:n-1]
    trapz_int = dt*(c[0:n-1]+c[1:n])/2.0
    trapz_int_2 = np.insert(trapz_int,0,0.0) 
    trapz_int_final = np.cumsum(trapz_int_2)
    return trapz_int_final

def EXPCONV(T, time, a):
    n = len(time)
    f = np.zeros(n)
    x = (time[1:n] - time[0:n-1])/T
    da = (a[1:n] - a[0:n-1])/x
    E = np.exp(-x)
    E0 = 1-E
    E1 = x-E0
    add = a[0:n-1]*E0 + da*E1
    for i in range(0,n-1):
        f[i+1] = E[i]*f[i] + add[i]      
    return f

def model_2CFM(P, t, ca):
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
        conc=Fp*Tpos*EXPCONV(Tpos, t, ca)
    else:
        Epos = (TT-Tneg)/(Tpos-Tneg)
        Eneg = 1-Epos
        conc = Fp*Epos*Tpos*EXPCONV(Tpos, t, ca) + Fp*Eneg*Tneg*EXPCONV(Tneg, t, ca)
    return conc  

def fun_2CFM(P, t, ct, ca):
    return model_2CFM(P, t, ca) - ct

def model_uptake(P, t, ca):
    Fp = P[1]
    PS = P[3]*Fp
    VE = P[2]*P[0]
    VP = P[0]-VE
    TP = VP/Fp
    E = PS/Fp
    conc = Fp*(1-E)*TP*EXPCONV(TP, t, ca) + Fp*E*INT_TRAP( t, ca)
    return conc  
def fun_uptake(P, t,ct, ca):
    return model_uptake(P, t, ca) - ct

def Concentration_SPGRESS(S, S0, T10, FA, TR, r):
    E = np.exp(-TR/T10)
    c = np.cos(FA*np.pi/180)
    Sn = (S/S0)*(1-E)/(1-c*E)	#normalized signal
    R1 = -np.log((1-Sn)/(1-c*Sn))/TR	#relaxation rate in 1/msec
    return 1000*(R1 - 1/T10)/r

def ShiftAif(aif, Time, Delay):
    ShAif = np.interp(Time-Delay, Time, aif)
    nneg = np.sum(Time <= Delay)
    if nneg > 0 :
        ShAif[0:nneg-1] = 0
    return ShAif

def add_delay(P,Time, aif,roi_new,s,nid,fit_model):
    n = 1+(s[1]-s[0])/s[2]
    Delay = s[0] + s[2]*np.arange(n)
    Error = Delay*0
   
    for i in range(int(n)):
       # print(i)
        if Delay[i] > 0 :
            aif_del = ShiftAif(aif,Time,Delay[i]) 
        else:
            aif_del = aif 
            
        if Delay[i] < 0 :
            roi_del = ShiftAif(roi_new,Time,-Delay[i]) 
        else:
            roi_del = roi_new
            
        if fit_model == 'uptake':
            res = least_squares(fun_uptake, P, method='lm',  args=(Time, roi_del,aif_del),max_nfev=20000, verbose=0)       
            fit = model_uptake(res.x,Time, aif_del)
        elif fit_model == '2CFM':    
            res = least_squares(fun_2CFM, P, method='lm',  args=(Time, roi_del,aif_del),max_nfev=20000, verbose=0) 
            fit = model_2CFM(res.x,Time,aif_del) 
          
        if Delay[i] < 0 :
            fit_shft = ShiftAif(fit,Time,Delay[i]) 
        else:
            fit_shft = fit
        #nd0 =(Delay[i]/(Time[1]-Time[0]))
        nd = abs(int(round(Delay[i]/(Time[1]-Time[0]))))
        nT = len(Time)
        Error[i] =  100*np.sum((roi_new[0:nT-nd-1]-fit_shft[0:nT-nd-1])**2)/(np.sum(roi_new[0:nT-nd-1]**2))
           
    imin = np.where(Error==Error.min())
    Delay = Delay[imin]
    #print('imin=',imin)
    print('Delay',Delay)
    if Delay > 0 :
        aif_del = ShiftAif(aif,Time,Delay) 
    else:
        aif_del = aif 
        
    if Delay < 0 :
        roi_del = ShiftAif(roi_new,Time,-Delay) 
    else:
        roi_del = roi_new
        
    if fit_model == 'uptake':
        res = least_squares(fun_uptake, P, method='lm',  args=(Time, roi_del,aif_del),max_nfev=20000, verbose=0)       
        fit = model_uptake(res.x,Time, aif_del)
    elif fit_model == '2CFM':
        res = least_squares(fun_2CFM, P, method='lm',  args=(Time, roi_del,aif_del),max_nfev=20000, verbose=1) 
        fit = model_2CFM(res.x,Time, aif_del)    
        
    if Delay < 0 :
        fit = np.interp(Time-Delay, Time, fit)    
    return  res, fit
