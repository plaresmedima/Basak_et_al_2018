#This code reads the output files of fitLinear.py, fitNonlinear.py, fitLinearDelay.py 
#and plots regreassion curve and Bland-Altman plot for single-kidney GFR (SK-GFR) 
#for 3T subgroup. 

##It also prints the correlation coefficient, mean difference, stdev difference,
#p-values of SK-GFR for the 3T subgroup.

#Specify 'model' as 'Linear', 'Delay' or 'Nonlinear' 

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

group = [ 'MRH008_wellcome','PANDA','NKRF']
gfr = []
isogfr=[]
gfr_LK = []
gfr_RK = []
patient = []
kidney = []

model = 'Linear'#'Delay'#'Nonlinear' #  

for name in group:    
    Z = np.genfromtxt('%s_%s.csv' %(name,model), dtype=None, delimiter=',',names=True, unpack = True)
    ind = np.where((Z['Parameter']== b'SK-GFR') & (Z['Access']==1))
    ind2 = np.where((Z['Parameter']== b'Iso-SK-GFR') & (Z['Access']==1))
     
    correction = 'Value_%s' %model 
    gfr.extend(Z['%s' %correction][ind])
    isogfr.extend(Z['%s' %correction][ind2])
    
slope, intercept, r_value, p_value, std_err = stats.linregress(isogfr,gfr)
gfr = np.array(gfr)
isogfr = np.array(isogfr)
mean_gfr = np.mean([gfr,isogfr], axis=0)
diff_gfr = gfr-isogfr
mean_diff = np.mean(diff_gfr)
std_diff = np.std(diff_gfr)
CI_upper = mean_diff+1.96*std_diff
CI_lower = mean_diff-1.96*std_diff
x1 = np.arange(np.min(mean_gfr),np.max(mean_gfr))
print('Mean Difference:', mean_diff,'Stdev:', std_diff)
print('Upper CI:',CI_upper,'Lower CI:', CI_lower)
t, p = stats.ttest_ind(gfr,isogfr,equal_var=False)
print('t:',t,'p:',p)

######### Plot regression curve ##############
plt.figure()
plt.scatter(isogfr,gfr)
plt.plot(isogfr, intercept + slope*isogfr)

########### Bland-Altman Plot ##############
plt.figure()
plt.plot(mean_gfr,diff_gfr,'ko',markerfacecolor='None',markeredgewidth=2)
#plt.ylim(-40,90)
plt.plot(x1,np.ones(len(x1))*mean_diff,'k--',linewidth=2)
plt.plot(x1,np.ones(len(x1))*CI_upper,'k--',linewidth=2)
plt.plot(x1,np.ones(len(x1))*CI_lower,'k--',linewidth=2)

################## Analysis within 30% of reference GFR #################
d1 = (abs(diff_gfr)/isogfr)#*100
n_30 = sum(d1 < 0.3)/len(d1)*100
print('Percentage of patients within 30% of reference SK-GFR= ', n_30)