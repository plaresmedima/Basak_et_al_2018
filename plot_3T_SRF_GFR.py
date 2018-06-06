#This code reads the output files of fitAll.py 
#and plots regreassion curve and Bland-Altman plot for SRF and total GFR for 3 T subgroup

#It also prints the correlation coefficient, mean difference, stdev difference,
#p-values of SRF and Total GFR for the 3T subgroup.

#Specify 'model' as 'Linear', 'Nonlinear', 'Delay' or 'NLT1' 
#'Linear': Linear model for signal to concentration conversion
#'Nonlinear': nonLinear model for signal to concentration conversion
#'Delay': linear model with delay correction
#'NLT1': nonlinear model with measured T1-values

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

group = [ 'MRH008_wellcome','PANDA','NKRF']
gfr = []
isogfr=[]
gfr_LK = []
gfr_RK = []
isogfr_LK = []
isogfr_RK = []

model = 'NLT1' # 'Delay'# 'Linear'#  'Nonlinear' #

for name in group:
    Z = np.genfromtxt('%s_%s.csv' %(name,model), dtype=None, delimiter=',',names=True, unpack = True)
    ind = np.where((Z['Parameter']== b'SK-GFR') & (Z['Access']==1) & (Z['AccessSF']==1))
    ind2 = np.where((Z['Parameter']== b'Iso-SK-GFR') & (Z['Access']==1)& (Z['AccessSF']==1))
    ind_LK = np.where((Z['Parameter']== b'SK-GFR') & (Z['Kidney']==b'LK') & (Z['Access']==1)& (Z['AccessSF']==1))
    
    ind_RK = np.where((Z['Parameter']== b'SK-GFR') & (Z['Kidney']==b'RK') & (Z['Access']==1)& (Z['AccessSF']==1))
    ind2_LK = np.where((Z['Parameter']== b'Iso-SK-GFR') & (Z['Kidney']==b'LK') & (Z['Access']==1) & (Z['AccessSF']==1))
    ind2_RK = np.where((Z['Parameter']== b'Iso-SK-GFR') & (Z['Kidney']==b'RK') & (Z['Access']==1)& (Z['AccessSF']==1))
    
    correction = 'Value_%s' %model 
    
    gfr.extend(Z['%s' %correction][ind])
    isogfr.extend(Z['%s' %correction][ind2])
   
    gfr_LK.extend(Z['%s' %correction][ind_LK])
    isogfr_LK.extend(Z['%s' %correction][ind2_LK])
    gfr_RK.extend(Z['%s' %correction][ind_RK])
    isogfr_RK.extend(Z['%s' %correction][ind2_RK])


gfr = np.array(gfr)
isogfr = np.array(isogfr)
gfr_LK = np.array(gfr_LK)
isogfr_LK = np.array(isogfr_LK)
gfr_RK = np.array(gfr_RK)
isogfr_RK = np.array(isogfr_RK)

gfr_tot = gfr_LK+gfr_RK
sf = gfr_LK/gfr_tot
isogfr_tot = isogfr_LK + isogfr_RK
isosf = isogfr_LK/isogfr_tot

slope, intercept, r_value, p_value, std_err = stats.linregress(isosf,sf)

mean_sf = np.mean([sf,isosf], axis=0)
diff_sf = sf-isosf
mean_diff = np.mean(diff_sf)
std_diff = np.std(diff_sf)
CI_upper = mean_diff+1.96*std_diff
CI_lower = mean_diff-1.96*std_diff
x1 = np.arange(np.min(mean_sf),np.max(mean_sf)+0.1,0.1)
print('Mean Difference:', mean_diff,'Stdev:', std_diff)
print('Upper CI:',CI_upper,'Lower CI:', CI_lower)

#############---- BA Plot for SRF --------###############

plt.figure()
plt.plot(mean_sf,diff_sf,'ko',markerfacecolor='None',markeredgewidth=2)
plt.plot(x1,np.ones(len(x1))*mean_diff,'k--',linewidth=2)
plt.plot(x1,np.ones(len(x1))*CI_upper,'k--',linewidth=2)
plt.plot(x1,np.ones(len(x1))*CI_lower,'k--',linewidth=2)

##############---- plot regression curve for SRF --------###############
plt.figure()
plt.scatter(isosf,sf)
plt.plot(isosf, intercept + slope*isosf)
plt.title('%s' %correction)
t, p = stats.ttest_ind(sf,isosf,equal_var=False)
print('t sf:',t,'p sf:',p)

slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(isogfr_tot,gfr_tot)

##############---- plot regression curve for total GFR --------###############
plt.figure()
plt.scatter(isogfr_tot,gfr_tot)
plt.plot(isogfr_tot, intercept1 + slope1*isogfr_tot)
t, p = stats.ttest_ind(gfr_tot,isogfr_tot,equal_var=False)
print('t GFR:',t,'p GFR:',p)

#############---- BA Plot for Total GFR --------###############

mean_totgfr = np.mean([gfr_tot,isogfr_tot], axis=0)
diff_totgfr = gfr_tot-isogfr_tot
mean_diff_tot = np.mean(diff_totgfr)
std_diff_tot = np.std(diff_totgfr)
CI_upper_tot = mean_diff_tot +1.96*std_diff_tot
CI_lower_tot = mean_diff_tot -1.96*std_diff_tot
x2 = np.arange(np.min(mean_totgfr),np.max(mean_totgfr))

print('Mean Difference:', mean_diff_tot,'Stdev:', std_diff_tot)
print('Upper CI:',CI_upper_tot,'Lower CI:', CI_lower_tot)
#
plt.figure()
plt.plot(mean_totgfr,diff_totgfr,'ko',markerfacecolor='None',markeredgewidth=2)
plt.plot(x2,np.ones(len(x2))*mean_diff_tot,'k--',linewidth=2)
plt.plot(x2,np.ones(len(x2))*CI_upper_tot,'k--',linewidth=2)
plt.plot(x2,np.ones(len(x2))*CI_lower_tot,'k--',linewidth=2)

################## Analysis within 30% of reference GFR #################
d1 = (abs(diff_sf)/isosf)#*100
n_30 = sum(d1 < 0.3)/len(d1)*100
print('Percentage of patients within 30% of reference SRF= ', n_30)

d1 = (abs(diff_totgfr)/isogfr_tot)#*100
n_30 = sum(d1 < 0.3)/len(d1)*100
print('Percentage of patients within 30% of reference GFR= ', n_30)