#This code reads the output files of fitAll.py for linear model and  
#calculates split renal function (SRF) and total GFR 
#and plots regreassion curve and Bland-Altman (BA) plot for SRF and total GFR. 

#It also prints the correlation coefficient, mean difference, stdev difference,
#p-values of SRF and total GFR for entire group and for 3T and 1T subgroup separately. 

##Choose model as 'Linear'

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

group = [ ['MRH008_wellcome','PANDA','NKRF'],['1Tdata']]
model = 'Linear' 

for ii in range(2):
    gfr = []
    isogfr=[]
    gfr_LK = []
    gfr_RK = []
    isogfr_LK = []
    isogfr_RK = []    
    for name in group[ii]:
        Z = np.genfromtxt('%s_%s.csv' %(name,model), dtype=None, delimiter=',',names=True, unpack = True)

        ind = np.where((Z['Parameter']== b'SK-GFR') & (Z['Access']==1)& (Z['AccessSF']==1))
        ind2 = np.where((Z['Parameter']== b'Iso-SK-GFR') & (Z['Access']==1)& (Z['AccessSF']==1))
        ind_LK = np.where((Z['Parameter']== b'SK-GFR') & (Z['Kidney']==b'LK') & (Z['Access']==1)& (Z['AccessSF']==1))
        #print(ind_LK)
        ind_RK = np.where((Z['Parameter']== b'SK-GFR') & (Z['Kidney']==b'RK') & (Z['Access']==1)& (Z['AccessSF']==1))
        ind2_LK = np.where((Z['Parameter']== b'Iso-SK-GFR') & (Z['Kidney']==b'LK') & (Z['Access']==1)& (Z['AccessSF']==1))
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
    sf_LK = gfr_LK/gfr_tot
    sf_RK = gfr_RK/gfr_tot

    isogfr_tot = isogfr_LK + isogfr_RK
    isosf_LK = isogfr_LK/isogfr_tot
    isosf_RK = isogfr_RK/isogfr_tot
    if ii==0:
        sf_3T = sf_LK 
        isosf_3T =isosf_LK
        gfr_tot_3T = gfr_tot
        isogfr_tot_3T = isogfr_tot
    else:
        sf_1T = sf_LK 
        isosf_1T =isosf_LK
        gfr_tot_1T = gfr_tot
        isogfr_tot_1T = isogfr_tot

gfr_tot =  np.concatenate((gfr_tot_3T,gfr_tot_1T))
isogfr_tot = np.concatenate((isogfr_tot_3T,isogfr_tot_1T))        
sf = np.concatenate((sf_3T,sf_1T))
isosf = np.concatenate((isosf_3T,isosf_1T))
slope, intercept, r_value, p_value, std_err = stats.linregress(isosf,sf)

mean_sf_3T = np.mean([sf_3T,isosf_3T], axis=0)
diff_sf_3T = np.array(sf_3T)-np.array(isosf_3T)
mean_sf_1T = np.mean([sf_1T,isosf_1T], axis=0)
diff_sf_1T =  np.array(sf_1T)- np.array(isosf_1T)
mean_sf = np.mean([sf,isosf], axis=0)
diff_sf = sf-isosf
mean_diff = np.mean(diff_sf)
std_diff = np.std(diff_sf)
CI_upper = mean_diff+1.96*std_diff
CI_lower = mean_diff-1.96*std_diff
x1 = np.arange(np.min(mean_sf),np.max(mean_sf)+0.1,0.1)

print("y=%.6fx+(%.6f)"%(slope,intercept))
print("r-squared:", r_value**2)
print("r:", r_value)
print('Mean Difference:', mean_diff,'Stdev:', std_diff)
print('Upper CI:',CI_upper,'Lower CI:', CI_lower)

t, p = stats.ttest_ind(sf,isosf,equal_var=False)
print('t:',t,'p:',p)

##############---- plot regression curve for SRF --------###############

plt.figure()
plt.plot(isosf_3T,sf_3T,'ro')
plt.plot(isosf_1T,sf_1T,'b*')
xx = np.arange(0,1.1,.05)
yy =np.arange(0,1.1,.05)
plt.plot(xx, intercept + slope*xx,'c', linewidth=2.0)
plt.plot(xx,yy,'k--', linewidth=2.0) #isosf,isosf
plt.title('%s' %correction)

##############---- BA Plot for SRF --------###############

plt.figure()
plt.plot(mean_sf_3T,diff_sf_3T,'ro',markeredgewidth=2)
plt.plot(mean_sf_1T,diff_sf_1T,'b*',markeredgewidth=2)
plt.plot(x1,np.ones(len(x1))*mean_diff,'k--',linewidth=2)
plt.plot(x1,np.ones(len(x1))*CI_upper,'k--',linewidth=2)
plt.plot(x1,np.ones(len(x1))*CI_lower,'k--',linewidth=2)


##############---- plot regression curve for Total GFR --------###############

slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(isogfr_tot,gfr_tot)
print("y=%.6fx+(%.6f)"%(slope1,intercept1))
print("r-squared:", r_value1**2)
print("r:", r_value1)

xx = np.arange(170)
yy = np.arange(170)
plt.figure()
plt.plot(isogfr_tot_3T,gfr_tot_3T,'ro')
plt.plot(isogfr_tot_1T,gfr_tot_1T,'b*')
plt.plot(xx, intercept1 + slope1*xx,'c', linewidth=2.0)
plt.plot(xx,yy,'k--', linewidth=2.0)
plt.xlim(-3,175)
plt.ylim(-3,175)

##############---- BA Plot for Total GFR --------###############

mean_totgfr_3T = np.mean([gfr_tot_3T ,isogfr_tot_3T], axis=0)
diff_totgfr_3T  = gfr_tot_3T -isogfr_tot_3T 
mean_totgfr_1T = np.mean([gfr_tot_1T ,isogfr_tot_1T], axis=0)
diff_totgfr_1T  = gfr_tot_1T -isogfr_tot_1T
mean_totgfr = np.mean([gfr_tot ,isogfr_tot], axis=0)
diff_totgfr  = gfr_tot -isogfr_tot
mean_diff_tot = np.mean(diff_totgfr)
std_diff_tot = np.std(diff_totgfr)
CI_upper_tot = mean_diff_tot +1.96*std_diff_tot
CI_lower_tot = mean_diff_tot -1.96*std_diff_tot
x2 = np.arange(np.min(mean_totgfr),np.max(mean_totgfr))
#print(x1)
print('Mean Difference:', mean_diff_tot,'Stdev:', std_diff_tot)
print('Upper CI:',CI_upper_tot,'Lower CI:', CI_lower_tot)

plt.figure()
plt.plot(mean_totgfr_3T,diff_totgfr_3T,'ro',markeredgewidth=2)
plt.plot(mean_totgfr_1T,diff_totgfr_1T,'b*',markeredgewidth=2)
plt.plot(x2,np.ones(len(x2))*mean_diff_tot,'k--',linewidth=2)
plt.plot(x2,np.ones(len(x2))*CI_upper_tot,'k--',linewidth=2)
plt.plot(x2,np.ones(len(x2))*CI_lower_tot,'k--',linewidth=2)

t, p = stats.ttest_ind(gfr_tot,isogfr_tot,equal_var=False)
print('t:',t,'p:',p)

############# Mean difference and stdev difference for 3T and 1T subgroup ##############
mean_diff_3T = np.mean(diff_totgfr_3T)
std_diff_3T = np.std(diff_totgfr_3T)
mean_diff_1T = np.mean(diff_totgfr_1T)
std_diff_1T = np.std(diff_totgfr_1T)

mean_diff_3Tsf = np.mean(diff_sf_3T)
std_diff_3Tsf = np.std(diff_sf_3T)
mean_diff_1Tsf = np.mean(diff_sf_1T)
std_diff_1Tsf = np.std(diff_sf_1T)

print('Mean Difference 3T:', mean_diff_3T,'Stdev Difference 3T:', std_diff_3T)
print('Mean Difference 1T:', mean_diff_1T,'Stdev  Difference  1T:', std_diff_1T)
print('Mean Difference 3T:', mean_diff_3Tsf,'Stdev Difference 3T:', std_diff_3Tsf)
print('Mean Difference 1T:', mean_diff_1Tsf,'Stdev  Difference  1T:', std_diff_1Tsf)

#t, p = stats.ttest_ind(gfr_tot_3T,isogfr_tot_3T,equal_var=True)
#print('t 3T:',t,'p 3T:',p)
#
#t, p = stats.ttest_ind(gfr_tot_1T,isogfr_tot_1T,equal_var=True)
#print('t 1T:',t,'p 1T:',p)

#t, p = stats.ttest_ind(sf_3T,isosf_3T,equal_var=True)
#print('t 3T:',t,'p 3T:',p)
#
#t, p = stats.ttest_ind(sf_1T,isosf_1T,equal_var=True)
#print('t 1T:',t,'p 1T:',p)

################## Analysis within 30% of reference GFR #################
d1 = (abs(diff_sf)/isosf)#*100
n_30 = sum(d1 < 0.3)/len(d1)*100
print('Percentage of patients within 30% of reference SRF= ', n_30)

d1 = (abs(diff_totgfr)/isogfr_tot)#*100
n_30 = sum(d1 < 0.3)/len(d1)*100
print('Percentage of patients within 30% of reference GFR= ', n_30)


