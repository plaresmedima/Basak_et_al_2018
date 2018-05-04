#This code reads the output files of fitLinear.py 
#and plots regreassion curve and Bland-Altman plot for single-kidney GFR (SK-GFR). 

#It also prints the correlation coefficient, eman difference, stdev difference,
#p-values of SK-GFR for entire group and for 3T and 1T separately. 

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

group_3T = [ 'MRH008_wellcome','PANDA','NKRF']
group_1T = ['1Tdata']
gfr_3T = []
isogfr_3T=[]
gfr_1T = []
isogfr_1T=[]

model = 'Linear' 


for name in group_3T:
    Z = np.genfromtxt('%s_%s.csv' %(name,model) , dtype=None, delimiter=',',names=True, unpack = True)
    ind = np.where((Z['Parameter']== b'SK-GFR') & (Z['Access']==1))
    ind2 = np.where((Z['Parameter']== b'Iso-SK-GFR') & (Z['Access']==1))
      
    correction = 'Value_%s' %model 
    gfr_3T.extend(Z['%s' %correction][ind])
    isogfr_3T.extend(Z['%s' %correction][ind2])
   

for name in group_1T:
    Z = np.genfromtxt('%s_%s.csv' %(name,model), dtype=None, delimiter=',',names=True, unpack = True)
    ind = np.where((Z['Parameter']== b'SK-GFR') & (Z['Access']==1))
    ind2 = np.where((Z['Parameter']== b'Iso-SK-GFR') & (Z['Access']==1))
      
    correction = 'Value_%s' %model 
    gfr_1T.extend(Z['%s' %correction][ind])
    isogfr_1T.extend(Z['%s' %correction][ind2])
    
gfr=np.concatenate((gfr_3T,gfr_1T))
isogfr=np.concatenate((isogfr_3T,isogfr_1T))   
slope, intercept, r_value, p_value, std_err = stats.linregress(isogfr,gfr)
gfr = np.array(gfr)
isogfr = np.array(isogfr)

mean_gfr_3T = np.mean([gfr_3T,isogfr_3T], axis=0)
diff_gfr_3T = np.array(gfr_3T)-np.array(isogfr_3T)
mean_gfr_1T = np.mean([gfr_1T,isogfr_1T], axis=0)
diff_gfr_1T =  np.array(gfr_1T)- np.array(isogfr_1T)
mean_gfr = np.mean([gfr,isogfr], axis=0)
diff_gfr = gfr-isogfr
mean_diff = np.mean(diff_gfr)
std_diff = np.std(diff_gfr)
CI_upper = mean_diff+1.96*std_diff
CI_lower = mean_diff-1.96*std_diff
x1 = np.arange(np.min(mean_gfr),np.max(mean_gfr))

print("y=%.6fx+(%.6f)"%(slope,intercept))
print("r-squared:", r_value**2)
print("r:", r_value)
print('Mean Difference:', mean_diff,'Stdev  Difference :', std_diff)
print('Upper CI:',CI_upper,'Lower CI:', CI_lower)
t, p = stats.ttest_ind(gfr,isogfr,equal_var=True)
print('t:',t,'p:',p)

######### Plot regression curve ##############
xx = np.arange(105)
yy = np.arange(105)
plt.figure()
plt.plot(isogfr_3T,gfr_3T,'ro')
plt.plot(isogfr_1T,gfr_1T,'b*')
plt.plot(xx,intercept + slope*xx,'c', linewidth=2.0)
plt.plot(xx,yy,'k--', linewidth=2.0)
plt.xlim(-3,130)
plt.ylim(-3,130)
#plt.title('%s' %correction)
#t1, p1 = stats.ttest_ind(gfr_3T,gfr_1T,equal_var=True)
#print('t1:',t1,'p1:',p1)

########### Bland-Altman Plot ##############
plt.figure()
plt.plot(mean_gfr_3T,diff_gfr_3T,'ro',markerfacecolor='None',markeredgewidth=2)
plt.plot(mean_gfr_1T,diff_gfr_1T,'b*',markerfacecolor='None',markeredgewidth=2)
#plt.ylim(-40,90)
plt.plot(x1,np.ones(len(x1))*mean_diff,'k--',linewidth=2)
plt.plot(x1,np.ones(len(x1))*CI_upper,'k--',linewidth=2)
plt.plot(x1,np.ones(len(x1))*CI_lower,'k--',linewidth=2)

############# Mean difference and stdev difference for 3T and 1T subgroup ##############
mean_diff_3T = np.mean(diff_gfr_3T)
std_diff_3T = np.std(diff_gfr_3T)
mean_diff_1T = np.mean(diff_gfr_1T)
std_diff_1T = np.std(diff_gfr_1T)

print('Mean Difference 3T:', mean_diff_3T,'Stdev Difference 3T:', std_diff_3T)
print('Mean Difference 1T:', mean_diff_1T,'Stdev  Difference  1T:', std_diff_1T)
t, p = stats.ttest_ind(gfr_3T,isogfr_3T,equal_var=True)
print('t 3T:',t,'p 3T:',p)

t, p = stats.ttest_ind(gfr_1T,isogfr_1T,equal_var=True)
print('t 1T:',t,'p 1T:',p)


################## Analysis within 30% of reference GFR #################
d1 = (abs(diff_gfr)/isogfr)#*100
n_30 = sum(d1 < 0.3)/len(d1)*100
print('Percentage of patients within 30% of reference SK-GFR= ', n_30)
