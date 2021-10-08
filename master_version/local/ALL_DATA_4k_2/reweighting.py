import numpy as np
import matplotlib.pyplot as plt
#import scipy as sp

MVN_4000_MASTER = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/MVN_4000_MASTER.npy')
chi2_array_ALL_DATA_4k = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/local/ALL_DATA_5k/chi2_array_ALL_DATA_4k.npy')

#to avoid overflow take data type as float 128 to handle exponentiation
chi2_array_ALL_DATA_4k = chi2_array_ALL_DATA_4k.astype(np.float128)
MVN_4000_MASTER = MVN_4000_MASTER.astype(np.float128)

#ignore overflow and division errors
np.seterr(divide='ignore', invalid='ignore', over='ignore')

#take log
chi2_array_ALL_DATA_4k=np.log(chi2_array_ALL_DATA_4k)




mean_chi2 = np.mean(chi2_array_ALL_DATA_4k)
chi2_diff = chi2_array_ALL_DATA_4k - mean_chi2 
chi2_diff, chi2_diff.shape
weights = np.empty((4000, 14))
for i in range(14):
    weights[:,i] = np.exp(-0.5 * (chi2_diff))/MVN_4000_MASTER[:,i]
    weights[:,i] = 4000 * weights[:,i]/np.sum(weights[:,i])
#    weights[:,i] = sp.special.expit(weights[:,i])
print(weights[:10,0])

#for parameter i: pairs_i = (param_val, weight_i, std_i), then select weights

list_of_tuples = []

        
for i in range(14):
    param_list_i=[]
    weight_list_i = []
    for k in range(4000):
        param_value = MVN_4000_MASTER[k, i] #at the kth point, for parameter i
        weight_value = weights[k,i]
        std_weight_value = np.std(weights[:,i])
        mean_weight = np.mean(weights[:,i])
        if (weight_value > (mean_weight - 4*std_weight_value)) and (weight_value < (mean_weight + 4*std_weight_value)):
            #if weight_value < (mean_weight + 4*std_weight_value):

            param_list_i.append(param_value)
            weight_list_i.append(weight_value)
    tuple_i = (param_list_i, weight_list_i)
    list_of_tuples.append(tuple_i)
#len(list_of_tuples)                
#list_of_tuples[1]

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 17})

params = np.array([-0.61856E-01 ,5.5593, 0.16618,-0.38300,0.81056,4.8239,9.9226,1.0301,4.8456,7.0603,1.5439 , 0.26877,-0.12732 , 9.5810])

# weights = np.empty((3999, 14))
# for i in range(14):
#     weights[:,i] = np.exp(-0.5 * (chi2_diff))/params[i]
#     weights[:,i] = 3999 * weights[:,i]/np.sum(weights[:,i])
# print(weights[:,0])
#There could be one weights that happens to be very large at 0
titles = ['$B_g$','$C_g$','$A_g$','$B_g$','$B_{u_v}$','$C_{u_v}$','$E_{u_v}$','$B_{d_v}$','$C_{d_v}$','$C_{Ubar}$','$D_U$','$A_{Dbar}$','$B_{Dbar}$','$C_{Dbar}$']
#['Bg','Cg','Aprig','Bprig','Buv','Cuv','Euv','Bdv','Cdv','CUbar','DUbar','ADbar','BDbar','CDbar']
#['$B_g$','$C_g$','$A_g$','$B_g$','$B_{u_v}$','$C_{u_v}$','$E_{u_v}$','$B_{d_v}$','$C_{d_v}$','$C_{Ubar}$','$D_U$','$A_{Dbar}$','$B_{Dbar}$','CDbar']


fig, axes = plt.subplots(nrows=14, ncols=3)
#for i, ax in enumerate(axes.flatten()):

#PLOT UNWEIGHTED DISTRIBUTIONS (AT COL 0)
for i in range(14):
    #axes[i,0].hist(list_of_tuples[i][0], bins=50)
    axes[i,0].hist(MVN_4000_MASTER[:,i].flatten(), bins=50, color='g')

    #axes[i,0].set(title=titles[i] + ' Unweighted', xlabel='value')
    axes[i,0].set_title(titles[i] + ' Unweighted')
    axes[i,0].set_xlabel('value')
    axes[i,0].set_ylim(0,320)

#PLOT WEIGHTED DISTRIBUTIONS
for i in range(14):
    axes[i,1].hist(MVN_4000_MASTER[:,i].flatten(), weights=weights[:,i], bins=50, color = 'r')
    #axes[i,1].set(title=titles[i] + ' Weighted', xlabel='value')
    axes[i,1].set_title(titles[i] + ' Weighted Unfiltered')
    axes[i,1].set_xlabel('value')
    axes[i,1].set_ylim(0,320)
    
##FILTER WEIGHTS

##PLOT WEIGHTED AND FILTERED    
for i in range(14):
    axes[i,2].hist(np.array(list_of_tuples[i][0]), weights=np.array(list_of_tuples[i][1]), bins=50, color = 'r')
    #axes[i,1].set(title=titles[i] + ' Weighted', xlabel='value')
    axes[i,2].set_title(titles[i] + ' Weighted Filtered')
    axes[i,2].set_xlabel('value')
    axes[i,2].set_ylim(0,320)
    
    #axes[i,0].legend()
# # plt.minorticks_on()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9 , top=0.9, wspace=0.2, hspace=0.4)
#plt.savefig('all_data_4k_all_params_FILTERED.png')
plt.show()