import numpy as np
import matplotlib.pyplot as plt


from IPython.display import Image
Image(filename='Best_fit_PDF_values.png')


for i in range(13):
    plt.hist(MVN_4000[:,i], bins=100)
plt.title('All HERAPDF Parameter Distributions')


import seaborn as sns
colors=sns.color_palette("rocket",3)
# sns.set_style("white")
plt.style.use('seaborn-paper')
#plt.rc('text', usetex=True)
fig, axes = plt.subplots(nrows=4, ncols=4,figsize=(10,15))
axes[0,0].hist(MVN_4000[:,0],bins=100, label='Bg')
axes[0,1].hist(MVN_4000[:,1],bins=100, label='Cg')
axes[0,2].hist(MVN_4000[:,2],bins=100,label='Aprig')
axes[0,3].hist(MVN_4000[:,3],bins=100, label='Buv')
axes[1,0].hist(MVN_4000[:,4],bins=100, label='Cuv')
axes[1,1].hist(MVN_4000[:,5],bins=100,label='Euv')
axes[1,2].hist(MVN_4000[:,6],bins=100, label='Bdv')
axes[1,3].hist(MVN_4000[:,7],bins=100, label='Cdv')
axes[2,0].hist(MVN_4000[:,8],bins=100, label='CUbar')
axes[2,1].hist(MVN_4000[:,9],bins=100,label='DUbar')
axes[2,2].hist(MVN_4000[:,10],bins=100,label='ADbar')
axes[2,3].hist(MVN_4000[:,11],bins=100,label='BDbar')
axes[3,0].hist(MVN_4000[:,12],bins=100,label='CDbar')
axes[3,1].hist(MVN_4000[:,13],bins=100,label='CDbar')
axes[3,2].hist(MVN_4000[:,13],bins=100,label='CDbar')
axes[3,3].hist(MVN_4000[:,13],bins=100,label='CDbar')
plt.tight_layout(); plt.suptitle('HERAPDF Parameters')
titles = ['$B_g$','$C_g$','$A_g$','$B_g$','$B_{u_v}$','$C_{u_v}$','$E_{u_v}$','$B_{d_v}$','$C_{d_v}$','$C_{Ubar}$','$D_U$','$A_{Dbar}$','$B_{Dbar}$','CDbar','CDbar','CDbar','CDbar']
for i, ax in enumerate(axes.flatten()):
    ax.set(title=titles[i], xlabel='value')
    ax.legend()
# plt.minorticks_on()
# plt.tick_params(direction='in',right=True, top=True)
# plt.tick_params(labelsize=14)
# plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9 , top=0.9, wspace=0.2, hspace=0.4)
#plt.savefig('HERAPDF_params_MVN_4000_unweughted.png', dpi=300, bbox_inches='tight')
plt.show()


mean_chi2 = np.mean(MVN_4000_chi2)
chi2_diff = MVN_4000_chi2 - mean_chi2
chi2_diff, chi2_diff.shape


Bg = MVN_4000[:-1,0]
weights=np.exp(-0.5*(chi2_diff))/Bg
weights = 4000*weights/np.sum(weights)
weights


plt.hist(weights.flatten(), bins=50, range=(0,10)); plt.title('$w_{B_g}$', fontsize=13)


-0.009 + 0.005


import matplotlib.pyplot as plt; import numpy as np
MVN_4000= np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/Compute_chi2/MVN_samples/MVN_4000.npy')
MVN_4000_chi2 = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/Compute_chi2/chi2_array_4000.npy')
# dof = 377
# MVN_4000_chi2_per_dof=MVN_4000_chi2/377
# MVN_4000_chi2

mean_chi2 = np.mean(MVN_4000_chi2)
chi2_diff = abs(MVN_4000_chi2 - mean_chi2)
chi2_diff, chi2_diff.shape
weights = np.empty((3999, 14))
for i in range(14):
    weights[:,i] = np.exp(-0.5 * (chi2_diff))/MVN_4000[:-1,i]
    weights[:,i] = 3999 * weights[:,i]/np.sum(weights[:,i])

Bg = MVN_4000[:-1,0]
weights_Bg=np.exp(-0.5*(chi2_diff))/Bg
weights_Bg = 3999*weights_Bg/np.sum(weights_Bg)

# plt, axs = plt.subplots(1,2,figsize=(14,7))
# axs[0].hist(Bg.flatten(), range=(-0.2,-0.003),bins=50)
# axs[0].set_title(r'$B_g$ Unweighted Distribution', size=18)
# axs[1].hist(Bg.flatten(), weights=weights_Bg, color='r',range=(-0.2,-0.003),bins=50)
# axs[1].set_title(r'$B_g$ Weighted Distribution', size=18)
# axs[1].set_ylim(0,280)
# axs[0].set_ylim(0,280)
plt.rcParams["figure.figsize"] = [7, 7]
plt.hist(Bg.flatten(), range=(-0.2,-0.03),bins=50, alpha=0.35, label=r'$B_g$ Unweighted Distribution')
n, bins, patches=plt.hist(Bg.flatten(), weights=weights_Bg, color='r',range=(-0.2,-0.03),bins=50, alpha=0.35, label=r'$B_g$ Weighted Distribution')
plt.legend(fontsize=13, loc='best')
print(weights_Bg)
#plt.savefig('1_data_Bg.png')
n


import matplotlib.pyplot as plt
Ag = MVN_4000[:-1,10]
weights_Ag=np.exp(-0.5*(chi2_diff))/Ag
weights_Ag = 3999*weights_Ag/np.sum(weights_Ag)

# plt, axs = plt.subplots(1,2,figsize=(14,7))
# axs[0].hist(Bg.flatten(), range=(-0.2,-0.003),bins=50)
# axs[0].set_title(r'$B_g$ Unweighted Distribution', size=18)
# axs[1].hist(Bg.flatten(), weights=weights_Bg, color='r',range=(-0.2,-0.003),bins=50)
# axs[1].set_title(r'$B_g$ Weighted Distribution', size=18)
# axs[1].set_ylim(0,280)
# axs[0].set_ylim(0,280)
plt.rcParams["figure.figsize"] = [7, 7]
plt.hist(Ag.flatten(),bins=50, alpha=0.35, label=r'$A_g$ Unweighted Distribution')
plt.hist(Ag.flatten(), weights=weights_Ag, color='r',bins=50, alpha=0.35, label=r'$A_g$ Weighted Distribution')
plt.legend(fontsize=13, loc='lower left')



COV = np.load('COV.npy')
cov_diag = COV.diagonal()
sigma_2=cov_diag
sigma_2_Bg= cov_diag[0]; sigma_Bg=np.sqrt(sigma_2_Bg);sigma_Bg


bg_bar = np.mean(Bg) 
print(r'the mean $\bar{B_g}=$ ', bg_bar, '$\sigma_{B_g}=\sqrt{\Sigma_{00}}=$', sigma_Bg)


z_68 = 1.8
B_g_L = bg_bar-(z_68*sigma_Bg/(np.sqrt(4000)))
B_g_U = bg_bar+(z_68*sigma_Bg/(np.sqrt(4000)))
print('the lower and upper bounds of $\hat{B_g}^{Gauss}$ are', B_g_L, 'and ', B_g_U, 'respectively')


import scipy.stats as st
z_95=st.norm.ppf((1-.95)/2)
z_95


MVN_4000= np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/Compute_chi2/MVN_samples/MVN_4000.npy')
MVN_4000_chi2 = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/Compute_chi2/chi2_array_4000.npy')
# dof = 377
# MVN_4000_chi2_per_dof=MVN_4000_chi2/377
# MVN_4000_chi2

mean_chi2 = np.mean(MVN_4000_chi2)
chi2_diff = abs(MVN_4000_chi2 - mean_chi2)
chi2_diff, chi2_diff.shape
weights = np.empty((3999, 14))
for i in range(14):
    weights[:,i] = np.exp(-0.5 * (chi2_diff))/MVN_4000[:-1,i]
    weights[:,i] = 3999 * weights[:,i]/np.sum(weights[:,i])
    #weights[:,i] = sp.special.expit(weights[:,i])
print(weights[:10,0])


#FILTER WEIGHTS
filtered_weights=[]

for i in range(14):
    #mean weight for parameter i
    mean_weight_i = np.mean(weights[:,i])
    std_weight_i = np.std(weights[:,i])
    final_list = [x for x in weights[:,i] if (x > mean_weight_i - 4 * std_weight_i)]
    final_list = [x for x in final_list if (x < mean_weight_i + 4 * std_weight_i)]
    filtered_weights.append(np.array(final_list))
    
filtered_weights_ = [np.array(x) for x in filtered_weights]
print('UNFILTERED WEIGHTS\n')
print(weights.shape, '\n\n', weights[:,0], '\n\n', weights[:,0].mean(), '\n\n', weights[:,0].std())
print('\n \n\n FILTERED WEIGHTS\n')
print(np.array(filtered_weights_).shape, '\n\n', filtered_weights_[0], '\n\n', filtered_weights_[0].mean(), '\n\n', filtered_weights_[0].std())
fig, ax = plt.subplots(1, 2, figsize=(10,10))
ax[0].hist(filtered_weights_[0], bins=100, range=(0,10), label=r'$w_k^i = \frac{N_{samples} exp^{-\frac{1}{2} (\chi_k ^2 - E[\chi^2])}}{\mathcal{N}(\theta_i;\mu_i, \sigma_i) \ \sum_{k=1}^{N_{samples}} w_k}$')
ax[0].set_xlabel('Filtered Weights HERA ONLY', fontsize=15)
ax[1].hist(weights[:,0], bins=100, range=(0,10), label=r'$w_k^i = \frac{N_{samples} exp^{-\frac{1}{2} (\chi_k ^2 - E[\chi^2])}}{\mathcal{N}(\theta_i;\mu_i, \sigma_i) \ \sum_{k=1}^{N_{samples}} w_k}$')
ax[1].set_xlabel('Uniltered Weights HERA ONLY', fontsize=15)
ax[0].legend(fontsize=13); ax[0].legend(fontsize=13)
print('shapes are', filtered_weights[0].shape, weights[:,0].shape)
plt.tight_layout()


np.array(list_of_tuples[i][1])
np.einsum()


get_ipython().getoutput("pwd")


import numpy as np;
MVN_4000_MASTER = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/MVN_4000_MASTER.npy')
chi2_array_ALL_DATA_4k = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/local/ALL_DATA_5k/chi2_array_ALL_DATA_4k.npy')

#to avoid overflow take data type as float 128 to handle exponentiation
chi2_array_ALL_DATA_4k = chi2_array_ALL_DATA_4k.astype(np.float128)
MVN_4000_MASTER = MVN_4000_MASTER.astype(np.float128)

mean_chi2 = np.mean(chi2_array_ALL_DATA_4k)
chi2_diff = abs(chi2_array_ALL_DATA_4k - mean_chi2)
chi2_diff, chi2_diff.shape


weights_OLD = np.empty((4000, 14))
for i in range(14):
    weights_OLD[:,i] = np.exp(-0.5 * (chi2_diff))/MVN_4000_MASTER[:,i]
    weights_OLD[:,i] = 4000 * weights[:,i]/np.sum(weights[:,i])
    #weights[:,i] = sp.special.expit(weights[:,i])
print(weights_OLD[:10,0])


np.log(4000)


log_numerator = np.empty((4000,14))
for i in range(14):
    log_numerator[:,i] =  - 0.5 * (chi2_diff)
log_numerator #no normalization factor


MVN_4000_MASTER[:,2]


log_den = np.empty((4000,14))
for i in range(14):
    log_den[:,i] = np.log(MVN_4000_MASTER[:,i])
log_den[:,0], log_den[:,2]


log_RHS = np.empty((4000,14))
for i in range(14):
    log_RHS[:,i] = abs(log_numerator[:,i] - log_den[:,i])
print('log(RHS) = ', log_RHS[:,0], '\n')
print('therefore, log(w) = log(RHS)')


mean_log_RHS = np.empty((4000,14))
for i in range(14):
    mean_log_RHS[:,i] = np.mean(log_RHS[:,i])

weights = np.empty((4000,14))
for i in range(14):
    weights[:,i] = abs(log_RHS[:,i] - mean_log_RHS[:,i])
weights[:,0]


for i in range(14):
    weights[:,i] = 4000 * weights[:,i]/np.sum(weights[:,i])
weights[:,i]


weights[:,2]


# import numpy as np;
# MVN_4000_MASTER = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/MVN_4000_MASTER.npy')
# chi2_array_ALL_DATA_4k = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/local/ALL_DATA_5k/chi2_array_ALL_DATA_4k.npy')

# #to avoid overflow take data type as float 128 to handle exponentiation
# chi2_array_ALL_DATA_4k = chi2_array_ALL_DATA_4k.astype(np.float128)
# MVN_4000_MASTER = MVN_4000_MASTER.astype(np.float128)
#np.seterr(divide='ignore', invalid='ignore', over='ignore')

# mean_chi2 = np.mean(chi2_array_ALL_DATA_4k)
# chi2_diff = abs(chi2_array_ALL_DATA_4k - mean_chi2)
# chi2_diff, chi2_diff.shape
# weights = np.empty((4000, 14))
# for i in range(14):
#     weights[:,i] = np.exp(-0.5 * (chi2_diff))/MVN_4000_MASTER[:,i]
#     weights[:,i] = 4000 * weights[:,i]/np.sum(weights[:,i])
#     #weights[:,i] = sp.special.expit(weights[:,i])
# print(weights[:10,0])



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
        if (weight_value > (mean_weight - 6*std_weight_value)) and (weight_value < (mean_weight + 6*std_weight_value)):
            #if weight_value < (mean_weight + 4*std_weight_value):

            param_list_i.append(param_value)
            weight_list_i.append(weight_value)
    tuple_i = (param_list_i, weight_list_i)
    list_of_tuples.append(tuple_i)
#len(list_of_tuples)                
#list_of_tuples[1]


# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 17})

# params = np.array([-0.61856E-01 ,5.5593, 0.16618,-0.38300,0.81056,4.8239,9.9226,1.0301,4.8456,7.0603,1.5439 , 0.26877,-0.12732 , 9.5810])

# # weights = np.empty((3999, 14))
# # for i in range(14):
# #     weights[:,i] = np.exp(-0.5 * (chi2_diff))/params[i]
# #     weights[:,i] = 3999 * weights[:,i]/np.sum(weights[:,i])
# # print(weights[:,0])


# #There could be one weights that happens to be very large at 0
# titles = ['$B_g$','$C_g$','$A_g$','$B_g$','$B_{u_v}$','$C_{u_v}$','$E_{u_v}$','$B_{d_v}$','$C_{d_v}$','$C_{Ubar}$','$D_U$','$A_{Dbar}$','$B_{Dbar}$','$C_{Dbar}$']
# #['Bg','Cg','Aprig','Bprig','Buv','Cuv','Euv','Bdv','Cdv','CUbar','DUbar','ADbar','BDbar','CDbar']
# #['$B_g$','$C_g$','$A_g$','$B_g$','$B_{u_v}$','$C_{u_v}$','$E_{u_v}$','$B_{d_v}$','$C_{d_v}$','$C_{Ubar}$','$D_U$','$A_{Dbar}$','$B_{Dbar}$','CDbar']


# fig, axes = plt.subplots(nrows=14, ncols=3, figsize=(40,60))
# #for i, ax in enumerate(axes.flatten()):

# #PLOT UNWEIGHTED DISTRIBUTIONS (AT COL 0)
# for i in range(14):
#     axes[i,0].hist(list_of_tuples[i][0], bins=50)

#     #axes[i,0].set(title=titles[i] + ' Unweighted', xlabel='value')
#     axes[i,0].set_title(titles[i] + ' Unweighted', size=25)
#     axes[i,0].set_xlabel('value', size=20)
#     axes[i,0].set_ylim(0,320)

# #PLOT WEIGHTED DISTRIBUTIONS
# for i in range(14):
#     axes[i,1].hist(MVN_4000_MASTER[:,i].flatten(), weights=weights[:,i], bins=50, color = 'r')
#     #axes[i,1].set(title=titles[i] + ' Weighted', xlabel='value')
#     axes[i,1].set_title(titles[i] + ' Weighted Unfiltered',size=25)
#     axes[i,1].set_xlabel('value', size=20)
#     axes[i,1].set_ylim(0,320)
    
# ##FILTER WEIGHTS

# ##PLOT WEIGHTED AND FILTERED    
                  
    
#     #axes[i,0].legend()
# # # plt.minorticks_on()
# plt.tight_layout()
# #plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9 , top=0.9, wspace=0.2, hspace=0.4)
# #plt.savefig('all_data_4k_all_params_FILTERED.png')
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
#import scipy as sp

MVN_4000_MASTER = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/MVN_4000_MASTER.npy')
chi2_array_ALL_DATA_4k = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/local/ALL_DATA_5k/chi2_array_ALL_DATA_4k.npy')

#to avoid overflow take data type as float 128 to handle exponentiation
chi2_array_ALL_DATA_4k = chi2_array_ALL_DATA_4k.astype(np.float128)
MVN_4000_MASTER = MVN_4000_MASTER.astype(np.float128)

#ignore overflow and division errors
#np.seterr(divide='ignore', invalid='ignore', over='ignore')

#take log


mean_chi2 = np.mean(chi2_array_ALL_DATA_4k)
chi2_diff =abs(chi2_array_ALL_DATA_4k - mean_chi2)


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
        if (weight_value > (mean_weight - 5*std_weight_value)) and (weight_value < (mean_weight + 5*std_weight_value)):
            #if weight_value < (mean_weight + 4*std_weight_value):

            param_list_i.append(param_value)
            weight_list_i.append(weight_value)
    tuple_i = (param_list_i, weight_list_i)
    list_of_tuples.append(tuple_i)
#len(list_of_tuples)                
#list_of_tuples[1]

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


fig, axes = plt.subplots(nrows=14, ncols=3, figsize=(20,30))
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
#plt.tight_layout()
plt.subplots_adjust(left=0.125, bottom=0, right=0.9 , top=0.9, wspace=0.2, hspace=0.9)
#plt.savefig('all_data_4k_all_params_FILTERED.png')
plt.show()


weights[:,0].shape, len(list_of_tuples[0][1])


import numpy as np
import matplotlib.pyplot as plt
#import scipy as sp

MVN_4000_MASTER = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/MVN_4000_MASTER.npy')
chi2_array_ALL_DATA_4k = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/local/ALL_DATA_5k/chi2_array_ALL_DATA_4k.npy')

#to avoid overflow take data type as float 128 to handle exponentiation
chi2_array_ALL_DATA_4k = chi2_array_ALL_DATA_4k.astype(np.float128)
MVN_4000_MASTER = MVN_4000_MASTER.astype(np.float128)

#ignore overflow and division errors
#np.seterr(divide='ignore', invalid='ignore', over='ignore')

#for parameter i: pairs_i = (param_val, weight_i, std_i), then select weights to be within 4 std of the weights mean. (only take parameter values corresponding to those weights)

list_of_tuples = []
        
for i in range(14):
    param_list_i=[]
    weight_list_i = []
    for k in range(4000):
        param_value = MVN_4000_MASTER[k, i] #at the kth point, for parameter i
        weight_value = weights[k,i]
        std_weight_value = np.std(weights[:,i])
        mean_weight = np.mean(weights[:,i])
        if (weight_value > (mean_weight - 5*std_weight_value)) and (weight_value < (mean_weight + 5*std_weight_value)):
            #if weight_value < (mean_weight + 4*std_weight_value):

            param_list_i.append(param_value)
            weight_list_i.append(weight_value)
    tuple_i = (param_list_i, weight_list_i)
    list_of_tuples.append(tuple_i)
#len(list_of_tuples)                
#list_of_tuples[1]

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


fig, axes = plt.subplots(nrows=14, ncols=3, figsize=(20,30))
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
#plt.tight_layout()
plt.subplots_adjust(left=0.125, bottom=0, right=0.9 , top=0.9, wspace=0.2, hspace=0.9)
#plt.savefig('all_data_4k_all_params_FILTERED.png')
plt.show()


weight_2 = np.exp(-0.5 * chi2_diff)/MVN_4000_MASTER[:,2]


weight_2 = 4000 * weight_2/(np.sum(weight_2))
plt.hist(weight_2, bins=50)


fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(20,20))

for i in range(7):
    axes[i,0].hist(MVN_4000_MASTER[:,i].flatten(), bins=100, color = 'r', alpha=0.4,label='Gaussian')
    axes[i,0].hist(np.array(list_of_tuples[i][0]), weights=np.array(list_of_tuples[i][1]), bins=100, color = 'g',alpha=0.3, label='Reweighted')
    #axes[i,1].set(title=titles[i] + ' Weighted', xlabel='value')
    axes[i,0].set_title('All Data '+ titles[i] )
    axes[i,0].set_xlabel('value')
    axes[i,0].set_ylim(0,320)
    axes[i,0].legend()
for j in range(0,7):
    axes[j,1].hist(MVN_4000_MASTER[:,j+7].flatten(), bins=100, color = 'r', alpha=0.4,label='Gaussian')
    axes[j,1].hist(np.array(list_of_tuples[j+7][0]), weights=np.array(list_of_tuples[j+7][1]), bins=100, color = 'g',alpha=0.3, label='Reweighted')
    #axes[i,1].set(title=titles[i] + ' Weighted', xlabel='value')
    axes[j,1].set_title('All Data ' +titles[j+7] )
    axes[j,1].set_xlabel('value')
    axes[j,1].set_ylim(0,320)
    axes[j,1].legend()
    
plt.tight_layout()
plt.show()


hists_unweighted=[]; hists_weighted = []
bins_unweighted=[]; bins_weighted=[]
for i in range(14):
    hist, bins, pathces = plt.hist(MVN_4000_MASTER[:,i].flatten(), bins=50)
    hists_unweighted.append(hist)
    bins_unweighted.append(bins)
#     axes[i,0].hist(MVN_4000_MASTER[:,i].flatten(), bins=50, color = 'r', alpha=0.4,label='Gaussian')
#     axes[i,0].hist(np.array(list_of_tuples[i][0]), weights=np.array(list_of_tuples[i][1]), bins=50, color = 'g',alpha=0.3, label='Reweighted')
    #axes[i,1].set(title=titles[i] + ' Weighted', xlabel='value')
#     axes[i,0].set_title(titles[i] )
#     axes[i,0].set_xlabel('value')
#     axes[i,0].set_ylim(0,320)
#     axes[i,0].legend(



# QQ Plot
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot

# generate univariate observations
data = MVN[:,0]
# q-q plot
qqplot(data, line='s')
qqplot(MVN[:,1], line='s')
plt.show()
qqplot()



# QQ Plot
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
import statsmodels.api as sm

fig = plt.figure()
#add_subplot(nrows, ncols, index,
for i in range(14):
    
    ax_i = fig.add_subplot(14,2,1)
    sm.graphics.qqplot(MVN_4000[:-1,0], ax=ax_i)

fig.tight_layout()




fig, axes = plt.subplots(nrows=4, ncols=4,figsize=(10,15))
plt.tight_layout(); plt.suptitle('Quantile-Quantile Plots for HERAPDF Parameters')

data=[MVN[:,1],
MVN[:,2],
MVN[:,3],
MVN[:,4],
MVN[:,5],
MVN[:,6],
MVN[:,7],
MVN[:,8],
MVN[:,9],
MVN[:,10],
MVN[:,11],
MVN[:,12],
MVN[:,13],
MVN[:,13],
MVN[:,13], MVN[:,13]]

for i, ax in enumerate(axes.flatten()):
    qqplot(data[i], ax = ax, line='s')
    ax.set(title=titles[i], xlabel='value')
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9 , top=0.9, wspace=0.2, hspace=0.4)
plt.savefig('QQPlots_HERAPDF_params_MVN.png', dpi=300, bbox_inches='tight')
plt.show()


MVN.mean(axis=0)


def get_weights(z, means, cov):
    z = np.array(z)
    
    mu = means # MVN.mean(axis=0)
    cov = np.array(cov)
    N = len(z)
    temp1 = np.linalg.det(cov) ** (-1/2)
    temp2 = np.exp(-.5 * (z - mu).T @ np.linalg.inv(cov) @ (z - mu))
    return (2 * np.pi) ** (-N/2) * temp1 * temp2

weights = [get_weights(z=MVN[i,:], means=MVN.mean(axis=0), cov=COV) for i in range(13)]
weights


np.at


MVN.shape


MVN[0,:]


means.shape


COV.shape


# data=[MVN[:,1],
# MVN[:,2],
# MVN[:,3],
# MVN[:,4],
# MVN[:,5],
# MVN[:,6],
# MVN[:,7],
# MVN[:,8],
# MVN[:,9],
# MVN[:,10],
# MVN[:,11],
# MVN[:,12],
# MVN[:,13]]

def f(z, μ, Σ):
    """
    The density function of multivariate normal distribution.

    Parameters
    ---------------
    z: ndarray(float, dim=2)
        random vector, N by 1
    μ: ndarray(float, dim=1 or 2)
        the mean of z, N by 1
    Σ: ndarray(float, dim=2)
        the covarianece matrix of z, N by 1
    """

    z = np.array(z)
    μ = np.array(μ)
    Σ = np.array(Σ)

    N = z.size

    temp1 = np.linalg.det(Σ) ** (-1/2)
    temp2 = np.exp(-.5 * (z - μ).T @ np.linalg.inv(Σ) @ (z - μ))

    return (2 * np.pi) ** (-N/2) * temp1 * temp2

f(MVN[1,:], means, COV)


def my_mv_pdf(x, mu, cov):
    k = len(x)
    uu = mu.reshape(k,1)
    xx = x.reshape(k,1)
    t1 = (2*np.pi)**2
    t2 = np.linalg.det(cov)
    t3 = 1.0/np.sqrt(t1*t2)
    t4 =(xx-uu).T
    t5 = np.linalg.inv(cov)
    t6= (xx-uu)
    t7 = -0.5 *(np.dot(t4, t5).dot(t6))
    result = t3* np.exp(t7)
    return result
my_mv_pdf(MVN[1,:], means, COV)


class MultivariateNormal:
    """
    Class of multivariate normal distribution.

    Parameters
    ----------
    μ: ndarray(float, dim=1)
        the mean of z, N by 1
    Σ: ndarray(float, dim=2)
        the covarianece matrix of z, N by 1

    Arguments
    ---------
    μ, Σ:
        see parameters
    μs: list(ndarray(float, dim=1))
        list of mean vectors μ1 and μ2 in order
    Σs: list(list(ndarray(float, dim=2)))
        2 dimensional list of covariance matrices
        Σ11, Σ12, Σ21, Σ22 in order
    βs: list(ndarray(float, dim=1))
        list of regression coefficients β1 and β2 in order
    """

    def __init__(self, μ, Σ):
        "initialization"
        self.μ = np.array(μ)
        self.Σ = np.atleast_2d(Σ)

    def partition(self, k):
        """
        Given k, partition the random vector z into a size k vector z1
        and a size N-k vector z2. Partition the mean vector μ into
        μ1 and μ2, and the covariance matrix Σ into Σ11, Σ12, Σ21, Σ22
        correspondingly. Compute the regression coefficients β1 and β2
        using the partitioned arrays.
        """
        μ = self.μ
        Σ = self.Σ

        self.μs = [μ[:k], μ[k:]]
        self.Σs = [[Σ[:k, :k], Σ[:k, k:]],
                   [Σ[k:, :k], Σ[k:, k:]]]

        self.βs = [self.Σs[0][1] @ np.linalg.inv(self.Σs[1][1]),
                   self.Σs[1][0] @ np.linalg.inv(self.Σs[0][0])]

    def cond_dist(self, ind, z):
        """
        Compute the conditional distribution of z1 given z2, or reversely.
        Argument ind determines whether we compute the conditional
        distribution of z1 (ind=0) or z2 (ind=1).

        Returns
        ---------
        μ_hat: ndarray(float, ndim=1)
            The conditional mean of z1 or z2.
        Σ_hat: ndarray(float, ndim=2)
            The conditional covariance matrix of z1 or z2.
        """
        β = self.βs[ind]
        μs = self.μs
        Σs = self.Σs

        μ_hat = μs[ind] + β @ (z - μs[1-ind])
        Σ_hat = Σs[ind][ind] - β @ Σs[1-ind][1-ind] @ β.T

        return μ_hat, Σ_hat
multi_normal = MultivariateNormal(means, COV)
multi_normal


cov_list[5] = [0.460E-04, -0.829E-04, -0.565E-04, -0.914E-05, 0.403E-04, 0.861E-03] 
cov_list[6]= [0.165E-03, 0.298E-04,-0.274E-03,-0.198E-04,-0.574E-03, 0.421E-02, 0.755E-01]
cov_list[7] = [0.510E-04, 0.855E-04,-0.644E-04,-0.101E-04,-0.111E-04,-0.975E-05,-0.446E-03, 0.540E-03]
cov_list[8] =[0.246E-03,-0.593E-03,-0.290E-03,-0.491E-04, 0.140E-04,-0.398E-03, 0.296E-02, 0.170E-02, 0.158E-01 ]
cov_list[9] =[0.935E-03,-0.214E-02,-0.136E-02,-0.173E-03, 0.242E-03, 0.473E-03, 0.194E-03, 0.662E-03, 0.120E-03, 0.498E-01]
cov_list[10] = [-0.191E-02, 0.429E-02, 0.264E-02, 0.335E-03,-0.221E-03,-0.661E-03, 0.638E-02,-0.487E-03,-0.170E-02, 0.253E-01, 0.982E-01 ]
cov_list[11] = [-0.774E-05, 0.139E-04, 0.497E-05, 0.232E-05, 0.327E-05,-0.318E-06, 0.436E-04, 0.570E-05, 0.243E-06, 0.168E-04,-0.739E-04, 0.156E-04]
cov_list[12] = [-0.558E-05,-0.582E-06, 0.425E-05, 0.155E-05, 0.768E-06,-0.388E-07, 0.580E-05, 0.193E-05,-0.315E-05, 0.182E-04,-0.388E-04, 0.518E-05, 0.304E-05]
cov_list[13] = [0.349E-02,-0.135E-01,-0.437E-02,-0.589E-03, 0.215E-03, 0.131E-04,-0.652E-02, 0.240E-02, 0.188E-02,-0.218E-01, 0.363E-01, 0.110E-03, 0.182E-04, 0.370E+00]


cov_list


cov_list_list = [[None]*14]*14;


COV = np.empty((14,14))
for i in range(len(COV)):
    cov_list_list[i]= cov_list[i]
#     for j in range(len(COV[i])):
#         COV[i][j] = cov_list_list[i][j]
            
#COV
cov_list_list


# for line in chain_cov_mat:
#     row = line.strip().split()
#     for i in range(14):
#         for row_val in row[i].split():
            
#             for j in range(14):
#                 for col_val in row[i][j].split():
#                     COV[row_val][col] = float(row[i][j])
cov_list=[]
COV = np.empty((14,14))

# delimeters = "-", " "
# regexPattern = '|'.join(map(re.escape, delimiters))
pattern = re.compile(r'[\s\S.\d\D\w\W]\d\.\d\d\d[E]-\d\d')
#matches = pattern.finditer(text_to_search)
# for match in matches:
#     print(match)
for line in lines[127:143]:
    rows = line.strip().split('\n')
    triang_rows = rows[0]
#     for row in triang_rows.split('\n'):

    for row in triang_rows.strip().split('\s'):

        matches = pattern.finditer(row)
        inner_list=[]
        for match in matches:
            inner_list.append(float(row[match.span()[0]:match.span()[1]]))
            #cov_list.append(float(row[match.span()[0]:match.span()[1]]))
        cov_list.append(inner_list)
#         split_row = re.split(regexPattern, row)
#         for val in split_row:
#             cov_list.append(float(val))
        
        #for val in row:
            
#     for value in value_0:
        
#         cov_list.append(float(value))
cov_list


cov_list[0] = [0.632e-03]
cov_list[1] = [0.872E-03, 0.117E-01]
cov_list[:4]


COV = np.empty((14,14))



len(cov_list)


length = max(map(len, cov_list))
cov = np.array([xi+[None]*(length-len(xi)) for xi in cov_list])
cov.shape


############WRITE GENERATED PARAMETERS INTO NEW MINUIT.IN FILE
with open('minuit_ex.in.txt', 'w') as second:
    second.write('set title\n')
    second.write('new  14p HERAPDF\n')
    second.write('parameters\n')
    #lets put 0 for the fourth column, meaning that this parameter is fixed
    second.write('    '+ '2'+ '    ' + "'Bg'"+'    '+str(generated_params[0])+ '    '+'0.\n')
    second.write('    '+ '3'+ '    ' + "'Cg'"+'    '+str(generated_params[1])+ '    '+'0.\n')
    second.write('    '+ '7'+ '    ' + "'Aprig'"+'    '+str(generated_params[2])+ '    '+'0.\n')
    second.write('    '+ '8'+ '    ' + "'Bprig'"+'    '+str(generated_params[3])+ '    '+'0.\n')
    second.write('    '+ '9'+ '    ' + "'Cprig'"+'    '+str(generated_params[4])+ '    '+'0.\n')
    second.write('    '+ '12'+ '    ' + "'Buv'"+'    '+str(generated_params[5])+ '    '+'0.\n')
    second.write('    '+ '13'+ '    ' + "'Cuv'"+'    '+str(generated_params[6])+ '    '+'0.\n')
    second.write('    '+ '15'+ '    ' + "'Euv'"+'    '+str(generated_params[7])+ '    '+'0.\n')
    second.write('    '+ '22'+ '    ' + "'Bdv'"+'    '+str(generated_params[8])+ '    '+'0.\n')
    second.write('    '+ '23'+ '    ' + "'Cdv'"+'    '+str(generated_params[9])+ '    '+'0.\n')
    second.write('    '+ '33'+ '    ' + "'CUbar'"+'    '+str(generated_params[10])+ '    '+'0.\n')
    second.write('    '+ '34'+ '    ' + "'DUbar'"+'    '+str(generated_params[11])+ '    '+'0.\n')
    second.write('    '+ '41'+ '    ' + "'ADbar'"+'    '+str(generated_params[12])+ '    '+'0.\n')
    second.write('    '+ '42'+ '    ' + "'BDbar'"+'    '+str(generated_params[13])+ '    '+'0.\n')
    second.write('    '+ '43'+ '    ' + "'CDbar'"+'    '+str(generated_params[14])+ '    '+'0.\n')
    second.write('\n\n\n')
    second.write('migrad 200000\n')
    second.write('hesse\n')
    second.write('hesse\n')
    second.write('set print 3\n\n')
    second.write('return')
#we dont have to close it since we are using a context manager "with open()"


import itertools as IT
with open(filename, 'r') as f:
    lines = IT.chain(IT.islice(f, 0, 4), IT.islice(f, 5, 14) )
arr = np.genfromtxt(lines)


import numpy as np
#make a list of dtypes for each of the columns that we want

dtype1 = np.dtype([('NO.', 'int'), ('NAME', 'str'), ('VALUE', 'float32'), ('ERROR', 'float32')])
a = np.loadtxt(filename, dtype=dtype1, skiprows=106,  max_rows=4, usecols=(0, 1, 2, 3))
#np.loadtxt(filename, dtype)
a['VALUE']


words


import pandas as pd
df = pd.read_csv(filename, names=['NO','NAME','VALUE','ERROR'])[95:112]
#pd.read_csv(filename)ERROR
df.NO.apply(lambda x: pd.Series(str(x).split("\s+")))
#df.columns=['NO','NAME','VALUE','ERROR']


len(np.array(means))
means=np.array(means).astype(np.float)
means



def covariance_matrix(X):
    m = len(X) 
    mean = np.mean(X)
    cov_matrix = (X - mean).T.dot((X - mean)) / m-1
    np.random.seed(2020)
    return cov_matrix + 0.00001 
cov_mat_sig = np.array(covariance_matrix(means))
cov_mat_sig


filename = 'minuit.out.txt'
infile = open(filename, 'r')
lines = infile.readlines()
with open('minuit.in.txt', 'w') as second:
    for line in lines[9:n]:
        split_line = line.strip().split('      ')#the delimeter is 6 spaces to separate the columns
        for i in split_line:
            second.write(i)
infile.close()
second.close()
