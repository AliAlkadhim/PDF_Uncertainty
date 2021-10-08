import numpy as np
import matplotlib.pyplot as plt


MVN_4000 = np.load('MVN_samples/MVN_4000.npy'); MVN_4000


MVN_4000[:,2].mean()


MVN_25k = np.load('MVN_samples/MVN_25k.npy'); MVN_25k


chi2_25k=np.load('chi2_array_25k.npy'); chi2_25k


chi2_25k.shape


MVN_4000.shape


Bg = MVN_4000[:,0]
print('The sample size=', len(Bg))
plt.hist(Bg.flatten(), bins=100); plt.title(r'$Bg$ unweighted distribtution')


from IPython.display import Image
Image(filename='Best_fit_PDF_values.png')


Cg = MVN_4000[:,1]
plt.hist(Cg.flatten(), bins=50)


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


MVN_4000_chi2 = np.load('chi2_array_4000.npy')
dof = 377
MVN_4000_chi2_per_dof=MVN_4000_chi2/377
MVN_4000_chi2


mean_chi2 = np.mean(MVN_4000_chi2)
chi2_diff = MVN_4000_chi2 - mean_chi2
chi2_diff, chi2_diff.shape


Bg = MVN_4000[:-1,0]
weights=np.exp(-0.5*(chi2_diff))/Bg
weights = 4000*weights/np.sum(weights)
weights


plt.hist(weights.flatten(), bins=50, range=(0,10)); plt.title('$w_{B_g}$', fontsize=13)


-0.009 + 0.005


import matplotlib.pyplot as plt
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
plt.hist(Bg.flatten(), range=(-0.2,-0.003),bins=50, alpha=0.35, label=r'$B_g$ Unweighted Distribution')
n, bins, patches=plt.hist(Bg.flatten(), weights=weights_Bg, color='r',range=(-0.2,-0.003),bins=50, alpha=0.35, label=r'$B_g$ Weighted Distribution')
plt.legend(fontsize=13, loc='best')
print(weights_Bg)
#plt.savefig('1_data_Bg.png')
n


import matplotlib.pyplot as plt
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
plt.hist(Bg.flatten(), range=(-0.2,-0.003),bins=50, alpha=0.35, label=r'$B_g$ Unweighted Distribution')
plt.hist(Bg.flatten()*weights_Bg, color='r',range=(-0.2,-0.003),bins=50, alpha=0.35, label=r'$B_g$ Weighted Distribution')
plt.legend(fontsize=13, loc='best')
print(weights_Bg)
#plt.savefig('1_data_Bg.png')
n


# a = np.histogram(Bg.flatten(), weights=weights_Bg); plt.hist(a[2], bins=50)





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


import matplotlib.pyplot as plt
Ag = MVN_4000[:-1,2]
weights_Ag=np.exp(-0.5*(chi2_diff))/Ag
weights_Ag = 3999*weights_Ag/np.sum(weights_Ag)

# plt, axs = plt.subplots(1,2,figsize=(14,7))
# axs[0].hist(Ag.flatten(), range=(-0.2,-0.003),bins=50)
# axs[0].set_title(r'$B_g$ Unweighted Distribution', size=18)
# axs[1].hist(Ag.flatten(), weights=weights_Ag, color='r',range=(-0.2,-0.003),bins=50)
# axs[1].set_title(r'$B_g$ Weighted Distribution', size=18)
# axs[1].set_ylim(0,280)
# axs[0].set_ylim(0,280)
plt.rcParams["figure.figsize"] = [7, 7]
plt.hist(Ag.flatten(),bins=50, alpha=0.35, label=r'$A^{\prime}_g$ Unweighted Distribution')
plt.hist(Ag.flatten(), weights=weights_Ag, color='r',bins=50, alpha=0.35, label=r'$A^{\prime}_g$ Weighted Distribution')
plt.legend(fontsize=13, loc='best')
print(weights_Ag)
plt.savefig('1_data_Ag.png')


import matplotlib.pyplot as plt
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


plt.hist(Bg.flatten(), range=(-0.2,-0.003),bins=50, alpha=0.35, label=r'$B_g$ Unweighted Distribution')
plt.hist(Bg.flatten(), weights=weights_Bg, color='r',range=(-0.2,-0.003),bins=50, alpha=0.35, label=r'$B_g$ Weighted Distribution')
plt.legend(fontsize=13, loc='best')
print(weights_Bg)
plt.savefig('1_data_Bg.png')


Bg = MVN_4000[:-1,0]; weights_Bg


params=[]
for i in range(14):
    params.append(np.array(MVN_4000[:-1,i]))
params[0]


# weights = np.empty((3999, 2))
# weights[:,0] = np.exp(-0.5 * (chi2_diff))
# weights[:,0] = 4000* weights[:,0]/ np.sum(weights[:,0])
# weights[:,0]


weights = np.empty((3999, 2))
weights.shape


weights = np.empty((3999, 14))
for i in range(14):
    weights[:,i] = np.exp(-0.5 * (chi2_diff))/params[i]
    weights[:,i] = 24998 * weights[:,i]/np.sum(weights[:,i])
weights[:,0]





for i in range(3):
    print(weights[:,i][i])


weights[:,0]


weights = np.empty((3999, 14))
for i in range(13):
    weights[:,i] = np.exp(-0.5 * (chi2_diff))/MVN_4000[:-1,i]
    weights[:,i] = 3999 * weights[:,i]/np.sum(weights[:,i])
print(weights.shape, '\n\n', weights[:,0], '\n\n', weights[:,0].mean(), '\n\n', weights[:,0].std())


import matplotlib.pyplot as plt
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
plt.hist(Bg.flatten(),bins=50, alpha=0.35, label=r'$B_g$ Unweighted Distribution')
plt.hist(Bg.flatten(), weights=weights_Bg**1.3, color='r',bins=50, alpha=0.35, label=r'$B_g$ Weighted Distribution')
plt.legend(fontsize=13, loc='best')
print(weights_Bg)
plt.savefig('all_data_Bg.png')


# MVN1=MVN_4000[:-1,i].flatten()-0.05
# MVN2=MVN_4000[:-1,i].flatten()+0.02
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 17})

# weights = np.empty((3999, 14))
# for i in range(14):
#     weights[:,i] = np.exp(-0.5 * (chi2_diff))/params[i]
#     weights[:,i] = 3999 * weights[:,i]/np.sum(weights[:,i])
# print(weights[:,0])
# #There could be one weights that happens to be very large at 0
# titles = ['$B_g$','$C_g$','$A_g$','$B_g$','$B_{u_v}$','$C_{u_v}$','$E_{u_v}$','$B_{d_v}$','$C_{d_v}$','$C_{Ubar}$','$D_U$','$A_{Dbar}$','$B_{Dbar}$','$C_{Dbar}$']
# #['Bg','Cg','Aprig','Bprig','Buv','Cuv','Euv','Bdv','Cdv','CUbar','DUbar','ADbar','BDbar','CDbar']
# #['$B_g$','$C_g$','$A_g$','$B_g$','$B_{u_v}$','$C_{u_v}$','$E_{u_v}$','$B_{d_v}$','$C_{d_v}$','$C_{Ubar}$','$D_U$','$A_{Dbar}$','$B_{Dbar}$','CDbar']


# fig, axes = plt.subplots(nrows=14, ncols=2, figsize=(40,60))
# #for i, ax in enumerate(axes.flatten()):

# #PLOT UNWEIGHTED DISTRIBUTIONS (AT COL 0)
# for i in range(14):
#     axes[i,0].hist(MVN_4000[:-1,i].flatten(), bins=100)

#     #axes[i,0].set(title=titles[i] + ' Unweighted', xlabel='value')
#     axes[i,0].set_title(titles[i] + ' Unweighted', size=25)
#     axes[i,0].set_xlabel('value', size=20)
#     axes[i,0].set_ylim(0,180)

# #PLOT WEIGHTED DISTRIBUTIONS
# for i in range(14):
#     axes[i,1].hist(MVN1, weights=weights[:,i], bins=100, color = 'r')
#     axes[i,1].hist(MVN2, weights=weights[:,i], bins=100, color = 'r')
#     #axes[i,1].set(title=titles[i] + ' Weighted', xlabel='value')
#     axes[i,1].set_title(titles[i] + ' Weighted',size=25)
#     axes[i,1].set_xlabel('value', size=20)
#     axes[i,1].set_ylim(0,180)
    
    
#     #axes[i,0].legend()
# # # plt.minorticks_on()
# plt.tight_layout()
# #plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9 , top=0.9, wspace=0.2, hspace=0.4)
# plt.savefig('4k_HERA_LHC.png')
# plt.show()


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
fig, ax = plt.subplots(1, 2)
ax[0].hist(filtered_weights_[0], bins=100, range=(0,10))
ax[0].set_xlabel('Filtered Weights', fontsize=15)
ax[1].hist(weights[:,0], bins=100, range=(0,10))
ax[1].set_xlabel('Uniltered Weights', fontsize=15)
print('shapes are', filtered_weights[0].shape, weights[:,0].shape)



#for parameter i: pairs_i = (param_val, weight_i, std_i), then select weights

list_of_tuples = []

        
for i in range(14):
    param_list_i=[]
    weight_list_i = []
    for k in range(3999):
        param_value = MVN_4000[k, i] #at the kth point, for parameter i
        weight_value = weights[k,i]
        std_weight_value = np.std(weights[:,i])
        mean_weight = np.mean(weights[:,i])
        if (weight_value > (mean_weight - 4*std_weight_value)) and (weight_value < (mean_weight + 4*std_weight_value)):
            #if weight_value < (mean_weight + 4*std_weight_value):

            param_list_i.append(param_value)
            weight_list_i.append(weight_value)
    tuple_i = (param_list_i, weight_list_i)
    list_of_tuples.append(tuple_i)
len(list_of_tuples)                



np.array(list_of_tuples[i][1])


import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 17})

params = np.array([-0.61856E-01 ,5.5593, 0.16618,-0.38300,0.81056,4.8239,9.9226,1.0301,4.8456,7.0603,1.5439 , 0.26877,-0.12732 , 9.5810])

# weights = np.empty((3999, 14))
# for i in range(14):
#     weights[:,i] = np.exp(-0.5 * (chi2_diff))/params[i]
#     weights[:,i] = 3999 * weights[:,i]/np.sum(weights[:,i])
# print(weights[:,0])
weights = np.empty((3999, 14))
for i in range(14):
    weights[:,i] = np.exp(-0.5 * (chi2_diff))/MVN_4000[:-1,i]
    weights[:,i] = 3999 * weights[:,i]/np.sum(weights[:,i])
print(weights[:,0])

#There could be one weights that happens to be very large at 0
titles = ['$B_g$','$C_g$','$A_g$','$B_g$','$B_{u_v}$','$C_{u_v}$','$E_{u_v}$','$B_{d_v}$','$C_{d_v}$','$C_{Ubar}$','$D_U$','$A_{Dbar}$','$B_{Dbar}$','$C_{Dbar}$']
#['Bg','Cg','Aprig','Bprig','Buv','Cuv','Euv','Bdv','Cdv','CUbar','DUbar','ADbar','BDbar','CDbar']
#['$B_g$','$C_g$','$A_g$','$B_g$','$B_{u_v}$','$C_{u_v}$','$E_{u_v}$','$B_{d_v}$','$C_{d_v}$','$C_{Ubar}$','$D_U$','$A_{Dbar}$','$B_{Dbar}$','CDbar']


fig, axes = plt.subplots(nrows=14, ncols=3, figsize=(40,60))
#for i, ax in enumerate(axes.flatten()):

#PLOT UNWEIGHTED DISTRIBUTIONS (AT COL 0)
for i in range(14):
    axes[i,0].hist(list_of_tuples[i][0], bins=50)

    #axes[i,0].set(title=titles[i] + ' Unweighted', xlabel='value')
    axes[i,0].set_title(titles[i] + ' Unweighted', size=25)
    axes[i,0].set_xlabel('value', size=20)
    axes[i,0].set_ylim(0,320)

#PLOT WEIGHTED DISTRIBUTIONS
for i in range(14):
    axes[i,1].hist(MVN_4000[:-1,i].flatten(), weights=weights[:,i], bins=50, color = 'r')
    #axes[i,1].set(title=titles[i] + ' Weighted', xlabel='value')
    axes[i,1].set_title(titles[i] + ' Weighted Unfiltered',size=25)
    axes[i,1].set_xlabel('value', size=20)
    axes[i,1].set_ylim(0,320)
    
##FILTER WEIGHTS

##PLOT WEIGHTED AND FILTERED    
for i in range(14):
    axes[i,2].hist(np.array(list_of_tuples[i][0]), weights=np.array(list_of_tuples[i][1]), bins=50, color = 'r')
    #axes[i,1].set(title=titles[i] + ' Weighted', xlabel='value')
    axes[i,2].set_title(titles[i] + ' Weighted Filtered',size=25)
    axes[i,2].set_xlabel('value', size=20)
    axes[i,2].set_ylim(0,320)
    
    #axes[i,0].legend()
# # plt.minorticks_on()
plt.tight_layout()
#plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9 , top=0.9, wspace=0.2, hspace=0.4)
plt.savefig('all_data_4k_all_params_FILTERED.png')
plt.show()


import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 17})

params = np.array([-0.61856E-01 ,5.5593, 0.16618,-0.38300,0.81056,4.8239,9.9226,1.0301,4.8456,7.0603,1.5439 , 0.26877,-0.12732 , 9.5810])

# weights = np.empty((3999, 14))
# for i in range(14):
#     weights[:,i] = np.exp(-0.5 * (chi2_diff))/params[i]
#     weights[:,i] = 3999 * weights[:,i]/np.sum(weights[:,i])
# print(weights[:,0])
weights = np.empty((3999, 14))
for i in range(14):
    weights[:,i] = np.exp(-0.5 * (chi2_diff))/MVN_4000[:-1,i]
    weights[:,i] = 3999 * weights[:,i]/np.sum(weights[:,i])
print(weights[:,0])

#There could be one weights that happens to be very large at 0
titles = ['$B_g$','$C_g$','$A_g$','$B_g$','$B_{u_v}$','$C_{u_v}$','$E_{u_v}$','$B_{d_v}$','$C_{d_v}$','$C_{Ubar}$','$D_U$','$A_{Dbar}$','$B_{Dbar}$','$C_{Dbar}$']
#['Bg','Cg','Aprig','Bprig','Buv','Cuv','Euv','Bdv','Cdv','CUbar','DUbar','ADbar','BDbar','CDbar']
#['$B_g$','$C_g$','$A_g$','$B_g$','$B_{u_v}$','$C_{u_v}$','$E_{u_v}$','$B_{d_v}$','$C_{d_v}$','$C_{Ubar}$','$D_U$','$A_{Dbar}$','$B_{Dbar}$','CDbar']


fig, axes = plt.subplots(nrows=14, ncols=3, figsize=(40,60))
#for i, ax in enumerate(axes.flatten()):

#PLOT UNWEIGHTED DISTRIBUTIONS (AT COL 0)
for i in range(14):
    axes[i,0].hist(MVN_4000[:-1,i].flatten(), bins=50)

    #axes[i,0].set(title=titles[i] + ' Unweighted', xlabel='value')
    axes[i,0].set_title(titles[i] + ' Unweighted', size=25)
    axes[i,0].set_xlabel('value', size=20)
    axes[i,0].set_ylim(0,320)

#PLOT WEIGHTED DISTRIBUTIONS
for i in range(14):
    axes[i,1].hist(MVN_4000[:-1,i].flatten(), weights=weights[:,i], bins=50, color = 'r')
    #axes[i,1].set(title=titles[i] + ' Weighted', xlabel='value')
    axes[i,1].set_title(titles[i] + ' Weighted',size=25)
    axes[i,1].set_xlabel('value', size=20)
    axes[i,1].set_ylim(0,520)
    
##FILTER WEIGHTS

#PLOT WEIGHTED AND FILTERED    
# for i in range(14):
#     axes[i,1].hist(MVN_4000[:-1,i].flatten(), weights=filtered_weights[i], bins=50, color = 'r')
#     #axes[i,1].set(title=titles[i] + ' Weighted', xlabel='value')
#     axes[i,1].set_title(titles[i] + ' Filtered Weighted',size=25)
#     axes[i,1].set_xlabel('value', size=20)
#     axes[i,1].set_ylim(0,520)
    
    #axes[i,0].legend()
# # plt.minorticks_on()
plt.tight_layout()
#plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9 , top=0.9, wspace=0.2, hspace=0.4)
#plt.savefig('all_data_4k_all_params.png')
plt.show()


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
fig, ax = plt.subplots(1, 2)
ax[0].hist(filtered_weights_[0], bins=100, range=(0,10))
ax[0].set_xlabel('Filtered Weights', fontsize=15)
ax[1].hist(weights[:,0], bins=100, range=(0,10))
ax[1].set_xlabel('Uniltered Weights', fontsize=15)


filtered_weights
#this is a list of arrays: each array in this list is the array of filtered weightes 


weights = np.empty((3999, 14))
for i in range(13):
    weights[:,i] = np.exp(-0.5 * (chi2_diff))/MVN_4000[:-1,i]
    weights[:,i] = 3999 * weights[:,i]/np.sum(weights[:,i])
print(weights.shape, '\n\n', weights[:,0], '\n\n', weights[:,0].mean(), '\n\n', weights[:,0].std())


# filter_mask = np.empty(np.shape(weights), dtype=bool)
# # filtered_weights_list =[]

# for i in range(14):
#     mean_i = np.mean(weights[:,i])
#     print(mean_i)
#     std_i = np.std(weights[:,i])
#     filter_mask[:,i] = weights[:,i] > (mean_i - 4 * std_i)
#     filter_mask[:,i] = filter_mask[:,i]  < (mean_i + 4 * std_i)
#     filtered_weights[:,i] = weights[:,i][filter_mask[:,i]]
#     filtered_weights_list.append(filtered_weights[:,i])

#filter_mask, filter_mask.shape
#weights[filter_mask]
#filtered_weights
#for i in range(14):

# filtered_weights = filtered_weights[filter_mask]
# filtered_weights

#print(filter_mask.shape, filter_mask, '\n\n')
# print(filtered_weights.shape, filtered_weights[:,0])
#filter_mask


filter_mask = np.empty(np.shape(weights), dtype=bool)
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
filter_mask = (weights > (mean_weights - 4 * std_weights))

filter_mask.reshape(np.shape(weights))
# w = weights[filter_mask]
# w.reshape(np.shape(weights))
weights[:,0][filter_mask[:,0]]


filtered_weights=weights[filter_mask]
filtered_weights


# uw_hist =  np.histogram(MVN_4000[:-1,0], bins=50)
# plt.hist(uw_hist[1].flatten(), bins=50)


unweighted_hists = []
for i in range(13):
    uw_hist, uwbins =  np.histogram(MVN[:-1,i])
    unweighted_hists.append()


import matplotlib.pyplot as plt

filter_mask = np.empty(np.shape(weights), dtype=bool)

#There could be one weights that happens to be very large at 0
titles = ['$B_g$','$C_g$','$A_g$','$B_g$','$B_{u_v}$','$C_{u_v}$','$E_{u_v}$','$B_{d_v}$','$C_{d_v}$','$C_{Ubar}$','$D_U$','$A_{Dbar}$','$B_{Dbar}$','$C_{Dbar}$']
#['Bg','Cg','Aprig','Bprig','Buv','Cuv','Euv','Bdv','Cdv','CUbar','DUbar','ADbar','BDbar','CDbar']
#['$B_g$','$C_g$','$A_g$','$B_g$','$B_{u_v}$','$C_{u_v}$','$E_{u_v}$','$B_{d_v}$','$C_{d_v}$','$C_{Ubar}$','$D_U$','$A_{Dbar}$','$B_{Dbar}$','CDbar']


fig, axes = plt.subplots(nrows=14, ncols=2, figsize=(40,60))
#for i, ax in enumerate(axes.flatten()):
# for i in range(14):
#     mean_i = np.mean(weights[:,i])
#     print(mean_i)
#     std_i = np.std(weights[:,i])
#     filter_mask[:,i] = weights[:,i] > (mean_i - 4 * std_i)
#     filter_mask[:,i] = filter_mask[:,i]  < (mean_i + 4 * std_i)
#     filtered_weights[:,i] = weights[:,i][filter_mask[:,i]]

#list of tuples (weight, index), then sort them in ascending weight, then cut off the list of thuples

#PLOT UNWEIGHTED DISTRIBUTIONS (AT COL 0)
for i in range(14):
    axes[i,0].hist(MVN_4000[:-1,i], bins=50)

    axes[i,0].set(title=titles[i] + ' Unweighted', xlabel='value')
    axes[i,0].set_ylim(0,280)

#PLOT WEIGHTED DISTRIBUTIONS
for i in range(14):
    #FILTER WEIGHTS
    mean_i = np.mean(weights[:,i])
    std_i = np.std(weights[:,i])
    filter_mask[:,i] = weights[:,i] > (mean_i - 2 * std_i)
    filter_mask[:,i] = filter_mask[:,i]  < (mean_i + 2 * std_i)
    #print(filter_mask[:,i].shape, weights[:,i].shape)
    axes[i,1].hist(MVN_4000[1:,i][filter_mask[:,i]], weights=weights[:,i][filter_mask[:,i]], bins=50, color = 'r')
    #axes[i,1].set(title=titles[i] + ' Weighted and Filtered', xlabel='value')
    axes[i,1].set_ylim(0,280)
    
    
    #axes[i,0].legend()
# # plt.minorticks_on()
plt.tight_layout()
#plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9 , top=0.9, wspace=0.2, hspace=0.4)
plt.show()


param_names =['Bg','Cg','Aprig','Bprig','Buv','Cuv','Euv','Bdv','Cdv','CUbar','DUbar','ADbar','BDbar','CDbar','CDbar','CDbar','CDbar']
Bg = MVN_4000[:-1,0]
Cg = MVN_4000[:-1,1]
params = [Bg, Cg]
weights = np.exp(-0.5 * (chi2_diff))/params
weights = 4000 * weights/ np.sum(weights)
weights




weights_Bg=np.exp(-0.5*(chi2_diff))/Bg
weights_Bg = 4000*weights_Bg/np.sum(weights_Bg)

plt, axs = plt.subplots(1,2,figsize=(14,7))
axs[0].hist(Bg.flatten(), bins=100)
axs[0].set_title('Bg Unweighted Distribution')
axs[1].hist(Bg.flatten(), bins=100, weights=weights_Bg, color='r')
axs[1].set_title('Bg Weighted Distribution')
axs[1].set_ylim(0,150)
axs[0].set_ylim(0,150)



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
