import numpy as np
import matplotlib.pyplot as plt


MVN_4000 = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/Compute_chi2/MVN_samples/MVN_4000.npy'); MVN_4000


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


MVN_4000_chi2 = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/local/HERA_4k/chi2_array_4000.npy')
dof = 377
#MVN_4000_chi2_per_dof=MVN_4000_chi2/377
#MVN_4000_chi2
MVN_4000_chi2=MVN_4000_chi2.astype('float64')


mean_chi2 = np.mean(MVN_4000_chi2); print(mean_chi2)

chi2_diff = MVN_4000_chi2 - mean_chi2
chi2_diff, chi2_diff.shape


Bg = MVN_4000[:,0]
weights=np.exp(-0.5*(chi2_diff))/Bg
weights = 4000*weights/np.sum(weights)
weights


plt.hist(weights.flatten(), bins=50, range=(0,2)); plt.title('weights w')


-0.009 + 0.005


import matplotlib.pyplot as plt
Bg = MVN_4000[:,0]
weights_Bg=np.exp(-0.5*(chi2_diff))/Bg
weights_Bg = 4000*weights_Bg/np.sum(weights_Bg)

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


import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 17})

params = np.array([-0.61856E-01 ,5.5593, 0.16618,-0.38300,0.81056,4.8239,9.9226,1.0301,4.8456,7.0603,1.5439 , 0.26877,-0.12732 , 9.5810])

weights = np.empty((3999, 14))
for i in range(14):
    weights[:,i] = np.exp(-0.5 * (chi2_diff))/params[i]
    weights[:,i] = 3999 * weights[:,i]/np.sum(weights[:,i])
print(weights[:,0])
#There could be one weights that happens to be very large at 0
titles = ['$B_g$','$C_g$','$A_g$','$B_g$','$B_{u_v}$','$C_{u_v}$','$E_{u_v}$','$B_{d_v}$','$C_{d_v}$','$C_{Ubar}$','$D_U$','$A_{Dbar}$','$B_{Dbar}$','$C_{Dbar}$']
#['Bg','Cg','Aprig','Bprig','Buv','Cuv','Euv','Bdv','Cdv','CUbar','DUbar','ADbar','BDbar','CDbar']
#['$B_g$','$C_g$','$A_g$','$B_g$','$B_{u_v}$','$C_{u_v}$','$E_{u_v}$','$B_{d_v}$','$C_{d_v}$','$C_{Ubar}$','$D_U$','$A_{Dbar}$','$B_{Dbar}$','CDbar']


fig, axes = plt.subplots(nrows=14, ncols=2, figsize=(40,60))
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
    axes[i,1].set_ylim(0,320)
    
    
    #axes[i,0].legend()
# # plt.minorticks_on()
plt.tight_layout()
#plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9 , top=0.9, wspace=0.2, hspace=0.4)
plt.savefig('all_data_4k_all_params.png')
plt.show()


for arr in weights:
    print(arr)
    break


filtered_weights=[]
for i in range(14):
    mean_i = np.mean(weights[:,i])
    std_i = np.std(weights[:,i])
    final_list = [x for x in weights[:,i] if (x > mean_i - 4 * std_i)]
    final_list = [x for x in final_list if (x < mean_i + 4 * std_i)]
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


5000*10/3600


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
