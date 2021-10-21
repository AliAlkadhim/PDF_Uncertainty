import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex":True})


MVN_4000_MASTER = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/MVN_25k_MASTER.npy')
MVN_4000_MASTER, MVN_4000_MASTER.shape


for i in range(MVN_4000_MASTER.shape[0]):
    print('first point', MVN_4000_MASTER[i,:])
    break


MVN_4000_MASTER = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/MVN_25k_MASTER.npy')
COV_MASTER= np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/COV_MASTER.npy')
params_MASTER= np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/params_MASTER.npy')
COV_MASTER[0], params_MASTER[0]


MVN_4000_MASTER[:,0]


from scipy.stats import multivariate_normal
def f(MVN, mu, sigma):
    """
    The density function of multivariate normal distribution.
    N = size of the mean vector, or number of parameter points (14)
    MVN = the 2D MV Gaussian function
    sigma = the covariance matrix from our best-fit values
    """

    MVN_per_point_l=[]; MVN_per_point_l_scipy=[]
    for i in range(MVN.shape[0]):
        # z is the vector of parameters
        z = MVN[i,:]


        N = z.size

        temp1 = np.linalg.det(sigma) ** (-1/2)
        temp2 = np.exp(-.5 * (z - mu).T @ np.linalg.inv(sigma) @ (z - mu))
        MVN_per_point = (2 * np.pi) ** (-N/2) * temp1 * temp2
        MVN_per_point_l.append(MVN_per_point)
        MVN_from_func = np.array(MVN_per_point_l)
        MVN_from_scipy = multivariate_normal.pdf(z, mean=mu, cov=sigma)
        MVN_per_point_l_scipy.append(MVN_from_scipy)
    return MVN_from_func, np.array(MVN_per_point_l_scipy)


#multivariate_normal(np.linspace(0,10), params_MASTER, COV_MASTER)
COV_MASTER


MVN_per_point_from_func, MVN_per_point_from_scipy = f(MVN_4000_MASTER, params_MASTER, COV_MASTER); MVN_per_point_l


plt.hist(MVN_per_point_from_scipy)


MVN_per_point_l[MVN_per_point_l <0]


plt.hist(MVN_per_point_l)
plt.title('Multivariate Normal per point', fontsize=14)


chi2_array_ALL_DATA_4k = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/local/ALL_DATA_25k/chi2_array_ALL_DATA_25k.npy')
chi2_array_ALL_DATA_4k, chi2_array_ALL_DATA_4k.shape


#to avoid overflow take data type as float 128 to handle exponentiation
chi2_array_ALL_DATA_4k = chi2_array_ALL_DATA_4k.astype(np.float128)
MVN_4000_MASTER = MVN_4000_MASTER.astype(np.float128)
plt.hist(chi2_array_ALL_DATA_4k, bins=100)
plt.title('$\chi^2$ for All data', fontsize=20)
print(r'the mean and std of $\chi^2$ are respectively', np.mean(chi2_array_ALL_DATA_4k), np.std(chi2_array_ALL_DATA_4k))


chi2_array_ALL_DATA_4k = chi2_array_ALL_DATA_4k-np.mean(chi2_array_ALL_DATA_4k)
plt.hist(np.exp(-0.5*chi2_array_ALL_DATA_4k))
plt.title('$e^{-0.5 \chi^2}$')


chi2_array_ALL_DATA_4k.mean(), chi2_array_ALL_DATA_4k.std()


chi2_array_ALL_DATA_4k.min(), chi2_array_ALL_DATA_4k.max(), chi2_array_ALL_DATA_4k.max()/chi2_array_ALL_DATA_4k.min()


# z = MVN_per_point_l
mu = params_MASTER
sigma = COV_MASTER
second_term_l = []
for i in range(MVN_4000_MASTER.shape[0]):
    second_term = (MVN_4000_MASTER[i,:] - mu).T @ np.linalg.inv(sigma) @ (MVN_4000_MASTER[i,:] - mu)
    second_term_l.append(second_term)
plt.hist(np.array(second_term_l))# second term dist is much narrower, which is what we expect
plt.title('Second Term', fontsize=15)


list_of_tuples = []; 
MVG_within_1_sigma=[]
chi2_within_1_sigma=[] #np.empty((4000,14))
for i in range(14):
    param_list_i=[]
    chi2_list=[]
#     weight_list_i = []
    chi2_list_param_i=[]
    for k in range(4000):
        param_value = MVN_4000_MASTER[k, i] #at the kth point, for parameter i
        
        MVG_point_within_1s = MVN_per_point_l[k]
        #std = np.std(MVN_per_point_l)
        #mean_MVG = np.mean(MVN_per_point_l)
        
        std_MVN_value = np.std(MVN_4000_MASTER[:,i])
        mean_MVN_value = np.mean(MVN_4000_MASTER[:,i])
        if (param_value > (mean_MVN_value - 1*std_MVN_value)) and (param_value < (mean_MVN_value + 1*std_MVN_value)):
            #if weight_value < (mean_weight + 4*std_weight_value):

            #param_list_i.append(param_value)
            param_list_i.append(MVG_point_within_1s)
            chi2_list_param_i.append(chi2_array_ALL_DATA_4k[k])
    MVG_within_1_sigma.append(param_list_i)
    chi2_within_1_sigma.append(chi2_list_param_i)
            #chi2_within_1_sigma[k,i] = chi2_array_ALL_DATA_4k[k]
    tuple_i = (param_list_i, chi2_list)
    list_of_tuples.append(tuple_i)
#len(list_of_tuples)                
#list_of_tuples[1]


np.mean(chi2_within_1_sigma[0]), np.mean(chi2_within_1_sigma[2]), np.mean(chi2_within_1_sigma[3])


plt.hist(MVN_per_point_l)


best_fitchi2_25k =3369.427
#subtract chi^2(best-fit)
chi2_array_ALL_DATA_4k_diff = chi2_array_ALL_DATA_4k - np.mean(chi2_array_ALL_DATA_4k)
#subtract chi^2 mean
chi2_array_ALL_DATA_4k_diff_param_1 =chi2_array_ALL_DATA_4k_diff- best_fitchi2_25k
chi2_array_ALL_DATA_4k_diff_param_1


def MVG_BestFit(MVN, mu, sigma):
    """
    The density function of multivariate normal distribution.
    N = size of the mean vector, or number of parameter points (14)
    MVN = the 2D MV Gaussian function
    sigma = the covariance matrix from our best-fit values
    """

#     MVN_per_point_l=[]

    #z = np.atleast_2d(z)
    z = MVN


    N = z.size

    temp1 = np.linalg.det(sigma) ** (-1/2)
    temp2 = np.exp(-.5 * (z - mu).T @ np.linalg.inv(sigma) @ (z - mu))
    MVN_per_point = (2 * np.pi) ** (-N/2) * temp1 * temp2

    return MVN_per_point


MVG_best_fit = MVG_BestFit(params_MASTER, params_MASTER, COV_MASTER)
MVG_best_fit


#subtract MVG(best-fit)
MVN_per_point_l_diff = MVN_per_point_l - MVG_best_fit
MVN_per_point_l_diff


#subtract mean
MVN_per_point_l_diff_min = MVN_per_point_l_diff - np.mean(MVN_per_point_l_diff)
MVN_per_point_l_diff_min


plt.hist(MVN_per_point_l_diff_min)


#take only positive values so that we dont get error when computing log
MVN_per_point_l = MVN_per_point_l - np.mean(MVN_per_point_l)

MVN_per_point_l_pos = MVN_per_point_l[MVN_per_point_l >0]


plt.hist(MVN_4000_MASTER[:,1][MVN_per_point_l_diff_min >0], weights=weight_param_1_normalized, bins=100, label='Reweighted', alpha=0.3, density=True, range=(0.07, 0.08))
plt.hist(MVN_4000_MASTER[:,1][MVN_per_point_l_diff_min >0], bins=100, label='Gaussian',  alpha=0.3, density=True,range=(0.07, 0.08))
plt.legend()

positive_MVN_per_point_l_diff = MVN_per_point_l_diff[MVN_per_point_l_diff >0]

positive_chi2_array_ALL_DATA_4k_diff_param_1 = chi2_array_ALL_DATA_4k_diff_param_1[MVN_per_point_l_diff >0]; positive_chi2_array_ALL_DATA_4k_diff_param_1


#take only positive values so that we dont get error when computing log

positive_MVN_per_point_l_diff_min = MVN_per_point_l_diff_min[MVN_per_point_l_diff_min >0]

positive_chi2_array_ALL_DATA_4k_diff_param_1 = chi2_array_ALL_DATA_4k_diff_param_1[MVN_per_point_l_diff_min >0]



log_weight_param_1 = (-0.5*positive_chi2_array_ALL_DATA_4k_diff_param_1)/np.log(positive_MVN_per_point_l_diff)
log_weight_param_1


log_weight_param_1 = (-0.5*positive_chi2_array_ALL_DATA_4k_diff_param_1)/np.log(positive_MVN_per_point_l_diff_min)
log_weight_param_1


weight_param_1 = np.exp(log_weight_param_1)

weight_param_1_normalized = len(weight_param_1) * weight_param_1/np.sum(weight_param_1)
weight_param_1_normalized


# weight_param_1 = len(weight_param_1) * weight_param_1/np.sum(weight_param_1)
# weight_param_1


plt.hist(MVN_4000_MASTER[:,1][MVN_per_point_l_diff_min >0], weights=weight_param_1_normalized, bins=100, label='Reweighted', alpha=0.3, density=True, range=(0.07, 0.08))
plt.hist(MVN_4000_MASTER[:,1][MVN_per_point_l_diff_min >0], bins=100, label='Gaussian',  alpha=0.3, density=True,range=(0.07, 0.08))
plt.legend()


from scipy.stats import multivariate_normal
def MVG(MVN, mu, sigma):
    """
    The density function of multivariate normal distribution.
    N = size of the mean vector, or number of parameter points (14)
    MVN = the 2D MV Gaussian for the sampling of the parameters 
    sigma = the covariance matrix from our best-fit values
    """

    MVN_per_point_l=[]; MVN_per_point_l_scipy=[]
    for i in range(MVN.shape[0]):
        # z is the vector of parameters
        z = MVN[i,:]


        N = z.size

        temp1 = np.linalg.det(sigma) ** (-1/2)
        temp2 = np.exp(-.5 * (z - mu).T @ np.linalg.inv(sigma) @ (z - mu))
        MVN_per_point = (2 * np.pi) ** (-N/2) * temp1 * temp2
        MVN_per_point_l.append(MVN_per_point)
        MVN_from_func = np.array(MVN_per_point_l)
        MVN_from_scipy = multivariate_normal.pdf(z, mu, sigma)
        MVN_per_point_l_scipy.append(MVN_from_scipy)
    return MVN_from_func, np.array(MVN_per_point_l_scipy)


def MVG_BestFit(MVN, mu, sigma):
    """
    The density function of multivariate normal distribution.
    N = size of the mean vector, or number of parameter points (14)
    MVN = the 2D MV Gaussian function
    sigma = the covariance matrix from our best-fit values
    """

#     MVN_per_point_l=[]

    #z = np.atleast_2d(z)
    z = MVN


    N = z.size

    temp1 = np.linalg.det(sigma) ** (-1/2)
    temp2 = np.exp(-.5 * (z - mu).T @ np.linalg.inv(sigma) @ (z - mu))
    MVN_per_point = (2 * np.pi) ** (-N/2) * temp1 * temp2
    
    MVN_per_point_scipy = multivariate_normal.pdf(z, mu, sigma)
    return np.array(MVN_per_point), MVN_per_point_scipy


#LOAD DATA
chi2_array_ALL_DATA_4k = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/local/ALL_DATA_25k/chi2_array_ALL_DATA_25k.npy')
MVN_4000_MASTER = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/MVN_25k_MASTER.npy')
COV_MASTER= np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/COV_MASTER.npy')
params_MASTER= np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/params_MASTER.npy')
#FLOAT 128 FOR HIGHER PRECISION
chi2_array_ALL_DATA_4k = chi2_array_ALL_DATA_4k.astype(np.float128)
MVN_4000_MASTER = MVN_4000_MASTER.astype(np.float128)

#EVALUAGE THE MV NORMAL AT EACH POINT (AT EACH SET OF PARAMETERS)
MVN_per_point_l, MVN_per_point_l_scipy = MVG(MVN_4000_MASTER, params_MASTER, COV_MASTER)
#EVALUAGE THE MV NORMAL AT THE BEST-FIT POINT
best_fitchi2_25k =3369.427
MVG_best_fit, MVG_best_fit_scipy = MVG_BestFit(params_MASTER, params_MASTER, COV_MASTER)



fix, axs = plt.subplots(1,2, figsize=(13,13))

axs[0].hist(MVN_per_point_l, label = 'MVN_per_point_l')
axs[1].hist(MVN_per_point_l_scipy, label='MVN_per_point_l_scipy')

print('MVG Best fit = ', MVG_best_fit, 'MVG Best fit from scipy = ', MVG_best_fit_scipy)
        


plt.hist(chi2_array_ALL_DATA_4k/best_fitchi2_25k)


MVN_per_point_l_diff_mean[MVN_per_point_l_diff_mean <0]


MVN_per_point_l
import scipy.stats as st

MVN_scipy = st.multivariate_normal(


m=MVN_per_point_l/MVG_best_fit 
n=chi2_array_ALL_DATA_4k/best_fitchi2_25k

plt.hist(n/m)


np.mean(m/)




def filter_within_bestfit(chi2_arr, MVG_arr, MVN_sample):
    

    MVG_within_1_sigma=[]
[54]:
￼
np.mean(m/)
[55]:
0.007298643418836985046
[288]:
￼
​
​
    chi2_within_1_sigma=[]
    
    MVN_within_1_sigma = []
    
    for i in range(14):
        MVG_list=[]
        chi2_list=[]
        
        MVN_param_i = []
        for k in range(MVN_sample.shape[0]):
            param_value = MVN_sample[k, i]#sample parameter value of MVN 
            MVG_at_k = MVG_arr[k]
            chi2_at_k = chi2_arr[k]
            
            std_MVN_value = np.std(MVN_sample[:,i])#for all the k's (number of points, at parameter label i
            mean_MVN_value = np.mean(MVN_sample[:,i])
            
            if (param_value > (mean_MVN_value - 1*std_MVN_value)) and (param_value < (mean_MVN_value + 1*std_MVN_value)):
                MVG_list.append(MVG_at_k)
                chi2_list.append(chi2_at_k)
                
                MVN_param_i.append(param_value)
                
        MVN_within_1_sigma.append(np.array(MVN_param_i))
                
        MVG_within_1_sigma.append(np.array(MVG_list))
        chi2_within_1_sigma.append(np.array(chi2_list))
    
    return chi2_within_1_sigma, MVG_within_1_sigma, MVN_within_1_sigma


        
chi2_within_1_sigma, MVG_within_1_sigma, MVN_within_1_sigma = filter_within_bestfit(chi2_array_ALL_DATA_4k, MVN_per_point_l, MVN_4000_MASTER)
                                   
# chi2_within_1_sigma=[] #np.empty((4000,14))
# for i in range(14):
#     param_list_i=[]
#     chi2_list=[]
# #     weight_list_i = []
#     chi2_list_param_i=[]
#     for k in range(4000):
#         param_value = MVN_4000_MASTER[k, i] #at the kth point, for parameter i
        
#         MVG_point_within_1s = MVN_per_point_l[k]
#         #std = np.std(MVN_per_point_l)
#         #mean_MVG = np.mean(MVN_per_point_l)
        
#         std_MVN_value = np.std(MVN_4000_MASTER[:,i])
#         mean_MVN_value = np.mean(MVN_4000_MASTER[:,i])
#         if (param_value > (mean_MVN_value - 1*std_MVN_value)) and (param_value < (mean_MVN_value + 1*std_MVN_value)):
#             #if weight_value < (mean_weight + 4*std_weight_value):

#             #param_list_i.append(param_value)
#             param_list_i.append(MVG_point_within_1s)
#             chi2_list_param_i.append(chi2_array_ALL_DATA_4k[k])
#     MVG_within_1_sigma.append(param_list_i)
#     chi2_within_1_sigma.append(chi2_list_param_i)
#             #chi2_within_1_sigma[k,i] = chi2_array_ALL_DATA_4k[k]
#     tuple_i = (param_list_i, chi2_list)
#     list_of_tuples.append(tuple_i)
#len(list_of_tuples)                
#list_of_tuples[1]


plt.hist(MVG_within_1_sigma[0], label='within 2 $\sigma$ for parameter 1', alpha=0.3)
plt.hist(MVN_per_point_l, label = 'unfiltered',  alpha=0.3)
plt.legend()


plt.hist(chi2_within_1_sigma[2])




def MVG_BestFit(MVN, mu, sigma):
    """
    The density function of multivariate normal distribution.
    N = size of the mean vector, or number of parameter points (14)
    MVN = the 2D MV Gaussian function
    sigma = the covariance matrix from our best-fit values
    """

    z = MVN
    N = z.size

    temp1 = np.linalg.det(sigma) ** (-1/2)
    temp2 = np.exp(-.5 * (z - mu).T @ np.linalg.inv(sigma) @ (z - mu))
    MVN_per_point = (2 * np.pi) ** (-N/2) * temp1 * temp2

    return MVN_per_point

best_fitchi2_25k =3369.427
MVG_best_fit = MVG_BestFit(params_MASTER, params_MASTER, COV_MASTER)

MVN_per_point_l = f(MVN_4000_MASTER, params_MASTER, COV_MASTER)

def delta_mean(chi2_arr, MVG_arr):
    mean_chi2 = chi2_arr - np.mean(chi2_arr)
    mean_MVG = MVG_arr - np.mean(MVG_arr)

    return mean_chi2, mean_MVG

def delta_best_fit(chi2_arr, MVG_arr):
    delta_chi2 = chi2_arr - best_fitchi2_25k
    delta_MVG = MVG_arr - MVG_best_fit
    return delta_chi2, delta_MVG

def pos_log_mask(chi2_arr, MVG_arr):
    mask = MVG_arr > 0
    pos_MVG = MVG_arr[MVG_arr > 0]
    pos_chi2 = chi2_arr[MVG_arr >0]
    return pos_chi2, pos_MVG, mask

def calc_weight_from_log(chi2_arr, MVG_arr):
    log_weight_unnormalized = (-0.5 * chi2_arr) - (np.log(MVG_arr))
    #delta_log_weight_unnormalized = log_weight_unnormalized - np.mean(log_weight_unnormalized)
    delta_log_weight_unnormalized = log_weight_unnormalized 
    weight_unnormalized = np.exp(delta_log_weight_unnormalized)
    weight_normalized = weight_unnormalized.size * weight_unnormalized/np.sum(weight_unnormalized)
    return weight_normalized

def calc_weight_normally(chi2_arr, MVG_arr):
    w = np.exp(-0.5 * chi2_arr)/MVG_arr
    
    w_norm = w.size * w/np.sum(w)
    return w_norm

chi, MVG = delta_mean(chi2_array_ALL_DATA_4k, MVN_per_point_l)

chi2, MVG2 = delta_best_fit(chi, MVG)

chi3, MVG3 = delta_best_fit(chi2_within_1_sigma[0], MVG_within_1_sigma[0])



#chi_pos, MVG_pos, mask = pos_log_mask(chi, MVG) #works

chi_pos, MVG_pos, mask = pos_log_mask(chi, MVG)

#chi_pos2, MVG_pos2, mask = pos_log_mask(chi2_within_1_sigma[0], MVG_within_1_sigma[0])

w = calc_weight_from_log(chi_pos, MVG_pos)

w_norm = calc_weight_normally(chi2, MVG2)
plt.hist(w_norm, range=(0,1))
print(w.mean())


w, w.size


mask.size, MVN_within_1_sigma[0].size, MVN_within_1_sigma[0][mask].size


plt.hist(MVN_4000_MASTER[:,0][mask], label = 'G', alpha=0.3)
plt.hist(MVN_4000_MASTER[:,0][mask], weights= w, label = 'R', alpha=0.3)
plt.legend()


fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(20,20))

titles = ['$B_g$','$C_g$','$A_g$','$B_g$','$B_{u_v}$','$C_{u_v}$','$E_{u_v}$','$B_{d_v}$','$C_{d_v}$','$C_{Ubar}$','$D_U$','$A_{Dbar}$','$B_{Dbar}$','CDbar','CDbar','CDbar','CDbar']


for i in range(7):
    axes[i,0].hist(MVN_within_1_sigma[i][mask], bins=100, color = 'r', alpha=0.4,label='Gaussian', range=(0.160,0.162))
    axes[i,0].hist(MVN_within_1_sigma[i][mask], weights=w, bins=100, color = 'g',alpha=0.3, label='Reweighted', range=(0.160,0.162))
    #axes[i,1].set(title=titles[i] + ' Weighted', xlabel='value')
    axes[i,0].set_title('All Data '+ titles[i] )
    axes[i,0].set_xlabel('value')
    axes[i,0].set_ylim(0,200)
    axes[i,0].legend()
for j in range(0,7):
    axes[j,1].hist(MVN_within_1_sigma[i][mask], bins=100, color = 'r', alpha=0.4,label='Gaussian', range=(0.160,0.162))
    axes[j,1].hist(MVN_within_1_sigma[i][mask], weights=w, bins=100, color = 'g',alpha=0.3, label='Reweighted', range=(0.160,0.162))
    #axes[i,1].set(title=titles[i] + ' Weighted', xlabel='value')
    axes[j,1].set_title('All Data ' +titles[j+7] )
    axes[j,1].set_xlabel('value')
    axes[j,1].set_ylim(0,200)
    axes[j,1].legend()
    
plt.tight_layout()
plt.show()


chi2_array_ALL_DATA_4k = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/local/ALL_DATA_25k/chi2_array_ALL_DATA_25k.npy')
MVN_4000_MASTER = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/MVN_25k_MASTER.npy')
COV_MASTER= np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/COV_MASTER.npy')
params_MASTER= np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/params_MASTER.npy')

chi2_array_ALL_DATA_4k = chi2_array_ALL_DATA_4k.astype(np.float128)
MVN_4000_MASTER = MVN_4000_MASTER.astype(np.float128)

best_fitchi2_25k =3369.427
MVG_best_fit = MVG_BestFit(params_MASTER, params_MASTER, COV_MASTER)

MVN_per_point_l = f(MVN_4000_MASTER, params_MASTER, COV_MASTER)


#first subtract the mean from both chi2 and MVG
# MVN_per_point_l_diff_mean = MVN_per_point_l - np.mean(MVN_per_point_l)

# pos_chi2_diff_mean =pos_chi2 - np.mean(pos_chi2)

# #now subtract best-fit values of chi2 and MVG
MVN_per_point_l_diff_mean = MVN_per_point_l - MVG_best_fit
pos_chi2_diff_mean = chi2_array_ALL_DATA_4k - best_fitchi2_25k

#HERE is where the positive masking should be done (right before taking the log)
MVN_per_point_l_diff_mean_pos = MVN_per_point_l_diff_mean[MVN_per_point_l_diff_mean > 0]
pos_chi2_diff_mean_pos = pos_chi2_diff_mean[MVN_per_point_l_diff_mean > 0]

#now calculate the log weight
log_weight_unnormalized = (-0.5 * pos_chi2_diff_mean_pos) - (np.log(MVN_per_point_l_diff_mean_pos))

log_weight_unnormalized = log_weight_unnormalized - np.mean(log_weight_unnormalized)
#exponentiate to get the weight
weight_unnormalized = np.exp(log_weight_unnormalized)
#normalize
weight_normalized = len(weight_unnormalized) * weight_unnormalized/np.sum(weight_unnormalized)


MVN_param_1_pos = MVN_4000_MASTER[:,][MVN_per_point_l_diff_mean > 0]

# plt.hist(MVN_param_1_pos, weights=weight_normalized, bins=50, label='Reweighted', alpha=0.3)
# plt.hist(MVN_param_1_pos, bins=50, label='Gaussian',  alpha=0.3)
# plt.ylim(0,5)
fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(20,20))

titles = ['$B_g$','$C_g$','$A_g$','$B_g$','$B_{u_v}$','$C_{u_v}$','$E_{u_v}$','$B_{d_v}$','$C_{d_v}$','$C_{Ubar}$','$D_U$','$A_{Dbar}$','$B_{Dbar}$','CDbar','CDbar','CDbar','CDbar']


for i in range(7):
    axes[i,0].hist(MVN_4000_MASTER[:,i][MVN_per_point_l_diff_mean > 0], bins=100, color = 'r', alpha=0.4,label='Gaussian')
    axes[i,0].hist(MVN_4000_MASTER[:,i][MVN_per_point_l_diff_mean > 0], weights=weight_normalized, bins=100, color = 'g',alpha=0.3, label='Reweighted')
    #axes[i,1].set(title=titles[i] + ' Weighted', xlabel='value')
    axes[i,0].set_title('All Data '+ titles[i] )
    axes[i,0].set_xlabel('value')
    axes[i,0].set_ylim(0,200)
    axes[i,0].legend()
for j in range(0,7):
    axes[j,1].hist(MVN_4000_MASTER[:,i][MVN_per_point_l_diff_mean > 0], bins=100, color = 'r', alpha=0.4,label='Gaussian')
    axes[j,1].hist(MVN_4000_MASTER[:,i][MVN_per_point_l_diff_mean > 0], weights=weight_normalized, bins=100, color = 'g',alpha=0.3, label='Reweighted')
    #axes[i,1].set(title=titles[i] + ' Weighted', xlabel='value')
    axes[j,1].set_title('All Data ' +titles[j+7] )
    axes[j,1].set_xlabel('value')
    axes[j,1].set_ylim(0,200)
    axes[j,1].legend()
    
plt.tight_layout()
plt.show()


MVN_param_1_pos.shape, weight_normalized.shape


weight_normalized


plt.hist(weight_normalized, density=True)
plt.title('weights', fontsize=14)




plt.hist(MVN_4000_MASTER[:,2], weights=weight_param_1, bins=100, label='Reweighted', alpha=0.3, density=True, range=(-0.1302,-0.130))
plt.hist(MVN_4000_MASTER[:,2], bins=100, label='Gaussian',  alpha=0.3, density=True, range=(-0.131,-0.130))
plt.legend()


np.mean(MVG_within_1_sigma[0]), np.mean(MVG_within_1_sigma[2]), np.mean(MVG_within_1_sigma[3])


np.log(np.mean(MVG_within_1_sigma[0]))


param_1 = MVN_4000_MASTER[:,0]
weight_1 = 
plt.hist(param_1)


plt.hist(chi2_within_1_sigma[0], bins=100)
plt.title('$\chi^2$ values within $1\ \sigma$ of the best-fit values of the parameters', fontsize=18)


len(MVG_within_1_sigma), len(MVG_within_1_sigma[0])


plt.hist(MVG_within_1_sigma[0], bins=100, range=(0,3e16))
plt.title('MVG parameter values within $1\ \sigma$ of the best-fit values of the parameters', fontsize=18)


chi2_array_ALL_DATA_4k - np.mean(chi2_array_ALL_DATA_4k)


chi2_diff = chi2_array_ALL_DATA_4k - np.mean(chi2_array_ALL_DATA_4k)
w = -0.5*(chi2_diff-np.array(second_term_l))
plt.hist(w)


np.mean(chi2_diff)


mean_chi2 = np.mean(chi2_array_ALL_DATA_4k)
chi2_best_fit = 3369.427
#remember to rename diff (no need to take diff at this point)
# chi2_diff = chi2_array_ALL_DATA_4k - mean_chi2
chi2_diff = chi2_array_ALL_DATA_4k - chi2_best_fit

chi2_diff = chi2_diff - np.mean(chi2_diff)

log_numerator = -0.5*chi2_diff
log_numerator.shape
# for i in range(14):
#     log_numerator[:,i] =  - 0.5 * (chi2_diff)
# log_numerator #no normalization factor
log_numerator, np.mean(log_numerator)


# log_numerator = log_numerator - np.mean(log_numerator)
# log_numerator


np.log(MVN_per_point_l)


#plt.hist(log_numerator, range=(-10**6,0)) #this is equivalent to plt.hist(log_numerator/(e7), range=(-0.01,0))
# plt.title('log numerator')
MVN_per_point_l_diff = MVN_per_point_l - np.mean(MVN_per_point_l)
log_den = np.log(MVN_per_point_l_diff)
log_den.shape
# plt.hist(log_den)
# plt.title('log denomenator')

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
axes[0,0].hist(log_numerator); axes[0,0].set_title('log numerator')
axes[0,1].hist(log_den); axes[0,1].set_title('log denomenator')

# axes[1,0].hist(-0.5*np.array(chi2_within_1_sigma[0])); axes[1,0].set_title('log numerator within 1 sigma of best-fit values')
# axes[1,1].hist(np.log(MVG_within_1_sigma[0])); axes[1,1].set_title('log denomenator within 1 sigma of best-fit values')




#plt.hist(log_numerator, range=(-10**6,0)) #this is equivalent to plt.hist(log_numerator/(e7), range=(-0.01,0))
# plt.title('log numerator')
log_den = np.log(MVN_per_point_l)
log_den = log_den- np.mean(log_den)


chi2_within_1_sigma[0] = np.array(chi2_within_1_sigma[0]) - np.mean(np.array(chi2_within_1_sigma[0]))

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
axes[0,0].hist(log_numerator); axes[0,0].set_title('log numerator with mean subtracted')
axes[0,1].hist(log_den); axes[0,1].set_title('log denomenator with mean subtracted')

axes[1,0].hist(-0.5*chi2_within_1_sigma[0]); axes[1,0].set_title('log numerator within 1 sigma of best-fit values with mean subtracted')
axes[1,1].hist(np.log(MVG_within_1_sigma[0])); axes[1,1].set_title('log denomenator within 1 sigma of best-fit values with mean subtracted')

plt.tight_layout()


# def log_den(MVN_per_point_l):
#     log_den_l = []
#     for i in range(MVN_per_point_l.shape[0]):
#         log_den = np.log(MVN_per_point_l)




chi2_array_ALL_DATA_4k = chi2_array_ALL_DATA_4k-np.mean(chi2_array_ALL_DATA_4k)
MVN_per_point_l = MVN_per_point_l - np.mean(MVN_per_point_l)
weights = np.exp(-0.5 * chi2_array_ALL_DATA_4k)/MVN_per_point_l
weights


weights = 4000 * weights/np.sum(weights)
plt.hist(weights, range=(0,2))


log_RHS = log_numerator - log_den
log_RHS, log_RHS.shape
plt.hist(log_RHS/(1e7), range=(-0.01,0))


plt.hist(log_RHS)


mean_log_RHS = np.mean(log_RHS)
log_weight =log_RHS- mean_log_RHS
log_weight, log_weight.shape


mean_log_RHS


plt.hist(log_weight)


log_weight = MVN_4000_MASTER.shape[0] * log_weight/np.sum(log_weight)
log_weight+






