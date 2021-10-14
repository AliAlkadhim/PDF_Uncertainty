import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex":True})


MVN_4000_MASTER = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/MVN_25k_MASTER.npy')
MVN_4000_MASTER, MVN_4000_MASTER.shape


for i in range(MVN_4000_MASTER.shape[0]):
    print('first point', MVN_4000_MASTER[i,:])
    break


COV_MASTER= np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/COV_MASTER.npy')
params_MASTER= np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/params_MASTER.npy')
COV_MASTER[0], params_MASTER[0]


def f(MVN, mu, sigma):
    """
    The density function of multivariate normal distribution.
    N = size of the mean vector, or number of parameter points (14)
    MVN = the 2D MV Gaussian function
    sigma = the covariance matrix from our best-fit values
    """

    MVN_per_point_l=[]
    for i in range(MVN.shape[0]):
        #z = np.atleast_2d(z)
        z = MVN[i,:]


        N = z.size

        temp1 = np.linalg.det(sigma) ** (-1/2)
        temp2 = np.exp(-.5 * (z - mu).T @ np.linalg.inv(sigma) @ (z - mu))
        MVN_per_point = (2 * np.pi) ** (-N/2) * temp1 * temp2
        MVN_per_point_l.append(MVN_per_point)
    return np.array(MVN_per_point_l)


MVN_per_point_l = f(MVN_4000_MASTER, params_MASTER, COV_MASTER); MVN_per_point_l


MVN_per_point_l.shape


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


MVN_per_point_l_diff_mean[MVN_per_point_l_diff_mean <0]


chi2_array_ALL_DATA_4k = chi2_array_ALL_DATA_4k.astype(np.float128)
MVN_4000_MASTER = MVN_4000_MASTER.astype(np.float128)

best_fitchi2_25k =3369.427
MVG_best_fit = MVG_BestFit(params_MASTER, params_MASTER, COV_MASTER)

MVN_per_point_l = f(MVN_4000_MASTER, params_MASTER, COV_MASTER)
#first step is masking both to take only positive values of MVG
# pos_MVN_per_point_l = MVN_per_point_l[MVN_per_point_l > 0]

# pos_chi2 = chi2_array_ALL_DATA_4k[MVN_per_point_l > 0]

#now subtract best-fit values of chi2 and MVG
# MVN_per_point_l = MVN_per_point_l - MVG_best_fit
# pos_chi2 = pos_chi2 - best_fitchi2_25k

#now subtract the mean from both chi2 and MVG
MVN_per_point_l_diff_mean = MVN_per_point_l - np.mean(MVN_per_point_l)

pos_chi2_diff_mean =pos_chi2 - np.mean(pos_chi2)


#HERE is where the positive masking should be done (right before taking the log)
MVN_per_point_l_diff_mean_pos = MVN_per_point_l_diff_mean[MVN_per_point_l_diff_mean > 0]
pos_chi2_diff_mean_pos = pos_chi2_diff_mean[MVN_per_point_l_diff_mean > 0]

#now calculate the log weight
log_weight_unnormalized = (-0.5 * pos_chi2_diff_mean_pos) - (np.log(MVN_per_point_l_diff_mean_pos))

weight_unnormalized = np.exp(log_weight_unnormalized)

weight_normalized = len(weight_unnormalized) * log_weight_unnormalized/np.sum(weight_unnormalized)

MVN_param_1_pos = MVN_4000_MASTER[:,2][MVN_per_point_l_diff_mean > 0]

plt.hist(MVN_param_1_pos, weights=weight_normalized, bins=100, label='Reweighted', alpha=0.3, density=True)
plt.hist(MVN_param_1_pos, bins=100, label='Gaussian',  alpha=0.3, density=True)
plt.legend()
MVN_param_1_pos.shape, weight_normalized.shape




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






