import numpy as np
import matplotlib.pyplot as plt


MVN_4000_MASTER = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/samples/MVN_4000_MASTER.npy')
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


chi2_array_ALL_DATA_4k = np.load('/home/ali/Desktop/Pulled_Github_Repositories/NNPDF_Uncertainty/master_version/local/ALL_DATA_5k/chi2_array_ALL_DATA_4k.npy')
chi2_array_ALL_DATA_4k, chi2_array_ALL_DATA_4k.shape


#to avoid overflow take data type as float 128 to handle exponentiation
chi2_array_ALL_DATA_4k = chi2_array_ALL_DATA_4k.astype(np.float128)
MVN_4000_MASTER = MVN_4000_MASTER.astype(np.float128)
plt.hist(chi2_array_ALL_DATA_4k, bins=100)
plt.title('$\chi^2$ for All data', fontsize=20)
chi2_array_ALL_DATA_4k, np.mean(chi2_array_ALL_DATA_4k), np.std(chi2_array_ALL_DATA_4k)


mu


# z = MVN_per_point_l
mu = params_MASTER
sigma = COV_MASTER
second_term_l = []
for i in range(MVN_4000_MASTER.shape[0]):
    second_term = (MVN_4000_MASTER[i,:] - mu).T @ np.linalg.inv(sigma) @ (MVN_4000_MASTER[i,:] - mu)
    second_term_l.append(second_term)
plt.hist(np.array(second_term_l))# second term dist is much narrower, which is what we expect


list_of_tuples = []
for i in range(14):
    param_list_i=[]
#     weight_list_i = []
    chi2_list=[]
    for k in range(4000):
        param_value = MVN_4000_MASTER[k, i] #at the kth point, for parameter i
        
        std_MVN_value = np.std(MVN_4000_MASTER[k,:])
        mean_MVN_value = np.mean(MVN_4000_MASTER[k,:])
        if (param_value > (mean_MVN_value - 1*param_value)) and (param_value < (mean_MVN_value + 1*param_value)):
            #if weight_value < (mean_weight + 4*std_weight_value):

            param_list_i.append(param_value)
            chi2_list.append(chi2_array_ALL_DATA_4k[k])
    tuple_i = (param_list_i, chi2_list)
    list_of_tuples.append(tuple_i)
#len(list_of_tuples)                
#list_of_tuples[1]


chi2_array_ALL_DATA_4k.mean(), chi2_array_ALL_DATA_4k.std()


chi2_array_ALL_DATA_4k.min()


for i in range(4000):
    print(list_of_tuples[i][1])


chi2_filtered = []
for i in range(4000):
    chi2_filtered.append(list_of_tuples[i][0])
plt.hist(np.array(chi2_filtered))


chi2_array_ALL_DATA_4k - np.mean(chi2_array_ALL_DATA_4k)


chi2_diff = chi2_array_ALL_DATA_4k - np.mean(chi2_array_ALL_DATA_4k)
w = -0.5*(chi2_diff-np.array(second_term_l))
plt.hist(w)


np.mean(chi2_diff)


mean_chi2 = np.mean(chi2_array_ALL_DATA_4k)
#remember to rename diff (no need to take diff at this point)
chi2_diff = chi2_array_ALL_DATA_4k - mean_chi2

log_numerator = -0.5*chi2_diff
log_numerator.shape
# for i in range(14):
#     log_numerator[:,i] =  - 0.5 * (chi2_diff)
# log_numerator #no normalization factor
log_numerator, np.mean(log_numerator)


log_numerator = log_numerator - np.mean(log_numerator)
log_numerator


plt.hist(log_numerator, range=(-10**6,0)) #this is equivalent to plt.hist(log_numerator/(e7), range=(-0.01,0))


# def log_den(MVN_per_point_l):
#     log_den_l = []
#     for i in range(MVN_per_point_l.shape[0]):
#         log_den = np.log(MVN_per_point_l)

log_den = np.log(MVN_per_point_l)
log_den.shape
plt.hist(log_den)





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






