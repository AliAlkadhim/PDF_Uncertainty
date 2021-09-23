import numpy as np
#k =np.load('job_0/chi2_array_100k.npy'); print(k)
f = open('collected_chi2.txt')
l = f.read()
ll=[]
for i in l.split(','):
    print(float(i))
    
	#ll.append(float(i))

#q = np.array(ll);
#print(q.shape)
