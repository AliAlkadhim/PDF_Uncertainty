import os
import re; import pandas as pd
params=[]
generated_params = []
error_list=[]
filename = 'minuit.out.txt'
infile = open(filename, 'r')
lines = infile.readlines()

from itertools import *
def main():

    for i in range(10):

        chain = chain(islice(lines, 106, 111), islice(lines, 111, 121))
        #for line in lines[106:121]:
        for line in chain:
            words = line.strip().split()
            
            #words = re.split(r"(?: '\s*)\s*", line.strip())
            #df = pd.read_table(words)
            #use re noncapture group, defined as (?:...)since we dont want the separators in our 
            #result.
            #print(words[2])
            values= words[2]
            errors=words[3]
            #print(errors)
            for value in values.split():
        #         #print(i)
                params.append(float(value))
            for error in errors.split():
                if error =='constant':
                    error_list.append(0.0)
                else:
                    error_list.append(float(error))
            
        infile.close()
        params = np.asarray(params); error_list =np.asarray(error_list)

        #############GENERATE PARAMETERS
        generated_params=[]
        for i in range(len(params)):
            param, error = params[i], error_list[i]
            generated_param = np.random.uniform(low = param-error, high=param+error)
            generated_params.append(generated_param)

        ############WRITE GENERATED PARAMETERS INTO NEW MINUIT.IN FILE
    with open('minuit_{d}.in.txt'.format(i), 'w') as second:
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


if name__=='__main__':
    main()

