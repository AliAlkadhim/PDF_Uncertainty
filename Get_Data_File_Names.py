import os
import glob
def main():
    path= "/home/ali/Desktop/Research/xfitter/xfitter-2.0.1/datafiles/cms/1609.05331"
    allfiles =os.listdir(path)
    files= [ fname for fname in allfiles if fname.endswith('.dat')]
    #os.chdir(path)
    file_names = "filenames_cms.txt"
    f = open(file_names, "w")
    f.write("NInputFiles = " + str(len(files)) +"\n"+"\n")

    for ind, file_ in enumerate(files):

        f.write("InputFileNames("+str(ind+1)+") = " + "'" + str(file_) + "'," + "\n")
        #ind+=1
    
    f.close()

if __name__ == "__main__":
    main()