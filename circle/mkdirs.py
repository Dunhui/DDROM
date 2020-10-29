import os
import sys
sys.path.append("/home/ray/.virtualenvs/venv_p3/lib/python3.6/site-packages")
import vtktools
import numpy as np
import shutil

# copy original files
def copyFiles(sourceDir,targetDir):
    if sourceDir.find("exceptionfolder")>0:
        return

    for file in os.listdir(sourceDir):
        sourceFile = os.path.join(sourceDir,file)
        targetFile = os.path.join(targetDir,file)

        if os.path.isfile(sourceFile):
            if not os.path.exists(targetDir):
                os.makedirs(targetDir)
            if not os.path.exists(targetFile) or (os.path.exists(targetFile) and (os.path.getsize(targetFile) !=os.path.getsize(sourceFile))):
                open(targetFile, "wb").write(open(sourceFile, "rb").read())
                # print(targetFile+" copy succeeded")

        if os.path.isdir(sourceFile):
            copyFiles(sourceFile, targetFile)

# create new folder
def mkdir(path):
	folder = os.path.exists(path)

	if folder:                   
		print ("---  We already have this folder name  ---")
		shutil.rmtree(path, ignore_errors=True)
		print ("---  We already delete this folder  ---")

	print ("---  create new folder...---")
	os.makedirs(path)
	("---  OK  ---")

def transform(predicted_data, originalFile, destinationFile):

	mkdir(destinationFile)     
	copyFiles(originalFile,destinationFile)

# 	# replace velocity with new output data 
	for i in range(predicted_data.shape[0]):

		f_filename=destinationFile + "/circle-2d-drag_" + str(i+1201)+ ".vtu"
		f_file = vtktools.vtu(f_filename) 

		predicted_uv = np.reshape(predicted_data[i],(2,-1))
		w = np.zeros(predicted_uv.shape[1])
		w_zero = np.reshape(w,(1,-1))
		velocity_uvw = np.vstack((predicted_uv,w)).T
		
		f_file.AddVectorField("Velocity_dim", velocity_uvw)
		f_file.Write(f_filename)
	
	print('transform succeed')	