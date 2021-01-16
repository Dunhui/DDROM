import os
import sys
# sys.path.append("/home/ray/.virtualenvs/venv_p3/lib/python3.6/site-packages")
import vtktools
import numpy as np
from keras.layers import Reshape


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


def transform_vector(data, num, originalFile, destinationFile):

    folder = os.path.exists(destinationFile)

    if not folder: 
        print('aaa')   
        os.makedirs(destinationFile)       
        copyFiles(originalFile,destinationFile)        
    print('start to restore data')
    # replace velocity with new output data 
    for i in range(num):

        f_filename = destinationFile + "/circle-2d-drag_" + str(i)+ ".vtu"
        f_file = vtktools.vtu(f_filename)

        if len(data.shape) == 2:
            predicted_uv = np.reshape(data[i],(2,-1))
            w = np.zeros(predicted_uv.shape[1])
            w_zero = np.reshape(w,(1,-1))
            data_i = np.vstack((predicted_uv,w)).T
        elif len(data.shape) == 3:
            data_i = data[i]

        f_file.AddVectorField('Velocity_dim', data_i)
        f_file.Write(f_filename)

    print('transform succeed')	
