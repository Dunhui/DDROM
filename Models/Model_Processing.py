import os
import vtktools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def save_model(model, model_name, save_dir):
# function for saving model
	
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	model_path = os.path.join(save_dir, model_name)
	model.save(model_path)

def draw_Acc_Loss(history):
# draw the plot for loss and acc
	plt.figure(1)
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	plt.figure(2)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()


def cc(ori_data,rom_data):
# draw the plot for correlation coefficient
    pearson_value = []
    if len(ori_data) != len(rom_data):
        print('the length of these two array do not match')
    else:
        for i in range(len(rom_data)):
            row_1 = np.reshape(ori_data[i],(-1,1))
            row_2 = np.reshape(rom_data[i],(-1,1))
            data = np.hstack((row_1,row_2))
            data = pd.DataFrame(data)
            pearson = data.corr() # pearson cc  # pearson = data.corr('spearman') # spearman cc  
            pear_value=pearson.iloc[0:1,1:2]
            value = pear_value.values
            if i == 0:
                pearson_value = value
            else:
                pearson_value = np.hstack((pearson_value,value)) 
        pearson_value = np.reshape(pearson_value,(-1,1))
       
    plt.figure(1)

    x = np.linspace(0,200,2000)
    y = pearson_value[0:2000,:]
    # plt.title('Correlation Coefficient')
    plt.plot(x, y)
    plt.xlim((-0.1, 200.1))# range
    plt.ylim((0.9, 1.01))
    plt.xlabel('Time(s)',{'size' : 11})
    plt.ylabel('Pearson Correlation Coefficient',{'size' : 11})
    plt.xticks(np.arange(0,200.1,25))
    plt.yticks(np.arange(0.9,1.01,0.01))
    plt.show()

# draw the plot for point over time series
def point_over_time(data_1, data_2, pointNo, fieldName):# 1: full model; 2: ROM model
    
    if data_1.shape[0] != data_2.shape[0]:
        print('the length of these two series do not match. Please check them.')

    else:
        if data_1.ndim == 3 and data_2.ndim == 3:
            x = np.linspace(0,200,2000)
            # x = np.arange(1,data_1.shape[0]+1)
            y_1_u = data_1[0:2000,pointNo,0] # u
            y_1_v = data_1[0:2000,pointNo,1] # v
            y_2_u = data_2[0:2000,pointNo,0] # u
            y_2_v = data_2[0:2000,pointNo,1] # v
            
            plt.figure(1)
            plt.plot(x, y_1_u, x, y_2_u)
            plt.xlim((-0.1, 200.1))# range
            plt.ylim((0, 1))
            plt.title('Magnitude of ' + fieldName + ' x axis, pointNo: '+ str(pointNo))
            plt.xlabel('Time(s)')
            plt.ylabel(fieldName)
            plt.legend(['Full Model', 'ROM Model'], loc='lower right')

            plt.figure(2)
            plt.plot(x, y_1_v, x, y_2_v)
            plt.title('Magnitude of ' + fieldName + ' y axis, pointNo: '+ str(pointNo))
            plt.xlim((-0.1, 200.1))# range
            plt.ylim((0, 1))
            plt.xlabel('Time(s)')
            plt.ylabel(fieldName)
            plt.legend(['Full Model', 'ROM Model'], loc='lower right')

            plt.show()

        elif data_1.ndim == 2 and data_2.ndim == 2:
            x = np.linspace(0,10,1000)
            # x = np.arange(1,data_1.shape[0]+1)
            y_1 = data_1[0:1000,pointNo] 
            y_2 = data_2[0:1000,pointNo]
            plt.figure()
            plt.plot(x,y_1,x,y_2)
            # plt.title('Magnitude of ' + fieldName + '    PointID: ' + str(pointNo))
            # plt.title('PointID: ' + str(pointNo))
            plt.xlabel('Time',{'size' : 11})
            plt.ylabel(fieldName,{'size' : 11})
            plt.xticks(np.arange(0,10.1,1))
            plt.yticks(np.arange(0,1.1,0.2))
            plt.legend(['Full Model', 'ROM'], loc='upper right')
            plt.show()

        else:
            print('the dimension of these two series are not equal. Please check them.')

def rmse_over_time(ori_data, rom_data):
    rmse_value = []
    if len(ori_data) != len(rom_data):
        print('the length of these two array do not match')
    else:
        for i in range(len(rom_data)):
            value = np.sqrt(mean_squared_error(ori_data[i], rom_data[i]))
            # value = mean_squared_error(ori_data[i], rom_data[i])
            if i == 0:
                rmse_value = value
            else:
                rmse_value = np.hstack((rmse_value,value))
        rmse_value = np.reshape(rmse_value,(-1,1))
       
    plt.figure(1)

    x = np.linspace(0,200,2000)
    y = rmse_value[0:2000,:]
    # plt.title('Correlation Coefficient')
    plt.plot(x, y)
    plt.xlim((-0.1, 200.1))# range
    plt.ylim((-0.005, 0.2005))
    plt.xlabel('Time(s)',{'size' : 11})
    plt.ylabel('RMSE',{'size' : 11})
    plt.xticks(np.arange(0,200.1,25))
    plt.yticks(np.arange(0,0.21,0.05))
    plt.show()

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


def transform_vector(data, num, originalFolder, destinationFolder, fileName, fieldName):

    folder = os.path.exists(destinationFolder)

    if not folder: 
        print('start to create the destination folder')   
        os.makedirs(destinationFolder)       
        copyFiles(originalFolder,destinationFolder) 

    print('start to store data as a new variable')
    if len(data.shape) == 3:
        w_zero = np.zeros((data.shape[0], data.shape[1],1))
        # print(w_zero.shape)
        data=np.concatenate((data,w_zero), axis = 2)
        # print(data.shape)
    i = 0
    for i in range(num-1):

        f_filename = destinationFolder + fileName + str(i)+ ".vtu"
        f_file = vtktools.vtu(f_filename)

        if len(data[i].shape) == 1:      
            f_file.AddScalarField(fieldName, data[i])
        
        elif len(data[i].shape) == 2:
            f_file.AddVectorField(fieldName, data[i])

        else:
            print('The shape of output and setted field are not matched')

        f_file.Write(f_filename)

    print('transform succeed')	



