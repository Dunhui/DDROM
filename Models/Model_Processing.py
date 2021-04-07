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

def pearson_value(ori_data, rom_data):
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
    return pearson_value

def ae_cc(ori_data, rom_data):
    pcc = pearson_value(ori_data, rom_data)
    print(pcc.shape[0], pcc.shape[1])
    plt.figure(1)
    x = np.linspace(0,pcc.shape[0], pcc.shape[0])
    plt.plot(x, pcc)
    plt.xlabel('Time(s)',{'size' : 11})
    plt.ylabel('Pearson Correlation Coefficient',{'size' : 11})
    plt.show()


def cc(ori_data, rom_data_0, rom_data_1, rom_data_2):
# draw the plot for correlation coefficient
    
    pcc_0 = pearson_value(ori_data, rom_data_0)
    pcc_1 = pearson_value(ori_data, rom_data_1)
    pcc_2 = pearson_value(ori_data, rom_data_2)
   
    plt.figure(1)
    x = np.linspace(0,200,2000)
    y_0 = pcc_0[0:2000,:]
    y_1 = pcc_1[0:2000,:]
    y_2 = pcc_2[0:2000,:]
    # plt.title('Correlation Coefficient')

    plt.plot(x, y_0, x, y_1, x, y_2, linewidth = 0.4)
    plt.xlim((-0.1, 200.1))# range
    plt.ylim((0.97, 1.0001))
    plt.xlabel('Time(s)',{'size' : 11})
    plt.ylabel('Pearson Correlation Coefficient',{'size' : 11})
    plt.xticks(np.arange(0,200.1,25))
    plt.yticks(np.arange(0.97,1.0001,0.01))
    plt.legend(['ROM dim=3', 'ROM dim=8', 'ROM dim=64'], loc='lower right')   
    
    plt.show()

# draw the plot for point over time series
def point_over_time(ori_data, rom_data_0, rom_data_1, rom_data_2, pointNo, fieldName):# 1: full model; 2: ROM model
    
    if ori_data.shape != rom_data_0.shape:
        print('the shape of these two series do not match. Please check them.')
        return 

    if ori_data.ndim == 3:

        if fieldName == 'Velocity': # flow_past_cylinder
            point = [1900,2000] # [start point, finish point]->200s
            rate = 10
            ylim_x = [0.45, 0.65] # ylim for first figure->data in x axis
            ylim_y = [0, 0.06] # ylim for second figure->data in y axis
            
        elif fieldName == 'Water::Velocity': # tephra_settling
            pass

        time = int((point[1]-point[0])/rate)
        x = np.linspace(0,time,time*rate) 
        y_u = ori_data[point[0]:point[1],pointNo,0] # u
        y_v = ori_data[point[0]:point[1],pointNo,1] # v
        y_0_u = rom_data_0[point[0]:point[1],pointNo,0] # u
        y_0_v = rom_data_0[point[0]:point[1],pointNo,1] # v
        y_1_u = rom_data_1[point[0]:point[1],pointNo,0] # u
        y_1_v = rom_data_1[point[0]:point[1],pointNo,1] # v
        y_2_u = rom_data_2[point[0]:point[1],pointNo,0] # u
        y_2_v = rom_data_2[point[0]:point[1],pointNo,1] # v
    
        plt.figure(1)
        plt.plot(x, y_u,'b-', x, y_0_u, 'y-', x, y_1_u, 'g-', x, y_2_u, 'r-', linewidth = 0.5)
    
        plt.xlim((-0.1, time))# range
        plt.ylim((ylim_x[0], ylim_x[1]))
        plt.title('Magnitude of ' + fieldName + ' x axis, pointNo: '+ str(pointNo))
        plt.xlabel('Time(s)')
        plt.ylabel(fieldName)
        plt.legend(['Full Model', 'ROM dim=3', 'ROM dim=8', 'ROM dim=64'], loc='lower right')

        plt.figure(2)
        plt.plot(x, y_v, x, y_0_v, x, y_1_v, x, y_2_v, linewidth = 0.5)
        plt.title('Magnitude of ' + fieldName + ' y axis, pointNo: '+ str(pointNo))
        plt.xlim((-0.1, time))# range
        plt.ylim((ylim_y[0], ylim_y[1]))
        plt.xlabel('Time(s)')
        plt.ylabel(fieldName)
        plt.legend(['Full Model', 'ROM dim=3', 'ROM dim=8', 'ROM dim=64'], loc='lower right')

        plt.show()

    elif ori_data.ndim == 2:

        if fieldName == 'Water::MaterialVolumeFraction': # water collapse
            pass
        elif fieldName == 'Temperature':
            point = [0,2000] # [start point, finish point]->50s
            rate = 40
            ylim = [-0.50, 0.50] # ylim for figure
           
        time = int((point[1]-point[0])/rate)
        x = np.linspace(0,time,time*rate) 
        y = ori_data[point[0]:point[1],pointNo] 
        y_0 = rom_data_0[point[0]:point[1],pointNo]
        y_1 = rom_data_1[point[0]:point[1],pointNo]
        y_2 = rom_data_2[point[0]:point[1],pointNo]
        plt.figure()
        plt.plot(x, y, x, y_0, x, y_1, x, y_2)
        plt.title('Magnitude of ' + fieldName + '    PointID: ' + str(pointNo))
        plt.xticks(np.arange(0,time,10))
        plt.yticks(np.arange(ylim[0],ylim[1],0.1))
        plt.xlabel('Time',{'size' : 11})
        plt.ylabel(fieldName,{'size' : 11})
        plt.legend(['Full Model', 'ROM dim=3', 'ROM dim=8', 'ROM dim=64'], loc='lower right')
        plt.show()

    else:
        print('the dimension of these two series are not equal. Please check them.')

def rmse(ori_data, rom_data):

    rmse_value = []
    if len(ori_data) != len(rom_data):
        print('the length of these two array do not match')
    else:
        for i in range(len(rom_data)):
            value = np.sqrt(mean_squared_error(ori_data[i], rom_data[i]))
            if i == 0:
                rmse_value = value
            else:
                rmse_value = np.hstack((rmse_value,value))
        rmse_value = np.reshape(rmse_value,(-1,1))
    return rmse_value

def ae_rmse(ori_data, rom_data):
    rmse_value = rmse(ori_data, rom_data)
    # print(pcc.shape[0], pcc.shape[1])
    plt.figure(1)
    x = np.linspace(0,rmse_value.shape[0], rmse_value.shape[0])
    plt.plot(x, rmse_value)
    plt.xlabel('Time(s)',{'size' : 11})
    plt.ylabel('RMSE',{'size' : 11})
    plt.show()



def rmse_over_time(ori_data, rom_data_0, rom_data_1, rom_data_2):
    rmse_0 = rmse(ori_data, rom_data_0)
    rmse_1 = rmse(ori_data, rom_data_1)
    rmse_2 = rmse(ori_data, rom_data_2)
       
    plt.figure(1)

    x = np.linspace(0,200,2000)
    y_0 = rmse_0[0:2000,:]
    y_1 = rmse_1[0:2000,:]
    y_2 = rmse_2[0:2000,:]
    # plt.title('Correlation Coefficient')
    plt.plot(x, y_0, x, y_1, x, y_2, linewidth = 0.4)
    plt.xlim((-0.1, 200.1))# range
    plt.ylim((-0.005, 0.081))
    plt.xlabel('Time(s)',{'size' : 11})
    plt.ylabel('RMSE',{'size' : 11})
    plt.xticks(np.arange(0,200.1,25))
    plt.yticks(np.arange(0,0.081,0.02))
    plt.legend(['ROM dim=3', 'ROM dim=8', 'ROM dim=64'], loc='upper left')  
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



