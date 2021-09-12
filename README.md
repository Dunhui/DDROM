# ROM-for-cfd  

## Data
full model example data:  https://fluidityproject.github.io/  
github address: https://github.com/FluidityProject/fluidity  
## Environment in venv
**install virtualenv**(ignore this if you have already install virtualenv)
```
# install pip first
sudo apt-get install python3-pip 
# install virtualenv
sudo pip3 install virtualenv
```
**create a new virtualenv called 'venv_ROM' with python3 for this project**
```
# enter the virtualenv folder  from home
cd .virtualenvs/      
# create a new virtualenv  
virtualenv -p python3 venv_ROM         
```
**enter the downloaded module folder and install the requirement file into the new virtualenv environment**
```
# open  a new commend in the downloaded module zip folder path
# enter this venv  
workon venv_ROM        
# unzip 
unzip DDROM-main.zip    
# enter this downloaded module folder  
cd DDROM-main/     
# remove the vtktools.py file into the lib in venv, ***the destination address is the venv just created
mv environment/vtktools.py ***/.virtualenvs/venv_ROM/lib/python3.6/site-packages/  
# remove the vtk.py file into the lib in venv, ***the destination address is the venv just created
mv environment/vtk.py ***/.virtualenvs/venv_ROM/lib/python3.6/site-packages/ 
# install the requirement file into this venv
pip install -r environment/requirements.txt  
  ```

