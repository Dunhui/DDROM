# ROM-for-cfd  

## Data
full model example data:  https://fluidityproject.github.io/  
github address: https://github.com/FluidityProject/fluidity  
## Environment in venv
**install virtualenv**(ignore this if you have already install virtualenv)
```
sudo apt-get install python3-pip //install pip first
sudo pip3 install virtualenv //install virtualenv
```
**create a new virtualenv called 'venv_ROM' with python3 for this project**
```
cd .virtualenvs/      //enter the virtualenv folder  from home
virtualenv -p python3 venv_ROM          //create a new virtualenv  
```
**enter the downloaded module folder and install the requirement file into the new virtualenv environment**
```
//open  a new commend in the downloaded module zip folder path
workon venv_ROM            //enter this venv  
unzip DDROM-main.zip    //unzip 
cd DDROM-main/     //enter this downloaded module folder  
mv environment/vtktools.py ***/.virtualenvs/venv_ROM/lib/python3.6/site-packages/   // ***the destination address is the venv just created
pip install -r environment/requirements.txt  //install the requirement file into this venv
  ```

