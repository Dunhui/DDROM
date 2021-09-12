# ROM-for-cfd  

## Data
full model example data:  https://fluidityproject.github.io/  
github address: https://github.com/FluidityProject/fluidity  
## Environment in venv
**install virtualenv**
```
sudo apt-get install python3-pip //install pip first
sudo pip3 install virtualenv //install virtualenv
```
**create a new virtualenv called 'venv_ROM' with python3 for this project**
```
        cd .virtualenvs/      //enter the virtualenv folder  from home
        virtualenv -p python3 venv_ROM          //create a new virtualenv  
```
**install the requirement file into the new virtualenv environment**
```
        cd DDROM-main/     //enter this downloaded module folder  
        workon venv_ROM            //enter this venv  
        pip install -r environment/requirements.txt  
  ```
others
