# DDROM

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)


This is the code base for DDROM. [2109.02126.pdf](https://arxiv.org/pdf/2109.02126.pdf)

## Table of Contents

- [Background](#background)
- [Data](#data)
- [Install](#install)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## Background

A new model reduction neural network architecture for fluid flow problem is presented. The SAE network compresses high-dimensional physical information into several much smaller sized representations in a latent space. These representations are expressed by a number of codes in the middle layer of SAE neural network. Then, those codes at different time levels are trained to construct a set of hyper-surfaces with multi variable response functions using attention-based deep learning methods. The inputs of the attention-based network are previous time levels’ codes and the outputs of the network are current time levels’ codes. The codes at current time level are then projected back to the original full space by the decoder layers in the SAE network.

## Data
**Data source: Fluidity**  
[main page](https://fluidityproject.github.io/) || [github address](https://github.com/FluidityProject/fluidity) || [fluidity manual](https://figshare.com/articles/journal_contribution/Fluidity_Manual/1387713)  
**Install data source**
```
sudo apt-add-repository -y ppa:fluidity-core/ppa
sudo apt-get update
sudo apt-get -y install fluidity  
```
**Generate data**
```
fluidity example.flml
```
## Install
**Environment**

*Install pip and virtualenv*(ignore this if you have already install virtualenv)
```
sudo apt-get install python3-pip 
```
```
sudo pip3 install virtualenv
```
*Create a new virtualenv called 'venv_ROM' with python3 for this project. Enter the virtualenv folder from home and create a new virtualenv*
```
cd .virtualenvs/      
```
```
virtualenv -p python3 venv_ROM         
```
*Enter the downloaded module folder and install the requirement file into the new virtualenv environment. Open a new commend in the downloaded module zip folder path, enter this venv and  install the requirement file into this venv.*
```
workon venv_ROM       
```
```
unzip DDROM-main.zip    
```
```
cd DDROM-main/
```
```
mv environment/vtktools.py ***/.virtualenvs/venv_ROM/lib/python3.6/site-packages/   
```
```
pip install -r environment/requirements.txt  
```



## Usage

```
```

Note: The `license` badge image link at the top of this file should be updated with the correct `:user` and `:repo`.

### Any optional sections



## Contributing


## License


