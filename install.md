# step1: update git     
## centos     
sudo yum update     
sudo yum install git-all     

## ubuntu     
sudo apt-get update     
sudo apt install git-all     

# step2: install python 3.9     
## download anaconda3     
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh     

## install conda     
sh Anaconda3-2022.05-Linux-x86_64.sh     
#### Notice: Select Yes to update ~/.bashrc      
source ~/.bashrc     

## create a virtual environment: python=3.9.13     
conda create -n lucaone_app python=3.9.13     


## activate lucaone      
conda activate lucaone_app

# step3:  install other requirements       
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple       