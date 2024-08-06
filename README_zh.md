# LucaOne APP      

## TimeLine
* 2024/08/01: add `checkpoint=17600000`, location: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/checkpoint-step17600000/'>checkpoint-step17600000</a>
  This project will download the checkpoint automatically according to the value of parameter **--llm_step**.


对核酸序列或者蛋白序列进行embedding，embedding有两种方式：matrix矩阵与vector向量。          
建议：如果使用matrix，在后续的下游网络中具体使用时，可以使用: 
* [CLS] Vector
* Avg Pooling
* Max Pooling(recommend)    
* Value-Level Attention Pooling(recommend) (ref: https://arxiv.org/abs/2210.03970)    
...    
Pooling: transform the embedding matrix into a vector.        
Recommend: Use pooling in downstream networks rather than in data pre-processing.    
**Notice**: If your task is sequence, use the pooling operation; otherwise, it is not required.


## 1. Environment installation          
### step1: update git
#### 1) centos
sudo yum update     
sudo yum install git-all

#### 2) ubuntu
sudo apt-get update     
sudo apt install git-all

### step2: install python 3.9
#### 1) download anaconda3
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

#### 2) install conda
sh Anaconda3-2022.05-Linux-x86_64.sh
##### Notice: Select Yes to update ~/.bashrc
source ~/.bashrc

#### 3) create a virtual environment: python=3.9.13
conda create -n lucaone_app python=3.9.13


#### 4) activate lucaone_app
conda activate lucaone_app

### step3:  install other requirements
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple       

## 2. Preparation
前置工作：   
Trained LucaOne Checkpoint FTP: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/'>TrainedCheckPoint for LucaOne</a>    

从上述的FTP中将目录`logs`与目录`models`拷贝到 `./models/llm/`目录下          

后续会持续更新checkpoint    

## 3. Inference          
脚本在目录`algorithms/`下：    
`inference_embedding_lucaone.py`: embedding using LucaOne(for nucleic acid or protein)         
`inference_embedding_dnabert2.py`: embedding using DNABert2(only for nucleic acid)      
`inference_embedding_esm.py`: embedding using ESM2(only for protein)     

机器：GPU可用的机器(如我们的A100可用的机器)         
位置：cd LucaOneApp/algorithms                

### Parameters      

#### 1) 模型版本参数(使用默认值即可):             
* llm_dir: 模型文件存放的路径，默认在../models/下         
* llm_type: 模型的类型，默认lucagplm       
* llm_version: 模型版本，默认v2.0       
* llm_task_level: 模型预训练的任务，默认token_level,span_level,seq_level,structure_level        
* llm_time_str: 模型开始训练的时间字符串，默认为20231125113045       
* llm_step: 当前使用的checkpoint step，默认为5600000，后面会随着训练进行更新          

#### 2) 重要参数: 
* embedding_type: matrix/vector, 分别为整个序列的矩阵或者[CLS]向量      
* trunc_type: 如果序列超过最大长度则阶段，right或者left     
* truncation_seq_length: 最大长度（不包括[CLS]与[SEP])，本身不限制长度，取决于embedding推理的显存          
* matrix_add_special_token: 如果embedding是matrix，则matrix是否包括[CLS]与[SEP]向量        
* seq_type: 输入序列的类型，gene表示核酸，prot表示蛋白          
* fasta fasta文件，id需要是命名文件名是合法的，因为使用id去命名embedding文件            
* save_path: embedding文件保存路径     
* embedding_complete: 当 `embedding_complete`被设置的时候, `truncation_seq_length`是无效的. 如果显存不够一次性推理整个序列，是否进行分段补全（如果不使用该参数，则每次截断0.95*len直到显卡可容纳的长度  
* embedding_complete_seg_overlap: 当`embedding_complete`被设置的时候, 使用对序列分段embedding的分段是否重叠(overlap sliding window)
* gpu: 使用哪一个gpu id       

#### 3) 可选参数:     
* 如果参数: fasta输入的是csv文件，需要使用id_idx与seq_idx来指定在csv中的列号(0开始)           

#### 4) 注意:    
一个序列输出一个embedding文件，使用seq_id进行命名(e.g. `matrix_seq_1000.pt`)，因此seq_id最好不能有特殊字符，比如空格或者"/"字符           

#### 5) Examples:              


```shell
# 对蛋白质进行embedding(输入csv文件，需要指明id与seq的列号)   
cd ./algorithms/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python inference_embedding_lucaone.py \
    --llm_dir ../models/  \
    --llm_type lucaone_gplm \
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 5600000 \
    --embedding_type matrix \
    --trunc_type right \
    --truncation_seq_length 100000 \
    --matrix_add_special_token \
    --seq_type prot \
    --input_file ../data/prot/test_prot_dataset.csv \
    --id_idx 0 \
    --seq_idx 1 \
    --save_path ../embedding/lucaone/test/prot \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu 0   
 
# 对核酸(DNA或者RNA)进行embedding(输入csv文件，需要指明id与seq的列号)   
cd ./algorithms/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python inference_embedding_lucaone.py \
    --llm_dir ../models/ \
    --llm_type lucaone_gplm \
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 5600000 \
    --embedding_type matrix \
    --trunc_type right \
    --truncation_seq_length 100000 \
    --matrix_add_special_token \
    --seq_type gene \
    --input_file ../data/gene/test_gene_dataset.csv \
    --id_idx 0 \
    --seq_idx 1 \
    --save_path ../embedding/lucaone/test/gene \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu 0   
```


```shell
# 对蛋白质进行embedding(输入fasta文件，seq头最好进行唯一id重命名，别包含特殊符号)   
cd ./algorithms/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python inference_embedding_lucaone.py \
    --llm_dir ../models/ \
    --llm_type lucaone_gplm \
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 5600000 \
    --embedding_type matrix \
    --trunc_type right \
    --truncation_seq_length 100000 \
    --matrix_add_special_token \
    --seq_type prot \
    --fasta ../data/prot/test_prot.fasta \
    --save_path ../embedding/lucaone/test/prot/ \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu 0   
```


```shell
# 对对核酸(DNA或者RNA)进行embedding(输入fasta文件，seq头最好进行唯一id重命名，别包含特殊符号)   
cd ./algorithms/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python inference_embedding_lucaone.py \
    --llm_dir ../models/ \
    --llm_type lucaone_gplm \
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 5600000 \
    --embedding_type matrix \
    --trunc_type right \
    --truncation_seq_length 100000 \
    --matrix_add_special_token \
    --seq_type gene \
    --fasta ../data/prot/test_gene.fasta \
    --save_path ../embedding/lucaone/test/gene/ \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu 0   
```


## 6. Contributor
<a href="https://scholar.google.com.hk/citations?user=RDbqGTcAAAAJ&hl=en" title="Yong He">Yong He</a>,
<a href="https://scholar.google.com/citations?user=lT3nelQAAAAJ&hl=en" title="Zhaorong Li">Zhaorong Li</a>,
<a href="https://scholar.google.com/citations?user=ODcOX4AAAAAJ&hl=zh-CN" title="Pan Fang">Pan Fang</a>

## 7. Citation
@article {LucaOne,        
author = {Yong He and Pan Fang and Yongtao Shan and Yuanfei Pan and Yanhong Wei and Yichang Chen and Yihao Chen and Yi Liu and Zhenyu Zeng and Zhan Zhou and Feng Zhu and Edward C. Holmes and Jieping Ye and Jun Li and Yuelong Shu and Mang Shi and Zhaorong Li},     
title = {LucaOne: Generalized Biological Foundation Model with Unified Nucleic Acid and Protein Language},      
elocation-id = {2024.05.10.592927},        
year = {2024},         
doi = {10.1101/2024.05.10.592927},        
publisher = {Cold Spring Harbor Laboratory},        
URL = {https://www.biorxiv.org/content/early/2024/05/14/2024.05.10.592927},        
eprint = {https://www.biorxiv.org/content/early/2024/05/14/2024.05.10.592927.full.pdf},        
journal = {bioRxiv}        
}    