# LucaOne APP   
## 1. Embedding     
Two embedding methods for nucleic acid or protein sequence: `matrix` or `vector`.        
suggestion: If `matrix` is applied, it can be converted to a vector in the downstream networks as follows when using the embedding matrix:
* [CLS] Vector(matrix[0, :])
* Avg Pooling    
* Max Pooling(recommend)           
* Value-Level Attention Pooling(recommend) (ref: https://arxiv.org/abs/2210.03970)    
...    
Pooling: transform the embedding matrix into a vector.             
Recommend: Use pooling in downstream networks rather than in data pre-processing.    
**Notice**: If your task is sequence, use the pooling operation; otherwise, it is not required.   

## 2. Environment Installation          
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


#### 4) activate lucaone
conda activate lucaone_app

### step3:  install other requirements
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple       

## 3. Preparation      
Trained LucaOne Checkpoint FTP: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/'>TrainedCheckPoint for LucaOne</a>

**Notice:**    
The project will download automatically LucaOne Trained-CheckPoint from **FTP**.

When downloading automatically failed, you can manually download:

Copy the **TrainedCheckPoint Files(`models/` + `logs/`)** from <href> http://47.93.21.181/lucaone/TrainedCheckPoint/* </href> into the directory: `./models/llm/`


## 4. Inference       
Scripts in `algorithms/`     
`inference_embedding_lucaone.py`: embedding using LucaOne(for nucleic acid or protein).           
`inference_embedding_dnabert2`: embedding using DNABert2(only for nucleic acid).     
`inference_embedding_esm`: embedding using ESM2(only for protein).       

### Parameters
1) LucaOne checkpoint parameters:      
    * llm_dir: the path for storing the checkpoint LucaOne model，default: `../models/`         
    * llm_type: the type of LucaOne, default: lucagplm         
    * llm_version: the version of LucaOne, default: v2.0         
    * llm_task_level: the pretrained tasks of LucaOne, default: token_level,span_level,seq_level,structure_level          
    * llm_time_str: the trained time str of LucaOne, default: 20231125113045         
    * llm_step:  the trained checkpoint of LucaOne, default: 5600000

2) Important parameters:     
    * embedding_type: `matrix` or `vector`, output the embedding matrix or [CLS] vector for the entire sequence, recommend: matrix.      
    * trunc_type: truncation type: `right` or `left`, truncation when the sequence exceeds the maximum length.    
    * truncation_seq_length: the maximum length for embedding(not including [CLS] and [SEP]), itself does not limit the length, depending on the capacity of GPU.            
    * matrix_add_special_token: if the embedding is matrix, whether the matrix includes [CLS] and [SEP] vectors.           
    * seq_type: type of input sequence: `gene` or `prot`, `gene` for nucleic acid(DNA or RNA), `prot` for protein.        
    * input_file: the input file path for embedding(format: csv or fasta). The seq_id in the file must be unique and cannot contain special characters.     
    * save_path: the saving dir for storing the embedding file.     
    * embedding_complete: When `embedding_complete` is set, `truncation_seq_length` is invalid. If the GPU memory is not enough to infer the entire sequence at once, it is used to determine whether to perform segmented completion (if this parameter is not used, 0.95*len is truncated each time until the CPU can process the length).       
    * embedding_complete_seg_overlap: When `embedding_complete` is set, whether the method of overlap is applicable to segmentation(overlap sliding window)
    * gpu: the gpu id to use(-1 for cpu).

3) Optional parameters:    
    * id_idx & seq_idx: when the input file format is csv file, need to use `id_idx` and `seq_idx` to specify the column index in the csv (starting with 0).

  

**Notice:**     
A sequence outputs one embedding file named with `seq_id`, so the `seq_id` must be unique and must not contain special characters, such as Spaces, "/", and so on.   


### Examples:

#### 1) the **csv** file format of input     

**Notice:**
1. need to specify the column index of the sequence id(*id_idx**) and sequence(**seq_idx**), starting index: 0 .
2. The **sequence id** must be globally unique in the input file and cannot contain special characters (because the embedding file stored is named by the sequence id).  

```shell
# for protein
cd ./algorithms/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python inference_embedding_lucaone.py \
    --llm_dir ../models  \
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
    --input_file ../data/test_data/prot/test_prot.csv \
    --id_idx 2 \
    --seq_idx 3 \
    --save_path ../embedding/lucaone/test_data/prot/test_prot \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu 0   
 
# for DNA or RNA
cd ./algorithms/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python inference_embedding_lucaone.py \
    --llm_dir ../models \
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
    --input_file ../data/test_data/gene/test_gene2.csv \
    --id_idx 0 \
    --seq_idx 1 \
    --save_path ../embedding/lucaone/test_data/gene/test_gene \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu 0   
```

#### 2) the **fasta** file format of input   

**Notice:**   
1. The **sequence id** must be globally unique in the input file and cannot contain special characters (because the embedding file stored is named by the sequence id).
```shell
# for protein
cd ./algorithms/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python inference_embedding_lucaone.py \
    --llm_dir ../models \
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
    --input_file ../data/test_data/prot/test_prot.fasta \
    --save_path ../embedding/lucaone/test_data/prot/test_prot \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu 0   
```


```shell
# for DNA or RNA
cd ./algorithms/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python inference_embedding_lucaone.py \
    --llm_dir ../models \
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
    --input_file ../data/test_data/gene/test_gene.fasta \
    --save_path ../embedding/lucaone/test_data/gene/test_gene \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu 0   
```   


## 5. Data and Code Availability
**FTP:**   
Pre-training data, code, and trained checkpoint of LucaOne, embedding inference code, downstream validation tasks data & code, and other materials are available: <a href='http://47.93.21.181/lucaone/'>FTP</a>.

**Details:**

The LucaOne's model code is available at: <a href='https://github.com/LucaOne/LucaOne'>LucaOne Github </a> or <a href='http://47.93.21.181/lucaone/LucaOne/'>LucaOne</a>.

The trained-checkpoint files are available at: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/'>TrainedCheckPoint</a>.

LucaOne's representational inference code is available at: <a href='https://github.com/LucaOne/LucaOneApp'>LucaOneApp Github</a> or <a href='http://47.93.21.181/lucaone/LucaOneApp'>LucaOneApp</a>.

The project of 8 downstream tasks is available at: <a href='https://github.com/LucaOne/LucaOneTasks'>LucaOneTasks Github</a> or <a href='http://47.93.21.181/lucaone/LucaOneTasks'>LucaOneTasks</a>.

The pre-training dataset of LucaOne is opened at: <a href='http://47.93.21.181/lucaone/PreTrainingDataset/'>PreTrainingDataset</a>.

The datasets of downstream tasks are available at: <a href='http://47.93.21.181/lucaone/DownstreamTasksDataset/'> DownstreamTasksDataset </a>.

Other supplementary materials are available at: <a href='http://47.93.21.181/lucaone/Others/'> Others </a>.


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

