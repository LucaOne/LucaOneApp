# LucaOne APP  
Embedding using LucaOne  

## TimeLine    
* 2025/04/08:
    * **LucaOne**          
      add `checkpoint=36000000` for `LucaOne`      
      location: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/latest/models/lucaone/lucaone/checkpoint-step36000000/'>checkpoint-step36000000</a>
    * **LucaOne-Gene**         
      add `checkpoint=36800000` for `LucaOne-Gene` (only trained using `DNA` and `RNA`)     
      location: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/latest/models/lucaone/lucaone-gene/checkpoint-step36800000/'>checkpoint-step36800000</a>
    * **LucaOne-Prot**        
      add `checkpoint=30000000` for `LucaOne-Prot` (only trained using `Protein`)       
      location: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/latest/models/lucaone/lucaone-prot/checkpoint-step30000000/'>checkpoint-step30000000</a>

* 2024/10/01: optimized embedding inference code: `src/llm/lucagplm/get_embedding.py`    
* 2024/08/01: add `checkpoint=17600000`, location: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/checkpoint-step17600000/'>checkpoint-step17600000</a>     

This project will download the checkpoint automatically from our `FTP` according to the value of parameter:   
* **--llm_step**   
* **--llm_version**  
* **--llm_step**


## Embedding Recommendation
| --llm_type | --llm_version  |              --llm_step              |                 Usage (seq_type)                 |
|:----------:|:--------------:|:------------------------------------:|:------------------------------------------------:|
| `lucaone`  |   `lucaone`    | `36000000`, `17600000`, or `5600000` | both `gene` (i.e. `DNA`, `RNA`) and `prot` sequences |
| `lucaone`  | `lucaone-gene` |              `36800000`              |    only for `gene` (i.e. `DNA`, `RNA`) sequences     |
| `lucaone`  | `lucaone-prot` |              `30000000`              |             only for `prot` sequence             | 



## 1. Embedding     
Two embedding methods for nucleic acid or protein sequence: `matrix` or `vector`.        
suggestion: If `matrix` is applied, it can be converted to a vector in the downstream networks as follows when using the embedding matrix:
* [CLS] Vector (embedding matrix[0, :], no parameterized pooling)
* Avg Pooling(No parameterized pooling)      
* Max Pooling(No parameterized pooling)            
* Value-Level Attention Pooling(recommended, parameterized pooling) (ref: https://arxiv.org/abs/2210.03970)    
...    
Pooling: transform the embedding matrix into a vector.             
Recommended: Use pooling in downstream networks rather than in data pre-processing.   
**Notice**: If your downstream task is sequence-level learning, use the pooling operation; otherwise, it is not required (for example: token-level task).  
If you are performing unsupervised analysis, such as clustering or T-SNE analysis, then directly use non-parametric pooing, such as [CLS] or Mean Pooling; 
if using embedding as the input to your downstream training task, such as classification or regression, it is recommended to use the parameterized pooing method, the parameters of pooling will be trainable in the downstream model.

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


#### 4) activate lucaone_app
conda activate lucaone_app

### step3:  install other requirements
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple       

## 3. Preparation      
Trained LucaOne Checkpoint FTP: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/'>TrainedCheckPoint for LucaOne</a>

**Notice:**    
The project will download automatically LucaOne Trained-CheckPoint from **FTP**.

When downloading automatically failed, you can manually download:

Copy the **TrainedCheckPoint Files(`models/` + `logs/`)** from <href> http://47.93.21.181/lucaone/TrainedCheckPoint/latest/ </href> into the directory: `./models/llm/`


## 4. Inference       
Scripts in `algorithms/`     
`inference_embedding_lucaone.py`: embedding using LucaOne(for nucleic acid (gene) or protein).           
`inference_embedding_dnabert2.py`: embedding using DNABert2(only for nucleic acid (gene)).     
`inference_embedding_esm.py`: embedding using ESM2(only for protein).       


**Suggestions and Instructions:**
1) Try to use a large GPU-memory machine for embedding inference, such as A100, H100, H200, etc., so that long sequences can be processed once.       
   LucaOne can process sequences of about `2800` in length at one time under A100;
2) For long sequences, LucaOne will do overlapped fragments in the sequence for embedding and finally merge them into a completed embedding matrix.        
   Please set `--embedding_complete` and `--embedding_complete_seg_overlap`;
3) If the GPU memory is not enough to process the longer sequence, it will use the CPU for embedding, so the speed will be reduced.       
   If your dataset is small, then you can set: `--gpu_id -1`;
4) If your dataset includes a lot of long sequences (more than 10,000 sequences), please set: `--embedding_complete`, `--embedding_complete_seg_overlap`, and `--embedding_fixed_len_a_time` (represent the maximum length for embedding at one-time).       
   If the sequence length is greater than the value of `--embedding_fixed_len_a_time`, fragment embedding is performed based on this value, and finally, the merge is performed; otherwise, according to the actual length of the sequence;
5) If `--embedding_complete` is not set, the code will truncate the sequence embedding according to the value of `--truncation_seq_length`;
6) For proteins, the length of most proteins is less than 1000; there are not many ultra-long protein sequences, so the value of `--embedding_fixed_len_a_time` can be set a large value or not be set;
7) For DNA, the DNA sequence of many tasks is very long; please set `--embedding_fixed_len_a_time`.  
   The larger the amount of ultra-long sequence, the smaller value should be set, such as `2800` under A100.      
   If the GPU embedding fails to process the longer sequence, the CPU will be called.      
   When the amount of dataset is not large, the spent time will not be long;
8) For RNA, most RNA is not very long, so the processing method can be consistent with the protein, so the `--embedding_fixed_len_a_time` can be set a larger value or not be set.

### Parameters
1) LucaOne checkpoint parameters:      
    * llm_dir: the path for storing the checkpoint LucaOne model，default: `../models/`         
    * llm_type: the llm type, default: `lucaone`         
    * llm_version: the version of LucaOne, default: `lucaone`, choices: [`lucaone`, `lucaone-gene`, `lucaone-prot`]
    * **llm_step:  the trained checkpoint of LucaOne**,  
    default:    
    `36000000` for `lucaone`, choices for `lucaone`: [`5600000`, `17600000`, `36000000`],        
    `36800000` for `lucaone-gene`,     
    `30000000` for `lucaone-prot` 

2) Important parameters:     
    * embedding_type: `matrix` or `vector`, output the embedding matrix or [CLS] vector for the entire sequence, recommend: matrix.      
    * trunc_type: truncation type: `right` or `left`, truncation when the sequence exceeds the maximum length.    
    * truncation_seq_length: the maximum length for embedding(not including [CLS] and [SEP]), itself does not limit the length, depending on the capacity of GPU.            
    * matrix_add_special_token: if the embedding is matrix, whether the matrix includes [CLS] and [SEP] vectors.           
    * seq_type: type of input sequence: `gene` or `prot`, `gene` for nucleic acid(DNA or RNA), `prot` for protein.        
    * input_file: the input file path for embedding(format: csv or fasta). The seq_id in the file must be unique and cannot contain special characters.     
    * save_path: the saving dir for storing the embedding file, one sequence for one embedding file.     
    * save_type: the embedding save type: `numpy` or `tensor`, default: `numpy`      
    * embedding_complete: When `embedding_complete` is set, `truncation_seq_length` is invalid. If the GPU memory is not enough to infer the entire sequence at once, it is used to determine whether to perform segmented completion (if this parameter is not used, 0.95*len is truncated each time until the CPU can process the length).       
    * embedding_complete_seg_overlap: When `embedding_complete` is set, whether the method of overlap is applicable to segmentation(overlap sliding window)
    * embedding_fixed_len_a_time: When the input sequence is too long for your GPU to complete the inference at once, you can specify the fixed length of the inference at once(default: None)     
    * gpu: the gpu id to use(-1 for cpu).

3) Optional parameters:    
    * id_idx & seq_idx: when the input file format is csv file, need to use `id_idx` and `seq_idx` to specify the column index in the csv (starting with 0).

  

**Notice:**     
A sequence outputs one embedding file named with `seq_id`, so the `seq_id` must be unique and must not contain special characters, such as Spaces, "/", and so on.   


### Examples:

#### 1) the **csv** file format of input     

**Notice:**
1. need to specify the column index of the sequence id(*id_idx**) and sequence(**seq_idx**), starting index: 0 .
2. The **sequence id** must be globally unique in the input file and cannot contain special characters (because the embedding file stored is named by the sequence id, e.g. `matrix_seq_1000.pt`).  

```shell

# for DNA or RNA
## using lucaone
cd ./algorithms/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8" 
python inference_embedding_lucaone.py \
    --llm_dir ../models \
    --llm_type lucaone \
    --llm_version lucaone \
    --llm_step 36000000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../data/test_data/gene/test_gene.csv \
    --id_idx 0 \
    --seq_idx 1 \
    --save_path ../embedding/lucaone/test_data/gene/test_gene/ \
    --save_type numpy \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0  
```

```shell
## using lucaone-gene
cd ./algorithms/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python inference_embedding_lucaone.py \
    --llm_dir ../models \
    --llm_type lucaone \
    --llm_version lucaone-gene \
    --llm_step 36800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../data/test_data/gene/test_gene.csv \
    --id_idx 0 \
    --seq_idx 1 \
    --save_path ../embedding/lucaone-gene/test_data/gene/test_gene/ \
    --save_type numpy \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0   
```

```shell
# for protein  
## using lucaone
cd ./algorithms/  
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python inference_embedding_lucaone.py \
    --llm_dir ../models  \
    --llm_type lucaone \
    --llm_version lucaone \
    --llm_step 36000000 \
    --truncation_seq_length 4096 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../data/test_data/prot/test_prot.csv \
    --id_idx 2 \
    --seq_idx 3 \
    --save_path ../embedding/lucaone/test_data/prot/test_prot/ \
    --save_type numpy \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0   
 ```

```shell
## using lucaone-prot
python inference_embedding_lucaone.py \
    --llm_dir ../models  \
    --llm_type lucaone \
    --llm_version lucaone-prot \
    --llm_step 30000000 \
    --truncation_seq_length 4096 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../data/test_data/prot/test_prot.csv \
    --id_idx 2 \
    --seq_idx 3 \
    --save_path ../embedding/lucaone-prot/test_data/prot/test_prot/ \
    --save_type numpy \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0   
```

#### 2) the **fasta** file format of input   

**Notice:**   
1. The **sequence id** must be globally unique in the input file and cannot contain special characters (because the embedding file stored is named by the sequence id).

```shell
# for DNA or RNA  
## using lucaone
cd ./algorithms/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python inference_embedding_lucaone.py \
    --llm_dir ../models \
    --llm_type lucaone \
    --llm_version lucaone \
    --llm_step 36000000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../data/test_data/gene/test_gene.fasta \
    --save_path ../embedding/lucaone/test_data/gene/test_gene/ \
    --save_type numpy \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0   
```

```shell
## using lucaone-gene  
python inference_embedding_lucaone.py \
    --llm_dir ../models \
    --llm_type lucaone \
    --llm_version lucaone-gene \
    --llm_step 36800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../data/test_data/gene/test_gene.fasta \
    --save_path ../embedding/lucaone-gene/test_data/gene/test_gene/ \
    --save_type numpy \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0  
```   

```shell
# for protein  
## using lucaone
cd ./algorithms/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python inference_embedding_lucaone.py \
    --llm_dir ../models \
    --llm_type lucaone \
    --llm_version lucaone \
    --llm_step 36000000 \
    --truncation_seq_length 4096 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../data/test_data/prot/test_prot.fasta \
    --save_path ../embedding/lucaone/test_data/prot/test_prot/ \
    --save_type numpy \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0   
```

```shell
## using lucaone-prot
python inference_embedding_lucaone.py \
    --llm_dir ../models \
    --llm_type lucaone \
    --llm_version lucaone-prot \
    --llm_step 30000000 \
    --truncation_seq_length 4096 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../data/test_data/prot/test_prot.fasta \
    --save_path ../embedding/lucaone-prot/test_data/prot/test_prot/ \
    --save_type numpy \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0   
```

## 5. Embedding using DNABert2 for Gene(DNA or RNA)            
**Notice：** Need to switch the virtual environment

### for DNABert2 Embedding
activate deactivate    
conda create -n lucaone_app_dnabert2 python=3.9.13    
conda activate lucaone_app_dnabert2      
pip install -r requirements_dnabert2.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


##### for `csv` format file as input
```shell
# for gene
cd ./algorithms/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python inference_embedding_dnabert2.py \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../data/test_data/prot/test_gene.csv \
    --id_idx 0 \
    --seq_idx 1 \
    --save_path ../embedding/danbert2/test_data/prot/test_gene/ \
    --save_type numpy \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --gpu_id 0    
```

##### for `fasta` format file as input
```shell
# for gene
cd ./algorithms/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python inference_embedding_dnabert2.py \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../data/test_data/prot/test_gene.fasta \
    --save_path ../embedding/danbert2/test_data/prot/test_gene/ \
    --save_type numpy \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --gpu_id 0   
```

## 6. Embedding using ESM2 for Prot      

##### for `csv` format file as input    
```shell
# for protein
cd ./algorithms/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python inference_embedding_esm.py \
    --llm_type esm2 \
    --llm_version 3B \
    --truncation_seq_length 4096 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../data/test_data/prot/test_prot.csv \
    --id_idx 0 \
    --seq_idx 1 \
    --save_path ../embedding/esm2/test_data/prot/test_prot/ \
    --save_type numpy \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0   
```

##### for `fasta` format file as input  
```shell
# for protein
cd ./algorithms/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python inference_embedding_esm.py \
    --llm_type esm2 \
    --llm_version 3B \
    --truncation_seq_length 4096 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../data/test_data/prot/test_prot.fasta \
    --save_path ../embedding/esm2/test_data/prot/test_prot/ \
    --save_type numpy \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0   
```

## 7. Data and Code Availability
**FTP:**   
Pre-training data, code, and trained checkpoint of LucaOne, embedding inference code, downstream validation tasks data & code, and other materials are available: <a href='http://47.93.21.181/lucaone/'>FTP</a>.

**Details:**

The LucaOne's model code is available at: <a href='https://github.com/LucaOne/LucaOne'>LucaOne Github </a> or <a href='http://47.93.21.181/lucaone/LucaOne/'>LucaOne</a>.

The trained-checkpoint files are available at: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/latest'>TrainedCheckPoint</a>.

LucaOne's representational inference code is available at: <a href='https://github.com/LucaOne/LucaOneApp'>LucaOneApp Github</a> or <a href='http://47.93.21.181/lucaone/LucaOneApp'>LucaOneApp</a>.

The project of 8 downstream tasks is available at: <a href='https://github.com/LucaOne/LucaOneTasks'>LucaOneTasks Github</a> or <a href='http://47.93.21.181/lucaone/LucaOneTasks'>LucaOneTasks</a>.

The pre-training dataset of LucaOne is opened at: <a href='http://47.93.21.181/lucaone/PreTrainingDataset/'>PreTrainingDataset</a>.

The datasets of downstream tasks are available at: <a href='http://47.93.21.181/lucaone/DownstreamTasksDataset/'> DownstreamTasksDataset </a>.

Other supplementary materials are available at: <a href='http://47.93.21.181/lucaone/Others/'> Others </a>.


## 8. Contributor    
<a href="https://scholar.google.com.hk/citations?user=RDbqGTcAAAAJ&hl=en" title="Yong He">Yong He</a>,
<a href="https://scholar.google.com/citations?user=lT3nelQAAAAJ&hl=en" title="Zhaorong Li">Zhaorong Li</a>,
<a href="https://scholar.google.com/citations?view_op=list_works&hl=en&user=uvrzUfEAAAAJ" title="Yongtao Shan">Yongtao Shan</a>, Yanhong Wei,
<a href="https://scholar.google.com.hk/citations?hl=zh-CN&pli=1&user=Zhlg9QkAAAAJ" title="Yuan-Fei Pan">Yuan-Fei Pan</a>,
<a href="https://scholar.google.com/citations?user=1KJOH7YAAAAJ&hl=zh-CN&oi=ao" title="Mang Shi">Mang Shi</a> 


## 9. Zenodo     
We have uploaded the model code, training scripts, and embedding inference scripts of LucaOne;    
The mode code, training and evaluation scripts, datasets, and trained models for downstream tasks,    
and additional supplementary materials to Zenodo (10.5281/zenodo.15171943).    
However, due to the substantial size of the pretraining dataset of LucaOne, it has not been included on Zenodo.     
Instead, it remains accessible via our publicly available FTP server (**<a href='http://47.93.21.181/lucaone/PreTrainingDataset/'>LucaOne Pretraining dataset</a>**).     
We are actively seeking an open FTP platform with sufficient storage capacity to host our pretraining dataset.

**<a href='https://doi.org/10.5281/zenodo.15171943'>LucaOne Zenodo</a>**


## 10. Citation
**<a href='https://www.biorxiv.org/content/10.1101/2024.05.10.592927v1'>LucaOne Biorxiv</a>**


@article {LucaOne,                
author = {Yong He and Pan Fang and Yongtao Shan and Yuanfei Pan and Yanhong Wei and Yichang Chen and Yihao Chen and Yi Liu and Zhenyu Zeng and Zhan Zhou and Feng Zhu and Edward C. Holmes and Jieping Ye and Jun Li and Yuelong Shu and Mang Shi and Zhaorong Li},     
title = {LucaOne: Generalized Biological Foundation Model with Unified Nucleic Acid and Protein Language},      
elocation-id = {2024.05.10.592927},        
year = {2024},         
doi = {10.1101/2024.05.10.592927},        
publisher = {Cold Spring Harbor Laboratory},        
URL = {https://www.biorxiv.org/content/early/2024/05/14/2024.05.10.592927 },        
eprint = {https://www.biorxiv.org/content/early/2024/05/14/2024.05.10.592927.full.pdf },        
journal = {bioRxiv}        
}


## 11. LucaTeam

<center>
<img alt="LucaTeam" src="./pics/LucaTeam.jpg"/>

Fig. 5 LucaTeam at the West Lake in Hangzhou.
</center>   


