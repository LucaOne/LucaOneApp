# LucaOne APP
使用LucaOne对序列进行embedding    

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
这个工程相应的checkpoint会自动从FTP根据embedding命令提供的下列参数值下载相应版本的checkpoint文件: 
* **--llm_type**
* **--llm_version**
* **--llm_step**


# Embedding Recommendation
| --llm_type | --llm_version  |              --llm_step              |             Usage (seq_type)              |
|:----------:|:--------------:|:------------------------------------:|:-----------------------------------------:|
| `lucaone`  |   `lucaone`    | `36000000`, `17600000`, 或者 `5600000` | 对`DNA`、`RNA`、或者`Protein`都可以<br/>(无差别embedding) |
| `lucaone`  | `lucaone-gene` |              `36800000`              |          只对`DNA`、`RNA`embedding           |
| `lucaone`  | `lucaone-prot` |              `30000000`              |        只对`Protein`embedding               |


## 1. Embedding
对核酸序列或者蛋白序列进行embedding，embedding有两种方式：matrix矩阵与vector向量。          
建议：如果使用matrix，在后续的下游模型中具体使用时，可以使用: 
* [CLS] Vector (embedding matrix[0, :], 无参数化pooling)
* Avg Pooling/Mean Pooling(无参数化pooling)
* Max Pooling(无参数化pooling)
* Value-Level Attention Pooling(参数化pooling, recommended) (ref: https://arxiv.org/abs/2210.03970)      
...    
Pooling: transform the embedding matrix into a vector.             
Recommended: Use pooling in downstream networks rather than in data pre-processing.    
如果你是进行无监督式的分析，比如聚类分析、比如T-SNE分析，那么直接使用无参数化的pooing，比如CLS，Mean Pooling，如果使用embedding
作为你的下游训练任务的输入，比如分类或者回归，那么建议使用参数化的pooing方法，pooling中的参数在下游模型中会进行学习。   
**Notice**: If your downstream task is sequence-level learning, use the pooling operation; otherwise, it is not required (for example: token-level task).


## 2. Environment installation          
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
Trained LucaOne Checkpoint FTP: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/latest/'>TrainedCheckPoint for LucaOne</a>    

从上述的FTP中将目录`logs`与目录`models`拷贝到 `./models/llm/`目录下          

后续会持续更新checkpoint    

## 3. Inference          
脚本在目录`algorithms/`下：    
`inference_embedding_lucaone.py`: embedding using LucaOne(for nucleic acid or protein)         
`inference_embedding_dnabert2.py`: embedding using DNABert2(only for nucleic acid)      
`inference_embedding_esm.py`: embedding using ESM2(only for protein)     

**建议与说明:**         
1）尽量使用显存大进行embedding 推理，如：A100，H100，H200等，这样一次性能够处理较长的序列，LucaOne在A100下可以一次性处理`2800`左右长度的序列；   
2）对于超长序列，LucaOne会进行Overlap分片进行embedding，最后合并成完整的embedding，请设置`--embedding_complete`与`--embedding_complete_seg_overlap`；    
3）如果显卡不足以处理输入的序列长度，会调用CPU进行处理，这样速度会变慢，如果你的数据集中长序列不是很多，那么可以使用这种方式: `--gpu_id -1`；      
4）如果你的数据集中长序列很多，比如: 万条以上，那么再设置`--embedding_complete`与`--embedding_complete_seg_overlap`之外，再加上设置`--embedding_fixed_len_a_time`，表示一次性embedding的最大长度。
如果序列长度大于这个长度，基于这个长度进行分片embedding，最后进行合并。否则根据序列的实际长度；    
5）如果不设置`--embedding_complete`，那么根据设置的`--truncation_seq_length`的值对序列进行截断embedding；  
6）对于蛋白，因为绝大部分蛋白长度在1000以下，因此超长蛋白序列不会很多，因此可以将`--embedding_fixed_len_a_time`设置长一点或者`不设置`；    
7）对于DNA，因为很多任务的DNA序列很长，那么请设置`--embedding_fixed_len_a_time`。    
如果数据集中超长序列数据量越多，该值设置越小一点，比如在A100下设置为`2800`，否则设置大一点，如果GPU根据这个长度embedding失败，则会调用CPU。如果数据集数不大，则时间不会很久；          
8）对于RNA，因为大部分RNA不会很长，因此与蛋白处理方式一致，因此可以将`--embedding_fixed_len_a_time`设置长一点或者不设置；


机器：GPU可用的机器(如我们的A100可用的机器)         
位置：cd LucaOneApp/algorithms                

### Parameters      

#### 1) 模型版本参数(使用默认值即可):             
* llm_dir: 模型文件存放的路径，默认在`../models/`下         
* llm_type: 模型的类型，默认: `lucaone`       
* llm_version: 模型版本，默认: `lucaone`，选项：[`lucaone`, `lucaone-gene`, `lucaone-prot`] 
* **llm_step: 需要使用的checkpoint step**，
默认:    
  `36000000` for `lucaone`, choices for `lucaone`: [`5600000`, `17600000`, `36000000`],        
  `36800000` for `lucaone-gene`,     
  `30000000` for `lucaone-prot`    

#### 2) 重要参数: 
* embedding_type: matrix/vector, 分别为整个序列的矩阵或者[CLS]向量       
* trunc_type: 如果序列超过最大长度则阶段，right或者left     
* truncation_seq_length: 最大长度（不包括[CLS]与[SEP])，本身不限制长度，取决于embedding推理的显存          
* matrix_add_special_token: 如果embedding是matrix，则matrix是否包括[CLS]与[SEP]向量        
* seq_type: 输入序列的类型，gene表示核酸，prot表示蛋白          
* input_file: 输入文件路径（fasta格式或者csv格式）           
* save_path: embedding文件保存路径，一个序列保存成一个文件     
* save_type: embedding保存类型: `numpy` or `tensor`, 默认: `numpy`      
* embedding_complete: 当 `embedding_complete`被设置的时候, `truncation_seq_length`是无效的. 如果显存不够一次性推理整个序列，是否进行分段补全（如果不使用该参数，则每次截断0.95*len直到显卡可容纳的长度  
* embedding_complete_seg_overlap: 当`embedding_complete`被设置的时候, 使用对序列分段embedding的分段是否重叠(overlap sliding window)    
* embedding_fixed_len_a_time: When the input sequence is too long for your GPU to complete the inference at once, you can specify the fixed length of the inference at once(default: None)       
* gpu: 使用哪一个gpu id       

#### 3) 可选参数:     
* 如果参数: fasta输入的是csv文件，需要使用id_idx与seq_idx来指定在csv中的列号(0开始)           

#### 4) 注意:    
一个序列输出一个embedding文件，使用seq_id进行命名(e.g. `matrix_seq_1000.pt`)，因此seq_id最好不能有特殊字符，比如空格或者"/"字符           

#### 5) Examples:       

##### for `csv` format file as input   
```shell
# 对核酸(DNA或者RNA)进行embedding(输入csv文件，需要指明id与seq的列号)     
## using lucaone
cd algorithms/
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
# 对蛋白质进行embedding(输入csv文件，需要指明id与seq的列号)   
## using lucaone
cd algorithms/  
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

##### for `fasta` format file as input   

```shell
# 对对核酸(DNA或者RNA)进行embedding(输入fasta文件，seq头最好进行唯一id重命名，别包含特殊符号)   
## using lucaone
cd algorithms/
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
# 对蛋白质进行embedding(输入fasta文件，seq头最好进行唯一id重命名，别包含特殊符号)   
## using lucaone
cd algorithms/
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
cd algorithms/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python inference_embedding_dnabert2.py \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../data/test_data/prot/test_gene.csv \
    --id_idx 0 \
    --seq_idx 1 \
    --save_path ../embedding/danbert2/test_data/prot/test_gene \
    --save_type numpy \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --gpu_id 0    
```

##### for `fasta` format file as input
```shell
# for gene
cd algorithms/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python inference_embedding_dnabert2.py \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../data/test_data/prot/test_gene.fasta \
    --save_path ../embedding/danbert2/test_data/prot/test_gene \
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
cd algorithms/
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
    --save_path ../embedding/esm2/test_data/prot/test_prot \
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
cd algorithms/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python inference_embedding_esm.py \
    --llm_type esm2 \
    --llm_version 3B \
    --truncation_seq_length 4096 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../data/test_data/prot/test_prot.fasta \
    --save_path ../embedding/esm2/test_data/prot/test_prot \
    --save_type numpy \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0   
```

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
**<a href='https://www.biorxiv.org/content/10.1101/2024.05.10.592927v2'>LucaOne Biorxiv</a>**   
**<a href='https://www.nature.com/articles/s42256-025-01044-4'>LucaOne NMI 2025</a>**


He, Y., Fang, P., Shan, Y. et al. Generalized biological foundation model with unified nucleic acid and protein language. Nat Mach Intell 7, 942–953 (2025). https://doi.org/10.1038/s42256-025-01044-4



## 11. LucaTeam

<center>
<img alt="LucaTeam" src="./pics/LucaTeam.jpg"/>

Fig. 5 LucaTeam at the West Lake in Hangzhou.
</center>   

