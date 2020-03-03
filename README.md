# Domain-specific Topic Model for Knowledge Discovery in Computational and Data-Intensive Scientific Communities
This implements the models and algorithms that describe in the paper "Domain-specific Topic Model for Knowledge Discovery in Computational and Data-Intensive Scientific Communities"

## Prerequisites
- Python: 2.7
- Numpy: 1.16.5
- tqdm: 4.35.0
- BeautifulSoup: 4.8.0

## Structure of the code
```
collector/
    bio/
        bmc_bio_collector.py
        bmc_genomics_collector.py
        geno_bio_collector.py
        nucleic_acids_collector.py
        plos_compbio_collector.py
    neuro/
        fcn_collector.py
        jocn_collector.py
        jon_collector.py
        neuron_collector.py
dataset/
    bio/
        *.*
    neuro/
        *.*
model/
    dstm.py
    lda.py
    plsa.py
output/
data_collector.py
data_processor.py
README.md
run_dstm.py
run_lda.py
run_plsa.py
text_utils.py
```
- collector/ : contains scripts for to extract papers from online scientific journals archives
    - bio/
        - bio/bmc_bio_collector.py      : script to collect papers from BMC Bioinformatics
        - bio/bmc_genomics_collector.py : script to collect papers from BMC Genomics 
        - bio/geno_bio_collector.py     : script to collect papers from Genome Biology
        - bio/nucleic_acids_collector.py: script to collect papers from Nucleic Acids Research
        - bio/plos_compbio_collector.py : script to collect papers from PLOS Computational Biology
    - neuro/
        - neuro/fcn_collector.py        : script to collect papers from Frontier of Computational Neuroscience
        - neuro/jocn_collector.py       : script to collect papers from Journal of Computational Neuroscience
        - neuro/jon_collector.py        : script to collect papers from Journal of Neuroscience
        - neuro/neuron_collector.py     : script to coolect papers from Neuron Journal
- dataset/ : contains all the data include raw and processed of the project
    - bio/   : contains raw and processed datasets for bioinformatics domains
    - neuro/ : contains raw and processed datasets for neuroscience domains
- model/   : implements core topic model algorithms for this project
    - dstm.py  : implements the our domain-specific topic model algorithm
    - lda.py   : implements state-of-the-art algorithm [Latent Dirichlet Allocation (LDA)](http://jmlr.org/papers/volume3/blei03a/blei03a.pdf)
    - plsa.py  : implements state-of-the-art algorithm [Probabilistic Latent Semantic Analysis (pLSA)](https://www.iro.umontreal.ca/~nie/IFT6255/Hofmann-UAI99.pdf)
- data_collector.py: utility functions to collect data from raw dataset (NSF Grant dataset) or from other websites (Google Scholar)
- data_processor.py: utility functions to transform raw dataset into required data format (such as bag-of-words) by our model. 
- run_dstm.py: main file to run DSTM model, which includes model parameter estimation and inference. 
- run_lda.py: main file to run state-of-the-art LDA model, which includes model parameter estimation and inference. 
- run_plsa.py: main file to run state-of-the-art pLSA model, which includes model parameter estimation and inference. 
- text_utils.py : utility functions for text processing

## Getting Started
### Data Collecting
In the data collecting stage, we collect three types of the dataset: a) papers,  we collect papers from specific scientific domains. In current project, we collect papers 
from two domains: bioinformatics, neuroscience; b) tools, we collect types of tools; c) datasets, we collect types of datasets. To collect papers, we 
provide scripts to automatically collect papers from websites; to collect tools or datasets, we need some domain knowledge to collect relevant datasets manually.
 
(Note, we have already collected the relevant dataset. it is not necessary to run the data collecting scripts unless you want to collect new dataset from new domain). 
- Collect papers from bioinformatics domain
```
python data_collector.py --domain bio
```
- Collect papers from neuroscience domain
```
python data_collector.py --domain neuro
```
- Tips: Collect papers from other domains and other scripts
    - You need to mimic the scripts (such as bmc_bio_collector.py, bmc_genomics_collector.py) under the folders to extract text from websites. 

### Data Processing
During the data collecting, you have collected paper texts from journals. In the data processing stage, you need to process raw text datasets for suitable dataset format
for the model. In our model, we use bag-of-words as our model input. Hence, for data processing, we need to transform the raw text format into bag-of-words format. In addtion, in data processing stage, we also need to generate the whole vocabulary, tool-to-doc, and dataset-to-doc tables. And the last thing, you need to generate vocabulary firstly before generating bag-of-words, tool-to-doc, or dataset-to-doc tables.

(Note, we have already processed the raw datasets based on the current data collections.. it is not necessary to run the data processing scripts unless you have new dataset to process). 

Using the bioinformatics domain as an example, the basic routines for data processing will be,
- Generating the vocabulary (this command should be run at first)
```
python data_processor.py --domain bio --operation vocabulary
```
- Generating the bag-of-words
```
python data_processor.py --domain bio --operation bag-of-words
```
- Generating the tool-to-doc table
```
python data_processor.py --domain bio --operation tool
```
- Generating the dataset-to-doc table 
```
python data_processor.py --domain bio --operation dataset
```

### Model Parameters Estimation
THe DSTM is probabilistic graphical model with latent variables. In our model, the latent variables are used to describe 
the patterns among research topics, tools, and datasets, which are unknown to us in the beginning. The goal of parameters 
estimation is to estimate these latent variables. In the model, we use Gibbs sampling algorithm to infer these latent 
patterns. 

Before staring parameters estimation, you need to define 


### Model Inference


## Visualization
Visualization or model representation are very important in unsupversied learning, which can help us to understand the latent patterns of our problem, and evaluate the feasibility of the model.

After model parameter estimation, the model files will be generated. 

## Model APIs


## Citations
```
@INPROCEEDINGS{dstm2018zhang,
author={Y. {Zhang} and P. {Calyam} and T. {Joshi} and S. {Nair} and D. {Xu}},
booktitle={2018 IEEE International Conference on Big Data (Big Data)},
title={Domain-specific Topic Model for Knowledge Discovery through Conversational Agents in Data Intensive Scientific Communities},
year={2018},
volume={},
number={},
pages={4886-4895},
doi={10.1109/BigData.2018.8622309},
ISSN={null},
month={Dec},}
```
