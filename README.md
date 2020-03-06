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
    common_tools.py
    stopwords.txt
model/
    dstm.py
    lda.py
    plsa.py
output/
    bio_base_model/
    neuro_base_model/
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
    common_tools.py  : list of the common tools (such as python, c++), which are not considered by our model
    stopwords.txt  : list of the stop words
- model/   : implements core topic model algorithms for this project
    - dstm.py  : implements the our domain-specific topic model algorithm
    - lda.py   : implements state-of-the-art algorithm [Latent Dirichlet Allocation (LDA)](http://jmlr.org/papers/volume3/blei03a/blei03a.pdf)
    - plsa.py  : implements state-of-the-art algorithm [Probabilistic Latent Semantic Analysis (pLSA)](https://www.iro.umontreal.ca/~nie/IFT6255/Hofmann-UAI99.pdf)
- output/  : the trained model files will be automatically saved in this folder
    - bio_base_model : pre-trained bioinformatics model file
    - neuro_base_model: pre-trained neuroscience model file
- data_collector.py: utility functions to collect data from raw dataset (NSF Grant dataset) or from other websites (Google Scholar)
- data_processor.py: utility functions to transform raw dataset into required data format (such as bag-of-words) by our model. 
- run_dstm.py: main file to run DSTM model, which includes model parameter estimation and inference. 
- run_lda.py: main file to run state-of-the-art LDA model, which includes model parameter estimation and inference. 
- run_plsa.py: main file to run state-of-the-art pLSA model, which includes model parameter estimation and inference. 
- text_utils.py : utility functions for text processing

## Getting Started
### Data Collecting
In the data collecting stage, we collect three types of the dataset: a) papers,  we collect papers from specific scientific domains. In the current project, we collect papers from two domains: bioinformatics, neuroscience; b) tools, we collect types of tools; c) datasets, we collect types of datasets. To collect papers, we 
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
During the data collecting, you have collected paper texts from journals. In the data processing stage, you need to process raw text datasets for a suitable dataset format
for the model. In our model, we use bag-of-words as our model input. Hence, for data processing, we need to transform the raw text format into the bag-of-words format. Besides, in the data processing stage, we also need to generate the whole vocabulary, tool-to-doc, and dataset-to-doc tables. And the last thing, you need to generate vocabulary firstly before generating bag-of-words, tool-to-doc, or dataset-to-doc tables.

(Note, we have already processed the raw datasets based on the current data collections.. it is not necessary to run the data processing scripts unless you have a new dataset to process). 

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
The DSTM is a probabilistic graphical model with latent variables. In our model, the latent variables are used to describe the patterns among research topics, tools, and datasets, which are unknown to us in the beginning. The goal of parameters estimation is to estimate these latent variables. In the model, we use the Gibbs sampling algorithm to infer these latent patterns. 

Before staring parameters estimation, you need to set up some parameters for running the program,
- the type of data source (data_source): it defines the type of data source you need to train your model, such as neuroscience (neuro), bioinformatics (bio). 
- the number of topics (num_topcis): it defines the number of topics you want to capture from your corpus. We usually choose the value from 50 to 200 for both bioinformatics and neuroscience domains.
- the number of iterations (num_iterations): it defines the number of iterations (epochs) your program will run. We usually choose the value from 50 to 100. 
- the mode (mode): it defines the mode to train the model, for example, the estimation mode (est) uses a certain ratio (such as 80%) of dataset for parameter estimation and rest of dataset for evaluation; the inference mode (inf) is used to infer the probability of new dataset after parameter estimation; and the demo mode (demo) is used to learn parameters for model demonstration, which uses all datasets. 
- the run mode of the program (run_mode): it defines the running mode to run the program. Our program supports two modes, which allow you to run the program from begining or run the program continuously. 
- the seed (seed): it is used by the random number generator, which can help you to reproduce the experimental results. 
- verbose (verbose): it shows performance debug information.
- the model output folder (model_folder): it specifies the model folder name for running continuously. If the program is running from the beginning, it will create a model by default.
- model evaluation (evaluate): it specifies whether model evaluation or not.
- save the model file (save): it choose whether to save the model file or not.

Typically, you need to set up the four key parameters(data_source, num_topcis, num_iterations, mode, save) and keep the rest of the parameters as the default. For example, if you train the model for demonstration using the corpus from the bioinformatics domain, the command will be
```
python run_dstm.py --data_source bio --mode demo --run_mode start --num_iterations 50 --num_topics 50 --save yes
```
Similarly, if you train the model using corpus from the neuroscience domain, the command will be
```
python run_dstm.py --data_source neuro --mode demo --run_mode start --num_iterations 50 --num_topics 70 --save yes
```
After finishing training, our program will output the path of model file for future use. 

Similarly, you can also train state-of-the-art model (PLSA, LDA) using similar commands.
```
python run_lda.py --data_source bio --mode demo --run_mode start --num_iterations 50 --num_topics 50 --save yes
python run_plsa.py --data_source bio --mode demo --num_iterations 50 --num_topics 50 --save yes
```
(Note, I haven't optimized our program. So, it will be slow to train the model. In addition, for each domain, we have provided trained model under the output/ folder, you can directly use it.)

### Model Inference
The model inference is to infer parameters of the new dataset based on the trained model. The basic step to use model inference will be,
- Run model parameters estimation (as described above) based on a dataset from a particular domain using estimation mode (est), and save the model file. Our program will automatically split the dataset into the estimation part and inference part, and choose the estimation part for training. 
```
python run_dstm.py --data_source bio --mode est --run_mode start --num_iterations 50 --num_topics 70 --save yes
```
- Then, run the model inference algorithm by changing the mode into inference(inf) and providing the name of model folder.
```
python run_dstm.py --data_source bio --mode inf --model_folder XXXXXX
```

## Visualization
Model visualization or model representation is very important in unsupervised learning, which can help us to understand the latent patterns of our problem, and evaluate the feasibility of our model.

After the model parameter estimation, the model files will be generated. Then, you can explore our model by using the visualization interfaces we provided. We have provided the pre-trained model for each domain (bioinformatics and neuroscience). 

- visualize the relationship between research topics, research tools, and research datasets. 

- visualize the trend of tools or datasets been investigated by researchers over the last ten years. 

- visualize the research topics in 2D-space for fiding similar topics among scientific communities, which could be applied to cross-domain. 


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
