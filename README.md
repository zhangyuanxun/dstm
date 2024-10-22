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
images/
model/
    dstm.py
    lda.py
    plsa.py
output/
    bio_base_model/
    neuro_base_model/
constants.py
data_collector.py
data_processor.py
input_fn.py
model_api.py
README.md
run_dstm.py
run_lda.py
run_plsa.py
text_utils.py
trend_utils.py
tsne.py
visualization.py
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
- imgs/ : contains some experimental images
- model/   : implements core topic model algorithms for this project
    - dstm.py  : implements the our domain-specific topic model algorithm
    - lda.py   : implements state-of-the-art algorithm [Latent Dirichlet Allocation (LDA)](http://jmlr.org/papers/volume3/blei03a/blei03a.pdf)
    - plsa.py  : implements state-of-the-art algorithm [Probabilistic Latent Semantic Analysis (pLSA)](https://www.iro.umontreal.ca/~nie/IFT6255/Hofmann-UAI99.pdf)
- output/  : the trained model files will be automatically saved in this folder
    - bio_base_model : pre-trained bioinformatics model file
    - neuro_base_model: pre-trained neuroscience model file
- constants.py : define some constant variables
- data_collector.py : utility functions to collect data from raw dataset (NSF Grant dataset) or from other websites (Google Scholar)
- data_processor.py : utility functions to transform raw dataset into required data format (such as bag-of-words) by our model
- input_fn.py : define input pipelines
- model_api.py : An example to demonstrate how to use DSTM model
- run_dstm.py : main file to run DSTM model, which includes model parameter estimation and inference
- run_lda.py : main file to run state-of-the-art LDA model, which includes model parameter estimation and inference
- run_plsa.py : main file to run state-of-the-art pLSA model, which includes model parameter estimation and inference
- text_utils.py : utility functions for text processing
- trend_utils.py : core utility functions for topics trend analysis
- tsne.py : tSNE algorithm to visualize topics in 2D space
- visualization.py : core functions for topic visualization and demonstration

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
python run_dstm.py --data_source bio --mode inf --model_folder <model folder name>
```

## Visualization
Model visualization or model representation is very important in unsupervised learning, which can help us to understand the latent patterns of our problem, and evaluate the feasibility of our model.

After the model parameter estimation, the model files will be generated. Then, you can explore our model by using the visualization interfaces we provided. We have provided the pre-trained model for each domain (bioinformatics and neuroscience). 

- visualize the relationship between research topics, research tools, and research datasets in tables. For example, you visualize the relationship in the bioinformatics domain based on a pre-trained model. Here we use the pre-trained model provided in our repo under the folder (\output).
And, the parameter (topK) is to describe the top K terms to be presented.
```
python visualization.py --data_source bio --topk 10 --model_folder bio_base_model
```
If anything goes well, you can get the 50 research topics and their relationships among research topics, research tools, and research datasets as below (Here, I list three topics.)
```
topic 0:
             protein    ---       0.0359              matlab    ---       0.6719                mrna    ---       0.3894
               state    ---       0.0318             Gromacs    ---       0.2117               mirna    ---       0.1380
            dynamics    ---       0.0291               BLAST    ---       0.0125                 plb    ---       0.0918
              single    ---       0.0133           clustalw2    ---       0.0111               mtdna    ---       0.0903
          activation    ---       0.0130                pfam    ---       0.0090                tcga    ---       0.0576
             complex    ---       0.0102             mercury    ---       0.0085              rnaseq    ---       0.0369
            proteins    ---       0.0093            Autodock    ---       0.0079               sirna    ---       0.0329
            involved    ---       0.0086              Bowtie    ---       0.0067                hmec    ---       0.0314
                 new    ---       0.0085             clustal    ---       0.0054        metabolomics    ---       0.0297
           mechanism    ---       0.0083               Glide    ---       0.0044            chipqpcr    ---       0.0186


topic 1:
               model    ---       0.0353              matlab    ---       0.7495                mrna    ---       0.4373
                time    ---       0.0246                 dss    ---       0.0909               mirna    ---       0.1212
         probability    ---       0.0152              MUSCLE    ---       0.0204               mtdna    ---       0.1039
              number    ---       0.0118                affy    ---       0.0130        metabolomics    ---       0.0957
            observed    ---       0.0092              glmnet    ---       0.0127              rnaseq    ---       0.0764
            approach    ---       0.0086               BLAST    ---       0.0111          encodedata    ---       0.0661
           inference    ---       0.0086              Picard    ---       0.0063                tcga    ---       0.0467
               rates    ---       0.0078                rfam    ---       0.0060             lncrnas    ---       0.0133
             studies    ---       0.0076             mercury    ---       0.0059                gdsc    ---       0.0067
                 low    ---       0.0076                STAR    ---       0.0057                hmec    ---       0.0051


topic 2:
                cell    ---       0.0560              matlab    ---       0.8432               mirna    ---       0.2823
               cells    ---       0.0217              glmnet    ---       0.0196                mrna    ---       0.2080
              target    ---       0.0187                 IMP    ---       0.0193          encodedata    ---       0.1162
              models    ---       0.0177                UCSC    ---       0.0115                tcga    ---       0.0771
             effects    ---       0.0145           Cufflinks    ---       0.0090              rnaseq    ---       0.0752
               noise    ---       0.0145             mercury    ---       0.0064        metabolomics    ---       0.0617
             control    ---       0.0135              Picard    ---       0.0061               mtdna    ---       0.0558
             results    ---       0.0129               BLAST    ---       0.0051                 hic    ---       0.0295
          functional    ---       0.0128              tophat    ---       0.0038               sirna    ---       0.0283
          associated    ---       0.0109             BioMart    ---       0.0036                ccle    ---       0.0179
```

Similarly, you can visualize the research topics in neuroscience domain as below.
```
python visualization.py --data_source neuro --topk 10 --model_folder neuro_base_model
```
Then, you will obtain 70 topics in neuroscience domain. (Here, I list three topics.) 
```
topic 0:
             protein    ---       0.0359              matlab    ---       0.6719                mrna    ---       0.3894
               state    ---       0.0318             Gromacs    ---       0.2117               mirna    ---       0.1380
            dynamics    ---       0.0291               BLAST    ---       0.0125                 plb    ---       0.0918
              single    ---       0.0133           clustalw2    ---       0.0111               mtdna    ---       0.0903
          activation    ---       0.0130                pfam    ---       0.0090                tcga    ---       0.0576
             complex    ---       0.0102             mercury    ---       0.0085              rnaseq    ---       0.0369
            proteins    ---       0.0093            Autodock    ---       0.0079               sirna    ---       0.0329
            involved    ---       0.0086              Bowtie    ---       0.0067                hmec    ---       0.0314
                 new    ---       0.0085             clustal    ---       0.0054        metabolomics    ---       0.0297
           mechanism    ---       0.0083               Glide    ---       0.0044            chipqpcr    ---       0.0186


topic 1:
               model    ---       0.0353              matlab    ---       0.7495                mrna    ---       0.4373
                time    ---       0.0246                 dss    ---       0.0909               mirna    ---       0.1212
         probability    ---       0.0152              MUSCLE    ---       0.0204               mtdna    ---       0.1039
              number    ---       0.0118                affy    ---       0.0130        metabolomics    ---       0.0957
            observed    ---       0.0092              glmnet    ---       0.0127              rnaseq    ---       0.0764
            approach    ---       0.0086               BLAST    ---       0.0111          encodedata    ---       0.0661
           inference    ---       0.0086              Picard    ---       0.0063                tcga    ---       0.0467
               rates    ---       0.0078                rfam    ---       0.0060             lncrnas    ---       0.0133
             studies    ---       0.0076             mercury    ---       0.0059                gdsc    ---       0.0067
                 low    ---       0.0076                STAR    ---       0.0057                hmec    ---       0.0051


topic 2:
                cell    ---       0.0560              matlab    ---       0.8432               mirna    ---       0.2823
               cells    ---       0.0217              glmnet    ---       0.0196                mrna    ---       0.2080
              target    ---       0.0187                 IMP    ---       0.0193          encodedata    ---       0.1162
              models    ---       0.0177                UCSC    ---       0.0115                tcga    ---       0.0771
             effects    ---       0.0145           Cufflinks    ---       0.0090              rnaseq    ---       0.0752
               noise    ---       0.0145             mercury    ---       0.0064        metabolomics    ---       0.0617
             control    ---       0.0135              Picard    ---       0.0061               mtdna    ---       0.0558
             results    ---       0.0129               BLAST    ---       0.0051                 hic    ---       0.0295
          functional    ---       0.0128              tophat    ---       0.0038               sirna    ---       0.0283
          associated    ---       0.0109             BioMart    ---       0.0036                ccle    ---       0.0179
```

- visualize the trend of tools or datasets been investigated by researchers for a particular research topic over the last ten years. 
To visualize the trend of tools or datasets over time, we need to run the trend analysis command based on our pre-trained model first by using the command below. 

(Note, we have already analyzed the trend for each domain (bio, neuro). So, you don't need to run the analysis command.)

```
python visualization.py --data_source bio --type trend_analysis --model_folder bio_base_model --trend_type tool
```
After finishing the analysis algorithm, you can plot the tool or dataset trend for each topic for a particular domain, by using the command below. 
```
python visualization.py --data_source bio --type trend --model_folder bio_base_model --trend_type tool --topic_id 38
```
If anything goes well, there will be a trend figure popped up that is similar to the figure below, and this figure will be also saved in the relevant location (\output\bio_base_model\tool_trend)
<p align="center">
<img src='imgs/topic-38.png' width="500px"/>
</p>
Feel free to explore other topics, for example, the topic_id 44 is relevant deep learning for bioinformatics.

```
python visualization.py --data_source bio --type trend --model_folder bio_base_model --trend_type tool --topic_id 44
```
<p align="center">
<img src='imgs/topic-44.png' width="500px"/>
</p>

- visualize the research topics in 2D-space for finding similar topics among scientific communities, which could be applied to cross-domain recommendations, or cross-domain knowledge sharing.
We use the tSNE algorithm to mapping high dimensional datasets (topics distribution) into low dimensional space (such as 2D space), which provides scientists to find similar topics for knowledge sharing or collaboration.
Our algorithm can be applied to a single domain demonstration or cross-domain demonstration. For example, you can map bioinformatics topics into 2D space by
using the command below,
```
python tsne.py --type single --model_folder1 bio_base_model --num_iterations 2000
```
If anything goes well, there will be a figure popped up that is similar to the figure below, and this figure will be also saved in the relevant location (\output\)

<p align="center">
    <img src='imgs/tsne_embedding_single_domain_bio.png' width="500px"/>
</p>

Or you can also explore neuro topics into 2D space by using this command,
```
python tsne.py --type single --model_folder1 neuro_base_model --num_iterations 2000
```
<p align="center">
    <img src='imgs/tsne_embedding_single_domain_neuro.png' width="500px"/>
</p>
Finally, you can also map topics from different into the same 2D space by using "cross" mode for cross domain demonstration

```
python tsne.py --type cross --model_folder1 bio_base_model --model_folder2 neuro_base_model --num_iterations 5000
```
<p align="center">
    <img src='imgs/tsne_embedding_cross_domain.png' width="500px"/>
</p>
In the figure above, the lighter color belongs to one domain, and the darker color belongs to another domain. 

## Model APIs
We can also use the pre-trained model to do some inferences or recommendations based on the user's queries. 
For example, you can ask the model to recommend tools or datasets for "neuron simulation". 
Here, we provide an example to demonstrate how to use our model. This example is included in the file (model_api.py). 
Basically, you just need two steps a) load the model; b) query from the model;
```
# a) load the model 
# define the model path to load the model (here, we use our bioinformatics pre-trained model)

model_folder = 'neuro_base_model/'
model_path = os.path.join(dirname(os.path.realpath(__file__)), 'output/') + model_folder

# load the model by initialize the DSTM model
model = DSTM_Model(model_path)

# b) define your query, and query from the model
s = 'neuron simulation in neuroscience'
model.query(s)
```
Then, our model will recommend some tools and datasets for relevant "neuron simulation" research topic. 
```
Highly matched topics is:
	 topic 38 : neurons channels neuron somatic bursting network
		 Suggested tools:  neurontool, xppaut, modeldb
		 Suggested datasets: somatic, bursting, stomatogastric

	 topic 50 : itch network scratching model models social
		 Suggested tools:  brian, genesistool, moosetool
		 Suggested datasets: scratch, integrators, spinal

	 topic 11 : effects behavioral number time similar state
		 Suggested tools:  matlab, helmholtztool, spm
		 Suggested datasets: inhibitory, excitatory, circuit
```
When use provides more meaningful queries, our model can also output more relevant topics.

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
