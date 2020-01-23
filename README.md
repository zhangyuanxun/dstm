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
README.md
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
- model/   : implements a few topic model algorithms for this project
    - dstm.py  : implements the our domain-specific topic model algorithm
    - lda.py   : implements state-of-the-art algorithm [Latent Dirichlet Allocation (LDA)](http://jmlr.org/papers/volume3/blei03a/blei03a.pdf)
    - plsa.py  : implements state-of-the-art algorithm [Probabilistic Latent Semantic Analysis (pLSA)](https://www.iro.umontreal.ca/~nie/IFT6255/Hofmann-UAI99.pdf)
- data_collector.py: utility to collect data from raw dataset (NSF Grant dataset) or from other websites (Google Scholar）


## Getting Started
### Data Collecting
In the data collecting stage, we collect three types of the dataset: a)papers,  we collect papers from specific scientific domains. In current project, we collect papers 
from two domains: bioinformatics, neuroscience; b) tools, we collect types of tools 
 
(Note, we have already collected the relevant dataset. it is not necessary to run the data collecting script unless you want to collect new dataset). 
- Collect papers from bioinformatics domain
```
python data_collector.py --domain bio
```
- Collect papers from neuroscience domain
```
python data_collector.py --domain neuro
```


