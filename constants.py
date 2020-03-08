import os
from os.path import dirname

# Papers from bmc bioinformatics
BIO_BMCINFOR_INFO_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/bio/bmc_bioinfo_info.json')
BIO_BMCINFOR_ORI_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/bio/bmc_bioinfo_original.json')

# Papers from bmc genomics
BIO_BMCGENO_INFO_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/bio/bmc_genomics_info.json')
BIO_BMCGENO_ORI_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/bio/bmc_genomics_original.json')

# Papers from PLOS Computational Biology
BIO_PLOS_COMPBIO_INFO_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/bio/plos_compbio_info.json')
BIO_PLOS_COMPBIO_ORI_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/bio/plos_compbio_original.json')

# Papers from Genome Biology
BIO_GENO_BIO_INFO_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/bio/geno_bio_info.json')
BIO_GENO_BIO_ORI_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/bio/geno_bio_original.json')

# Papers from Nucleic Acids Research
BIO_NUCLEIC_INFO_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/bio/nucleic_info.json')
BIO_NUCLEIC_ORI_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/bio/nucleic_original.json')

# Bioinformatics paper bag-of-words and paper index
BIO_PAPERS_BOW_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/bio/bio_papers_bow.json')
BIO_PAPERS_IDX = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/bio/bio_papers_idx.json')

# Bioinformatics tools paper mapping
BIO_PAPER_TOOL_MAP = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/bio/bio_paper_tool_map.json')
BIO_PAPER_DATASET_MAP = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/bio/bio_paper_dataset_map.json')

# Papers from Journal of Computational Neuroscience
NEURO_JOCN_INFO_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/neuro/jocn_info.json')
NEURO_JOCN_ORI_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/neuro/jocn_original.json')

# Papers from Front. Computational Neuroscience
NEURO_FCN_INFO_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/neuro/fcn_info.json')
NEURO_FCN_ORI_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/neuro/fcn_original.json')

# Papers from Journal of Neuroscience
NEURO_JON_INFO_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/neuro/jon_info.json')
NEURO_JON_ORI_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/neuro/jon_original.json')

# Papers from NEURON
NEURO_NEURON_INFO_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/neuro/neuron_info.json')
NEURO_NEURON_ORI_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/neuro/neuron_original.json')

# Neuroscience paper bag-of-words and paper index
NEURO_PAPERS_BOW_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/neuro/neuro_papers_bow.json')
NEURO_PAPERS_IDX = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/neuro/neuro_papers_idx.json')

# Neuroscience tools paper mapping
NEURO_PAPER_TOOL_MAP = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/neuro/neuro_paper_tool_map.json')
NEURO_PAPER_DATASET_MAP = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/neuro/neuro_paper_dataset_map.json')

# Stop words
STOP_WORDS = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/stopwords.txt')

# Vocabulary
NEURO_VOCAB_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/neuro/neuro_vocab.txt')
BIO_VOCAB_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/bio/bio_vocab.txt')
ALL_VOCAB_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/all_vocab.txt')

# model folder path
MODELS_FOLDER = os.path.join(dirname(os.path.realpath(__file__)), 'output/')