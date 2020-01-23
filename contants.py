import os
from os.path import dirname

# Papers from bmc bioinformatics
BIO_BMCINFOR_INFO_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'bio/bmc_bioinfo_info.json')
BIO_BMCINFOR_ORI_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'bio/bmc_bioinfo_original.json')

# Papers from bmc genomics
BIO_BMCGENO_INFO_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'bio/bmc_genomics_info.json')
BIO_BMCGENO_ORI_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'bio/bmc_genomics_original.json')

# Papers from PLOS Computational Biology
BIO_PLOS_COMPBIO_INFO_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'bio/plos_compbio_info.json')
BIO_PLOS_COMPBIO_ORI_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'bio/plos_compbio_original.json')

# Papers from Genome Biology
BIO_GENO_BIO_INFO_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'bio/geno_bio_info.json')
BIO_GENO_BIO_ORI_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'bio/geno_bio_original.json')

# Papers from Nucleic Acids Research
BIO_NUCLEIC_INFO_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'bio/nucleic_info.json')
BIO_NUCLEIC_ORI_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'bio/nucleic_original.json')

# Bioinformatics paper bag-of-words and paper index
BIO_PAPERS_BOW_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'bio/bio_papers_bow.json')
BIO_PAPERS_IDX = os.path.join(dirname(os.path.realpath(__file__)), 'bio/bio_papers_idx.json')

# Bioinformatics tools paper mapping
BIO_PAPER_TOOL_MAP = os.path.join(dirname(os.path.realpath(__file__)), 'bio/bio_paper_tool_map.json')
BIO_PAPER_DATASET_MAP = os.path.join(dirname(os.path.realpath(__file__)), 'bio/bio_paper_dataset_map.json')

# Papers from Journal of Computational Neuroscience
NEURO_JOCN_INFO_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'neuro/jocn_info.json')
NEURO_JOCN_ORI_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'neuro/jocn_original.json')

# Papers from Front. Computational Neuroscience
NEURO_FCN_INFO_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'neuro/fcn_info.json')
NEURO_FCN_ORI_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'neuro/fcn_original.json')

# Papers from Journal of Neuroscience
NEURO_JON_INFO_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'neuro/jon_info.json')
NEURO_JON_ORI_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'neuro/jon_original.json')

# Papers from NEURON
NEURO_NEURON_INFO_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'neuro/neuron_info.json')
NEURO_NEURON_ORI_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'neuro/neuron_original.json')

# Neuroscience paper bag-of-words and paper index
NEURO_PAPERS_BOW_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'neuro/neuro_papers_bow.json')
NEURO_PAPERS_IDX = os.path.join(dirname(os.path.realpath(__file__)), 'neuro/neuro_papers_idx.json')

# Neuroscience tools paper mapping
NEURO_PAPER_TOOL_MAP = os.path.join(dirname(os.path.realpath(__file__)), 'neuro/neuro_paper_tool_map.json')
NEURO_PAPER_DATASET_MAP = os.path.join(dirname(os.path.realpath(__file__)), 'neuro/neuro_paper_dataset_map.json')

# Stop words
STOP_WORDS = os.path.join(dirname(os.path.realpath(__file__)), 'stopwords.txt')

# Vocabulary
NEURO_VOCAB_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'neuro/neuro_vocab.txt')
BIO_VOCAB_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'bio/bio_vocab.txt')
ALL_VOCAB_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'all_vocab.txt')

# information retrieval
IR_QUERY_DATASET = os.path.join(dirname(os.path.realpath(__file__)), 'ir_query.json')
IR_RETRIEVAL_DATASET = os.path.join(dirname(os.path.realpath(__file__)), 'ir_retrieval.json')

IR_QUERY_DOCS = os.path.join(dirname(os.path.realpath(__file__)), 'ir_query_docs.dat')
IR_RETRIEVAL_DOCS = os.path.join(dirname(os.path.realpath(__file__)), 'ir_retrieval_docs.dat')
SIMILARITY_MATRIX = os.path.join(dirname(os.path.realpath(__file__)), 'similarity_matrix.npy')
DSTP_RETRIEVAL_SCORE = os.path.join(dirname(os.path.realpath(__file__)), 'ir_dstp_score.npy')
DSTP_RETRIEVAL_SCORE_01 = os.path.join(dirname(os.path.realpath(__file__)), 'ir_dstp_score_01.npy')  # 1%
DSTP_RETRIEVAL_SCORE_05 = os.path.join(dirname(os.path.realpath(__file__)), 'ir_dstp_score_05.npy')  # 5%
DSTP_RETRIEVAL_SCORE_1 = os.path.join(dirname(os.path.realpath(__file__)), 'ir_dstp_score_1.npy')  # 10%
DSTP_RETRIEVAL_SCORE_2 = os.path.join(dirname(os.path.realpath(__file__)), 'ir_dstp_score_2.npy')  # 20%
DSTP_RETRIEVAL_SCORE_5 = os.path.join(dirname(os.path.realpath(__file__)), 'ir_dstp_score_5.npy')  # 50%
LDA_RETRIEVAL_SCORE = os.path.join(dirname(os.path.realpath(__file__)), 'ir_lda_score.npy')
LDA_RETRIEVAL_SCORE_01 = os.path.join(dirname(os.path.realpath(__file__)), 'ir_lda_score_01.npy')  # 1%
LDA_RETRIEVAL_SCORE_05 = os.path.join(dirname(os.path.realpath(__file__)), 'ir_lda_score_05.npy')  # 5%
LDA_RETRIEVAL_SCORE_1 = os.path.join(dirname(os.path.realpath(__file__)), 'ir_lda_score_1.npy')  # 10%
LDA_RETRIEVAL_SCORE_2 = os.path.join(dirname(os.path.realpath(__file__)), 'ir_lda_score_2.npy')  # 20%
LDA_RETRIEVAL_SCORE_5 = os.path.join(dirname(os.path.realpath(__file__)), 'ir_lda_score_5.npy')  # 50%
PLSA_RETRIEVAL_SCORE = os.path.join(dirname(os.path.realpath(__file__)), 'plsa_lda_score.npy')
PLSA_RETRIEVAL_SCORE_01 = os.path.join(dirname(os.path.realpath(__file__)), 'plsa_lda_score_01.npy')  # 5%
PLSA_RETRIEVAL_SCORE_05 = os.path.join(dirname(os.path.realpath(__file__)), 'plsa_lda_score_05.npy')  # 5%
PLSA_RETRIEVAL_SCORE_1 = os.path.join(dirname(os.path.realpath(__file__)), 'plsa_lda_score_1.npy')  # 10%
PLSA_RETRIEVAL_SCORE_2 = os.path.join(dirname(os.path.realpath(__file__)), 'plsa_lda_score_2.npy')  # 20%
PLSA_RETRIEVAL_SCORE_5 = os.path.join(dirname(os.path.realpath(__file__)), 'plsa_lda_score_5.npy')  # 50%

# model folder path
MODELS_FOLDER = os.path.join(dirname(os.path.realpath(__file__)), '../models/')