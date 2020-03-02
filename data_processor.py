"""
Dataset processing, such as removing punctuation, removing stop words,
"""
import json
import argparse
from collections import Counter

from text_utils import *
from tqdm import tqdm
from dataset.bio.bio_tools import *
from dataset.bio.bio_dataset import *
from dataset.neuro.neuro_tools import *
from dataset.neuro.neuro_datasets import *

parser = argparse.ArgumentParser('Description the Command Line of Dataset processing')
parser.add_argument('--domain', default='bio', choices=['bio', 'neuro', 'all'],
                    help="Provide domain name for dataset processing (bio, neuro)")
parser.add_argument('--operation', default='dataset', choices=['vocabulary', 'bag-of-word', 'tool', 'dataset'],
                    help="Provide operation name for processing (bag-of-word, vocabulary, tool, dataset)")


RATIO_OF_PAPER = 0.13
PAPER_LESS_THAN = 10


def generate_bag_of_word(inputs):
    print "Generating Bag of Words:"
    papers = {}
    papers_bow = []
    papers_idx = {}
    journals = inputs['journals']

    for journal in journals:
        with open(journal, 'r') as fp:
            papers.update(json.load(fp))

    # get relevant vocabulary path
    if inputs['domain'] == 'bio':
        vocab_path = BIO_VOCAB_PATH
        paper_bow_path = BIO_PAPERS_BOW_PATH
        paper_idx_path = BIO_PAPERS_IDX
    elif inputs['domain'] == 'neuro':
        vocab_path = NEURO_VOCAB_PATH
        paper_bow_path = NEURO_PAPERS_BOW_PATH
        paper_idx_path = NEURO_PAPERS_IDX
    else:
        vocab_path = ALL_VOCAB_PATH

    with open(vocab_path) as fp:
        vocabs = fp.readlines()
    vocabs = [x.strip() for x in vocabs]

    # construct vocabulary as dictionary
    dict_vocab = {key: idx for idx, key in enumerate(vocabs)}

    # (1) Transform specials words
    tool_map = inputs['tool_map']
    dataset_map = inputs['dataset_map']

    for k in papers.keys():
        text = papers[k]

        # transform tool
        for t in tool_map:
            if t[2]:
                p = re.compile(t[0])
            else:
                p = re.compile(t[0], re.IGNORECASE)

            text = p.sub(t[1], text)

        # transform dataset
        for d in dataset_map:
            if d[2]:
                p = re.compile(d[0])
            else:
                p = re.compile(d[0], re.IGNORECASE)

            text = p.sub(d[1], text)

        papers[k] = text[:int(RATIO_OF_PAPER * len(text))]

    # (2) Generate bag-of-words, and paper index
    with tqdm(total=len(papers), unit="docs", desc="Bag-of-Words") as p_bar:
        idx = 0
        for k in papers.keys():
            tokens = papers[k].split()
            words_count = dict(Counter(tokens))

            d = {}
            for w in words_count.keys():
                if w in dict_vocab:
                    d[dict_vocab[w]] = words_count[w]

            papers_bow.append(d)
            papers_idx[idx] = k
            idx += 1
            p_bar.update(1)

    # write bow to json file
    with open(paper_bow_path, 'w') as fp:
        json.dump(papers_bow, fp)

    # write paper idx to json file
    with open(paper_idx_path, 'w') as fp:
        json.dump(papers_idx, fp)


def generate_vocabulary(inputs):
    print "Generating vocabulary:"
    vocabulary = []
    papers = {}
    journals = inputs['journals']

    for journal in journals:
        with open(journal, 'r') as fp:
            papers.update(json.load(fp))
        fp.close()

    # (1) Add tools, datasets names into vocabulary
    tool_map = inputs['tool_map']
    dataset_map = inputs['dataset_map']

    for t in tool_map:
        vocabulary.append(t[1])

    for d in dataset_map:
        vocabulary.append(d[1])

    # (2) Text pro-processing: remove url, punctuation, stop words
    words = []                        # store all the words for removing all less frequent words
    with tqdm(total=len(papers), unit="docs", desc="Pre-processing") as p_bar:
        for k in papers.keys():
            text = papers[k]
            # extract paper
            text = text[:int(RATIO_OF_PAPER * len(text))]
            text = remove_url(text)
            text = remove_underline(text)
            text = remove_non_ascii(text)
            text = remove_punctuation(text)
            text = remove_digits(text)
            text = remove_extra_space(text)
            text = remove_extra_space(text)
            text = to_lowercase(text)

            # change text to words tokens
            tokens = text.split()
            tokens = remove_stopwords(tokens)
            words += list(set(tokens))

            # update text with tokens
            papers[k] = tokens
            p_bar.update(1)

    # (3) Generate less frequent words
    less_freq = []
    words_count = dict(Counter(words))
    for k in words_count.keys():
        if words_count[k] <= PAPER_LESS_THAN:              # words less than 10 papers
            less_freq.append(k)

    less_freq.sort(key=len)

    # (4) Remove less frequent words
    with tqdm(total=len(papers), unit="docs", desc="Remove less frequent") as p_bar:
        for k in papers.keys():
            tokens = papers[k]
            tokens = remove_less_frequent_words(tokens, less_freq)
            vocabulary.extend(tokens)
            p_bar.update(1)

    # (5) generate vocabulary
    vocabulary = list(set(vocabulary))
    vocabulary.sort(key=len)

    with open(inputs['vocabulary'], 'w') as fp:
        fp.write("\n".join(vocabulary))


def generate_paper_tool_map(inputs):
    print "Generating paper tool map:"

    tool_map = inputs['tool_map']
    tools = []
    for t in tool_map:
        tools.append(t[1])

    # get journals
    papers = {}
    for journal in inputs['journals']:
        with open(journal, 'r') as fp:
            papers.update(json.load(fp))

    # transform tool name
    for k in papers.keys():
        text = papers[k]

        # transform tool
        for t in tool_map:
            if t[2]:                # true
                p = re.compile(t[0])
            else:
                p = re.compile(t[0], re.IGNORECASE)
            text = p.sub(t[1], text)

        papers[k] = text

    # get paper idx
    with open(inputs['paper_idx'], 'r') as fp:
        paper_idx = json.load(fp)

    # reverse key value
    paper_idx = {v: int(k) for k, v in paper_idx.iteritems()}
    paper_tool_map = [[]] * len(paper_idx)

    with tqdm(total=len(papers), unit="docs", desc="Generate paper tool map") as p_bar:
        for k in papers.keys():
            text_tokens = papers[k].split()

            tool_list = []
            for t in tools:
                if t in text_tokens:
                    tool_list.append(tools.index(t))

            p_idx = paper_idx[k]
            paper_tool_map[p_idx] = tool_list
            p_bar.update(1)

    # write to json file
    with open(inputs['paper_tool_map'], 'w') as fp:
        json.dump(paper_tool_map, fp)


def generate_paper_datasets_map(inputs):
    print "Generating paper dataset map"
    dataset_map = inputs['dataset_map']
    datasets = []
    for d in dataset_map:
        datasets.append(d[1])

    # get journals
    papers = {}
    for journal in inputs['journals']:
        with open(journal, 'r') as fp:
            papers.update(json.load(fp))

    # transform dataset name
    for k in papers.keys():
        text = papers[k]

        # transform dataset
        for d in dataset_map:
            if d[2]:                   # true
                p = re.compile(d[0])
            else:
                p = re.compile(d[0], re.IGNORECASE)
            text = p.sub(d[1], text)

        papers[k] = text

    # get paper idx
    with open(inputs['paper_idx'], 'r') as fp:
        paper_idx = json.load(fp)

    # reverse key value
    paper_idx = {v: int(k) for k, v in paper_idx.iteritems()}
    paper_dataset_map = [[]] * len(paper_idx)

    with tqdm(total=len(papers), unit="docs", desc="Generate paper dataset map") as p_bar:
        for k in papers.keys():
            text_tokens = papers[k].split()

            dataset_list = []
            for d in datasets:
                if d in text_tokens:
                    dataset_list.append(datasets.index(d))

            p_idx = paper_idx[k]
            paper_dataset_map[p_idx] = dataset_list
            p_bar.update(1)

    # write to json file
    with open(inputs['paper_dataset_map'], 'w') as fp:
        json.dump(paper_dataset_map, fp)


if __name__ == "__main__":
    args = parser.parse_args()
    inputs = {}

    if args.domain == 'bio':
        inputs['domain'] = 'bio'
        inputs['journals_info'] = [BIO_GENO_BIO_INFO_PATH, BIO_BMCINFOR_INFO_PATH,
                                   BIO_PLOS_COMPBIO_INFO_PATH, BIO_GENO_BIO_INFO_PATH, BIO_NUCLEIC_INFO_PATH]
        inputs['journals'] = [BIO_GENO_BIO_ORI_PATH, BIO_BMCINFOR_ORI_PATH,
                              BIO_PLOS_COMPBIO_ORI_PATH, BIO_GENO_BIO_ORI_PATH, BIO_NUCLEIC_ORI_PATH]
        inputs['tool_map'] = BIO_TOOLS_MAP
        inputs['dataset_map'] = BIO_DATASETS_MAP
        inputs['vocabulary'] = BIO_VOCAB_PATH
        inputs['paper_idx'] = BIO_PAPERS_IDX
        inputs['paper_tool_map'] = BIO_PAPER_TOOL_MAP
        inputs['paper_dataset_map'] = BIO_PAPER_DATASET_MAP
    elif args.domain == 'neuro':
        inputs['domain'] = 'neuro'
        inputs['journals_info'] = [NEURO_FCN_INFO_PATH, NEURO_JOCN_INFO_PATH, NEURO_NEURON_INFO_PATH,
                                   NEURO_JON_INFO_PATH]
        inputs['journals'] = [NEURO_FCN_ORI_PATH, NEURO_JOCN_ORI_PATH,
                              NEURO_NEURON_ORI_PATH, NEURO_JON_ORI_PATH]
        inputs['tool_map'] = NEURO_TOOLS_MAP
        inputs['dataset_map'] = NEURO_DATASETS_MAP
        inputs['vocabulary'] = NEURO_VOCAB_PATH
        inputs['paper_idx'] = NEURO_PAPERS_IDX
        inputs['paper_tool_map'] = NEURO_PAPER_TOOL_MAP
        inputs['paper_dataset_map'] = NEURO_PAPER_DATASET_MAP
    else:
        inputs['domain'] = 'all'
        inputs['journals_info'] = [BIO_GENO_BIO_INFO_PATH, BIO_BMCINFOR_INFO_PATH,
                                   BIO_PLOS_COMPBIO_INFO_PATH, BIO_GENO_BIO_INFO_PATH, BIO_NUCLEIC_INFO_PATH,
                                   NEURO_FCN_INFO_PATH, NEURO_JOCN_INFO_PATH, NEURO_NEURON_INFO_PATH,
                                   NEURO_JON_INFO_PATH]
        inputs['journals'] = [BIO_GENO_BIO_ORI_PATH, BIO_BMCINFOR_ORI_PATH, BIO_PLOS_COMPBIO_ORI_PATH,
                              BIO_GENO_BIO_ORI_PATH, BIO_NUCLEIC_ORI_PATH, NEURO_FCN_ORI_PATH, NEURO_JOCN_ORI_PATH,
                              NEURO_NEURON_ORI_PATH, NEURO_JON_ORI_PATH]
        inputs['tool_map'] = BIO_TOOLS_MAP + NEURO_TOOLS_MAP
        inputs['dataset_map'] = BIO_DATASETS_MAP + NEURO_DATASETS_MAP
        inputs['vocabulary'] = ALL_VOCAB_PATH

    if args.operation == 'bag-of-word':
        generate_bag_of_word(inputs)
    elif args.operation == 'vocabulary':
        generate_vocabulary(inputs)
    elif args.operation == 'tool':
        generate_paper_tool_map(inputs)
    elif args.operation == 'dataset':
        generate_paper_datasets_map(inputs)
