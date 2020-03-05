from contants import *
import json
from dataset.bio.bio_tools import *
from dataset.bio.bio_dataset import *
from dataset.neuro.neuro_tools import *
from dataset.neuro.neuro_datasets import *
from dataset.common_tools import *


def input_fn(mode, data_source):
    if data_source == 'bio':
        with open(BIO_VOCAB_PATH, 'r') as fp:
            vocabs = fp.readlines()
        vocabs = [x.strip() for x in vocabs]

        # docs with bag of words
        with open(BIO_PAPERS_BOW_PATH, 'r') as fp:
            docs = json.load(fp)

        # change to integer key
        for d in xrange(len(docs)):
            docs[d] = {int(k): int(v) for k, v in docs[d].items()}

        with open(BIO_PAPER_TOOL_MAP, 'r') as fp:
            doc_tool_map = json.load(fp)

        with open(BIO_PAPER_DATASET_MAP, 'r') as fp:
            doc_dataset_map = json.load(fp)

        # load paper idx
        with open(BIO_PAPERS_IDX, 'r') as fp:
            doc_idx = json.load(fp)

        doc_idx = {int(k): v for k, v in doc_idx.items()}

        tools = [t[0] for t in BIO_TOOLS_MAP]
        datasets = [s[1] for s in BIO_DATASETS_MAP]

        common_tools = []
        for i in range(len(BIO_DATASETS_MAP)):
            if BIO_DATASETS_MAP[i][1] in COMMON_TOOLS_MAP:
                common_tools.append(i)
    else:
        with open(NEURO_VOCAB_PATH, 'r') as fp:
            vocabs = fp.readlines()
        vocabs = [x.strip() for x in vocabs]

        # docs with bag of words
        with open(NEURO_PAPERS_BOW_PATH, 'r') as fp:
            docs = json.load(fp)

        # change to integer key
        for d in xrange(len(docs)):
            docs[d] = {int(k): int(v) for k, v in docs[d].items()}

        with open(NEURO_PAPER_TOOL_MAP, 'r') as fp:
            doc_tool_map = json.load(fp)

        with open(NEURO_PAPER_DATASET_MAP, 'r') as fp:
            doc_dataset_map = json.load(fp)

        # load paper idx
        with open(NEURO_PAPERS_IDX, 'r') as fp:
            doc_idx = json.load(fp)

        doc_idx = {int(k): v for k, v in doc_idx.items()}

        tools = [t[1] for t in NEURO_TOOLS_MAP]
        datasets = [s[1] for s in NEURO_DATASETS_MAP]

        common_tools = []
        for i in range(len(NEURO_TOOLS_MAP)):
            if NEURO_TOOLS_MAP[i][1] in COMMON_TOOLS_MAP:
                common_tools.append(i)

    # Currently, in order to improve efficiency of inference algorithm, remove documents with out tools and dataset
    new_docs = []
    new_doc_tool_map = []
    new_doc_dataset_map = []
    new_doc_idx = {}

    for i in range(len(docs)):
        if data_source == 'bio':
            if len(doc_tool_map[i]) == 0 and len(doc_dataset_map[i]) == 0:
                continue
        else:
            if len(doc_tool_map[i]) == 0:
                continue
        new_docs.append(docs[i])
        new_doc_tool_map.append(doc_tool_map[i])
        new_doc_dataset_map.append(doc_dataset_map[i])
        new_doc_idx[len(new_docs) - 1] = doc_idx[i]

    docs = new_docs
    doc_tool_map = new_doc_tool_map
    doc_dataset_map = new_doc_dataset_map
    doc_idx = new_doc_idx

    # if run algorithm as performance analysis, use 80% for estimation and 20% for inference
    # if run algorithm as results demonstration, use all dataset
    total_docs = len(docs)
    ratio = 0.8

    if mode == "est":
        inputs = {'vocab': vocabs, 'tools': tools, 'datasets': datasets,
                  'docs': docs[:int(total_docs * ratio)], 'docs_idx': doc_idx,
                  'doc_tool_map': doc_tool_map[:int(total_docs * ratio)],
                  'doc_dataset_map': doc_dataset_map[:int(total_docs * ratio)],
                  'common_tool': common_tools}

    elif mode == "inf":
        tools_in_trained = doc_tool_map[:int(total_docs * ratio)]
        dataset_in_trained = doc_dataset_map[:int(total_docs * ratio)]
        trained_tool = []
        trained_dataset = []

        for d in tools_in_trained:
            trained_tool.extend(d)

        for d in dataset_in_trained:
            trained_dataset.extend(d)

        trained_tool = list(set(trained_tool))
        trained_dataset = list(set(trained_dataset))

        inputs = {'vocab': vocabs, 'tools': tools, 'datasets': datasets,
                  'docs': docs[int(total_docs * ratio):], 'docs_idx': doc_idx,
                  'doc_tool_map': doc_tool_map[int(total_docs * ratio):],
                  'doc_dataset_map': doc_dataset_map[int(total_docs * ratio):],
                  'trained_tool': trained_tool, 'trained_dataset': trained_dataset,
                  'common_tool': common_tools}
    else:
        inputs = {'vocab': vocabs, 'tools': tools, 'datasets': datasets, 'docs_idx': doc_idx,
                  'docs': docs, 'doc_tool_map': doc_tool_map, 'doc_dataset_map': doc_dataset_map,
                  'common_tool': common_tools}

    return inputs
