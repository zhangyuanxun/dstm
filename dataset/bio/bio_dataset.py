"""
Bioinformatics datasets list,
format in ['dataset name', 'replacements', 'Uppercase/Lowercase sensitive', 'dataset url']
"""
BIO_DATASETS_MAP = [
    ['Whole Exome Sequencing',                  'wes',             False,   ''],
    ['Whole Genome Shotgun',                    'wgs',             False,   ''],
    ['The Cancer Genome Atlas',                 'tcga',            False,   ''],
    ['Cancer Cell Line Encyclopedia',           'ccle',            False,   ''],
    ['Genomics of Drug Sensitivity',            'gdsc',            False,   ''],
    ['Messenger RNA',                           'mrna',            False,   ''],
    ['Piwi-interacting RNA',                    'pirna',           False,   ''],
    ['Toxic Small RNA',                         'tsrna',           False,   ''],
    ['RNA sequencing',                          'rnaseq',          False,   ''],
    ['small nuclear RNA',                       'snrna',           False,   ''],
    ['microRNA',                                'mirna',           False,   ''],
    ['Small interfering RNA',                   'sirna',           False,   ''],
    ['Long non-coding RNAs',                    'lncrnas',         False,   ''],
    ['Small nucleolar RNAs',                    'snornas',         False,   ''],
    ['DNA affinity purification sequencing',    'dapseq',          False,   ''],
    ['ChIP-sequencing',                         'chipseq',         False,   ''],
    ['MiscRna',                                 'miscrna',         False,   ''],
    ['ATAC-seq',                                'atacseq',         False,   ''],
    ['FAIRE-seq',                               'faireseq',        False,   ''],
    ['MNase-seq',                               'mnaseseq',        False,   ''],
    ['DNase-seq',                               'dnaseseq',        False,   ''],
    ['ChIP-exo',                                'chipexo',         False,   ''],
    ['Bisulfite sequencing',                    'bisulfiteseq',    False,   ''],
    ['Methylation Sequencing',                  'methylseq',       False,   ''],
    ['metabolomics',                            'metabolomics',    False,   ''],
    ['ENCODE',                                  'encodedata',      True,    'https://www.encodeproject.org/'],
    ['Hi-C',                                    'hic',             True,    'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3149993/'],
    ['protein-ligand binding',                  'plb',             False,   'https://github.com/yfCuiFaith/DeepCSeqSite'],
    ['Mitochondrial DNA',                       'mtdna',           False,   ''],
    ['ChIP-qPCR',                               'chipqpcr',        False,   ''],
    ['HMEC',                                    'hmec',            False,   ''],
    ['Short hairpin RNA',                       'shrna',           False,   ''],
]


def check_validity():
    d = {}
    for i in BIO_DATASETS_MAP:
        # check length
        if len(i) != 4:
            print "Info not compeleted: " + str(i)

        # check repeated
        if i[1] in d:
            print "Repeated tool %s found!" % str(i[1])
        else:
            d[i[1]] = 1

        # check lower case
        if i[1].lower() != i[1]:
            print "%s is not lowercase" % str(i[1])


if __name__ == "__main__":
    check_validity()