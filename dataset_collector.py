"""
Collector dataset (publications)
"""

import argparse
from collector.bio import bmc_bio_collector, bmc_genomics_collector, plos_compbio_collector, \
    geno_bio_collector, nucleic_acids_collector
from collector.neuro import fcn_collector, jocn_collector, jon_collector, neuron_collector

parser = argparse.ArgumentParser()
parser.add_argument('--domain', default='bio', choices=['bio', 'neuro'],
                    help="Provide domain name of dataset for collection (bio, neuro)")

if __name__ == "__main__":
    args = parser.parse_args()

    if args.domain == 'bio':
        bmc_bio_collector.collector()
        bmc_genomics_collector.collector()
        plos_compbio_collector.collector()
        geno_bio_collector.collector()
        nucleic_acids_collector.collector()
    else:
        fcn_collector.collector()
        jon_collector.collector()
        jocn_collector.collect()
        neuron_collector.collector()
