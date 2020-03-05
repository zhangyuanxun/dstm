import argparse
from input_fn import *
from model.plsa import *


def parse_args():
    parser = argparse.ArgumentParser(description='Description the Command Line of DSTP Model')
    parser.add_argument('--data_source', help='Select data source: neuroscience(neuro), bioinformatics(bio), '
                                              '(default = %(default)s)', default='bio', choices=['bio', 'neuro'])
    parser.add_argument('--mode', help='Choose mode: estimate (est), inference (inf), or demonstration (demo) '
                                       '(default = %(default)s)', default='demo', choices=['est', 'inf', 'demo'])
    parser.add_argument('--num_iterations', help='Number of iterations (default = %(default)s)', type=int,
                        default=10, choices=range(0, 2000), metavar='(0, ..., 2000)')
    parser.add_argument('--num_topics', help='Number of topics (default = %(default)s)', type=int,
                        default=100, choices=range(1, 5001), metavar='(1, ..., 5000)')
    parser.add_argument('--seed', help='Set the seeds (default = %(default)s)', type=float,
                        default=7, choices=range(1, 1001), metavar='(1, ..., 1000)')
    parser.add_argument('--verbose', help='show performance debug information (default = %(default)s)', type=int,
                        default=1, choices=[0, 1])
    parser.add_argument('--model_folder', help='Specify the model folder name for running continuously. '
                                               'If running from start it will create a model by default', type=str)
    parser.add_argument('--save', help='choose whether save model or not '
                                       '(default = %(default)s)', default='yes', choices=['no', 'yes'])
    args = parser.parse_args()

    # provide basic check
    if args.mode == 'inf' and args.model_folder is None:
        parser.error('model_folder is required when mode is inf')

    return args


def main():
    print "Program start"

    # get parameters from arguments
    args = parse_args()

    if args.mode == 'demo':
        # get inputs data
        inputs = input_fn(args.mode, args.data_source)

        inputs['docs'] = inputs['docs']
        # set the hyper-parameters
        inputs['mode'] = args.mode
        inputs['num_topics'] = args.num_topics
        inputs['num_iterations'] = args.num_iterations
        inputs['verbose'] = args.verbose
        inputs['model_folder'] = args.model_folder
        inputs['seed'] = args.seed
        inputs['data_source'] = args.data_source
        inputs['num_vocabs'] = len(inputs['vocab'])  # number of vocabularies
        inputs['num_docs'] = len(inputs['docs'])  # number of documents

        print "The model mode is: " + str(inputs['mode'])
        print "The data source is: " + str(inputs['data_source'])
        print "The number of vocabularies: " + str(inputs['num_vocabs'])
        print "The number of documents: " + str(len(inputs['docs']))
        print "The number of topics: " + str(inputs['num_topics'])
        print "The number of iterations: " + str(inputs['num_iterations'])
        sys.stdout.flush()

        # (1) Model initialization
        model = PLSA(inputs)
        model.model_init(inputs)

        # (2) Run plsa algorithm
        model.run()
    elif args.mode == 'est':
        # get inputs data
        inputs = input_fn(args.mode, args.data_source)
        # set the hyper-parameters
        inputs['mode'] = args.mode
        inputs['num_topics'] = args.num_topics
        inputs['num_iterations'] = args.num_iterations
        inputs['verbose'] = args.verbose
        inputs['model_folder'] = args.model_folder
        inputs['seed'] = args.seed
        inputs['data_source'] = args.data_source
        inputs['num_vocabs'] = len(inputs['vocab'])  # number of vocabularies
        inputs['num_docs'] = len(inputs['docs'])  # number of documents

        print "The model mode is: " + str(inputs['mode'])
        print "The data source is: " + str(inputs['data_source'])
        print "The number of vocabularies: " + str(inputs['num_vocabs'])
        print "The number of documents: " + str(len(inputs['docs']))
        print "The number of topics: " + str(inputs['num_topics'])
        print "The number of iterations: " + str(inputs['num_iterations'])
        sys.stdout.flush()

        # step (1) Model initialization
        model = PLSA(inputs)
        model.model_init(inputs)

        # step (2) Run plsa algorithm
        model.run()

        # step (3) Run inference
        # get inputs data
        inputs = input_fn('inf', args.data_source)

        # run model inference
        model.inference(test_docs=inputs['docs'], num_iterations=10)

    print "Program end"


if __name__ == "__main__":
    main()
