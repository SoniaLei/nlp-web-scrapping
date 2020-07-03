from mlpipeline import read_config_file, Context, Experiment
import argparse
import sys
import os

def get_parser_args():
    """
    Creates a new argument parser and returns the arguments being passed.
    """
    parser = argparse.ArgumentParser(description='Machine Learning [NLP] Pipeline outputting ml Metrics in MLFlow.')
    parser.add_argument('-cf', '--config', '--configname',
                        help='Experiment configuration file path', type=str, required=True, dest='config_file')
    parser.add_argument('-n', '--name',
                        help='Experiment name used to experiment artifacts', type=str, required=False, dest='file_name')
    args = parser.parse_args(sys.argv[1:])
    return args


def main():
    """
    Orchestrates the whole ml program.
    """
    args = get_parser_args()

    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"{args.config_file} is not found in the root filesystem.")

    print("Starting ml pipeline program.")

    config_context = Context(exp_name=args.file_name, conf_file=args.config_file)
    config = config_context.validate_configuration_parameters()
    experiment = Experiment(config)
    experiment.run(probabilities=True)
    # print(experiment.results.confusion_matrix)
    # print(experiment.results.accuracy_score)
    # experiment.results.plot_confusion_matrix
    # experiment.save_to_mlflow()
    # print(experiment.results.predictions)
    print("SUCCESSFUL!!!!!")


if __name__ == '__main__':
    main()
