from mlpipeline.context import Context
from mlpipeline.pipeline import Pipelines
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
    Starts the whole ml nlp programme.
    """
    args = get_parser_args()

    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"{args.config_file} is not found in the root filesystem.")

    config_context = Context(exp_name=args.file_name, conf_file=args.config_file)




if __name__ == '__main__':
    print("Starting ml pipeline program.")
    #main() # Main() reads arguments being passed to the script.

    # Testing purposes:
    # Running from __main__ and not passing arg for now.
    context = Context('', conf_file='experiment_configs/exp005.yml')
    print(context.exp_name)

    pipelines = Pipelines(context.exp_name,
                          context.data,
                          context.transformers,
                          context.vectorizers,
                          context.estimators)
    pipelines.start_runs(safe_run=True)
    experiments = pipelines.collect_experiments()
    exp_combinations = experiments.compute_exp_combinations()
    experiments.save_exp_combinations(exp_combinations)

