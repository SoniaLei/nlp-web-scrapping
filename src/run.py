from mlpipeline import *
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

    print(f"Reading configurations for experiment in path: {args.config_file}.")
    context = Context(exp_name=args.file_name, conf_file=args.config_file)

    print("Setting nlp pipelines from conf file.")
    pipelines = Pipelines(exp_name=context.exp_name,
                          data=context.data,
                          transformers=context.transformers,
                          vectorizers=context.vectorizers,
                          estimators=context.estimators)

    print("Number of Pipelines created: ", len(pipelines))
    print("Pipeline names: ")
    [print("- ", name) for name in pipelines.names]
    print("\n")

    print("Fitting and predicting pipelines.")
    experiments = pipelines.start_runs(safe_run=True)

    # computes and saves each experiment
    experiments.add_experiment_combinations()

    print("End of pipeline. ")


# Remember to run this experiment using
# python run.py -cf experiment_configs/exp005.yml
# as an example
if __name__ == '__main__':
    print("\n")
    print("Starting ml pipeline program.")
    main()
