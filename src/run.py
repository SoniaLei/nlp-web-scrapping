from src.experiment import Experiment

# we'll need to pass the arg to main
# hardcoded for now

if __name__ == '__main__':
    print("Starting ml pipeline program.")
    e = Experiment()
    e.read_json('../experiments/exp_test.json')
    e.run()
