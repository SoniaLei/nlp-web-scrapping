from mlpipeline.experiment import Experiment, Data
from mlpipeline.context import Context
from mlpipeline.pipeline import Pipeline
import pandas as pd

#from mlflow import MLFlow

if __name__ == '__main__':

    print("Starting ml pipeline program.")

    # AUTOMATIC APPROACH #######################################
    config = Context(conf_file='./experiment_configs/experiment002.yml')
    config = config.set_configuration_parameters()
    experiment = Experiment(config)
    experiment.run()
    print(experiment.results.confusion_matrix)
    print(experiment.results.accuracy_score)
    #print(experiment.results.predictions)
    
# MLFlow??????
    mlflow = MLFlow(
        experiment_name = "Whatever is wanted for this run of the experiment",
        tracking_uri = "Could probably be in .yml file"
    )
    
    mlflow.Logging(
        params_dictionary="Dictionary - need to reconfigure experiment results for this output",
        metrics_dictionary="Dictionary - need to reconfigure experiment results for this output",
        artifact_filepaths="List - Elia was changing to a temporary artifact system"
    )    
    
    print("SUCCESSFUL!!!!!")
    # END AUTOMATIC APPROACH ####################################

    # context params
    """print(config.exp_name)
    print(config.conf_file)
    print(config.train)
    print(config.test)
    print(config.target)
    print(config.features)
    print(config.vectorizer)
    print(config.estimators)
    print(config.transformers)
    print(" ")


    # experiment params
    print(experiment.name)  # returns experiment.config.exp_name
    print(experiment.config)  # Context object with all params from yaml
    print(experiment.data)  # Data object from conf.params
    print(experiment.data.train_X)
    print(experiment.data.train_Y)
    print(experiment.data.test_Y)
    print(experiment.data.test_Y)
    print(experiment.pipeline)  # Pipeline Object from yaml
    print(experiment.pipeline.transformers)
    print(experiment.pipeline.vectorizers)
    print(experiment.pipeline.estimators)
    print(" ")

    # Experiment.results params
    print(experiment.results.accuracy_score)
    print(experiment.results.classification_report)
    print(experiment.results.confusion_matrix)
    experiment.results.dump_results_csv()
    experiment.save_to_mlflow()
    print(" ")



    # NOT AUTOMATIC APROACH  #######################################
    # MODULARITY USING CLASSES AS WE PLEASE

    # Instantiating Data without context using cleaned data since Transformers not implemented
    df_train = pd.read_csv("../data/processed/cleanedDataV1.csv")
    df_test = pd.read_csv("../data/processed/cleanedDataV1.csv")
    target = 'sentiment'
    features = 'text'
    d = Data(train=df_train,
             test=df_test,
             target=target,
             features=features)
    print(d)

    # Instantiating Pipeline without context
    class Test:
        def fit(self):
            pass
        def transform(self):
            pass
    class VectTest:
        def fit(self):
            pass
        def transform(self):
            pass
    class EstimatorTest:
        def fit(self):
            pass

    t = [('t1', Test()), ('t1', Test())]
    v = [('vec', VectTest()), ('vec', VectTest())]
    e = [('est', EstimatorTest()), ('est', EstimatorTest())]
    p = Pipeline(t, v, e).init()
    print(p)

    print(p._pipeline) # SKLEARN PIPELINE

    # Instanciating Experiment:
    experiment = Experiment(data=d, pipeline=p)
    print(experiment)
    print(experiment.pipeline)
    print(type(experiment.pipeline))  # Pipeline class
    experiment.pipeline.init()
    print(type(experiment.pipeline._pipeline))  # SKLEARN PIPELINE"""

