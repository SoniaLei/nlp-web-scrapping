from src.context import Context
from datetime import datetime
import pandas as pd
import pytest

def test_context_name():
    # experiment name passed and used
    c = Context('ex_name', conf_file='$PROJECT_PATHS$/src/experiment_template.yml')
    date_format = f'_%Y_%m_%d'
    date = datetime.today().strftime(date_format)
    assert c.exp_name == 'ex_name' + date

    # default name extracted from yml file name
    c = Context(conf_file='$PROJECT_PATHS$/src/experiment_template.yml')
    assert c.exp_name == 'experiment_template' + date

    # exp_name not a str type
    with pytest.raises(ValueError) as ex:
        Context(exp_name=100, conf_file='$PROJECT_PATHS$/src/experiment_template.yml')
        assert ex == "Experiment name must be string value."

    # exp_name not using _ scores
    with pytest.raises(ValueError) as ex:
        Context(exp_name='experiment-test', conf_file='$PROJECT_PATHS$/src/experiment_template.yml')
        assert ex == "File name must contain _ for readability."


def test_context_conf_file():
    c = Context(exp_name=None, conf_file='folder1/sub_folder/path_to_the_context.yml')
    assert c.conf_file == 'folder1/sub_folder/path_to_the_context.yml'

    # passing a non yml file
    with pytest.raises(TypeError) as ex:
        Context(exp_name=None, conf_file='folder1/sub_folder/path_to_the_context.csv')
        assert ex == 'Configuration file must be a yml file'

def test_read_conf_file(tmpdir):
    file = """
            data_files:
                train:  'tweets-train.csv'
                test: 'tweets-test.csv'
                features: 'fe1'
                target: 'sent'
            transformers:
                - transformer1:
                  param1: 'value_trans_1_param_1'
                  param2: 'value_trans_1_param_2'
                - transformer2:
                  param1: 'value_trans_2_param_1'
                  param2: 'value_trans_2_param_2'
            """
    p = tmpdir.mkdir("sub").join("conf_file.yml")
    p.write(file)
    parsed_file = Context.read_config_file(p)
    assert isinstance(parsed_file, dict)
    assert list(parsed_file.keys()) == ['data_files', 'transformers']
    assert list(parsed_file['data_files'].keys()) == ['train', 'test', 'features', 'target']
    assert parsed_file['data_files']['train'] == 'tweets-train.csv'
    assert parsed_file['data_files']['test'] == 'tweets-test.csv'
    assert parsed_file['data_files']['features'] == 'fe1'
    assert parsed_file['transformers'] == [{'transformer1': None,
                                            'param1': 'value_trans_1_param_1',
                                            'param2': 'value_trans_1_param_2',
                                           },
                                           {'transformer2': None,
                                           'param1': 'value_trans_2_param_1',
                                           'param2': 'value_trans_2_param_2',
                                           }]


class TestSetConfigurationParams:

    def test_config_params_missing_keys(self, tmpdir):
        file = """
                data_files:
                    train:  'tweets-train.csv'
                    test: 'tweets-test.csv'
                    features: 'fe1'
                    target: 'sent'
                transformers:
                    - transformer1:
                      param1: 'value_trans_1_param_1'
                      param2: 'value_trans_1_param_2'
                    - transformer2:
                      param1: 'value_trans_2_param_1'
                      param2: 'value_trans_2_param_2'
                """
        p = tmpdir.mkdir("test").join("conf_file.yml")
        p.write(file)
        c = Context('exp_name_param', conf_file=str(p))

        with pytest.raises(ValueError) as ex:
            c.set_configuration_parameters()
            assert ex == "Missing {'estimators', 'vectorizer'} compulsory keys in configuration file."

    def test_data_files_keys(self, tmpdir):
        file = """
                data_files:
                    train:  'tweets-train.csv'
                    test: 'tweets-test.csv'
                transformers:
                estimators:
                vectorizer:
                """
        p = tmpdir.mkdir("test").join("conf_file.yml")
        p.write(file)
        c = Context('exp_name_param', conf_file=str(p))
        with pytest.raises(ValueError) as ex:
            c.set_configuration_parameters()
            assert ex == "Missing {'target', 'features'} keys under data_files from context file."

    def test_data_files_keys_train_test(self, tmpdir):
        file = """
                data_files:
                    train:  ''
                    test: 
                    target:
                    features: 
                transformers:
                estimators:
                vectorizer:
                """
        p = tmpdir.mkdir("test").join("conf_file.yml")
        p.write(file)
        c = Context('exp_name_param', conf_file=str(p))
        with pytest.raises(ValueError) as ex:
            c.set_configuration_parameters()
        assert "train cannot be empty." == str(ex.value)

    def test_data_files_keys_train_csv_format(self, tmpdir):
        file = """
                data_files:
                    train:  'tweets-train.notcsv'
                    test: 
                    target:
                    features: 
                transformers:
                estimators:
                vectorizer:
                """
        p = tmpdir.mkdir("test").join("conf_file.yml")
        p.write(file)
        c = Context('exp_name_param', conf_file=str(p))
        with pytest.raises(TypeError) as ex:
            c.set_configuration_parameters()
        assert "train data must be a csv file with .csv extension." == str(ex.value)


    def test_data_files_keys_train_csv_format_file_no_exist(self, tmpdir):
        file = """
                data_files:
                    train:  'tweets-train.csv'
                    test: 
                    target:
                    features: 
                transformers:
                estimators:
                vectorizer:
                """
        p = tmpdir.mkdir("test").join("conf_file.yml")
        p.write(file)
        c = Context('exp_name_param', conf_file=str(p))
        with pytest.raises(FileNotFoundError) as ex:
            c.set_configuration_parameters()

        assert "[Errno 2] File tweets-train.csv does not exist: 'tweets-train.csv'" == str(ex.value)


    # # TODO create a csv temp file
    # def test_data_files_keys_train_csv_format_file_does_exist(self, tmpdir):
    #     df = tempfile.NamedTemporaryFile('w')
    #     df.name = 'test.csv'
    #     df.write('{1:[1,2,3]}')
    #     file = f"""
    #             data_files:
    #                 train:  {df.name}
    #                 test:
    #                 target:
    #                 features:
    #             transformers:
    #             estimators:
    #             vectorizer:
    #             """
    #     p = tmpdir.mkdir("test").join("conf_file.yml")
    #     p.write(file)
    #     c = Context('exp_name_param', conf_file=str(p))
    #     c.set_configuration_parameters()


    def test_data_files_keys_target_empty(self, tmpdir):
        file = """
                data_files:
                    train: '$PROJECT_PATHS$/../data/raw/tweets-train.csv'
                    test: '$PROJECT_PATHS$/../data/raw/tweets-train.csv'
                    target:
                    features:
                transformers:
                estimators:
                vectorizer:
                """
        p = tmpdir.mkdir("test").join("conf_file.yml")
        p.write(file)
        c = Context('exp_name_param', conf_file=str(p))
        with pytest.raises(ValueError) as ex:
            c.set_configuration_parameters()
        assert 'target cannot be empty.' == str(ex.value)

    def test_data_files_keys_features_empty(self, tmpdir):
        file = """
                data_files:
                    train: '$PROJECT_PATHS$/../data/raw/tweets-train.csv'
                    test: '$PROJECT_PATHS$/../data/raw/tweets-train.csv'
                    target: 'target_name'
                    features: 
                transformers:
                estimators:
                vectorizer:
                """
        p = tmpdir.mkdir("test").join("conf_file.yml")
        p.write(file)
        c = Context('exp_name_param', conf_file=str(p))
        with pytest.raises(ValueError) as ex:
            c.set_configuration_parameters()
        assert 'features cannot be empty.' == str(ex.value)
        assert isinstance(c.target, str)
        assert isinstance(c.train, pd.DataFrame)
        assert isinstance(c.test, pd.DataFrame)


    def test_data_files_key_transformers(self, tmpdir):
        file = """
                   data_files:
                       train: '$PROJECT_PATHS$/../data/raw/tweets-train.csv'
                       test: '$PROJECT_PATHS$/../data/raw/tweets-train.csv'
                       target: 'target_name'
                       features: 'features_name'
                   transformers:
                   estimators:
                   vectorizer:
                   """
        p = tmpdir.mkdir("test").join("conf_file.yml")
        p.write(file)
        c = Context('exp_name_param', conf_file=str(p))
        with pytest.raises(TypeError) as ex:
            c.set_configuration_parameters()
        assert " from config file expected to be a list got " in str(ex.value)

    def test_data_files_key_transformers(self, tmpdir):
        file = """
                   data_files:
                       train: '$PROJECT_PATHS$/../data/raw/tweets-train.csv'
                       test: '$PROJECT_PATHS$/../data/raw/tweets-train.csv'
                       target: 'target_name'
                       features: 'features_name'
                   transformers:
                     - StopWordsRemoval:
                   estimators:
                     - s
                   vectorizer:
                     - CountVectorizer:
                   """
        p = tmpdir.mkdir("test").join("conf_file.yml")
        p.write(file)
        c = Context('exp_name_param', conf_file=str(p))
        with pytest.raises(TypeError) as ex:
            c.set_configuration_parameters()
        assert "Parameter object: s must be of type dict " \
               "found <class 'str'> instead." == str(ex.value)


    def test_data_files_key_error_factory_object(self, tmpdir):
        file = """
                   data_files:
                       train: '$PROJECT_PATHS$/../data/raw/tweets-train.csv'
                       test: '$PROJECT_PATHS$/../data/raw/tweets-train.csv'
                       target: 'target_name'
                       features: 'features_name'
                   transformers:
                     - StopWordsRemoval:
                   estimators:
                     - s:
                   vectorizer:
                     - CountVectorizer:
                   """
        p = tmpdir.mkdir("test").join("conf_file.yml")
        p.write(file)
        c = Context('exp_name_param', conf_file=str(p))
        with pytest.raises(KeyError) as ex:
            c.set_configuration_parameters()
