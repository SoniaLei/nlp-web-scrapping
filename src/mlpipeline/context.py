"""Context module to Validate conf.yml file and store context objects"""
from datetime import datetime
import yaml
import pandas as pd
from .factory import ObjectFactory


class Context:
    """
    Context class stores hardcoded keys present in conf.yaml,
    reads, validate and parse conf parameters found in conf yml,
    and sets those as attributes of the Context class.
    """

    split = '_'
    # names used in configuration file as keys
    data_files = 'data_files'
    transformers = 'transformers'
    vectorizer = 'vectorizer'
    estimators = 'estimators'
    conf_keys = {data_files, transformers, vectorizer, estimators}
    # names/keys used under data_files key
    train = 'train'
    test = 'test'
    target = 'target'
    features = 'features'
    data_key_files = {train, test}

    def __init__(self, exp_name=None, *, conf_file):
        """Sets default exp name if None is passed,
        and stores conf filename.
        """
        # check if name none # else provide default
        if exp_name is None or len(str(exp_name).strip()) == 0:
            exp_name = conf_file.split('.')[0]
        self.exp_name = exp_name

        self.conf_file = conf_file
        self._validated_parameters = False

    @property
    def exp_name(self):
        """exp_name property"""
        return self._exp_name

    @exp_name.setter
    def exp_name(self, name):
        """Assert file name path passed follow name conventions \
        and appends today's date to name file.
        """
        if not isinstance(name, str):
            raise ValueError("Experiment name must be string value.")
        if '/' in name:
            name = name.split('/')[-1]
        # if Context.split not in name:
        #     raise ValueError(f"File name must contain {Context.split} "
        #                      f"for readability.")

        date_format = f'{Context.split}%Y{Context.split}%m{Context.split}%d'
        date = datetime.today().strftime(date_format)
        self._exp_name = name + date

    @property
    def conf_file(self):
        """conf_file property returns the yml file path."""
        return self._conf_file

    @conf_file.setter
    def conf_file(self, file_name):
        """Assert file_name extension is of yml type."""
        if 'yml' not in file_name.split(".")[-1]:
            raise ValueError(f"Configuration file must be a yml file.")
        self._conf_file = file_name

    @staticmethod
    def read_config_file(file_name):
        """Reads yaml or Json files and returns a python `dict` object.
        """
        file = open(file_name)
        try:
            parsed_file = yaml.safe_load(file)
        finally:
            file.close()
        return parsed_file

    def set_configuration_parameters(self):
        """If parameters are not validated, reads conf file, \
        validates parameters in conf.yml file and sets \
        them as instance properties.
        """
        if not self._validated_parameters:
            parsed_file = Context.read_config_file(self.conf_file)
            self.validate_and_set_config_params(parsed_file)
            self._validated_parameters = True
        return self

    def validate_and_set_config_params(self, data):
        """Validates compulsory parameter names are found in conf. file
        and validates the content of those."""
        if not isinstance(data, dict):
            raise TypeError(f"Configuration file must be a dict \
                             found {type(data)} instead.")
        if Context.conf_keys - data.keys() != set():
            raise ValueError(f"Missing {Context.conf_keys - data.keys()} "
                             f"compulsory keys in configuration file.")

        self.validate_set_data_files(data[Context.data_files])

        factory = ObjectFactory()
        Context.conf_keys.remove(Context.data_files)
        for conf_key in Context.conf_keys:
            self.validate_set_parameters(data[conf_key], factory, conf_key)

    def validate_set_data_files(self, data):
        """Validates data_keys are present and sets data_file parameters"""
        if Context.data_key_files - data.keys() != set():
            raise ValueError(f"Missing {Context.data_key_files - data.keys()} "
                             f"keys under data_files from context file.")
        for key in data.keys():
            if key in Context.data_key_files:  # if key not ['target', 'features']
                file = data[key]
                self.validate_and_set_csv(key, file)
            else:
                setattr(self, key, data[key])

    def validate_and_set_csv(self, property_name, file_name):
        """Validates csv file names and converts them into a `pd.DataFrame`\
        object. Assigns them to an instance attribute
        """
        if file_name is None or len(str(file_name).strip()) == 0:
            raise ValueError(f'{property_name} cannot be empty.')
        if 'csv' not in str(file_name).split("."):
            raise TypeError(f"Train data must be a csv file with \
                             .csv extension.")
        df = pd.read_csv(file_name)
        setattr(self, property_name, df)

    def validate_set_parameters(self, parameters, factory, attr_name):
        """Validates each parameter of type `dict` from the list parameters,
        parse the string to an object using the factory class and appends\
        a tuple ('parm_name', object) to a list of parsed parameters.
        Sets an attribute instance to store them.
        """
        if not isinstance(parameters, list):
            raise TypeError(f"{attr_name} from config file expected to be a list"
                            f" got {type(parameters)} instead.")
        params_parsed = []
        for parameter in parameters:
            if not isinstance(parameter, dict):
                raise TypeError(f"Parameter object: {parameter} must be of"
                                f"type dict found {type(parameter)} instead.")
            for param_name, param_values in parameter.items():
                objectified_param = factory.create_object(param_name, param_values)
                params_parsed.append((param_name, objectified_param))
        setattr(self, attr_name, params_parsed)
