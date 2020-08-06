"""
Context module to Validate conf.yml file, initialize context component objects,
and store those as context attributes ready to be injected to pipelines.
"""
from datetime import datetime
from .utiles import validate_path, read_config_file
from .components import Data, Transformers, Vectorizers, Estimators


class Context:
    """
    Context class stores hardcoded keys present in conf.yaml,
    reads, validate and parse conf parameters found in conf yml,
    and sets those as attributes of the Context class.
    """

    def __init__(self,
                 exp_name=None,
                 *,
                 conf_file,
                 data_key='data',
                 trasformers_key='transformers',
                 vectorizers_key='vectorizers',
                 estimators_key='estimators'):
        """
        Sets default exp name if None is passed,
        sets data, transformers, vectorizers and
        estimators parameters and initializes each
        respective class.
        """
        self._context_keys = {data_key, estimators_key}
        self.config = conf_file
        exp_name = exp_name or str(conf_file).split('.')[-2]
        self.exp_name = exp_name

        if bool(self._context_keys - set(self.config.keys())):
            raise KeyError(f"Missing compulsory configuration keys: ",
                           self._context_keys - set(self.config.keys()))

        self.data = Data(**self.config[data_key])
        # Transformers can be None
        transformers_section = self.config.get(trasformers_key, {})
        self.transformers = Transformers(**transformers_section)
        # Vectorizers can be None
        vectorizers_section = self.config.get(vectorizers_key, {})
        self.vectorizers = Vectorizers(**vectorizers_section)

        self.estimators = Estimators(**self.config[estimators_key])

    @property
    def exp_name(self):
        """
        exp_name property
        """
        return self._exp_name

    @exp_name.setter
    def exp_name(self, name):
        """
        Assert file name path passed follow name conventions \
        and appends today's date to name file.
        """
        if not isinstance(name, str):
            raise ValueError("Experiment name must be string value.")

        if '/' in name:
            # If using file path as name get
            # file name only
            name = name.split('/')[-1]

        date_format = f'_%Y_%m_%d'
        date = datetime.today().strftime(date_format)
        name_date = name + date
        self._exp_name = name_date

    @property
    def config(self):
        """
        conf_file property returns the yml file path.
        """
        return self._config

    @config.setter
    def config(self, conf_info):
        """
        Assert file_name extension is of yml type.
        """
        if not isinstance(conf_info, str):
            raise ValueError("config file must be str path name.")

        validate_path(conf_info, 'yml')
        config_dic = read_config_file(conf_info)

        self._config = config_dic
