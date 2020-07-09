import yaml


def read_config_file(file_name):
    """Reads yaml or Json files and returns a python `dict` object.
    """
    file = open(file_name)
    try:
        parsed_file = yaml.safe_load(file)
    finally:
        file.close()
    return parsed_file


def validate_path(file_name, extension):
    """Validates csv file names and converts them into a `pd.DataFrame`\
            object. Assigns them to an instance attribute
    """
    if file_name is None or len(str(file_name).strip()) == 0:
        raise ValueError(f'{"file_name"} cannot be empty.')

    if extension not in str(file_name).split("."):
        raise TypeError(f"File name {file_name} must be of type {extension}")
