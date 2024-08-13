import json

def read_json_file(file_path):
    """
    Reads a json file at a given path

    Parameters:
        file_path (string): Path of the JSON file to read

    Returns:
        object: JSON content
    """
    with open(file_path, 'r') as file:
        return json.load(file)