import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model', type=str, help='Model name')
parser.add_argument('--test', dest='test', type=str, help='CyberMetric test dataset name')
args = parser.parse_args()

def cmd_args():
    """
    Returns parsed command arguments

    Returns:
        object: Parsed cmd args
    """
    return args