import yaml
import argparse

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def dict2namespace(d):
    namespace = argparse.Namespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(namespace, k, dict2namespace(v))
        else:
            setattr(namespace, k, v)
    return namespace