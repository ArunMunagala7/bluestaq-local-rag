import yaml, os

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

CFG = load_config(os.path.join(os.getcwd(), "config.yaml"))
