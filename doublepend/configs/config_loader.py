import yaml

def load_config(file): 

    with open(file) as f:
        config = yaml.full_load(f)
    
        
    return config

def print_config (config):
    return print(yaml.dump(config,default_flow_style=False, sort_keys=False))
        
def save_config(config, file_path):
    
    with open(file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)
    