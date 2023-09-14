# run_train_imitation.py
import json
from imitation.scripts import train_imitation 

def run_train_imitation(config, save_path):
    run = train_imitation.train_imitation_ex.run(**config)
    
    # dump dict to save_path
    with open(save_path, 'w') as f:
        json.dump(run, f)

