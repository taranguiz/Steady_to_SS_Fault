# import writer function
from model_config import ModelConfig
import json
import os
from updated_steady import build_steady_topo
from geomorph_dynamics_loop_trying_something import run_geomorf_loop
import imageio
    
config = ModelConfig()
config_dict = vars(config)

# create output folder and move to it
try:
    os.chdir(config.save_location)
except:
    os.mkdir(config.save_location)
    os.chdir(config.save_location)

#Saving parameters into a file
with open(f'config.json', "w") as file:
    json.dump(config_dict, file, indent=4)

def main():

    writer = imageio.get_writer(f'{config.home_path}/{config.save_location}/{config.model_name}_evolution.mp4', fps=20)
    #changed name of the vide maker when started from a differnt time 
    #build_steady_topo(config, writer)

    run_geomorf_loop(config, writer)

    writer.close()


if __name__ == "__main__":
    main()