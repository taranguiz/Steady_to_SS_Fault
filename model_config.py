import yaml
import numpy as np
import os

# Get the directory containing this script
_script_dir = os.path.dirname(__file__)
# Construct the full path to the config file
_config_path = os.path.join(_script_dir, 'parameters_trying_something.yaml')
    
class ModelConfig:
    
    def __init__(self):
        self.config = yaml.safe_load(
            open(
                _config_path,
                'r'
            )
        )
        #General 
        self.model_name = self.config['saving']['model_name']
        self.alt_name = self.config['comments']['alt_name']
        self.home_path = self.config['saving']['home_path']
        self.save_location = 'output/%s'%self.model_name

        #Shape
        self.ymax=self.config['shape']['ymax']
        self.xmax=self.config['shape']['xmax']
        self.dxy=self.config['shape']['dxy']

        self.nrows = int(self.ymax/self.dxy)
        self.ncols = int(self.xmax/self.dxy)
        
        #Geomorphology
        self.uplift_rate= self.config['geomorphology']['uplift_rate']
        self.H0= self.config['geomorphology']['H0'] #initial soil

        #Hillsope Geomorphology for DDTD component
        self.Sc=self.config['geomorphology']['Sc']
        self.Hstar= self.config['geomorphology']['Hstar'] # characteristic transport depth, m
        self.V0= self.config['geomorphology']['V0'] #transport velocity coefficient changed this
        self.D= self.Hstar*self.V0#V0 *Hstar  #effective(maximum) diffusivity
        self.P0=self.config['geomorphology']['P0']
        
        #Flow router
        self.run_off=self.config['geomorphology']['run_off']
        
        #Fluvial Erosion for SPACE Large Scale Eroder
        self.K_sed=self.config['geomorphology']['K_sed'] #sediment erodibility
        self.K_br= self.config['geomorphology']['K_br'] #bedrock erodibility
        self.F_f=self.config['geomorphology']['F_f']#fraction of fine sediment
        self.phi= self.config['geomorphology']['phi'] #sediment porosity
        self.H_star=self.config['geomorphology']['H_star'] #sediment entrainment lenght scale
        self.Vs= self.config['geomorphology']['Vs'] #velocity of sediment
        self.m_sp= self.config['geomorphology']['m_sp'] #exponent ondrainage area stream power
        self.n_sp= self.config['geomorphology']['n_sp'] #exponent on channel slope in the stream power framework
        self.sp_crit_sed= self.config['geomorphology']['sp_crit_sed'] #sediment erosion threshold
        self.sp_crit_br= self.config['geomorphology']['sp_crit_br'] #bedrock erosion threshold
        
        #Strike Slip Fault 
        self.total_slip= float(self.config['tectonics']['total_slip']) #total distance to slip
        self.method= self.config['tectonics']['method'] #roll or drop 

        #Time
        self.total_model_time= float(self.config['time']['total_model_time'])
        self.dt_model= float(self.config['time']['dt_model'])
        self.total_steady_time= float(self.config['time']['total_steady_time'])
        self.dt_steady= float(self.config['time']['dt_steady'])

        #Climate
        self.fluvial_freq=float(self.config['climate']['fluvial_freq']) #how often the humid period occurs
        self.fluvial_len=float(self.config['climate']['fluvial_len']) #how long the humid period last