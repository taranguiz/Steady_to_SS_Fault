import yaml
from yaml.loader import SafeLoader
    
class ModelConfig:
    
    def __init__(self):
        self.config = yaml.safe_load(
            open(
                'parameters_trying_something.yaml',
                'r'
            )
        )
        #General 
        self.model_name = self.config['saving']['model_name']
        self.alt_name = self.config['comments']['alt_name']
        #Shape
        self.ymax=self.config['shape']['ymax']
        self.xmax=self.config['shape']['xmax']
        self.dxy=self.config['shape']['dxy']

        self.nrows = int(self.ymax/self.dxy)
        self.ncols = int(self.xmax/self.dxy)
        
        #Geomorphology
        self.uplift_rate= self.config['geomorphology']['uplift_rate']

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
        self.total_slip= self.config['tectonics']['total_slip'] #total distance to slip
        self.method= self.config['tectonics']['method'] #roll or drop 

        #Time
        self.total_model_time= self.config['time']['total_model_time']
        self.dt_model= self.config['time']['dt_model']
        self.total_steady_time= self.config['time']['total_steady_time']
        self.dt_steady= self.config['time']['dt_steady']

        #Climate
        self.fluvial_freq=self.config['climate']['fluvial_freq'] #how often the humid period occurs
        self.fluvial_len=self.config['climate']['fluvial_len'] #how long the humid period last

