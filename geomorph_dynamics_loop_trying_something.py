#!/usr/bin/env python3
# import time
import numpy as np
import matplotlib.pyplot as plt
import yaml
from yaml.loader import SafeLoader
import time
import imageio
import glob
import os

#from Landlab
from landlab import RasterModelGrid, imshow_grid, imshowhs_grid
from landlab.io import read_esri_ascii
from landlab.io.netcdf import write_raster_netcdf
from landlab.io.netcdf import read_netcdf
from landlab.io.netcdf import write_netcdf

#Hillslope geomorphology
from landlab.components import ExponentialWeatherer
from landlab.components import DepthDependentTaylorDiffuser
from landlab.components import DepthDependentDiffuser

#Fluvial Geomorphology and Flow routing
from landlab.components import FlowDirectorMFD #trying the FlowDirectorMFD
from landlab.components import FlowAccumulator, Space, FastscapeEroder, PriorityFloodFlowRouter
from landlab.components.space import SpaceLargeScaleEroder

from ss_fault_function import ss_fault

#reading the file with parameters
config = yaml.safe_load(open('parameters_trying_something.yaml','r'))#,Loader=yaml.FullLoader)

model_name= config['saving']['model_name']
alt_name= config['comments']['alt_name']
save_location = 'output_model_run/%s'%model_name # location to save output
home_path= '/home/jupyter-taranguiz/StrikeSlip/faulting/steady-to-faulting'

try:
    os.chdir(save_location)
except:
    os.mkdir('output_model_run/%s'%model_name)
    os.chdir(save_location)

print('saving into: %s' %(os.getcwd()), file=open('out_%s.txt' %model_name, 'w')) # print to a file

#grid
ymax=config['shape']['ymax']
xmax=config['shape']['xmax']
dxy=config['shape']['dxy']

nrows = int(ymax/dxy)
ncols = int(xmax/dxy)

#Instantiate model grid
mg= RasterModelGrid ((nrows,ncols), dxy)
#add field topographic elevation
mg.add_zeros("node", "topographic__elevation")

np.random.seed(seed=5000)
#creating initial model topography
random_noise = (np.random.rand(len(mg.node_y)))

#add the topo to the field
mg["node"]["topographic__elevation"] +=random_noise

# add field 'soil__depth' to the grid
mg.add_zeros("node", "soil__depth", clobber=True)

# Set  5m of initial soil depth at core nodes
mg.at_node["soil__depth"][mg.core_nodes] = 10.0  # meters

# Add field 'bedrock__elevation' to the grid
mg.add_zeros("bedrock__elevation", at="node")
# Sum 'soil__depth' and 'bedrock__elevation'
# to yield 'topographic elevation'
mg.at_node["bedrock__elevation"][:] = mg.at_node["topographic__elevation"]
mg.at_node["topographic__elevation"][:] += mg.at_node["soil__depth"]
mg.add_zeros("node", "soil_production__rate", clobber=True)
# soil_production_rate= mg.at_node["soil_production__rate"]

rock= mg.at_node["bedrock__elevation"]
soil=mg.at_node["soil__depth"]
z=mg.at_node["topographic__elevation"]


# Geomorphic parameters

# uplift
uplift_rate= config['geomorphology']['uplift_rate']

#Hillsope Geomorphology for DDTD component
Sc=config['geomorphology']['Sc']
Hstar= config['geomorphology']['Hstar'] # characteristic transport depth, m
V0= config['geomorphology']['V0'] #transport velocity coefficient changed this
D= Hstar*V0#V0 *Hstar  #effective(maximum) diffusivity
P0=config['geomorphology']['P0']
run_off=config['geomorphology']['run_off']

#Fluvial Erosion for SPACE Large Scale Eroder
K_sed=config['geomorphology']['K_sed'] #sediment erodibility
K_br= config['geomorphology']['K_br'] #bedrock erodibility
F_f=config['geomorphology']['F_f']#fraction of fine sediment
phi= config['geomorphology']['phi'] #sediment porosity
H_star=config['geomorphology']['H_star'] #sediment entrainment lenght scale
Vs= config['geomorphology']['Vs'] #velocity of sediment
m_sp= config['geomorphology']['m_sp'] #exponent ondrainage area stream power
n_sp= config['geomorphology']['n_sp'] #exponent on channel slope in the stream power framework
sp_crit_sed= config['geomorphology']['sp_crit_sed'] #sediment erosion threshold
sp_crit_br= config['geomorphology']['sp_crit_br'] #bedrock erosion threshold

#Initialize video maker
writer = imageio.get_writer(f'{home_path}faulting/output_model_run/{model_name}/{alt_name}.mp4', fps=20)

def add_file_to_writer(filename):
    image=imageio.imread(filename)
    writer.append_data(image)

def get_file_sequence(time_step):
    num_digits = len(str(total_model_time))
    file_sequence_num = time_step * 10 ** -num_digits
    float_convert = f'{{:.{num_digits}f}}'
    file_sequence_num = float_convert.format(file_sequence_num).split('.')[1]
    return file_sequence_num

#READING STEADY STATE TOPO
# (mg,z)=read_esri_ascii('/Users/taranguiz/Research/CSDMS_summer_2022/output_new_topo_ddd_5/finaltopo_topographic__elevation.asc', name="topographic__elevation")
# (mg1,soil_0)=read_esri_ascii('/Users/taranguiz/Research/CSDMS_summer_2022/output_new_topo_ddd_5/finaltopo_soil__depth.asc', name='soil__depth')
# (mg2,bed_0)=read_esri_ascii('/Users/taranguiz/Research/CSDMS_summer_2022/output_new_topo_ddd_5/finaltopo_bedrock__elevation.asc', name='bedrock__elevation')
# # (mg3, dra_0)=re
# mg.add_field("soil__depth", soil_0, at='node')
# mg.add_field("bedrock__elevation", bed_0, at='node')

mg = read_netcdf(f'{home_path}steady/output_topo/Topo_3/steady-state.nc')

z = mg.at_node['topographic__elevation']
soil=mg.at_node['soil__depth']
bed= mg.at_node['bedrock__elevation']
print(mg.fields())

mg.set_closed_boundaries_at_grid_edges(bottom_is_closed=False, left_is_closed=True,
                                       right_is_closed=True, top_is_closed=True)
shrink = 0.5
# # fig= plt.figure(figsize=[8,8])
# imshow_grid(mg,z, cmap='viridis', shrink=shrink)
# plt.title('Topography after 0 years')
# #write_raster_netcdf(f'{model_name}0000000.nc', mg, format="NETCDF4")
# initial_topo_img = f'/Users/taranguiz/Research/Lateral_advection/output_model_run/{model_name}/{model_name}0000000.png'
# plt.savefig(initial_topo_img, dpi=300, facecolor='white')
# # plt.show()
#
# add_file_to_writer(initial_topo_img)
# plt.clf()



# print (mg.at_node['topographic__elevation'][mg.core_nodes])
# print (mg.at_node['bedrock__elevation'][mg.core_nodes])
# print (mg.at_node['soil__depth'][mg.core_nodes])
#TECTONICS AND TIME PARAMETERS
total_slip= config['tectonics']['total_slip']
method= config['tectonics']['method']
total_model_time= config['time']['total_model_time']
dt= config['time']['dt']
iterations= np.arange(0,total_model_time+dt,dt)
print(iterations)
desired_slip_per_event=(total_slip/total_model_time)*dt
shrink = 0.5

fault_loc_y=int(mg.number_of_node_rows / 3.)
fault_nodes = np.where(mg.node_y==(fault_loc_y*10))[0]
print(fault_nodes)

#instantiate components
expweath=ExponentialWeatherer(mg, soil_production_maximum_rate=P0, soil_production_decay_depth=Hstar)

# Hillslope with Taylor Diffuser
ddtd=DepthDependentTaylorDiffuser(mg,slope_crit=Sc,
                                  soil_transport_velocity=V0,
                                  soil_transport_decay_depth=Hstar,
                                  nterms=2,
                                  dynamic_dt=True,
                                  if_unstable='warn', courant_factor=0.1)
#Flow Router #Initialize with more arguments than before
fr=PriorityFloodFlowRouter(mg,
                           flow_metric='D8',
                           separate_hill_flow=True,
                           hill_flow_metric="Quinn",
                           update_hill_flow_instantaneous=True,
                           suppress_out=True, runoff_rate=run_off)
#SPACE Large Scale
space= SpaceLargeScaleEroder(mg,
                             K_sed=K_sed,
                             K_br=K_br,
                            F_f=F_f,
                            phi=phi,
                            H_star=H_star,
                            v_s=Vs,
                            m_sp=m_sp,
                            n_sp=n_sp,
                            sp_crit_sed=sp_crit_sed,
                             sp_crit_br=sp_crit_br)


#fluvial array (look up table)
fluvial_freq=config['climate']['fluvial_freq'] #how often the humid period occurs
fluvial_len=config['climate']['fluvial_len'] #how long the humid period last
fluvial_0=np.arange(fluvial_freq,total_model_time, fluvial_freq)
fluvial_n=fluvial_0+fluvial_len
fluvial_times=np.vstack((fluvial_0,fluvial_n)).T
print(fluvial_times)


#### print parameters to file ####
print('name of the model is: %s' %alt_name, file=open('out_%s.txt' %model_name, 'a'))
print('uplift_rate: %s' %uplift_rate, file=open('out_%s.txt' %model_name, 'a'))
print('Sc: %s' %Sc, file=open('out_%s.txt' %model_name, 'a'))
print('Hstar: %s' %Hstar, file=open('out_%s.txt' %model_name, 'a'))
print('V0: %s' %V0, file=open('out_%s.txt' %model_name, 'a'))
print('P0: %s' %P0, file=open('out_%s.txt' %model_name, 'a'))
print('run_off: %s' %run_off, file=open('out_%s.txt' %model_name, 'a'))
print('K_sed: %s' %K_sed, file=open('out_%s.txt' %model_name, 'a'))
print('K_br: %s' %K_br, file=open('out_%s.txt' %model_name, 'a'))
print('F_f: %s' %F_f, file=open('out_%s.txt' %model_name, 'a'))
print('phi: %s' %phi, file=open('out_%s.txt' %model_name, 'a'))
print('H_star: %s' %H_star, file=open('out_%s.txt' %model_name, 'a'))
print('Vs: %s' %Vs, file=open('out_%s.txt' %model_name, 'a'))
print('m_sp: %s' %m_sp, file=open('out_%s.txt' %model_name, 'a'))
print('n_sp: %s' %n_sp, file=open('out_%s.txt' %model_name, 'a'))
print('sp_crit_sed: %s' %sp_crit_sed, file=open('out_%s.txt' %model_name, 'a'))
print('sp_crit_br: %s' %sp_crit_br, file=open('out_%s.txt' %model_name, 'a'))
print('total_slip: %s' %total_slip, file=open('out_%s.txt' %model_name, 'a'))
print('method: %s' %method, file=open('out_%s.txt' %model_name, 'a'))
print('total_model_time: %s' %total_model_time, file=open('out_%s.txt' %model_name, 'a'))
print('dt: %s' %dt, file=open('out_%s.txt' %model_name, 'a'))
print('fluvial_freq: %s' %fluvial_freq, file=open('out_%s.txt' %model_name, 'a'))
print('fluvial_len: %s' %fluvial_len, file=open('out_%s.txt' %model_name, 'a'))
print('',file=open('out_%s.txt' %model_name, 'a'))


#things for the loop
time=0 #time counter
f=0 #index counter for fluvial
h=0 #index counter for hillslope
accumulate=0
Mean_da= [] #Mean drainage area
Mean_elev=[] #mean elevation
Mean_soil=[] #mean soil depth
Mean_da= np.append(Mean_da, np.mean(mg.at_node['drainage_area']))
Mean_soil= np.append(Mean_soil, np.mean(mg.at_node['soil__depth']))
Mean_elev= np.append(Mean_elev, np.mean(mg.at_node['topographic__elevation']))
quakes_times=[]

while time < total_model_time:

      z[mg.core_nodes]+= (uplift_rate*dt) #do uplift all the time
      bed[mg.core_nodes]+= (uplift_rate*dt)
      expweath.calc_soil_prod_rate()
      ddtd.run_one_step(dt)
      fr.run_one_step()
      space.run_one_step(dt)

      accumulate += desired_slip_per_event
      print('is accumulating')

      Mean_da= np.append(Mean_da, np.mean(mg.at_node['drainage_area']))
      Mean_soil= np.append(Mean_soil, np.mean(mg.at_node['soil__depth']))
      Mean_elev= np.append(Mean_elev, np.mean(mg.at_node['topographic__elevation']))

      if accumulate >= mg.dx:
         ss_fault(grid=mg, fault_loc_y=fault_loc_y, total_slip=total_slip,
                  total_time=total_model_time, method=method, accumulate=accumulate)
         accumulate = accumulate % mg.dx
          # expweath.maximum_weathering_rate=1*1e-6
          # expweath.calc_soil_prod_rate()
          # ddtd.run_one_step(dt=1250)
         quakes_times=np.append(quakes_times, time)
         print('one slip')

      # if time >= fluvial_times[f,0] and time <= fluvial_times[f,1]:
      #     print('is: ' + str(time) +' fluvial time')
      #     fr.run_one_step()
      #     space.run_one_step(dt)
      #     if time == fluvial_times[f,1] and f < (len(fluvial_times)-1):
      #         f=f+1
      #     if f==(len(fluvial_times)-1):
      #         pass

      # if time >= hillslope_times[h,0] and time <= hillslope_times[h,1]:
      #     print ('is: ' + str(time) +' so is hola colluvial')
      #     # expweath.maximum_weathering_rate=1*1e-5
      #     expweath.calc_soil_prod_rate()
      #     ddtd.run_one_step(dt)
      #     if time == hillslope_times[h,1] and h < (len(hillslope_times)-1):
      #         h=h+1
      #     if h == (len(hillslope_times)-1):
      #         pass

      if time%5000 == 0: #time>0 and
          # fig1 = plt.figure(figsize=[8, 8])
          imshow_grid(mg, z, cmap='viridis', shrink=shrink)
          plt.title('Topography after ' + str(time) + ' years')
          #plt.show()
          # print(mg.at_node.keys())
          #write_raster_netcdf(f'{model_name}{get_file_sequence(time)}.nc', mg, format="NETCDF4",
                              # names=['surface_water__discharge', 'drainage_area',
                              #        'bedrock__elevation', 'soil__depth',
                              #        'sediment__flux', 'soil_production__rate', 'topographic__steepest_slope'])
          loop_topo_img  = f'{home_path}faulting/output_model_run/{model_name}/{model_name}{get_file_sequence(time)}.png'
          plt.savefig(
              loop_topo_img,
              dpi=300, facecolor='white'
          )
          add_file_to_writer(loop_topo_img)

          plt.clf()
          
      if time%100000 == 0:
        write_raster_netcdf(f'{home_path}faulting/output_model_run/{model_name}{get_file_sequence(time)}.nc', mg,
                            format='NETCDF3_64BIT', 
                            names=[
                                'bedrock__elevation',
                                'drainage_area',
                                'flood_status_code',
                                'flow__link_to_receiver_node',
                                'flow__receiver_node',
                                'flow__receiver_proportions',
                                'flow__upstream_node_order',
                                'soil__depth',
                                'soil_production__rate',
                                'surface_water__discharge',
                                'topographic__elevation',
                                'topographic__steepest_slope',
                                'water__unit_flux_in'])
                              # names=['surface_water__discharge', 'drainage_area',
                              #        'bedrock__elevation', 'soil__depth',
                              #        'sediment__flux', 'soil_production__rate', 'topographic__steepest_slope'])

      time = time + dt
      print(time)

print(time)
# fig2 = plt.figure(figsize=[8, 8])
imshow_grid(mg, z, cmap='viridis', shrink=shrink)
plt.title('Topography after ' + str(total_model_time) + ' years')
# plt.show()
# write_raster_netcdf(
#     f'{model_name}{total_model_time}.nc',
#     mg,
#     format="NETCDF4",
#     names=[
#         'surface_water__discharge',
#         'drainage_area',
#         'bedrock__elevation',
#         'soil__depth',
#         'sediment__flux',
#         'soil_production__rate',
#         'topographic__steepest_slope'
#     ]
#)
final_topo_img= f'{home_path}faulting/output_model_run/{model_name}/{model_name}{total_model_time}.png'
plt.savefig(final_topo_img, dpi=300, facecolor='white')
add_file_to_writer(final_topo_img)
writer.close()

fig3= plt.figure(figsize=[8,8])
plt.plot(iterations, Mean_elev)
plt.xlabel('time [years]')
plt.ylabel('mean elevation[m]')
plt.savefig(f'{home_path}faulting/output_model_run/{model_name}/{model_name}mean_elevation.jpg')
fig4= plt.figure(figsize=[8,8])
plt.plot(iterations, Mean_da)
plt.xlabel('time [years]')
plt.ylabel('mean drainage area [m2]')
plt.savefig(f'{home_path}faulting/output_model_run/{model_name}/{model_name}mean_drainage.jpg')
fig5= plt.figure(figsize=[8,8])
plt.plot(iterations, Mean_soil)
plt.xlabel('time [years]')
plt.ylabel('mean soil_depth [m]')
plt.savefig(f'{home_path}faulting/output_model_run/{model_name}/{model_name}mean_soil.jpg')

np.savetxt(f'{home_path}faulting/output_model_run/{model_name}/{model_name}mean_elev.txt',
           (Mean_elev),
           delimiter=',',
           header='Mean_elev',
           comments='')
np.savetxt(f'{home_path}faulting/output_model_run/{model_name}/{model_name}mean_da.txt',
           (Mean_da),
           delimiter=',',
           header='Mean_da',
           comments='')
np.savetxt(f'{home_path}faulting/output_model_run/{model_name}/{model_name}mean_soil.txt',
           (Mean_soil),
           delimiter=',',
           header='Mean_soil',
           comments='')
np.savetxt(f'{home_path}faulting/output_model_run/{model_name}/{model_name}quakes.txt',
           (quakes_times),
           delimiter=',',
           header='quakes_times',
           comments='')

write_netcdf(f'{home_path}faulting/output_model_run/{model_name}/{model_name}final-topo.nc', mg, 
        format='NETCDF3_64BIT', 
        names=[
            'bedrock__elevation',
            'drainage_area',
            'flood_status_code',
            'flow__link_to_receiver_node',
            'flow__receiver_node',
            'flow__receiver_proportions',
            'flow__upstream_node_order',
            'soil__depth',
            'soil_production__rate',
            'surface_water__discharge',
            'topographic__elevation',
            'topographic__steepest_slope',
            'water__unit_flux_in'
        ]
            )

# try: 
#     mg.save(f'/home/jupyter-taranguiz/StrikeSlip/faulting/output_topo/{model_name}/final-topo2.nc')
# except Exception as e:
#     print(str(e))

# mg.save(f'/Users/taranguiz/Research/Lateral_advection/output_model_run/{model_name}/{model_name}_{total_model_time}.asc')

# figsize = [16,4] # size of grid plots
# fig, ax = plt.subplots(figsize=figsize)

# x = mg.node_x[fault_nodes]
# # soil_level = bed + soil
#
# ax.plot(x, mg.at_node['soil__depth'][fault_nodes], 'orange', linewidth=2, markersize=12, label='soil')
# ax.plot(x, mg.at_node['bedrock__elevation'][fault_nodes], linewidth=2, markersize=12, label='bedrock')
# ax.plot(x, mg.at_node['topographic__elevation'][fault_nodes],'red', linewidth=2, markersize=12, label='topo')
#
#
# plt.title('Final cross-Profile topography at fault location')
# #plt.text(480, 9, 'air')
# #plt.text(480, 7, 'soil')
# #plt.text(470, 2, 'bedrock')
#
# # plt.xlim(0,1000)
# # plt.ylim(0, 10)
# ax.set_xlabel('X (m)')
# ax.set_ylabel('Depth (m)')
# ax.legend(loc='upper right')
# plt.show()




