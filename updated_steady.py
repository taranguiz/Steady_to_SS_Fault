#!/usr/bin/env python3
# import time
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

#from Landlab
from landlab import RasterModelGrid, imshow_grid, imshowhs_grid
from landlab.io import read_esri_ascii
from landlab.io.netcdf import write_netcdf

#Hillslope geomorphology
from landlab.components import ExponentialWeatherer
from landlab.components import DepthDependentTaylorDiffuser
from landlab.components import DepthDependentDiffuser

#Fluvial Geomorphology and Flow routing
from landlab.components import FlowDirectorMFD #trying the FlowDirectorMFD
from landlab.components import FlowAccumulator, Space, FastscapeEroder, PriorityFloodFlowRouter
from landlab.components.space import SpaceLargeScaleEroder

model_name= 'Topo_4'
save_location = 'output_topo/%s'%model_name # location to save output

try:
    os.chdir(save_location)
except:
    os.mkdir('output_topo/%s'%model_name)
    os.chdir(save_location)

print('saving into: %s' %(os.getcwd()), file=open('out_%s.txt' %model_name, 'w')) # print to a file

# Model dimensions
xmax= 3000
ymax= 1000
dxy= 10
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
uplift_rate= 9 *1e-5

#Hillsope Geomorphology for DDTD component
H=5 # original soil depth
Sc= 0.7 #critical slope
Hstar= 0.1 # characteristic transport depth, m
V0= 0.1 #transport velocity coefficient
D= Hstar*V0 #V0 *Hstar  #effective(maximum) diffusivity
P0= 1*1e-4

#Fluvial Erosion for SPACE Large Scale Eroder
run_off=0.5

K_sed=10*1e-6 #sediment erodibility
K_br= 8*1e-6 #bedrock erodibility
F_f=0.5 #fraction of fine sediment
phi= 0.5 #sediment porosity
H_star=Hstar #sediment entrainment lenght scale
Vs= 1 #velocity of sediment
m_sp= 0.5 #exponent ondrainage area stream power
n_sp= 1 #exponent on channel slope in the stream power framework
sp_crit_sed=0 #sediment erosion threshold
sp_crit_br=0 #bedrock erosion threshold

print('name of the model is: %s' %model_name, file=open('out_%s.txt' %model_name, 'a'))
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
# instantiate components

#Weathering
expweath=ExponentialWeatherer(mg, soil_production_maximum_rate=P0, soil_production_decay_depth=Hstar)

# Hillslope with Diffuser
# ddd=DepthDependentDiffuser(mg, linear_diffusivity=D,
#                                   soil_transport_decay_depth=Hstar)
# Hillslope with Taylor Diffuser
ddtd=DepthDependentTaylorDiffuser(mg,slope_crit=Sc,
                                  soil_transport_velocity=V0,
                                  soil_transport_decay_depth=Hstar,
                                  nterms=2,
                                  dynamic_dt=True,
                                  if_unstable='warn')
#Flow Router
fr=PriorityFloodFlowRouter(mg, flow_metric='D8', suppress_out=True, runoff_rate=run_off)
#SPACE Large Scale
space= SpaceLargeScaleEroder(mg,
                             K_sed=K_sed,
                             K_br=K_br,
                            F_f=F_f,
                            phi=phi,
                            H_star=Hstar,
                            v_s=Vs,
                            m_sp=m_sp,
                            n_sp=n_sp,
                            sp_crit_sed=0,
                             sp_crit_br=0)

mg.set_closed_boundaries_at_grid_edges(
    bottom_is_closed=False,
    left_is_closed=True,
    right_is_closed=True,
    top_is_closed=True,
)

#Initialize video maker
writer = imageio.get_writer(f'/home/jupyter-taranguiz/StrikeSlip/steady/output_topo/{model_name}/evolution.mp4', fps=20)

def add_file_to_writer(filename):
    image=imageio.imread(filename)
    writer.append_data(image)


#REMEMBER TO UNCOMMENT
# Now the for loop to do landscape evolution
# fig = plt.figure(figsize=[8, 8])
# imshow_grid(mg, z, cmap='terrain', grid_units=['m', 'm'])
# plt.show()

#timing
tmax= 8000000 #5000000
dt=200 #200
model_time=np.arange(0,tmax,dt)
iterations=len(model_time)
# print(iterations)
#to track elevation changes
Mean_da= [] #Mean drainage area
Mean_elev=[] #mean elevation
Mean_soil=[] #mean soil depth
# Mean_da= np.append(Mean_da, np.mean(mg.at_node['drainage_area']))
# Mean_soil= np.append(Mean_soil, np.mean(mg.at_node['soil__depth']))
# Mean_elev= np.append(Mean_elev, np.mean(mg.at_node['topographic__elevation']))

def build_steady_topo():

    for i in range(iterations):
        # print(i)

        # to track surface
    #     z_beg=sum(z[mg.core_nodes])

        #Insert uplift at core nodes
        rock[mg.core_nodes] += uplift_rate * dt
        z[:] = rock + soil
        #z[mg.core_nodes] += uplift_rate * dt
        #rock[mg.core_nodes] += uplift_rate * dt

        #hillslope
        # expweath.run_one_step()
        expweath.calc_soil_prod_rate()
        ddtd.run_one_step(dt)

        #run flow router
        fr.run_one_step()

        #run space
        space.run_one_step(dt)


        #track surface
    #     z_end=sum(z[mg.core_nodes])

    #     dif_z= z_beg - z_end


        Mean_da= np.append(Mean_da, np.mean(mg.at_node['drainage_area']))
        Mean_soil= np.append(Mean_soil, np.mean(mg.at_node['soil__depth']))
        Mean_elev= np.append(Mean_elev, np.mean(mg.at_node['topographic__elevation']))
        #print(Mean_elev)


        if i%2000 ==0:
            #print('the change in z is: ' + str(dif_z) + 'and it is iteration ' + str(i))
            #fig = plt.figure(figsize=[8, 8])
            imshow_grid(mg, z, cmap='terrain', grid_units=['m', 'm'])
            plt.title('Topography after ' + str(int((i * dt))) + ' years')
            plt.show()
            loop_topo_img = f'/home/jupyter-taranguiz/StrikeSlip/steady/output_topo/{model_name}/topo_%s_yrs.png' % (int(i * dt))
            plt.savefig(loop_topo_img, dpi=300, facecolor='white')
            add_file_to_writer(loop_topo_img)
            plt.clf()
            
build_steady_topo()

#fig = plt.figure(figsize=[8, 8])
imshow_grid(mg, z, cmap='terrain', grid_units=['m', 'm'])
plt.title('Topography after ' + str(int((tmax))) + ' years')
#plt.show()        
#imshow_grid(mg, z, cmap='terrain', grid_units=['m', 'm'])
final_topo_img= f'/home/jupyter-taranguiz/StrikeSlip/steady/output_topo/{model_name}/final.png'
plt.savefig(final_topo_img, dpi=300, facecolor='white')
add_file_to_writer(final_topo_img)
writer.close()

print(type(Mean_elev))
print(Mean_elev.size)
print(type(model_time))
print(model_time.size)

fig2= plt.figure(figsize=[8,8])
plt.plot(model_time, Mean_elev)
plt.xlabel('model iterations')
plt.ylabel('mean elevation[m]')
plt.savefig(f'/home/jupyter-taranguiz/StrikeSlip/steady/output_topo/{model_name}/mean_elevation.jpg')

fig3= plt.figure(figsize=[8,8])
plt.plot(model_time, Mean_da)
plt.xlabel('time [years]')
plt.ylabel('mean drainage area [m2]')
plt.savefig(f'/home/jupyter-taranguiz/StrikeSlip/steady/output_topo/{model_name}/mean_drainage.jpg')

fig4= plt.figure(figsize=[8,8])
plt.plot(model_time, Mean_soil)
plt.xlabel('time [years]')
plt.ylabel('mean soil_depth [m]')
plt.savefig(f'/home/jupyter-taranguiz/StrikeSlip/steady/output_topo/{model_name}/mean_soil.jpg')

np.savetxt(f'/home/jupyter-taranguiz/StrikeSlip/steady/output_topo/{model_name}/{model_name}mean_elev.txt',
           (Mean_elev),
           delimiter=',',
           header='Mean_elev',
           comments='')
np.savetxt(f'/home/jupyter-taranguiz/StrikeSlip/steady/output_topo/{model_name}/{model_name}mean_da.txt',
           (Mean_da),
           delimiter=',',
           header='Mean_da',
           comments='')
np.savetxt(f'/home/jupyter-taranguiz/StrikeSlip/steady/output_topo/{model_name}/{model_name}mean_soil.txt',
           (Mean_soil),
           delimiter=',',
           header='Mean_soil',
           comments='')

print(mg.fields())
try: 
    write_netcdf(
        f'/home/jupyter-taranguiz/StrikeSlip/steady/output_topo/{model_name}/steady-state.nc', 
        mg, 
        format='NETCDF4', 
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
except Exception as e:
    print(str(e))

try: 
    mg.save(f'/home/jupyter-taranguiz/StrikeSlip/steady/output_topo/{model_name}/steady-state.nc')
except Exception as e:
    print(str(e))

    # if float(diff) < float(uplift_rate):
    #     print('steady state reached')
    #     break


