#!/usr/bin/env python3
# import time
import numpy as np
import matplotlib.pyplot as plt
# import time
import pickle
from collections import defaultdict
import os # Added import
import copy # Make sure copy is imported (it might already be as deepcopy)

#from Landlab
from landlab import RasterModelGrid, imshow_grid, imshowhs_grid
# from landlab.io import read_esri_ascii
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
from util import get_file_sequence, add_file_to_writer
from copy import deepcopy # Ensure deepcopy is specifically imported
#READING STEADY STATE TOPO

def read_grid(config):
    
    """Read the final steady state from pickle file efficiently"""
    print("reading steady topo")
    # Read the final steady state
    with open(f'{config.home_path}/output/steady_state_files/final_state.pkl', 'rb') as f:
        # Load only the last state
        final_state = pickle.load(f)
        #print(final_state)
        #final_time = max(final_state.keys())
        #final_state = steady_states[final_time]
        print("Loaded final state")
    
    final_state.set_closed_boundaries_at_grid_edges(
        bottom_is_closed=False, 
        left_is_closed=True,
        right_is_closed=True, 
        top_is_closed=True
    )
    return final_state

def save_grid_state(mg, time):
    state = {}
    for field_name, field_data in mg.at_node.items():
        state[f'at_node_{field_name}'] = field_data.copy()
    return state

def run_geomorf_loop(config, writer):
    mg = read_grid(config)
    print(mg.at_node.keys())
    #exit()
    
    # Define NetCDF fields to save (if using netcdf format)
    netcdf_field_names = [
        'bedrock__elevation', 'drainage_area', 'flood_status_code',
        'flow__link_to_receiver_node', 'flow__receiver_node', 'flow__receiver_proportions',
        'flow__upstream_node_order', 'soil__depth', 'soil_production__rate',
        'surface_water__discharge', 'topographic__elevation', 'topographic__steepest_slope',
        'water__unit_flux_in', 'sediment__influx', 'sediment__outflux', 'sediment__flux'
    ]

    # Define base output directory for pickle files
    
    if hasattr(config, 'save_format') and config.save_format == 'pickle':
        pickle_output_dir = os.path.join(config.home_path, config.save_location, 'pickle_outputs', config.model_name)
        os.makedirs(pickle_output_dir, exist_ok=True)
        pickle_output_filename = os.path.join(pickle_output_dir, f"{config.model_name}_dynamic_states.pkl")
    else:
        # Fallback or define if only netcdf is primary, though loading below is pickle-specific
        pickle_output_filename = None # Or some default non-saving path

    grid_states = defaultdict(dict) # Initialize grid_states

    # Try to load existing states if they exist (Pickle specific)
    if hasattr(config, 'save_format') and config.save_format == 'pickle' and pickle_output_filename:
        try:
            with open(pickle_output_filename, 'rb') as f:
                grid_states = pickle.load(f)
                print(f"Loaded {len(grid_states)} existing grid states from {pickle_output_filename}")
        except FileNotFoundError:
            print(f"Pickle file {pickle_output_filename} not found, starting fresh.")
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Error loading pickle file {pickle_output_filename}: {e}. Starting fresh.")
    elif not hasattr(config, 'save_format'):
        print("Warning: config.save_format is not set. Saving and Loading behavior may be undefined.")

    z = mg.at_node['topographic__elevation']
    soil = mg.at_node['soil__depth']
    rock = mg.at_node['bedrock__elevation']
    
    #TECTONICS AND TIME PARAMETERS
    total_slip= config.total_slip
    method= config.method
    iterations= np.arange(0,config.total_model_time+config.dt_model,config.dt_model)
    print(iterations)
    
    desired_slip_per_event=(config.total_slip/config.total_model_time)*config.dt_model
    shrink = 0.5
    fault_loc_y=int(mg.number_of_node_rows / 3.)
    fault_nodes = np.where(mg.node_y==(fault_loc_y*10))[0]
    #print(fault_nodes)

    #fluvial array (look up table)
    fluvial_0= np.arange(config.fluvial_freq,config.total_model_time, config.fluvial_freq)
    fluvial_n=fluvial_0 + config.fluvial_len
    fluvial_times=np.vstack((fluvial_0,fluvial_n)).T
    print(f"Fluvial times array shape: {fluvial_times.shape}")
    print(f"Fluvial times array: {fluvial_times}")

    #things for the loop
    time=0 #time counter
    fluvial_idx=0 #index counter for fluvial periods
    accumulate=0 #counter for fault earthquake times
    Mean_da= [] #Mean drainage area
    Mean_elev=[] #mean elevation
    Mean_soil=[] #mean soil depth
    Mean_da= np.append(Mean_da, np.mean(mg.at_node['drainage_area']))
    Mean_soil= np.append(Mean_soil, np.mean(mg.at_node['soil__depth']))
    Mean_elev= np.append(Mean_elev, np.mean(mg.at_node['topographic__elevation']))
    quakes_times=[]
    
    #instantiate components
    print("inititializing components")

    expweath=ExponentialWeatherer(mg, 
                                soil_production_maximum_rate=config.P0, 
                                soil_production_decay_depth=config.Hstar)

    # Hillslope with Taylor Diffuser
    ddtd=DepthDependentTaylorDiffuser(mg,slope_crit=config.Sc,
                                    soil_transport_velocity=config.V0,
                                    soil_transport_decay_depth=config.Hstar,
                                    nterms=2,
                                    dynamic_dt=True,
                                    if_unstable='warn', 
                                    courant_factor=0.1)

    #Flow Router #Initialize with more arguments than before
    fr=PriorityFloodFlowRouter(mg,
                            flow_metric='D8',
                            separate_hill_flow=True,
                            hill_flow_metric="Quinn",
                            update_hill_flow_instantaneous=True,
                            suppress_out=True, runoff_rate=config.run_off)
    #SPACE Large Scale
    space= SpaceLargeScaleEroder(mg,
                                K_sed=config.K_sed,
                                K_br=config.K_br,
                                F_f=config.F_f,
                                phi=config.phi,
                                H_star=config.H_star,
                                v_s=config.Vs,
                                m_sp=config.m_sp,
                                n_sp=config.n_sp,
                                sp_crit_sed=config.sp_crit_sed,
                                sp_crit_br=config.sp_crit_br)

    while time <= config.total_model_time:

        #z[mg.core_nodes]+= (config.uplift_rate*config.dt_model) #do uplift all the time
        #rock[mg.core_nodes]+= (config.uplift_rate*config.dt_model)
                
        rock[mg.core_nodes]+= (config.uplift_rate*config.dt_model)
        z[:] = rock + soil
        #z[mg.core_nodes]+= (config.uplift_rate*config.dt_model) #do uplift all the time
        
        expweath.calc_soil_prod_rate()
        ddtd.run_one_step(config.dt_model)
        
        
        fr.run_one_step()
        
        #comment next line if is pulse climate
        space.run_one_step(config.dt_model)

        accumulate += desired_slip_per_event
        print('is accumulating')

        Mean_da= np.append(Mean_da, np.mean(mg.at_node['drainage_area']))
        Mean_soil= np.append(Mean_soil, np.mean(mg.at_node['soil__depth']))
        Mean_elev= np.append(Mean_elev, np.mean(mg.at_node['topographic__elevation']))

        if accumulate >= mg.dx:
            ss_fault(grid=mg, 
                     fault_loc_y=fault_loc_y, 
                     total_slip=config.total_slip,
                     total_time= config.total_model_time, 
                     method=config.method, 
                     accumulate=accumulate)
            accumulate = accumulate % mg.dx
            # expweath.maximum_weathering_rate=1*1e-6
            # expweath.calc_soil_prod_rate()
            # ddtd.run_one_step(dt=1250)
            quakes_times=np.append(quakes_times, time)
            print('one slip')
        
        # accumulate += desired_slip_per_event
        # print('is accumulating')

        # #comment next if when doing cont climate
        # if len(fluvial_times) > 0 and fluvial_idx < len(fluvial_times):  # Add safety check
        #     if time >= fluvial_times[fluvial_idx,0] and time <= fluvial_times[fluvial_idx,1]:
        #         print(f'is: {time} fluvial time, index {fluvial_idx}')
        #         #fr.run_one_step()
        #         space.run_one_step(config.dt_model)
        #         if time == fluvial_times[fluvial_idx,1] and fluvial_idx < (len(fluvial_times)-1):
        #             fluvial_idx += 1
        #             print(f"Incremented fluvial index to {fluvial_idx}")
        #         if fluvial_idx == (len(fluvial_times)-1):
        #             print("Reached last fluvial period")
        #             pass

        
        if time%1000 == 0: #time>0 and
            # fig1 = plt.figure(figsize=[8, 8])
            imshow_grid(mg, z, cmap='coolwarm', shrink=shrink, grid_units=['m', 'm'])
            plt.title('Topography after ' + str(int(config.total_steady_time + time)) + ' years')
            # plt.show()
            loop_topo_img  = f'{config.home_path}/{config.save_location}/{config.model_name}{get_file_sequence(int(config.total_steady_time + time), config)}.png'
            plt.savefig(
                loop_topo_img,
                dpi=300, facecolor='white'
            )
            add_file_to_writer(writer, loop_topo_img)

            plt.clf()
            
        if time%config.frequency_output == 0: # Removed time>0 condition to allow saving at t=0 if frequency allows
            if hasattr(config, 'save_format'):
                if config.save_format == 'pickle':
                    # Save grid state to memory (useful for current session, e.g. for final pickle dump if selected)
                    grid_states[time] = deepcopy(mg)
                    if pickle_output_filename:
                        print(f"Saving grid states to Pickle at time {time}...")
                        with open(pickle_output_filename, 'wb') as f: # Overwrites pickle file with full history
                            pickle.dump(grid_states, f)
                        print(f"Saved {len(grid_states)} grid states to {pickle_output_filename}")
                    else:
                        print("Pickle output filename not defined. Skipping save.")
                elif config.save_format == 'netcdf':
                    print(f"Saving grid state to NetCDF at time {time}...")
                    netcdf_output_dir = os.path.join(config.home_path, config.save_location, 'netcdf_outputs', config.model_name)
                    os.makedirs(netcdf_output_dir, exist_ok=True)
                    current_time_for_filename = int(config.total_steady_time + time)
                    sequence_str = get_file_sequence(current_time_for_filename, config)
                    netcdf_filename = os.path.join(netcdf_output_dir, f"{config.model_name}{sequence_str}.nc")

                    # Create a copy specifically for saving to avoid modifying the live grid
                    mg_to_save = deepcopy(mg)

                    # Ensure known integer fields are explicitly int64 in the copy before saving
                    int_fields_to_cast = ['flood_status_code', 'flow__link_to_receiver_node', 'flow__receiver_node', 'flow__upstream_node_order']
                    for field_name in int_fields_to_cast:
                        # Operate on the copy (mg_to_save)
                        if field_name in mg_to_save.at_node and mg_to_save.at_node[field_name] is not None:
                            try:
                                mg_to_save.at_node[field_name] = mg_to_save.at_node[field_name].astype(np.int64)
                            except Exception as e:
                                print(f"Warning: Could not cast {field_name} in the copy to np.int64 before periodic save: {e}")

                    try:
                        # Save the modified copy (mg_to_save)
                        write_netcdf(netcdf_filename, mg_to_save, names=netcdf_field_names, format="NETCDF3_64BIT")
                        print(f"Saved NetCDF to {netcdf_filename}")
                    except Exception as e:
                        print(f"Error saving NetCDF file {netcdf_filename}: {e}")
                else:
                    print(f"Unknown save_format: {config.save_format}. No data saved for time {time}.")
            else:
                print("Warning: config.save_format not set. No data saved for time {time}.")

        print(time)
        time = time + config.dt_model
        # print(time) 
    
    # Save final timestep
    grid_states[config.total_model_time] = deepcopy(mg) # Store final state in memory

    if hasattr(config, 'save_format'):
        if config.save_format == 'pickle':
            if pickle_output_filename:
                print(f"Final save of grid states to Pickle...")
                with open(pickle_output_filename, 'wb') as f:
                    pickle.dump(grid_states, f) # Dumps the entire history
                print(f"\nFinal save: {len(grid_states)} grid states saved to {pickle_output_filename}")
            else:
                print("Pickle output filename not defined. Skipping final pickle save.")
        elif config.save_format == 'netcdf':
            print(f"Final save of grid state to NetCDF...")
            netcdf_output_dir = os.path.join(config.home_path, config.save_location, 'netcdf_outputs', config.model_name)
            os.makedirs(netcdf_output_dir, exist_ok=True)
            final_time_for_filename = int(config.total_steady_time + config.total_model_time)
            sequence_str = get_file_sequence(final_time_for_filename, config)
            netcdf_filename = os.path.join(netcdf_output_dir, f"{config.model_name}{sequence_str}.nc")

            # Create a copy specifically for final saving
            mg_to_save = deepcopy(mg)

            # Ensure known integer fields are explicitly int64 in the copy before final saving
            int_fields_to_cast = ['flood_status_code', 'flow__link_to_receiver_node', 'flow__receiver_node', 'flow__upstream_node_order']
            for field_name in int_fields_to_cast:
                # Operate on the copy (mg_to_save)
                if field_name in mg_to_save.at_node and mg_to_save.at_node[field_name] is not None:
                    try:
                        mg_to_save.at_node[field_name] = mg_to_save.at_node[field_name].astype(np.int64)
                    except Exception as e:
                        print(f"Warning: Could not cast {field_name} in the copy to np.int64 before final save: {e}")

            try:
                # Save the modified copy (mg_to_save)
                write_netcdf(netcdf_filename, mg_to_save, names=netcdf_field_names, format="NETCDF3_64BIT")
                print(f"\nFinal NetCDF saved to {netcdf_filename}")
            except Exception as e:
                print(f"Error saving final NetCDF file {netcdf_filename}: {e}")
        else:
            print(f"Unknown save_format: {config.save_format}. No final data saved.")
    else:
        print("Warning: config.save_format not set. No final data saved.")
    
    imshow_grid(mg, z, cmap='coolwarm', shrink=shrink, grid_units=['m', 'm'])
    plt.title('Topography after ' + str(int(config.total_steady_time + config.total_model_time)) + ' years')
    loop_topo_img  = f'{config.home_path}/{config.save_location}/{config.model_name}{get_file_sequence(int(config.total_steady_time + config.total_model_time), config)}.png'
    plt.savefig(loop_topo_img,
                dpi=300, 
                facecolor='white'
                )
    add_file_to_writer(writer, loop_topo_img)
    plt.clf()

    # Save timeseries data and event data to text files
    tabular_output_dir = os.path.join(config.home_path, config.save_location, 'tabular_outputs', config.model_name)
    os.makedirs(tabular_output_dir, exist_ok=True)
    print(f"\nSaving tabular data to {tabular_output_dir}...")

    # Ensure Mean_da, Mean_elev, Mean_soil are numpy arrays
    mean_da_arr = np.array(Mean_da)
    mean_elev_arr = np.array(Mean_elev)
    mean_soil_arr = np.array(Mean_soil)
    iterations_arr = np.array(iterations) # Defined earlier in the script

    # Check if lengths match for stacking. They should if logic is correct.
    if not (len(iterations_arr) == len(mean_da_arr) == len(mean_elev_arr) == len(mean_soil_arr)):
        print("Warning: Length mismatch between iterations and mean values. Timeseries data might be misaligned.")
        # Fallback: Save them separately or adjust logic if this warning appears frequently

    # Save mean values timeseries
    try:
        timeseries_data = np.column_stack((iterations_arr, mean_da_arr, mean_elev_arr, mean_soil_arr))
        timeseries_header = "Time,Mean_Drainage_Area,Mean_Elevation,Mean_Soil_Depth"
        timeseries_filename = os.path.join(tabular_output_dir, f"{config.model_name}_timeseries_means.csv")
        np.savetxt(timeseries_filename, timeseries_data, delimiter=",", header=timeseries_header, comments='', fmt='%s') # Using %s to handle potential mixed types or ensure precision
        print(f"Saved timeseries means to {timeseries_filename}")
    except ValueError as e:
        print(f"Error stacking or saving timeseries data: {e}. Check array lengths and contents.")
        print(f"Iterations length: {len(iterations_arr)}")
        print(f"Mean_da length: {len(mean_da_arr)}")
        print(f"Mean_elev length: {len(mean_elev_arr)}")
        print(f"Mean_soil length: {len(mean_soil_arr)}")

    # Save quake times
    quakes_times_arr = np.array(quakes_times)
    quakes_header = "Quake_Time"
    quakes_filename = os.path.join(tabular_output_dir, f"{config.model_name}_quake_times.csv")
    np.savetxt(quakes_filename, quakes_times_arr, delimiter=",", header=quakes_header, comments='', fmt='%s')
    print(f"Saved quake times to {quakes_filename}")

    # Save fluvial event times
    fluvial_header = "Fluvial_Start_Time,Fluvial_End_Time"
    fluvial_filename = os.path.join(tabular_output_dir, f"{config.model_name}_fluvial_event_times.csv")
    np.savetxt(fluvial_filename, fluvial_times, delimiter=",", header=fluvial_header, comments='', fmt='%s')
    print(f"Saved fluvial event times to {fluvial_filename}")

    return mg
# print(time)
# fig2 = plt.figure(figsize=[8, 8])
# imshow_grid(mg, z, cmap='viridis', shrink=shrink)
# plt.title('Topography after ' + str(total_model_time) + ' years')
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
# final_topo_img= f'{home_path}faulting/output_model_run/{model_name}/{model_name}{total_model_time}.png'
# plt.savefig(final_topo_img, dpi=300, facecolor='white')
# add_file_to_writer(writer, final_topo_img)
# writer.close()

# fig3= plt.figure(figsize=[8,8])
# plt.plot(iterations, Mean_elev)
# plt.xlabel('time [years]')
# plt.ylabel('mean elevation[m]')
# plt.savefig(f'{home_path}faulting/output_model_run/{model_name}/{model_name}mean_elevation.jpg')
# fig4= plt.figure(figsize=[8,8])
# plt.plot(iterations, Mean_da)
# plt.xlabel('time [years]')
# plt.ylabel('mean drainage area [m2]')
# plt.savefig(f'{home_path}faulting/output_model_run/{model_name}/{model_name}mean_drainage.jpg')
# fig5= plt.figure(figsize=[8,8])
# plt.plot(iterations, Mean_soil)
# plt.xlabel('time [years]')
# plt.ylabel('mean soil_depth [m]')
# plt.savefig(f'{home_path}faulting/output_model_run/{model_name}/{model_name}mean_soil.jpg')

# np.savetxt(f'{home_path}faulting/output_model_run/{model_name}/{model_name}mean_elev.txt',
#            (Mean_elev),
#            delimiter=',',
#            header='Mean_elev',
#            comments='')
# np.savetxt(f'{home_path}faulting/output_model_run/{model_name}/{model_name}mean_da.txt',
#            (Mean_da),
#            delimiter=',',
#            header='Mean_da',
#            comments='')
# np.savetxt(f'{home_path}faulting/output_model_run/{model_name}/{model_name}mean_soil.txt',
#            (Mean_soil),
#            delimiter=',',
#            header='Mean_soil',
#            comments='')
# np.savetxt(f'{home_path}faulting/output_model_run/{model_name}/{model_name}quakes.txt',
#            (quakes_times),
#            delimiter=',',
#            header='quakes_times',
#            comments='')

# write_netcdf(f'{home_path}faulting/output_model_run/{model_name}/{model_name}final-topo.nc', mg, 
#         format='NETCDF3_64BIT', 
#         names=[
#             'bedrock__elevation',
#             'drainage_area',
#             'flood_status_code',
#             'flow__link_to_receiver_node',
#             'flow__receiver_node',
#             'flow__receiver_proportions',
#             'flow__upstream_node_order',
#             'soil__depth',
#             'soil_production__rate',
#             'surface_water__discharge',
#             'topographic__elevation',
#             'topographic__steepest_slope',
#             'water__unit_flux_in'
#         ]
#             )

# # try: 
# #     mg.save(f'/home/jupyter-taranguiz/StrikeSlip/faulting/output_topo/{model_name}/final-topo2.nc')
# # except Exception as e:
# #     print(str(e))

# # mg.save(f'/Users/taranguiz/Research/Lateral_advection/output_model_run/{model_name}/{model_name}_{total_model_time}.asc')

# # figsize = [16,4] # size of grid plots
# # fig, ax = plt.subplots(figsize=figsize)

# # x = mg.node_x[fault_nodes]
# # # soil_level = bed + soil
# #
# # ax.plot(x, mg.at_node['soil__depth'][fault_nodes], 'orange', linewidth=2, markersize=12, label='soil')
# # ax.plot(x, mg.at_node['bedrock__elevation'][fault_nodes], linewidth=2, markersize=12, label='bedrock')
# # ax.plot(x, mg.at_node['topographic__elevation'][fault_nodes],'red', linewidth=2, markersize=12, label='topo')
# #
# #
# # plt.title('Final cross-Profile topography at fault location')
# # #plt.text(480, 9, 'air')
# # #plt.text(480, 7, 'soil')
# # #plt.text(470, 2, 'bedrock')
# #
# # # plt.xlim(0,1000)
# # # plt.ylim(0, 10)
# # ax.set_xlabel('X (m)')
# # ax.set_ylabel('Depth (m)')
# # ax.legend(loc='upper right')
# # plt.show()

