import imageio
import numpy as np


def add_file_to_writer(writer, filename):
    image=imageio.imread(filename)
    writer.append_data(image)

def get_file_sequence(time, config):
    final_time = config.total_model_time + config.total_steady_time
    num_digits = len(str(final_time))
    file_sequence_num = time * 10 ** -num_digits
    float_convert = f'{{:.{num_digits}f}}'
    file_sequence_num = float_convert.format(file_sequence_num).split('.')[1]
    return file_sequence_num

def save_grid_state(grid, time):
    """
    Save current grid state including all available fields and time.
   
    Parameters
    ----------
    grid : RasterModelGrid
        The Landlab grid to save state from
    time : float
        Current model time
       
    Returns
    -------
    dict
        Dictionary containing all grid state fields
    """
    # Add time to grid fields
    grid.at_node['time'] = np.full_like(grid.at_node['topographic__elevation'], time)
   
    # Initialize state dictionary
    state = {}
   
    # Save all fields at node
    for field_name in grid.at_node.keys():
        state[field_name] = grid.at_node[field_name].copy()
       
    # Optionally save fields at other grid positions if they exist
    for position in ['at_link', 'at_patch', 'at_corner', 'at_face', 'at_cell']:
        if hasattr(grid, position):
            fields = getattr(grid, position)
            for field_name in fields.keys():
                state[f"{position}_{field_name}"] = fields[field_name].copy()
   
    # Clean up - remove time field from grid since we just added it
    grid.at_node.pop('time')
   
    return state

