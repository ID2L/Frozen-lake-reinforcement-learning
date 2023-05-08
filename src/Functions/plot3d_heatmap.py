
# importing required libraries
from typing import Dict, List
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pylab
def draw_3D(data: List[Dict[str, str | int]], x_key = "x", y_key = "y", z_key = "z", scalar_key = "value", color_key = None):
    # Project data
    vectorized_data = {
        "x": [], 
        "y": [], 
        "z": [], 
        "value": []
    }
    for item in data:
        assert x_key in item
        assert y_key in item
        assert z_key in item
        assert scalar_key in item
        if color_key is not None:
            assert color_key in item
        vectorized_data["x"].append(item[x_key])
        vectorized_data["y"].append(item[y_key])
        vectorized_data["z"].append(item[z_key])
        vectorized_data["value"].append(item[scalar_key])
        pass

    # TODO adapt https://www.geeksforgeeks.org/3d-heatmap-in-python/ 
    figure = plt.figure(figsize=(10, 10))
    # axes = figure.add_subplot(111, projection='3d')
    axes = Axes3D(figure)
    cmap = cm.get_cmap('Greys')
    color_map = cm.ScalarMappable(cmap=cmap)
    color_map.set_array(vectorized_data["value"])
    # creating the heatmap
    img = axes.scatter(vectorized_data["x"],
                       vectorized_data["y"],
                       vectorized_data["z"],
                       color='green')
    plt.colorbar(color_map)
    # adding title and labels
    axes.set_title("3D Heatmap")
    axes.set_xlabel('X-axis')
    axes.set_ylabel('Y-axis')
    axes.set_zlabel('Z-axis')
    
    # displaying plot
    plt.show()
    pass