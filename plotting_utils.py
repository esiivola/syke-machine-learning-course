import numpy as np # numpy comes from "numeric python" and it is a very popular library for numerical operations
import pandas as pd # pandas provides more advanced data structures and methods for manipulating data
import matplotlib.pyplot as plt # a widely used visualiation library
import matplotlib
import cartopy.crs as ccrs # a geographic stuffwhich we use for plotting
import cartopy

import requests # for querying the data from internet
import io # for io operations
import urllib # for building the query string
import urllib.parse # --||--

import sklearn.preprocessing # sklearn is a good library for doing basic machine learning, in addition to that, it contains some neat preprocessing stuff

import seaborn as sns


from sklearn.metrics import roc_curve, auc

from shapely.geometry import Point

from typing import List, Set, Dict, Tuple, Optional, Any, Callable

SUBPLOTWIDTH = 4
SUBPLOTHEIGHT = 2.5

MAPHEIGHT = 10
MAPWIDTH = 10

# Bounds in lon_min, lat_min, lon_max, lat_max
BOUNDS_BALTIC_SEA = [52.5,5,67.5,40]
BOUNDS_NORTHERN_BALTIC_SEA = [58,12,65.5,40]

def get_empty_axes(nplots:int, ncols:int):
    """
    Returns figure and axis object for subplots
    
    Parameters:
    nplots (int): number of subplots
    ncols (int): number of columns in the subplot
    
    Returns:
    fig (matplotlib.pyplot.Figure): Figure
    axs (matplotlib.axes.Axes): Axes
    """
    fig, axs = plt.subplots(int(np.ceil(nplots/ncols) ), ncols, constrained_layout=True, figsize=(ncols*SUBPLOTWIDTH, int(np.ceil(nplots/ncols) )*SUBPLOTHEIGHT), squeeze=False)
    
    for ax in axs.ravel():
        ax.set_visible(False)
    return fig, axs
    
def plot_histogram(data_numpy:np.ndarray, xlabel:str, bins:str=40, label:str=None, color:str='grey', ax:matplotlib.axes.Axes=None):
    """
    Visualizes data as a histogram
    
    Parameters:
    data_numpy (numpy.ndarray): numpy ndarray of shape Nx1
    xlabel (str): text that is displayed under the x axis
    bins (int): number of bins in the histogram (default 40)
    label (str): title of the histogram (default None)
    color (str): color of the barplot
    ax (matplotlib.axes.Axes): Axes to be used for plotting (default: create new)
    
    Returns:
    None
    """
    if ax==None:
        ax = plt.gca()
    ax.hist(data_numpy, bins=bins, color=color, label=label)
    ax.set_yticks([])
    ax.set_ylabel("Frequency")
    ax.set_xlabel(xlabel)
    ax.tick_params(left = False, bottom = False)

def plot_histograms(data: pd.DataFrame, plotted_columns:List[Tuple[str,str]]=[], ncols:int=3):
    """
    Visualizes data as histograms that are presented in an array. Parameter data contains all data and plotted columns defines the columns of that data that are included in teh figure.
    
    Parameters:
    data (pandas.DataFrame): Pandas Dataframe which columns are visualized as histograms
    plotted_columns (List[Tuple[str,srt]]): List of tuples containing the visualized columns. The first item of each tuple indicates the name of the column in the pandas dataframe and the second item indicates the visualized name in the histogram
    ncols (int): number of columns in the histogram array, defaults to 3.
    Returns:
    None
    """
    if len(plotted_columns)==0:
        plotted_columns = [(col,col) for col in data.columns]
    nplots = len(plotted_columns)
    fig, axs = get_empty_axes(nplots, ncols)
    for i, (variable, name) in enumerate(plotted_columns):
        row, col = i // ncols, i % ncols
        axs[row,col].set_visible(True)
        plot_histogram(data[variable], name, ax=axs[row,col])

def plot_region_in_map(bounding_box:Tuple[int,int,int,int], bounding_box_context:Tuple[int,int,int,int]=BOUNDS_BALTIC_SEA, projection:ccrs.CRS=ccrs.EuroPP(), coastline:bool=True, stock_img:bool=True, title:str=None):
    """
    The function plots a map in a specific projection (parameter projection) so that the map is limited by bounding_box_context and within those bounds bounding_box is visualized with a solid line
    box = lon_min, lat_min, lon_max, lat_max
    Parameters:
    bounding_box (Tuple[int,int,int,int]): a Tuple with (longitude min, latitude min, longitude max, latitude max) for the visualized bounding box
    bounding_box_context (Tuple[int,int,int,int]): a Tuple with (longitude min, latitude min, longitude max, latitude max) for the bounds of the plot
    projection (ccrs.CRS): Projection used in plotting. defaults to ccrs.EuroPP
    coastline (bool): If the plot visualizes the coastline
    stock_img (bool): If the plot visualizes the background
    title (str): The title of the plot
    Returns:
    matplotlib.figure that contains the region
    """
    fig = plt.figure(figsize=(MAPHEIGHT, MAPWIDTH))
    ax = plt.axes(projection = projection)
    if stock_img:
        ax.stock_img()
    if coastline:
        ax.coastlines(resolution='50m')
    if bounding_box is not None:
        lats = np.array([bounding_box[1], bounding_box[1], bounding_box[3], bounding_box[3], bounding_box[1]])
        lons = np.array([bounding_box[0], bounding_box[2], bounding_box[2], bounding_box[0], bounding_box[0]])

        plt.plot(lons, lats, transform=ccrs.PlateCarree(), c='k')

    proj_coords = projection.transform_points(ccrs.PlateCarree(), np.array([bounding_box_context[1],bounding_box_context[3]]), np.array([bounding_box_context[0], bounding_box_context[2]]))
    ax.set_ylim([proj_coords[0,1], proj_coords[1,1]])
    ax.set_xlim([proj_coords[0,0], proj_coords[1,0]])

    ax.gridlines(draw_labels=True)
    if title is not None:
        plt.title(title)
    return fig

def scatterplot_in_map(longitudes:np.array, latitudes:np.array, bounding_box:Tuple[int,int,int,int], bounding_box_context:Tuple[int,int,int,int]=BOUNDS_BALTIC_SEA, projection:ccrs.CRS=ccrs.EuroPP(), coastline:bool=True, stock_img:bool=True, title:str=None, **kwargs):
    """
    The function plots scatterplot within a map in a specific projection (parameter projection) so that the map is limited by bounding_box_context and within those bounds bounding_box is visualized with a solid line
    box = lon_min, lat_min, lon_max, lat_max
    Parameters:
    longitudes (np.array): a Nx1 array containing the longitudes
    latitudes (np.array): a Nx1 array containing the latitudes
    bounding_box (Tuple[int,int,int,int]): a Tuple with (longitude min, latitude min, longitude max, latitude max) for the visualized bounding box
    bounding_box_context (Tuple[int,int,int,int]): a Tuple with (longitude min, latitude min, longitude max, latitude max) for the bounds of the plot
    projection (ccrs.CRS): Projection used in plotting. defaults to ccrs.EuroPP
    coastline (bool): If the plot visualizes the coastline
    stock_img (bool): If the plot visualizes the background
    title (str): The title of the plot
    **kwargs: extra parameters to be passed to the matplotlib pyplot scatter-function 
    Returns:
    matplotlib.figure that contains the region
    """
    plot_region_in_map(bounding_box, bounding_box_context=bounding_box_context, projection=projection,coastline=coastline, stock_img=stock_img, title=title)
    plt.scatter(longitudes, latitudes, transform=ccrs.PlateCarree(), **kwargs)

def plot_class_marginals(data: pd.DataFrame, class_column: str, plotted_columns:List[Tuple[str,str]] = [], ncols:int=3, flip_xy:bool=False, **kwargs):
    """
    Visualizes  the marginal distributions of each class for different columns in the data. Parameter data contains all data and plotted columns defines the columns of that data that are included in the figure. parameter class_column defines the column which contains the class labels. The marginals in each subplot are plotted for all classes separately.
    
    Parameters:
    data (pandas.DataFrame): Pandas Dataframe which columns are visualized as histograms
    class_column (int): The column in data that contains the class labels
    plotted_columns (List[Tuple[str,srt]]): List of tuples containing the visualized columns. The first item of each tuple indicates the name of the column in the pandas dataframe and the second item indicates the visualized name 
    ncols (int): number of columns in the histogram array, defaults to 3.
    flip_xy (bool): default behaviour is to plot different classes on y-axis
    **kwargs: extra parameters to be passed to the seaborn stripplot-function 
    Returns:
    None
    """
    if len(plotted_columns)==0:
        plotted_columns = [(col, col) for col in data.columns if col not in [class_column]]
    
    nplots = len(plotted_columns)
    fig, axs = get_empty_axes(nplots, ncols)
 
    for i, (variable, name) in enumerate(plotted_columns):
        row, col = i // ncols, i % ncols
        axs[row,col].set_visible(True)
        if not flip_xy:
            sns.stripplot(x=class_column, y=variable, data=data, ax=axs[row,col], **kwargs)
        else:
            sns.stripplot(y=class_column, x=variable, data=data, ax=axs[row,col], **kwargs)
            if col != 0:
                axs[row, col].set_ylabel(None)

def plot_residuals_classification(data: pd.DataFrame, y_pred: np.ndarray, y_true: np.ndarray, plotted_columns:List[Tuple[str,str]]=[], label_names:List[Tuple[int,str]]=None , ncols:int=3, **kwargs):
    """
    Visualizes the predicted classes as a function of one column at a time so that the tru labels are visualized with different colors. The purpose of the function is to help find biased predictions for different classes.
    
    Parameters:
    data (pd.DataFrame): Data of size N x D the predictions have been made with
    y_pred (np.ndarray): N x 1 predictions for the data
    y_true (np.ndarray): N x 1 true labels for the data
    plotted_columns (List[Tuple[str,srt]]): List of tuples containing the visualized columns. The first item of each tuple indicates the name of the column in the pandas dataframe and the second item indicates the visualized name
    label_names (List[Tuple[int, str]]): List of tuples with label names. The first member of the tuple contains the integers differentiating the classes and the second member contains the name of that class
    ncols (int): number of columns in the histogram array, defaults to 3.
    Returns:
    None
    """
    if label_names is None:
        label_names = [ (i,i) for i in np.unique(y_true)]
    if len(plotted_columns)==0:
        plotted_columns = [(col, col) for col in data.columns]
    
    nplots = len(plotted_columns)
    fig, axs = get_empty_axes(nplots, ncols)
    
    colors = sns.color_palette()
    for i, (variable, name) in enumerate(plotted_columns):
        row, col = i // ncols, i % ncols
        axs[row,col].set_visible(True)
        for ij, (c , label) in enumerate(label_names):
            axs[row, col].scatter(data[variable][y_true==c], y_pred[y_true==c] + np.random.uniform(low=-0.1, high=0.1, size=y_pred[y_true==c].shape), color=colors[ij], label=label, **kwargs)
        
        axs[row, col].set_yticks([t[0] for t in label_names])
        axs[row, col].set_yticklabels([t[1] for t in label_names])
        axs[row, col].set_xlabel(name)
        if col==0:
            axs[row, col].set_ylabel("Predicted label")
        if i==len(plotted_columns)-1:
            axs[row, col].legend(title="True label")

def plot_scatter(data: pd.DataFrame, columns_x:List[Tuple[str,str]] = [], columns_y:List[Tuple[str,str]] = [], ncols:int=3, **kwargs):
    """
    Visualizes the data in scatterplot. The visualized columns of data are defined in columns_x and columns_y.
    
    Parameters:
    data (pd.DataFrame): Data that is visualized
    columns_x (List[Tuple[str,str]]): list that defines the data on x-axises. The first item of each tuple in the list defines the column name and the second item defines the visualized name. If the list contains only one item, same column is used in all figures
    columns_y (List[Tuple[str,str]]): list that defines the data on y-axises. The first item of each tuple in the list defines the column name and the second item defines the visualized name. If the list contains only one item, same column is used in all figures
    ncols (int): number of columns in the histogram array, defaults to 3.
    **kwargs: extra parameters to be passed to the matplotlib pyplot scatter-function
    Returns:
    None
    """
    if len(columns_y)==0 and len(columns_x)==0:
        raise ValueError("Both columns_x and columns_y cannot be empty")
    
    if len(columns_x)==1 and len(columns_y)>0:
        columns_x = [(columns_x[0][0], columns_x[0][1]) for _ in range(len(columns_y)) ]
    elif len(columns_y)==1 and len(columns_x)>0:
        columns_y = [(columns_y[0][0], columns_y[0][1]) for _ in range(len(columns_x)) ]
    
    nplots = len(columns_x)
    fig, axs = get_empty_axes(nplots, ncols)
    
    for i, ((col_x, label_x), (col_y, label_y)) in enumerate(zip(columns_x, columns_y)):
        row, col = i // ncols, i % ncols
        axs[row,col].set_visible(True)
        axs[row,col].scatter(data[col_x], data[col_y],**kwargs)
        axs[row,col].set_ylabel(label_y)
        axs[row,col].set_xlabel(label_x)

def plot_effects(data: pd.DataFrame, alg:Callable, input_columns:List[str],
                 normalizer:Callable, normalized_columns:List[str], plotted_columns:List[Tuple[str,str]]=[],
                 periodic_columns:List[str]=[], ncols:int=3, coordinates:Tuple[str,str]=None,
                 bounding_box:Tuple[int,int,int,int]=None, bounding_box_context:Tuple[int,int,int,int]=BOUNDS_NORTHERN_BALTIC_SEA, 
                 projection:ccrs.CRS=ccrs.EuroPP(), coastline:bool=False, stock_img:bool=False):
    """
    Visualizes the predictions with respect to different columns by varying one column at a time and setting the rest of the columns to their median values.
    
    Parameters:
    data (pd.DataFrame): Data that is visualized
    alg (Callable): The prediction algorithm 
    input_columns (List[str]): Names of the columns that the prediction algorithm takes as an input
    plotted_columns (List[Tuple[str,str]]): List of tuples containing the visualized columns. The first item of each tuple indicates the name of the column in the pandas dataframe and the second item indicates the visualized name
    normalizer (Callable): The normalizer object that normalizes the data
    normalized_columns (List[str]): The columns that are normalized
    periodic_columns (List[str]): The columns that are periodic in nature
    ncols (int): number of columns in the histogram array, defaults to 3.
    coordinates (Tuple[str,str]): names of the columns defining latitudes and longitudes
    bounding_box (Tuple[int,int,int,int]): a Tuple with (longitude min, latitude min, longitude max, latitude max) for the visualized bounding box
    bounding_box_context (Tuple[int,int,int,int]): a Tuple with (longitude min, latitude min, longitude max, latitude max) for the bounds of the plot
    projection (ccrs.CRS): Projection used in plotting. defaults to ccrs.EuroPP
    coastline (bool): If the plot visualizes the coastline
    stock_img (bool): If the plot visualizes the background
    title (str): The title of the plot
    Returns:
    matplotlib.figure that contains the region
    """
    if len(plotted_columns)==0:
        tmp = coordinates or []
        plotted_columns = [(col, col) for col in data.columns if col not in tmp]
    
    data_median = np.median(data,axis=0)
    
    #Set the periodic medians 
    for col in periodic_columns:
        ind = list(data.columns).index(col)
        ind_cos, ind_sin = list(data.columns).index("COS"+col), list(data.columns).index("SIN"+col)
        data_median[ind_cos] = np.cos(data_median[ind]*2*np.pi) 
        data_median[ind_sin] = np.sin(data_median[ind]*2*np.pi)
    
    if coordinates is not None:
        if len(coordinates) != 2:
            raise(ValueError)
        if bounding_box is None:
            bounding_box = bounding_box_context

        LAT, LON = np.meshgrid(np.linspace(bounding_box[1], bounding_box[3], num=100),
                               np.linspace(bounding_box[0], bounding_box[2], num=100))
        sh = LAT.shape
        X_ = np.tile(data_median, (sh[0]*sh[1],1))
        X_ = pd.DataFrame(X_, columns=data.columns)
        X_[coordinates[0]] = LAT.reshape(-1)
        X_[coordinates[1]] = LON.reshape(-1)
        X_[normalized_columns] = normalizer.transform(X_[normalized_columns])
        Y_pred = alg.predict(X_[input_columns]).reshape(sh)
        Y_pred = Y_pred- np.min(Y_pred)
        
        fig = plt.figure(figsize=(MAPHEIGHT, MAPWIDTH))
        ax = plt.axes(projection = projection)
        if stock_img:
            ax.stock_img()
        if coastline:
            ax.coastlines(resolution='50m')

        proj_coords = projection.transform_points(ccrs.PlateCarree(), np.array([bounding_box_context[1],bounding_box_context[3]]), np.array([bounding_box_context[0], bounding_box_context[2]]))
        ax.set_ylim([proj_coords[0,1], proj_coords[1,1]])
        ax.set_xlim([proj_coords[0,0], proj_coords[1,0]])

        ax.gridlines(draw_labels=True)
        
        #plot_region_in_map(bounding_box, bounding_box_context=bounding_box_context, projection=projection,coastline=coastline, stock_img=coastline)
    
        land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='gray',
                                                    facecolor='white')

        plt.contourf( LON, LAT, Y_pred, transform=ccrs.PlateCarree(), cmap='gray')
        ax.add_feature(land_50m)
        
        #scatterplot_in_map(LON, LAT, Y_pred, bounding_box_context=plotting_utils.BOUNDS_NORTHERN_BALTIC_SEA)
        cbar = plt.colorbar(fraction=0.03, pad=0.1)
        cbar.set_label('Effect on output', rotation=270)
        
    
    nplots = len(plotted_columns)
    fig, axs = get_empty_axes(nplots, ncols)
    for i, (column, column_label) in enumerate(plotted_columns):
        row, col = i // ncols, i % ncols
        axs[row,col].set_visible(True)
        
        x = np.linspace(np.min(data[column]), np.max(data[column]), num=100)
        X_ = np.tile(data_median, (x.shape[0],1))
        X_ = pd.DataFrame(X_, columns=data.columns)
        X_[column] = x
        if column in periodic_columns:
            X_['SIN'+column] = np.sin(2*np.pi*x)
            X_['COS'+column] = np.cos(2*np.pi*x)
        X_[normalized_columns] = normalizer.transform(X_[normalized_columns])
        y_pred = alg.predict(X_[input_columns]).reshape(-1)
        axs[row, col].plot(x, y_pred-np.min(y_pred), c='k')
        axs[row, col].set_xlabel(column_label)
        axs[row, col].set_ylabel('Effect on output')

def draw_binary_roc(y:np.ndarray , y_score:np.ndarray, **kwargs):
    """
    Visualizes the ROC curve
    
    Parameters:
    y (np.ndarray): The true labels
    y_score (np.ndarray): Confidence of class prediction
    """
    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc = auc(fpr, tpr) 
    # Plot the roc-curve
    plt.figure(figsize=(SUBPLOTWIDTH,SUBPLOTHEIGHT))
    plt.plot(fpr, tpr, 'k',
             label='ROC curve (area under curve = {b:0.2f})'.format( b=roc_auc),
             **kwargs)
    plt.plot([0, 1], [0, 1], ':k')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
