import numpy as np # numpy comes from "numeric python" and it is a very popular library for numerical operations
import pandas as pd # pandas provides more advanced data structures and methods for manipulating data
import matplotlib.pyplot as plt # a widely used visualiation library
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


SUBPLOTWIDTH = 4
SUBPLOTHEIGHT = 2.5

MAPHEIGHT = 10
MAPWIDTH = 10

# Bounds in lon_min, lat_min, lon_max, lat_max
BOUNDS_BALTIC_SEA = [52.5,5,67.5,40]
BOUNDS_NORTHERN_BALTIC_SEA = [58,12,65.5,40]

def get_empty_axes(nplots, ncols):
    fig, axs = plt.subplots(int(np.ceil(nplots/ncols) ), ncols, constrained_layout=True, figsize=(ncols*SUBPLOTWIDTH, int(np.ceil(nplots/ncols) )*SUBPLOTHEIGHT), squeeze=False)
    
    for ax in axs.ravel():
        ax.set_visible(False)
    return fig, axs
    
def plot_histogram(data_numpy, xlabel, bins=40, label=None, color='grey', ax=None):
    """
    TODO: write documentation
    """
    if ax==None:
        ax = plt.gca()
    ax.hist(data_numpy, bins=bins, color=color, label=label)
    ax.set_yticks([])
    ax.set_ylabel("Frequency")
    ax.set_xlabel(xlabel) #title(title)
    ax.tick_params(left = False, bottom = False)

def plot_histograms(data, plotted_columns=[], ncols=3):
    """
    TODO: write documentation
    """
    if len(plotted_columns)==0:
        plotted_columns = [(col,col) for col in data.columns]
    nplots = len(plotted_columns)
    fig, axs = get_empty_axes(nplots, ncols)
    for i, (variable, name) in enumerate(plotted_columns):
        row, col = i // ncols, i % ncols
        axs[row,col].set_visible(True)
        plot_histogram(data[variable], name, ax=axs[row,col])

def plot_region_in_map(bounding_box, bounding_box_context=BOUNDS_BALTIC_SEA, projection=ccrs.EuroPP(), coastline=True, stock_img=True, title=None):
    """
    box = lon_min, lat_min, lon_max, lat_max
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

def scatterplot_in_map(longitudes, latitudes, bounding_box=None, bounding_box_context=BOUNDS_BALTIC_SEA, projection=ccrs.EuroPP(), coastline=True, stock_img=True, title=None, **kwargs):
    """
    TODO: write documentation
    """
    plot_region_in_map(bounding_box, bounding_box_context=bounding_box_context, projection=projection,coastline=coastline, stock_img=stock_img, title=title)
    plt.scatter(longitudes, latitudes, transform=ccrs.PlateCarree(), **kwargs)

def plot_class_kdes(data, class_column, plotted_columns = [], ncols=3, **kwargs):
    if len(plotted_columns)==0:
        plotted_columns = [(col, col) for col in data.columns if col not in [class_column]]
    
    nplots = len(plotted_columns)

    fig, axs = get_empty_axes(nplots, ncols)
    
    for i, (variable, name) in enumerate(plotted_columns):
        row, col = i // ncols, i % ncols
        axs[row,col].set_visible(True)
        
        if i==nplots-1:
            legend = True
        else:
            legend = False
        if (data[variable].fillna(-9999) % 1  == 0).all() and np.unique(data[variable][~np.isnan(data[variable])]).shape[0]<10: # check if all variables as discrete
            sns.histplot(data=data, x=variable, ax = axs[row,col], hue=class_column, fill=True, common_norm=True, element="bars",
                                alpha=.5, linewidth=0, legend=legend, discrete=True)
            uniques = np.unique(data[variable][~np.isnan(data[variable])])
            axs[row,col].set_xticks(np.arange(np.min(uniques), np.max(uniques)+1))
        else:
            sns.kdeplot(data=data, x=variable, ax = axs[row,col], hue=class_column, fill=True, common_norm=True,
                    alpha=.5, linewidth=0, legend=legend)
        axs[row, col].set_yticks([])
        if col != 0:
            axs[row, col].set_ylabel(None)

def plot_class_marginals(data, class_column, plotted_columns = [], ncols=3, flip_xy=False, **kwargs):
    """
    TODO: write documentation
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

def plot_residuals_classification(data, y_pred, y_true, plotted_columns =[], label_names=None , ncols=3, **kwargs):
    """
    TODO: write documentation
    """
    if label_names is None:
        label_names = np.unique(y_true)
    if len(plotted_columns)==0:
        plotted_columns = [(col, col) for col in data.columns]
    
    nplots = len(plotted_columns)
    fig, axs = get_empty_axes(nplots, ncols)
    
    colors = sns.color_palette()
    for i, (variable, name) in enumerate(plotted_columns):
        row, col = i // ncols, i % ncols
        axs[row,col].set_visible(True)
        for ij, j in enumerate(np.unique(y_true)):
            axs[row, col].scatter(data[variable][y_true==j], y_pred[y_true==j] + np.random.uniform(low=-0.1, high=0.1, size=y_pred[y_true==j].shape), color=colors[ij], label=label_names[ij], **kwargs)
        axs[row, col].set_yticks(np.unique(y_true))
        axs[row, col].set_yticklabels(label_names)
        axs[row, col].set_xlabel(name)
        if col==0:
            axs[row, col].set_ylabel("Predicted label")
        if i==len(plotted_columns)-1:
            axs[row, col].legend(title="True label")

def plot_scatter(data, columns_x = [], columns_y = [], ncols=3, **kwargs):
    """
    TODO: write documentation
    """
    if len(columns_y)==0 and len(columns_x)==0:
        raise ValueError("Both columns_x and columns_y cannot be empty")
    
    if len(columns_x)==0:
        columns_x = [(col, col) for col in data.columns if col not in [t[0] for t in columns_y]]
    elif len(columns_y)==0:
        columns_y = [(col, col) for col in data.columns if col not in [t[0] for t in columns_x]]
    
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

def plot_effects(data, alg, input_columns,
                      normalizer, normalized_columns, plotted_columns=[],
                      periodic_columns=[], ncols=3, coordinates=None,
                      bounding_box=None, bounding_box_context=BOUNDS_NORTHERN_BALTIC_SEA, 
                      projection=ccrs.EuroPP(), coastline=False, stock_img=False):
    """
    TODO: write documentation
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

def draw_binary_roc(y, y_score, **kwargs):
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
