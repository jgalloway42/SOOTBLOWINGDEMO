import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from itertools import chain
import os
from datetime import datetime
import joblib
import logging

"""
Data Analysis Utility Functions

A comprehensive utility module providing helper functions for data analysis, 
visualization, and logging operations. This module streamlines common data 
science workflows with reusable functions for DataFrame operations, plotting, 
and file management.

Author: [Your Name]
Created: [Date]
Version: 1.0
"""

# =============================================================================
# General Helpers
# =============================================================================

def search_columns(search_str, df):
    """
    Search for DataFrame columns containing a specified substring (case-insensitive).
    
    This function performs a case-insensitive search across all column names
    in a DataFrame and returns those that contain the specified search string.
    
    Parameters
    ----------
    search_str : str
        The substring to search for in column names. Search is case-insensitive.
    df : pandas.DataFrame
        The DataFrame whose columns will be searched.
    
    Returns
    -------
    list of str
        List of column names that contain the search string.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'Price_USD': [1,2], 'sale_price': [3,4], 'quantity': [5,6]})
    >>> search_columns('price', df)
    ['Price_USD', 'sale_price']
    
    >>> search_columns('PRICE', df)
    ['Price_USD', 'sale_price']
    
    Notes
    -----
    - The search is case-insensitive
    - Returns empty list if no matches found
    - Useful for exploring unknown datasets or filtering columns for analysis
    """
    return list(filter(lambda x: search_str.upper() in x.upper(), [y for y in df.columns]))


def filter_list(search_str, searchable_list):
    """
    Filter a list for items containing a specified substring (case-insensitive).
    
    This function searches through a list and returns only items that contain
    the specified search string, regardless of case.
    
    Parameters
    ----------
    search_str : str
        The substring to search for in list items. Search is case-insensitive.
    searchable_list : list
        The list to be filtered.
    
    Returns
    -------
    list
        New list containing only items that contain the search string.
    
    Examples
    --------
    >>> features = ['age', 'salary', 'Age_Group', 'department']
    >>> filter_list('age', features)
    ['age', 'Age_Group']
    
    >>> filter_list('AGE', features)
    ['age', 'Age_Group']
    
    Notes
    -----
    - The search is case-insensitive
    - Returns empty list if no matches found
    - Original list is not modified
    """
    return list(filter(lambda x: search_str.upper() in x.upper(), [y for y in searchable_list]))


def flatten_list(list_of_lists):
    """
    Flatten a nested list structure into a single-level list.
    
    This function takes a list containing sublists and returns a single
    flat list with all elements from the sublists.
    
    Parameters
    ----------
    list_of_lists : list of lists
        A list where each element is itself a list.
    
    Returns
    -------
    list
        A flattened single-level list containing all elements from sublists.
    
    Examples
    --------
    >>> nested = [[1, 2], [3, 4], [5, 6]]
    >>> flatten_list(nested)
    [1, 2, 3, 4, 5, 6]
    
    >>> nested = [['a', 'b'], ['c'], ['d', 'e', 'f']]
    >>> flatten_list(nested)
    ['a', 'b', 'c', 'd', 'e', 'f']
    
    Notes
    -----
    - Only flattens one level deep
    - Empty sublists are handled gracefully
    - Uses itertools.chain for efficient flattening
    """
    return list(chain(*list_of_lists))


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_df(data, cols_list, save_to_path=None, figsize=(15, 20), linestyle='none', marker=','):
    """
    Create subplot visualizations for specified DataFrame columns with optional file caching.
    
    This function creates individual subplots for each specified column and can
    save/load plots to/from disk for performance optimization. If a saved plot
    exists, it loads the existing plot instead of regenerating it.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The source DataFrame containing the data to plot.
    cols_list : list of str
        List of column names from the DataFrame to plot. Each column gets its own subplot.
    save_to_path : str, optional
        File path for saving/loading plots. If None, plot is displayed without saving.
        If path exists, loads existing plot. If path doesn't exist, saves new plot.
    figsize : tuple of (int, int), optional
        Figure size as (width, height) in inches. Default is (15, 20).
    linestyle : str, optional
        Line style for the plots. Default is 'none' (scatter plot style).
    marker : str, optional
        Marker style for data points. Default is ',' (pixel marker).
    
    Returns
    -------
    str
        Status message indicating the operation performed:
        - 'Read From File: {path}' if existing plot was loaded
        - 'Saved to File: {path}' if new plot was created and saved
        - 'graph not saved to file...' if plot was created but not saved
    
    Examples
    --------
    >>> # Create subplots without saving
    >>> status = plot_df(df, ['col1', 'col2', 'col3'])
    'graph not saved to file...'
    
    >>> # Create and save subplots
    >>> status = plot_df(df, ['temperature', 'humidity'], save_to_path='weather.png')
    'Saved to File: weather.png'
    
    >>> # Load existing plot (if weather.png already exists)
    >>> status = plot_df(df, ['temperature', 'humidity'], save_to_path='weather.png')
    'Read From File: weather.png'
    
    Notes
    -----
    - Uses matplotlib for plotting
    - Each column gets its own subplot in a vertical arrangement
    - File caching helps avoid regenerating identical plots
    - Supports various matplotlib line styles and markers
    """
    if save_to_path:
        if os.path.exists(save_to_path):
            plt.imshow(plt.imread(save_to_path), aspect='auto', interpolation='nearest')
            return f'Read From File: {save_to_path}'
        else:
            _ = data[cols_list].plot(subplots=True, linestyle=linestyle,
                                   marker=marker, figsize=figsize)
            plt.savefig(save_to_path)
            return f'Saved to File: {save_to_path}'
    else:
        _ = data[cols_list].plot(subplots=True, linestyle=linestyle, marker=',', figsize=(15, 30))
        return 'graph not saved to file...'


def plotly_graph(df, cols_to_graph, left_legend=False, save_to_path=None):
    """
    Create interactive line plots using Plotly Express.
    
    This function creates interactive line plots with zoom, pan, and hover
    capabilities. The plot can be displayed in the notebook or saved as
    an HTML file for sharing.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The source DataFrame containing the data to plot. Index is used as x-axis.
    cols_to_graph : list of str
        List of column names to include as separate lines in the plot.
    left_legend : bool, optional
        If True, positions the legend on the left side of the plot.
        If False (default), uses default legend positioning.
    save_to_path : str, optional
        HTML file path for saving the interactive plot. If None, displays
        the plot in the current environment (e.g., Jupyter notebook).
    
    Returns
    -------
    None
        This function doesn't return a value. It either displays the plot
        or saves it to the specified file path.
    
    Examples
    --------
    >>> # Display interactive plot
    >>> plotly_graph(stock_data, ['AAPL', 'GOOGL', 'MSFT'])
    
    >>> # Save interactive plot with left legend
    >>> plotly_graph(stock_data, ['AAPL', 'GOOGL'], 
    ...               left_legend=True, save_to_path='stocks.html')
    ...saving graph to stocks.html
    
    Notes
    -----
    - Creates interactive plots with hover information
    - Uses DataFrame index as x-axis values
    - HTML files can be opened in any web browser
    - Plotly plots support zoom, pan, and data point inspection
    - Left legend positioning is useful for plots with many series
    """
    f = px.line(data_frame=df, x=df.index, y=cols_to_graph)
    if left_legend:
        f.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    if save_to_path:
        print(f'...saving graph to {save_to_path}')
        f.write_html(save_to_path)
    else:
        f.show()


# =============================================================================
# Logger and Saves
# =============================================================================

def save_joblib(object, folder_path, file_name, add_timestamp=False):
    """
    Save Python objects using joblib with optional timestamp functionality.
    
    This function serializes and saves Python objects (such as trained models,
    processed datasets, or analysis results) using joblib. Optionally adds
    a timestamp to the filename for version control.
    
    Parameters
    ----------
    object : any
        The Python object to be serialized and saved. Can be any pickleable
        object including models, DataFrames, lists, dictionaries, etc.
    folder_path : str
        The directory path where the file will be saved. Directory must exist.
    file_name : str
        The base filename for the saved object. Should include file extension.
    add_timestamp : bool, optional
        If True, appends a timestamp to the filename in format YYYY_MM_DD_HH_MM_SS.
        If False (default), uses the original filename.
    
    Returns
    -------
    None
        Prints a confirmation message with the full file path.
    
    Examples
    --------
    >>> # Save without timestamp
    >>> save_joblib(my_model, './models', 'random_forest.pkl')
    File Saved: ./models/random_forest.pkl
    
    >>> # Save with timestamp
    >>> save_joblib(processed_data, './data', 'clean_data.pkl', add_timestamp=True)
    File Saved: ./data/clean_data_2024_03_15_14_30_45.pkl
    
    Notes
    -----
    - Uses joblib for efficient serialization, especially good for scikit-learn models
    - Timestamp format: YYYY_MM_DD_HH_MM_SS
    - Preserves original file extension when adding timestamp
    - Folder must exist before saving
    - Joblib is more efficient than pickle for NumPy arrays and scikit-learn models
    """
    if add_timestamp:
        current_timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        root, ext = os.path.splitext(file_name)
        file_name = root + '_' + current_timestamp + ext  # ext retains the .xxxx including the .
    joblib.dump(object, os.path.join(folder_path, file_name))
    print(f'File Saved: {os.path.join(folder_path, file_name)}')


def get_logger(folder_path, file_name, logging_level=logging.DEBUG, add_timestamp=True):
    """
    Create a configured file logger for tracking analysis operations.
    
    This function sets up a file-based logger with customizable logging level
    and optional timestamp in the filename. Useful for tracking the progress
    and debugging data analysis pipelines.
    
    Parameters
    ----------
    folder_path : str
        Directory path where the log file will be created. Directory must exist.
    file_name : str
        Base filename for the log file. Should include .log extension.
    logging_level : int, optional
        The logging level threshold. Only messages at this level or higher
        will be logged. Default is logging.DEBUG (logs everything).
        Common levels: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    add_timestamp : bool, optional
        If True (default), appends timestamp to log filename in format
        YYYY_MM_DD_HH_MM_SS. If False, uses original filename.
    
    Returns
    -------
    logging.Logger
        Configured logger instance ready for use.
    
    Examples
    --------
    >>> # Create debug logger with timestamp
    >>> logger = get_logger('./logs', 'analysis.log')
    Log Started: ./logs/analysis_2024_03_15_14_30_45.log
    
    >>> # Create info-level logger without timestamp
    >>> logger = get_logger('./logs', 'process.log', 
    ...                     logging_level=logging.INFO, add_timestamp=False)
    Log Started: ./logs/process.log
    
    >>> # Use the logger
    >>> logger.info('Starting data processing')
    >>> logger.warning('Missing values detected')
    >>> logger.error('Processing failed')
    
    Notes
    -----
    - Log format: 'YYYY-MM-DD HH:MM:SS,mmm - Jupyter Notebook - LEVEL - MESSAGE'
    - Logger name is set to 'Jupyter Notebook'
    - Multiple loggers can be created for different purposes
    - Log files are created in append mode
    - Timestamp format in filename: YYYY_MM_DD_HH_MM_SS
    
    Logging Levels
    --------------
    - DEBUG (10): Detailed diagnostic information
    - INFO (20): General informational messages
    - WARNING (30): Warning messages for potential issues
    - ERROR (40): Error messages for serious problems
    - CRITICAL (50): Critical errors that may stop execution
    """
    if add_timestamp:
        current_timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        root, ext = os.path.splitext(file_name)
        file_name = root + '_' + current_timestamp + ext  # ext retains the .xxxx including the .
    
    logger = logging.getLogger('Jupyter Notebook')
    logger.setLevel(logging_level)
    handler = logging.FileHandler(os.path.join(folder_path, file_name))
    handler.setLevel(logging_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    print(f'Log Started: {os.path.join(folder_path, file_name)}')
    return logger