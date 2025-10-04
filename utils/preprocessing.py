import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import skew, kurtosis, zscore
import matplotlib.pyplot as plt
import seaborn as sns
import math

# =====================================================================Functions for data pre-processing========================================================================

## Checking basic data information
def check_data_information(data, cols):
    list_item = []
    for col in cols:
        # Convert unique values to string representation
        unique_sample = ', '.join(map(str, data[col].unique()[:5]))
        
        list_item.append([
            col,                                           # The column name
            str(data[col].dtype),                          # The data type as string
            data[col].isna().sum(),                        # The count of null values
            round(100 * data[col].isna().sum() / len(data[col]), 2),  # The percentage of null values
            data.duplicated().sum(),                       # The count of duplicated rows
            data[col].nunique(),                           # The count of unique values
            unique_sample                                  # Sample of unique values as string
        ])

    desc_df = pd.DataFrame(
        data=list_item,
        columns=[
            'Feature',
            'Data Type',
            'Null Values',
            'Null Percentage',
            'Duplicated Values',
            'Unique Values',
            'Unique Sample'
        ]
    )
    return desc_df

## Drop columns function
def drop_columns(data, columns):
    return data.drop(columns=columns, errors='ignore')

## Handle missing values function
def handle_missing_values(data, columns, strategy='fill', imputation_method='median'):
    # Return the original data if the column is empty
    if columns is None:
        return data
    
    # Impute missing values based on the strategy
    if strategy == 'fill':
        if imputation_method == 'median':
            return data[columns].fillna(data[columns].median())
        elif imputation_method == 'mean':
            return data[columns].fillna(data[columns].mean())
        elif imputation_method == 'mode':
            return data[columns].fillna(data[columns].mode().iloc[0])
        elif imputation_method == 'ffill':
            return data[columns].fillna(method='ffill')
        elif imputation_method == 'bfill':
            return data[columns].fillna(method='bfill')
        else:
            return data[columns].fillna(data[columns].median())

    # Remove rows with missing values
    elif strategy == 'remove':
        return data.dropna(subset=columns)

## Handle outliers function
def filter_outliers(data, col_series, method='iqr', threshold=3):
    # Return the original data if the column series is empty
    if col_series is None:
        return data

    # Validate the method parameter
    if method.lower() not in ['iqr', 'zscore']:
        raise ValueError("Method must be either 'iqr' or 'zscore'")
    
    # Start with all rows marked as True (non-outliers)
    filtered_entries = np.array([True] * len(data))
    
    # Loop through each column
    for col in col_series:
        if method.lower() == 'iqr':
            # IQR method
            Q1 = data[col].quantile(0.25)  # First quartile (25th percentile)
            Q3 = data[col].quantile(0.75)  # Third quartile (75th percentile)
            IQR = Q3 - Q1  # Interquartile range
            lower_bound = Q1 - (IQR * 1.5)  # Lower bound for outliers
            upper_bound = Q3 + (IQR * 1.5)  # Upper bound for outliers

            # Create a filter that identifies non-outliers for the current column
            filter_outlier = ((data[col] >= lower_bound) & (data[col] <= upper_bound))
            
        elif method.lower() == 'zscore':  # zscore method
            # Calculate Z-Scores and create filter
            z_scores = np.abs(stats.zscore(data[col]))

            # Create a filter that identifies non-outliers
            filter_outlier = (z_scores < threshold)
        
        # Update the filter to exclude rows that have outliers in the current column
        filtered_entries = filtered_entries & filter_outlier
    
    return data[filtered_entries]

# =====================================================================Functions for statistical summary========================================================================

# Describe numerical columns
def describe_numerical_combined(data, col_series, target_col=None):
    """
    Generate descriptive statistics for numerical columns in a dataframe,
    both overall and optionally grouped by target variable.
    
    Parameters:
    data (pd.DataFrame): The dataframe containing the numerical columns
    col_series (list): The list of numerical columns to describe
    target_col (str, optional): The name of the target column for classification
    
    Returns:
    pd.DataFrame: A dataframe containing descriptive statistics, with target classes if specified
    """
    # Overall statistics (original approach)
    overall_summary = data[col_series].describe().transpose().reset_index()
    overall_summary = overall_summary.rename(columns={'index': 'Feature'})
    
    # Add additional statistics for overall data
    overall_summary['range'] = overall_summary['max'] - overall_summary['min']
    overall_summary['IQR'] = overall_summary['75%'] - overall_summary['25%']
    overall_summary['CV'] = (overall_summary['std'] / overall_summary['mean']) * 100
    
    # Calculate skewness and kurtosis for numerical columns
    numerical_data = data[col_series].select_dtypes(include=['int64', 'float64']).dropna()
    overall_summary['skewness'] = [skew(numerical_data[col]) for col in numerical_data.columns]
    overall_summary['kurtosis'] = [kurtosis(numerical_data[col]) for col in numerical_data.columns]
    
    # Rename columns to indicate these are overall statistics
    overall_summary.columns = ['Feature'] + [f'overall_{col}' if col != 'Feature' else col 
                                           for col in overall_summary.columns[1:]]
    
    final_summary = overall_summary
    
    # If target column is provided, add class-specific statistics
    if target_col is not None:
        target_classes = sorted(data[target_col].unique())
        class_summaries = []
        
        for target_class in target_classes:
            # Filter data for current class
            class_data = data[data[target_col] == target_class]
            
            # Calculate basic statistics
            class_summary = class_data[col_series].describe().transpose().reset_index()
            class_summary = class_summary.rename(columns={'index': 'Feature'})
            
            # Add additional statistics
            class_summary['range'] = class_summary['max'] - class_summary['min']
            class_summary['IQR'] = class_summary['75%'] - class_summary['25%']
            class_summary['CV'] = (class_summary['std'] / class_summary['mean']) * 100
            
            # Calculate skewness and kurtosis
            numerical_class_data = class_data[col_series].select_dtypes(include=['int64', 'float64']).dropna()
            class_summary['skewness'] = [skew(numerical_class_data[col]) for col in numerical_class_data.columns]
            class_summary['kurtosis'] = [kurtosis(numerical_class_data[col]) for col in numerical_class_data.columns]
            
            # Rename columns to indicate which class they belong to
            class_summary.columns = ['Feature'] + [f'class_{target_class}_{col}' if col != 'Feature' else col 
                                                 for col in class_summary.columns[1:]]
            
            class_summaries.append(class_summary)
        
        # Combine all class summaries
        all_class_summaries = class_summaries[0]
        for summary in class_summaries[1:]:
            all_class_summaries = pd.merge(all_class_summaries, summary, on='Feature')
            
        # Merge with overall summary
        final_summary = pd.merge(final_summary, all_class_summaries, on='Feature')
        
        # Reorder columns to group statistics by type rather than by class
        # Get all column names except 'Feature'
        cols = final_summary.columns.tolist()
        cols.remove('Feature')
        
        # Group similar statistics together
        stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range', 'IQR', 'CV', 'skewness', 'kurtosis']
        new_cols = ['Feature']
        
        for stat in stats:
            stat_cols = [col for col in cols if stat in col]
            new_cols.extend(stat_cols)
            
        final_summary = final_summary[new_cols]
    
    return final_summary

# Describe categorical columns
def describe_categorical_combined(data, col_series, target_col=None):
    """
    Generate descriptive statistics for categorical columns in a dataframe,
    both overall and optionally grouped by target variable.
    
    Parameters:
    data (pd.DataFrame): The dataframe containing the categorical columns
    col_series (list): The list of categorical columns to describe
    target_col (str, optional): The name of the target column for classification
    
    Returns:
    pd.DataFrame: A dataframe containing descriptive statistics, with target classes if specified
    """
    # Overall statistics
    cats_summary = data[col_series].describe().transpose().reset_index().rename(columns={'index': 'Feature'})
    
    # Add additional statistics for overall data
    cats_summary['bottom'] = [data[col].value_counts().idxmin() for col in col_series]
    cats_summary['freq_bottom'] = [data[col].value_counts().min() for col in col_series]
    cats_summary['top_percentage'] = [round(data[col].value_counts().max() / len(data) * 100, 2) 
                                    for col in col_series]
    cats_summary['bottom_percentage'] = [round(data[col].value_counts().min() / len(data) * 100, 2) 
                                       for col in col_series]
    
    # Add number of unique categories
    cats_summary['n_categories'] = [data[col].nunique() for col in col_series]
    
    # Rename columns to indicate these are overall statistics
    cats_summary.columns = ['Feature'] + [f'overall_{col}' if col != 'Feature' else col 
                                        for col in cats_summary.columns[1:]]
    
    final_summary = cats_summary
    
    # If target column is provided, add class-specific statistics
    if target_col is not None:
        target_classes = sorted(data[target_col].unique())
        class_summaries = []
        
        for target_class in target_classes:
            # Filter data for current class
            class_data = data[data[target_col] == target_class]
            
            # Calculate basic statistics
            class_summary = class_data[col_series].describe().transpose().reset_index()
            class_summary = class_summary.rename(columns={'index': 'Feature'})
            
            # Add additional statistics
            class_summary['bottom'] = [class_data[col].value_counts().idxmin() for col in col_series]
            class_summary['freq_bottom'] = [class_data[col].value_counts().min() for col in col_series]
            class_summary['top_percentage'] = [round(class_data[col].value_counts().max() / len(class_data) * 100, 2) 
                                             for col in col_series]
            class_summary['bottom_percentage'] = [round(class_data[col].value_counts().min() / len(class_data) * 100, 2) 
                                                for col in col_series]
            class_summary['n_categories'] = [class_data[col].nunique() for col in col_series]
            
            # Rename columns to indicate which class they belong to
            class_summary.columns = ['Feature'] + [f'class_{target_class}_{col}' if col != 'Feature' else col 
                                                 for col in class_summary.columns[1:]]
            
            class_summaries.append(class_summary)
        
        # Combine all class summaries
        all_class_summaries = class_summaries[0]
        for summary in class_summaries[1:]:
            all_class_summaries = pd.merge(all_class_summaries, summary, on='Feature')
            
        # Merge with overall summary
        final_summary = pd.merge(final_summary, all_class_summaries, on='Feature')
        
        # Reorder columns to group statistics by type rather than by class
        cols = final_summary.columns.tolist()
        cols.remove('Feature')
        
        # Group similar statistics together
        stats = ['count', 'unique', 'top', 'freq', 'bottom', 'freq_bottom', 
                'top_percentage', 'bottom_percentage', 'n_categories']
        new_cols = ['Feature']
        
        for stat in stats:
            stat_cols = [col for col in cols if stat in col]
            new_cols.extend(stat_cols)
            
        final_summary = final_summary[new_cols]
    
    return final_summary

# =====================================================================Functions for data visualization========================================================================

# Hisplot and kdeplot analysis
def plot_dynamic_hisplots_kdeplots(df, col_series, plot_type='histplot', ncols=6, figsize=(26, 18), hue=None, multiple='layer', fill=None):
    """
    Creates a dynamic grid of histogram plots (with KDE) or KDE plots for multiple numerical columns.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.
    col_series : list of str
        List of column names to include in the plots.
    plot_type : str, optional, default='histplot'
        Type of plot to generate. Options are:
        - 'histplot': Histogram with KDE overlay
        - 'kdeplot': Kernel Density Estimation plot
    ncols : int, optional, default=6
        Number of columns in the subplot grid. Adjust this value to change grid width.
    figsize : tuple, optional, default=(26, 18)
        Size of the figure to control plot dimensions.
    hue : str, optional, default=None
        Column name to use for color encoding. Creates separate distributions for each category.
    multiple : str, optional, default='layer'
        How to display multiple distributions. Options are:
        - 'layer': Distributions are overlaid
        - 'dodge': Distributions are placed side by side

    Returns:
    -------
    None
        Displays a grid of distribution plots.

    Examples:
    --------
    >>> # Create histogram plots with KDE
    >>> plot_dynamic_hisplots_kdeplots(df, ['col1', 'col2'], plot_type='histplot')

    >>> # Create KDE plots with categorical splitting
    >>> plot_dynamic_hisplots_kdeplots(
    ...     df,
    ...     ['col1', 'col2'],
    ...     plot_type='kdeplot',
    ...     hue='category',
    ...     multiple='layer'
    ... )

    Notes:
    -----
    - For histplots, KDE (Kernel Density Estimation) is automatically enabled
    - The y-axis label adjusts automatically based on the plot type
    """

    # Validate plot_type parameter
    if plot_type not in ['histplot', 'kdeplot']:
        raise ValueError("plot_type must be either 'histplot' or 'kdeplot'")

    # Calculate required number of rows based on number of plots and specified columns
    num_plots = len(col_series)
    nrows = math.ceil(num_plots / ncols)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    # Convert ax to array if it's a single subplot
    if num_plots == 1:
        ax = np.array([ax])
    else: 
        ax = ax.flatten()  # Flatten the axes array for easy indexing

    # Generate plots for each column
    for i, col in enumerate(col_series):
        if plot_type == 'histplot':
            sns.histplot(data=df, ax=ax[i], x=col, kde=True, hue=hue, multiple=multiple)
        else:  # kdeplot
            sns.kdeplot(data=df, ax=ax[i], x=col, hue=hue, multiple=multiple, fill=fill)

        ax[i].set_title(f'Distribution of {col}')
        ax[i].set_ylabel(f'{"Count" if plot_type == "histplot" else "Density"} of {col}')
        ax[i].set_xlabel(f'{col}')

    # Remove any unused subplots if total subplots exceed columns in cols
    for j in range(num_plots, len(ax)):
        fig.delaxes(ax[j])

    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.show()

# Distribution type analysis
def identify_distribution_types(df, col_series, uniform_cols=None, multimodal_cols=None):
    """
    Identifies and categorizes the distribution type of each numerical column in the DataFrame based on skewness and kurtosis.
    Allows manual specification of columns suspected to be uniform or bimodal/multimodal.

    Parameters:
    df : pd.DataFrame
        The input DataFrame containing the data.
    col_series : list of str
        List of column names to analyze for distribution type.
    uniform_cols : list of str, optional
        List of column names suspected to be uniform. Default is None.
    bimodal_cols : list of str, optional
        List of column names suspected to be bimodal/multimodal. Default is None.

    Returns:
    pd.DataFrame: A DataFrame containing the columns' names, skewness values, kurtosis values, and identified distribution type.
    """
    # Initialize lists to store results
    mean_list = []
    median_list = []
    mode_list = []
    skew_type_list = []
    skew_val_list = []
    kurtosis_val_list = []

    # Loop through each column to calculate distribution metrics
    for col in col_series:
        data = df[col].dropna()  # Remove any NaN values

        # Calculate summary statistics
        mean = round(data.mean(), 3)
        median = data.median()
        mode = data.mode()[0] if not data.mode().empty else median  # Handle case where mode is empty
        skew_val = round(skew(data, nan_policy="omit"), 3)
        kurtosis_val = round(kurtosis(data, nan_policy="omit"), 3)

        # Identify distribution type based on skewness and summary statistics
        if (mean == median == mode) or (-0.2 < skew_val < 0.2):
            skew_type = "Normal Distribution (Symmetric)"
        elif mean < median < mode:
            if skew_val <= -1:
                skew_type = "Highly Negatively Skewed"
            elif -0.5 >= skew_val > -1:
                skew_type = "Moderately Negatively Skewed"
            else:
                skew_type = "Moderately Normal Distribution (Symmetric)"
        else:
            if skew_val >= 1:
                skew_type = "Highly Positively Skewed"
            elif 0.5 <= skew_val < 1:
                skew_type = "Moderately Positively Skewed"
            else:
                skew_type = "Moderately Normal Distribution (Symmetric)"

        # Append the results to the lists
        mean_list.append(mean)
        median_list.append(median)
        mode_list.append(mode)
        skew_type_list.append(skew_type)
        skew_val_list.append(skew_val)
        kurtosis_val_list.append(kurtosis_val)

    # Create a DataFrame to store the results
    dist = pd.DataFrame({
        "Column Name": col_series,
        "Mean": mean_list,
        "Median": median_list,
        "Mode": mode_list,
        "Skewness": skew_val_list,
        "Kurtosis": kurtosis_val_list,
        "Type of Distribution": skew_type_list
    })

    # Manually assign specific distributions based on user-provided column names
    if uniform_cols:
        dist.loc[dist['Column Name'].isin(uniform_cols), 'Type of Distribution'] = 'Uniform Distribution'
    if multimodal_cols:
        dist.loc[dist['Column Name'].isin(multimodal_cols), 'Type of Distribution'] = 'Multi-modal Distribution'

    return dist