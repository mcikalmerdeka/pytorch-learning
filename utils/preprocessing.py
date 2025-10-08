import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import skew, kurtosis, zscore
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.impute import KNNImputer

# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                       Functions for Data Pre-Processing                          ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝

## Checking basic data information
def check_data_information(data, cols):
    """
    Check basic data information including data types, null values, duplicates, and unique values.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame to analyze
    cols : list
        List of column names to check
    
    Returns:
    --------
    pd.DataFrame
        A DataFrame containing information about each column
    
    Examples:
    ---------
    >>> # Check information for all columns
    >>> info_df = check_data_information(df, df.columns.tolist())
    
    >>> # Check information for specific columns
    >>> info_df = check_data_information(df, ['age', 'salary', 'department'])
    """
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
    """
    Drop specified columns from a DataFrame.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame from which to drop columns
    columns : str or list of str
        Column name(s) to drop. Can be a single column name or a list of column names.
    
    Returns:
    --------
    pd.DataFrame
        A new DataFrame with specified columns removed
    
    Examples:
    ---------
    >>> # Drop single column
    >>> df_new = drop_columns(df, 'column_to_drop')
    
    >>> # Drop multiple columns
    >>> df_new = drop_columns(df, ['col1', 'col2', 'col3'])
    """
    # Convert single column to list for consistent handling
    if isinstance(columns, str):
        columns = [columns]
    
    return data.drop(columns=columns, errors='ignore')

## Change binary column data type
def change_binary_dtype(data, column, target_type='categorical'):
    """
    Convert binary columns between numerical (0/1) and categorical (No/Yes) representations.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame containing the binary column
    column : str
        The name of the column to convert
    target_type : str, default='categorical'
        The desired data type for the column:
        - 'categorical': Convert 0/1 to 'No'/'Yes'
        - 'numerical': Convert 'No'/'Yes' back to 0/1
    
    Returns:
    --------
    pd.Series
        The converted column
    
    Examples:
    ---------
    >>> # Convert numerical to categorical
    >>> df['status'] = change_binary_dtype(df, 'status', target_type='categorical')
    >>> # Result: 0 -> 'No', 1 -> 'Yes'
    
    >>> # Convert categorical back to numerical
    >>> df['status'] = change_binary_dtype(df, 'status', target_type='numerical')
    >>> # Result: 'No' -> 0, 'Yes' -> 1
    """
    if target_type == 'categorical':
        return data[column].map({0: 'No', 1: 'Yes'})
    elif target_type == 'numerical':
        return data[column].map({'No': 0, 'Yes': 1})
    else:
        raise ValueError("target_type must be either 'categorical' or 'numerical'")

## Handle missing values function
def handle_missing_values(data, columns, strategy='median', imputer=None, n_neighbors=5):
    """
    Handle missing values using various imputation strategies.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataframe to process
    columns : list
        List of column names to impute
    strategy : str, default='median'
        Imputation method:
        - 'median', 'mean', 'mode': Simple imputation
        - 'ffill', 'bfill': Forward/backward fill
        - 'knn': K-Nearest Neighbors imputation (advanced)
        - 'remove': Drop rows with missing values
    imputer : sklearn imputer object, default=None
        Pre-fitted imputer for test data (for 'knn')
    n_neighbors : int, default=5
        Number of neighbors for KNN imputation
    
    Returns:
    --------
    df_imputed : pd.DataFrame
        Dataframe with imputed values
    imputer : sklearn imputer object or None
        The fitted imputer (for 'knn' or 'iterative' methods)
    
    Example:
    --------
    # Simple imputation
    df_imputed, _ = handle_missing_values(df, columns=['col1', 'col2'], strategy='median')
    
    # KNN imputation on training data
    X_train_imputed, imputer = handle_missing_values(X_train, columns=['col1', 'col2'], 
                                                       strategy='knn', n_neighbors=5)
    
    # Apply same imputer on test data
    X_test_imputed, _ = handle_missing_values(X_test, columns=['col1', 'col2'], 
                                               imputer=imputer)
    """
    if columns is None or len(columns) == 0:
        return data, None
    
    df_imputed = data.copy()
    
    # Validate columns exist
    missing_cols = [col for col in columns if col not in df_imputed.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")
    
    # Remove rows with missing values
    if strategy == 'remove':
        return df_imputed.dropna(subset=columns), None
    
    # Simple imputation methods
    elif strategy in ['median', 'mean', 'mode']:
        if strategy == 'median':
            df_imputed[columns] = df_imputed[columns].fillna(df_imputed[columns].median())
        elif strategy == 'mean':
            df_imputed[columns] = df_imputed[columns].fillna(df_imputed[columns].mean())
        elif strategy == 'mode':
            for col in columns:
                mode_val = df_imputed[col].mode()
                if len(mode_val) > 0:
                    df_imputed[col] = df_imputed[col].fillna(mode_val.iloc[0])
        return df_imputed, None
    
    # Forward/backward fill
    elif strategy in ['ffill', 'bfill']:
        df_imputed[columns] = df_imputed[columns].fillna(method=strategy)
        return df_imputed, None
    
    # KNN imputation (advanced)
    elif strategy == 'knn':
        if imputer is None:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_imputed[columns] = imputer.fit_transform(df_imputed[columns])
        else:
            df_imputed[columns] = imputer.transform(df_imputed[columns])
        return df_imputed, imputer
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'median', 'mean', 'mode', 'ffill', 'bfill', 'knn', or 'remove'")

## Handle and detect outliers function
def filter_outliers(data, columns, method='iqr', threshold=1.5, detect_only=False, return_mask=False, verbose=True):
    """
    Unified function to detect and/or filter outliers using IQR or Z-score method.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataframe to process
    columns : list
        List of column names to check for outliers
    method : str, default='iqr'
        Method to detect outliers: 'iqr' or 'zscore'
    threshold : float, default=1.5
        For IQR: multiplier for IQR range (default 1.5, typically 1.5-3)
        For Z-score: max absolute z-score (default 1.5, typically 2-3)
    detect_only : bool, default=False
        If True, returns summary DataFrame of outliers (detection mode)
        If False, returns filtered dataframe (filtering mode)
    return_mask : bool, default=False
        Only for filtering mode: if True, returns (filtered_data, mask)
    verbose : bool, default=True
        Only for detection mode: if True, prints summary statistics
    
    Returns:
    --------
    Detection mode (detect_only=True):
        pd.DataFrame: Summary with columns, bounds, counts, percentages
    
    Filtering mode (detect_only=False):
        pd.DataFrame or tuple: Filtered data, or (filtered_data, mask) if return_mask=True
    
    Examples:
    --------
    # Detection mode - analyze outliers
    summary = filter_outliers(df, columns=['col1', 'col2'], method='iqr', detect_only=True)
    
    # Filtering mode - remove outliers
    df_clean = filter_outliers(df, columns=['col1', 'col2'], method='iqr', threshold=1.5)
    
    # With mask for debugging
    df_clean, mask = filter_outliers(df, columns=['col1'], return_mask=True)
    """
    if columns is None or len(columns) == 0:
        if detect_only:
            return pd.DataFrame()
        return data if not return_mask else (data, np.array([True] * len(data)))

    if method.lower() not in ['iqr', 'zscore']:
        raise ValueError("Method must be either 'iqr' or 'zscore'")
    
    # Validate columns
    missing_cols = [col for col in columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")
    
    # Initialize tracking variables for detection mode
    if detect_only:
        outlier_counts = []
        non_outlier_counts = []
        is_outlier_list = []
        low_bounds = []
        high_bounds = []
        outlier_percentages = []
    
    # Start with all rows marked as True (non-outliers)
    filtered_entries = np.array([True] * len(data))
    
    # Loop through each column
    for col in columns:
        # IQR method
        if method.lower() == 'iqr':
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (IQR * threshold)
            upper_bound = Q3 + (IQR * threshold)
            filter_outlier = ((data[col] >= lower_bound) & (data[col] <= upper_bound))
            
        # Z-score method
        elif method.lower() == 'zscore':
            z_scores = np.abs(stats.zscore(data[col]))
            filter_outlier = (z_scores < threshold)
            
            mean = data[col].mean()
            std = data[col].std()
            lower_bound = mean - (threshold * std)
            upper_bound = mean + (threshold * std)
        
        # Store detection statistics
        if detect_only:
            outlier_count = len(data[~filter_outlier])
            non_outlier_count = len(data[filter_outlier])
            outlier_pct = round((outlier_count / len(data)) * 100, 2)
            
            outlier_counts.append(outlier_count)
            non_outlier_counts.append(non_outlier_count)
            is_outlier_list.append(data[col][~filter_outlier].any())
            low_bounds.append(lower_bound)
            high_bounds.append(upper_bound)
            outlier_percentages.append(outlier_pct)
        
        # Combine filters
        filtered_entries = filtered_entries & filter_outlier
    
    # Detection mode - return summary DataFrame
    if detect_only:
        if verbose:
            print(f'Amount of Rows: {len(data)}')
            print(f'Amount of Outlier Rows (Across All Columns): {len(data[~filtered_entries])}')
            print(f'Amount of Non-Outlier Rows (Across All Columns): {len(data[filtered_entries])}')
            print(f'Percentage of Outliers: {round(len(data[~filtered_entries]) / len(data) * 100, 2)}%')
            print()
        
        return pd.DataFrame({
            'Column Name': columns,
            'Outlier Exist': is_outlier_list,
            'Lower Limit': low_bounds,
            'Upper Limit': high_bounds,
            'Outlier Data': outlier_counts,
            'Non-Outlier Data': non_outlier_counts,
            'Outlier Percentage (%)': outlier_percentages
        })
    
    # Filtering mode - return filtered data
    if return_mask:
        return data[filtered_entries], filtered_entries
    return data[filtered_entries]

## Feature scaling function
def feature_scaling(data, scaling_config=None, columns=None, method='standard', 
                   scaler=None, apply_log=False):
    """
    General feature scaling function with flexible options.
    Supports applying different scaling methods to different columns.
    
    SCALING METHOD GUIDE:
    ---------------------
    1. 'standard' (StandardScaler): Mean=0, Std=1
       - Best for: Normally distributed data, models assuming normal distribution (linear/logistic regression)
       - Sensitive to outliers
    
    2. 'minmax' (MinMaxScaler): Scale to [0, 1] range
       - Best for: Bounded features (counts, percentages), neural networks, image data
       - Very sensitive to outliers
    
    3. 'robust' (RobustScaler): Uses median and IQR
       - Best for: Data with many outliers, skewed distributions
       - More resistant to outliers than standard/minmax
    
    4. 'power' (PowerTransformer): Box-Cox or Yeo-Johnson transformation
       - Best for: Highly skewed data, making distributions more Gaussian
       - Automatically reduces skewness
       - Use 'yeo-johnson' for any data (handles zeros/negatives)
       - Use 'box-cox' only for strictly positive data
    
    5. 'quantile' (QuantileTransformer): Maps to uniform or normal distribution
       - Best for: Heavily skewed/non-linear data, reducing impact of extreme outliers
       - Most robust to outliers
       - Output: 'normal' for Gaussian-like, 'uniform' for evenly spread [0,1]
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataframe to scale
    scaling_config : dict, optional
        Dictionary mapping scaling methods to column lists and options.
        Format: {
            'standard': {'columns': ['col1', 'col2'], 'apply_log': False},
            'minmax': {'columns': ['col3', 'col4'], 'apply_log': False},
            'robust': {'columns': ['col5', 'col6'], 'apply_log': True},
            'power': {'columns': ['col7'], 'method': 'yeo-johnson'},
            'quantile': {'columns': ['col8'], 'n_quantiles': 1000, 'output_distribution': 'normal'}
        }
        If provided, overrides 'columns', 'method', and 'apply_log' parameters.
    columns : list, optional
        List of column names to scale (used when scaling_config is None)
    method : str, default='standard'
        Scaling method: 'standard', 'minmax', 'robust', 'power', or 'quantile' (used when scaling_config is None)
    scaler : dict or sklearn scaler object, default=None
        For single method: sklearn scaler object
        For multiple methods: dict mapping method names to fitted scaler objects
        If None, fits new scaler(s) (for training data)
    apply_log : bool, default=False
        Whether to apply log1p transformation before scaling (used when scaling_config is None)
    
    Returns:
    --------
    df_scaled : pd.DataFrame
        Dataframe with scaled features
    scaler : dict or sklearn scaler object
        For single method: the fitted scaler
        For multiple methods: dict of fitted scalers by method name
    
    Examples:
    ---------
    # Single method (legacy API - still works)
    X_train_scaled, scaler = feature_scaling(
        X_train, 
        columns=['col1', 'col2'], 
        method='standard'
    )
    X_test_scaled, _ = feature_scaling(X_test, columns=['col1', 'col2'], scaler=scaler)
    
    # Multiple methods (new API) - Training data
    X_train_scaled, scalers = feature_scaling(
        X_train,
        scaling_config={
            'minmax': {
                'columns': ['amount_of_bathrooms', 'amount_of_bedrooms']
            },
            'standard': {
                'columns': ['land_size_sqm', 'building_size_sqm'],
                'apply_log': True
            },
            'power': {
                'columns': ['price'],
                'method': 'yeo-johnson'  # or 'box-cox' for positive-only data
            },
            'quantile': {
                'columns': ['area'],
                'n_quantiles': 1000,
                'output_distribution': 'normal'  # or 'uniform'
            }
        }
    )
    
    # Multiple methods - Test data (reuse fitted scalers)
    X_test_scaled, _ = feature_scaling(
        X_test,
        scaling_config={
            'minmax': {'columns': ['amount_of_bathrooms', 'amount_of_bedrooms']},
            'standard': {'columns': ['land_size_sqm', 'building_size_sqm'], 'apply_log': True},
            'power': {'columns': ['price'], 'method': 'yeo-johnson'},
            'quantile': {'columns': ['area'], 'n_quantiles': 1000, 'output_distribution': 'normal'}
        },
        scaler=scalers
    )
    """
    df_scaled = data.copy()
    
    # --- MODE 1: Multiple scaling methods (new API) ---
    if scaling_config is not None:
        # Validate scaling_config structure
        if not isinstance(scaling_config, dict):
            raise ValueError("scaling_config must be a dictionary")
        
        # Collect all columns being scaled
        all_cols = []
        for method_name, config in scaling_config.items():
            if 'columns' not in config:
                raise ValueError(f"scaling_config['{method_name}'] must have 'columns' key")
            all_cols.extend(config['columns'])
        
        # Check for duplicates
        if len(all_cols) != len(set(all_cols)):
            duplicates = [col for col in set(all_cols) if all_cols.count(col) > 1]
            raise ValueError(f"Columns cannot be scaled with multiple methods: {duplicates}")
        
        # Validate all columns exist
        missing_cols = [col for col in all_cols if col not in df_scaled.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in dataframe: {missing_cols}")
        
        # Initialize scaler dict if not provided (training mode)
        if scaler is None:
            scaler = {}
            is_training = True
        else:
            if not isinstance(scaler, dict):
                raise ValueError("When using scaling_config, scaler must be a dict or None")
            is_training = False
        
        # Apply each scaling method
        for method_name, config in scaling_config.items():
            cols = config['columns']
            log_transform = config.get('apply_log', False)
            
            # Convert to float
            df_scaled[cols] = df_scaled[cols].astype(float)
            
            # Apply log transformation if requested
            if log_transform:
                for col in cols:
                    df_scaled[col] = np.log1p(df_scaled[col])
            
            # Get or create scaler
            if is_training:
                if method_name == 'standard':
                    method_scaler = StandardScaler()
                elif method_name == 'minmax':
                    method_scaler = MinMaxScaler()
                elif method_name == 'robust':
                    method_scaler = RobustScaler(quantile_range=(5, 95))
                elif method_name == 'power':
                    # PowerTransformer: auto handles skewness, makes data more Gaussian
                    power_method = config.get('method', 'yeo-johnson')  # 'yeo-johnson' or 'box-cox'
                    method_scaler = PowerTransformer(method=power_method, standardize=True)
                elif method_name == 'quantile':
                    # QuantileTransformer: maps to uniform or normal distribution
                    n_quantiles = config.get('n_quantiles', 1000)
                    output_dist = config.get('output_distribution', 'normal')  # 'normal' or 'uniform'
                    method_scaler = QuantileTransformer(n_quantiles=n_quantiles, 
                                                       output_distribution=output_dist,
                                                       random_state=42)
                else:
                    raise ValueError(f"Unknown scaling method: {method_name}. Use 'standard', 'minmax', 'robust', 'power', or 'quantile'")
                
                # Fit and transform
                df_scaled[cols] = method_scaler.fit_transform(df_scaled[cols])
                scaler[method_name] = method_scaler
            else:
                # Transform only (test data)
                if method_name not in scaler:
                    raise ValueError(f"Scaler for method '{method_name}' not found in provided scaler dict")
                df_scaled[cols] = scaler[method_name].transform(df_scaled[cols])
        
        return df_scaled, scaler
    
    # --- MODE 2: Single scaling method (legacy API) ---
    else:
        if columns is None:
            raise ValueError("Must specify 'columns' when not using scaling_config")
        
        # Validate columns exist
        missing_cols = [col for col in columns if col not in df_scaled.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in dataframe: {missing_cols}")
        
        # Convert to float
        df_scaled[columns] = df_scaled[columns].astype(float)
        
        # Apply log transformation if requested
        if apply_log:
            for col in columns:
                df_scaled[col] = np.log1p(df_scaled[col])
        
        # Initialize scaler if not provided (training mode)
        if scaler is None:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler(quantile_range=(5, 95))
            elif method == 'power':
                scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            elif method == 'quantile':
                scaler = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42)
            else:
                raise ValueError(f"Unknown scaling method: {method}. Use 'standard', 'minmax', 'robust', 'power', or 'quantile'")
            
            # Fit and transform (training data)
            df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        else:
            # Only transform (test data)
            df_scaled[columns] = scaler.transform(df_scaled[columns])
        
        return df_scaled, scaler

## Feature encoding function
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def feature_encoding(data, ordinal_columns=None, nominal_columns=None, 
                     ordinal_categories=None, drop_first=True, 
                     preserve_dtypes=True, handle_unknown='error'):
    """
    General feature encoding function using sklearn ColumnTransformer.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataframe to encode
    ordinal_columns : list, optional
        List of column names for ordinal encoding
    nominal_columns : list, optional
        List of column names for one-hot encoding
    ordinal_categories : dict, optional
        Dictionary mapping ordinal column names to their category order lists
        Example: {'Education': ['SMA', 'D3', 'S1', 'S2', 'S3']}
    drop_first : bool, default=True
        Whether to drop first category in one-hot encoding (avoid dummy trap)
    preserve_dtypes : bool, default=True
        Whether to preserve original dtypes for non-encoded columns
    handle_unknown : str, default='error'
        How to handle unknown categories: 'error', 'use_encoded_value', 'ignore'
        
    Returns:
    --------
    pd.DataFrame
        Encoded dataframe with preserved column order and dtypes
        
    Examples:
    ---------
    >>> # Simple one-hot encoding
    >>> df_encoded = feature_encoding(df, nominal_columns=['Color', 'Size'])
    
    >>> # Ordinal encoding with custom order
    >>> df_encoded = feature_encoding(
    ...     df, 
    ...     ordinal_columns=['Education'],
    ...     ordinal_categories={'Education': ['SMA', 'D3', 'S1', 'S2', 'S3']}
    ... )
    
    >>> # Mixed encoding
    >>> df_encoded = feature_encoding(
    ...     df,
    ...     ordinal_columns=['Education', 'Age_Group'],
    ...     nominal_columns=['Marital_Status'],
    ...     ordinal_categories={
    ...         'Education': ['SMA', 'D3', 'S1', 'S2', 'S3'],
    ...         'Age_Group': ['Young Adult', 'Middle Adult', 'Senior Adult']
    ...     }
    ... )
    """
    # Initialize defaults
    ordinal_columns = ordinal_columns or []
    nominal_columns = nominal_columns or []
    ordinal_categories = ordinal_categories or {}
    
    # Validate inputs
    all_encoding_cols = ordinal_columns + nominal_columns
    missing_cols = [col for col in all_encoding_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")
    
    if not ordinal_columns and not nominal_columns:
        raise ValueError("Must specify at least one column to encode (ordinal_columns or nominal_columns)")
    
    # Validate ordinal categories
    for col in ordinal_columns:
        if col not in ordinal_categories:
            raise ValueError(f"Ordinal column '{col}' requires category order in ordinal_categories")
        
        unique_vals = data[col].unique()
        if not all(val in ordinal_categories[col] for val in unique_vals):
            print(f"Warning: Some values in '{col}' are not in the specified category order")
    
    # Copy dataframe
    df_preprocessed = data.copy()
    
    # Store original dtypes
    original_dtypes = df_preprocessed.dtypes if preserve_dtypes else None
    
    # Identify datetime columns to preserve separately
    datetime_columns = df_preprocessed.select_dtypes(include=['datetime64']).columns.tolist()
    datetime_data = df_preprocessed[datetime_columns].copy() if datetime_columns else None
    
    # Build transformers list
    transformers = []
    
    # Add ordinal encoders
    for col in ordinal_columns:
        transformers.append((
            f'ordinal_{col}',
            OrdinalEncoder(
                categories=[ordinal_categories[col]], 
                dtype=np.float64,
                handle_unknown=handle_unknown
            ),
            [col]
        ))
    
    # Add one-hot encoders
    for col in nominal_columns:
        transformers.append((
            f'onehot_{col}',
            OneHotEncoder(
                drop='first' if drop_first else None,
                sparse_output=False,
                dtype=np.float64,
                handle_unknown='ignore' if handle_unknown == 'ignore' else 'error'
            ),
            [col]
        ))
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    
    # Remove datetime columns before transformation
    df_for_transform = df_preprocessed.drop(columns=datetime_columns) if datetime_columns else df_preprocessed
    
    # Apply transformation
    df_encoded = preprocessor.fit_transform(df_for_transform)
    
    # Build column names
    encoded_column_names = []
    
    # Add ordinal column names
    encoded_column_names.extend(ordinal_columns)
    
    # Add one-hot encoded column names
    for col in nominal_columns:
        categories = df_for_transform[col].unique()
        if drop_first:
            # Sort to ensure consistent ordering
            categories = sorted(categories)
            encoded_column_names.extend([f'{col}_{cat}' for cat in categories[1:]])
        else:
            encoded_column_names.extend([f'{col}_{cat}' for cat in sorted(categories)])
    
    # Add passthrough columns
    passthrough_cols = [c for c in df_for_transform.columns if c not in all_encoding_cols]
    encoded_column_names.extend(passthrough_cols)
    
    # Convert to DataFrame
    df_encoded = pd.DataFrame(df_encoded, columns=encoded_column_names, index=data.index)
    
    # Add back datetime columns
    if datetime_columns:
        for col in datetime_columns:
            df_encoded[col] = datetime_data[col]
    
    # Preserve original dtypes for non-encoded columns
    if preserve_dtypes:
        encoded_cols = ordinal_columns + [col for col in df_encoded.columns if any(col.startswith(f'{nc}_') for nc in nominal_columns)]
        
        for col in df_encoded.columns:
            if col in original_dtypes and col not in encoded_cols:
                try:
                    df_encoded[col] = df_encoded[col].astype(original_dtypes[col])
                except Exception as e:
                    print(f"Warning: Could not convert column '{col}' back to {original_dtypes[col]}. Error: {e}")
    
    return df_encoded

# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                       Functions for Statistical Summary                          ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝

## Describe numerical columns
def describe_numerical_combined(data, col_series, hue=None):
    """
    Generate descriptive statistics for numerical columns in a dataframe,
    both overall and optionally grouped by a categorical variable.
    
    Parameters:
    data (pd.DataFrame): The dataframe containing the numerical columns
    col_series (list): The list of numerical columns to describe
    hue (str, optional): The name of the categorical column to group by
    
    Returns:
    pd.DataFrame: A dataframe containing descriptive statistics, with hue classes if specified
    
    Examples:
    ---------
    >>> # Overall statistics only
    >>> stats = describe_numerical_combined(df, ['age', 'salary', 'experience'])
    
    >>> # Statistics grouped by a categorical variable
    >>> stats = describe_numerical_combined(df, ['age', 'salary'], hue='department')
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
    
    # If hue column is provided, add class-specific statistics
    if hue is not None:
        target_classes = sorted(data[hue].unique())
        class_summaries = []
        
        for target_class in target_classes:
            # Filter data for current class
            class_data = data[data[hue] == target_class]
            
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
def describe_categorical_combined(data, col_series, hue=None):
    """
    Generate descriptive statistics for categorical columns in a dataframe,
    both overall and optionally grouped by a categorical variable.
    
    Parameters:
    data (pd.DataFrame): The dataframe containing the categorical columns
    col_series (list): The list of categorical columns to describe
    hue (str, optional): The name of the categorical column to group by
    
    Returns:
    pd.DataFrame: A dataframe containing descriptive statistics, with hue classes if specified
    
    Examples:
    ---------
    >>> # Overall statistics only
    >>> stats = describe_categorical_combined(df, ['gender', 'department', 'job_title'])
    
    >>> # Statistics grouped by a categorical variable
    >>> stats = describe_categorical_combined(df, ['gender', 'job_title'], hue='department')
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
    
    # If hue column is provided, add class-specific statistics
    if hue is not None:
        target_classes = sorted(data[hue].unique())
        class_summaries = []
        
        for target_class in target_classes:
            # Filter data for current class
            class_data = data[data[hue] == target_class]
            
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

# Describe date columns
from datetime import datetime

def describe_date_columns(df, date_columns):
    """
    Comprehensive analysis of date columns including various temporal features
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing date columns
    date_columns (list): List of column names containing dates
    
    Returns:
    pandas.DataFrame: dates_summary statistics and temporal features for each date column
    
    Examples:
    ---------
    >>> # Analyze single date column
    >>> date_stats = describe_date_columns(df, ['purchase_date'])
    
    >>> # Analyze multiple date columns
    >>> date_stats = describe_date_columns(df, ['start_date', 'end_date', 'modified_date'])
    """
    dates_summary = df[date_columns].describe().transpose()
    
    # Basic range calculations
    dates_summary['date_range_days'] = dates_summary['max'] - dates_summary['min']
    dates_summary['date_range_months'] = dates_summary['date_range_days'].apply(lambda x: x.days / 30)
    dates_summary['date_range_years'] = dates_summary['date_range_days'].apply(lambda x: x.days / 365)
    
    # Additional temporal features
    for col in date_columns:
        # Distribution across time periods
        dates = df[col].dropna()
        dates_summary.loc[col, 'unique_years'] = dates.dt.year.nunique()
        dates_summary.loc[col, 'unique_months'] = dates.dt.to_period('M').nunique()
        dates_summary.loc[col, 'unique_days'] = dates.dt.date.nunique()
        
        # Temporal patterns
        dates_summary.loc[col, 'weekend_percentage'] = (dates.dt.dayofweek.isin([5, 6]).mean()) * 100
        dates_summary.loc[col, 'business_hours_percentage'] = (
            dates[dates.dt.hour.between(9, 17)].count() / dates.count()
        ) * 100 if hasattr(dates.dt, 'hour') else np.nan
        
        # Seasonality indicators
        dates_summary.loc[col, 'most_common_month'] = dates.dt.month.mode().iloc[0]
        dates_summary.loc[col, 'most_common_weekday'] = dates.dt.day_name().mode().iloc[0]
        
        # Gaps analysis
        sorted_dates = dates.sort_values()
        gaps = sorted_dates.diff().dropna()
        dates_summary.loc[col, 'max_gap_days'] = gaps.dt.days.max()
        dates_summary.loc[col, 'median_gap_days'] = gaps.dt.days.median()
        
        # Future/Past analysis
        now = datetime.now()
        dates_summary.loc[col, 'future_dates_percentage'] = (dates > now).mean() * 100
        
        # Regularity check
        dates_summary.loc[col, 'is_regular_interval'] = gaps.dt.days.std() < 1
    
    return dates_summary

# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                       Functions for Data Visualization                          ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝

## Hisplot and kdeplot analysis
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

# Boxplot and violinplot analysis
def plot_dynamic_boxplots_violinplots(df, col_series, plot_type='boxplot', ncols=6, figsize=(26, 18), orientation='v', hue=None):
    """
    Creates a dynamic grid of either boxplots or violin plots for multiple numerical columns.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.
    col_series : list of str
        List of column names to include in the plots.
    plot_type : str, optional, default='boxplot'
        Type of plot to generate. Options are 'boxplot' or 'violinplot'.
    ncols : int, optional, default=6
        Number of columns in the subplot grid. Adjust this value to change grid width.
    figsize : tuple, optional, default=(26, 18)
        Size of the figure to control plot dimensions.
    orientation : str, optional, default='v'
        Orientation of the plots. Use 'v' for vertical and 'h' for horizontal.
    hue : str, optional, default=None
        Column name to use for color encoding. Creates separate plots for each category.

    Returns:
    -------
    None
        Displays a grid of plots.

    Examples:
    --------
    >>> # Create vertical boxplots
    >>> plot_dynamic_boxplots_violinplots(df, ['col1', 'col2'], plot_type='boxplot', orientation='v')

    >>> # Create horizontal violin plots with categorical splitting
    >>> plot_dynamic_boxplots_violinplots(df, ['col1', 'col2'], plot_type='violinplot',
                                        orientation='h', hue='category')
    """
    # Validate plot_type parameter
    if plot_type not in ['boxplot', 'violinplot']:
        raise ValueError("plot_type must be either 'boxplot' or 'violinplot'")

    # Calculate required number of rows based on number of plots and specified columns
    num_plots = len(col_series)
    nrows = math.ceil(num_plots / ncols)

    # Adjust figsize based on orientation
    if orientation == 'h':
        figsize = (figsize[1], figsize[0])  # Swap width and height for horizontal plots

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    # Convert ax to array if it's a single subplot
    if num_plots == 1:
        ax = np.array([ax])
    else: 
        ax = ax.flatten()  # Flatten the axes array for easy indexing

    # Generate plots for each column
    for i, col in enumerate(col_series):
        if plot_type == 'boxplot':
            if orientation == 'v':
                sns.boxplot(data=df, ax=ax[i], y=col, orient='v', hue=hue)
                ax[i].set_title(f'Boxplot of {col}')
            else:  # orientation == 'h'
                sns.boxplot(data=df, ax=ax[i], x=col, orient='h', hue=hue)
                ax[i].set_title(f'Boxplot of {col}')
        else: # violinplot
            if orientation == 'v':
                sns.violinplot(data=df, ax=ax[i], y=col, orient='v', hue=hue, inner_kws=dict(box_width=15, whis_width=2))
                ax[i].set_title(f'Violinplot of {col}')
            else:  # orientation == 'h'
                sns.violinplot(data=df, ax=ax[i], x=col, orient='h', hue=hue, inner_kws=dict(box_width=15, whis_width=2))
                ax[i].set_title(f'Violinplot of {col}')

    # Remove any unused subplots if total subplots exceed columns in cols
    for j in range(num_plots, len(ax)):
        fig.delaxes(ax[j])

    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.show()

## Distribution type analysis
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
    multimodal_cols : list of str, optional
        List of column names suspected to be bimodal/multimodal. Default is None.

    Returns:
    pd.DataFrame: A DataFrame containing the columns' names, skewness values, kurtosis values, and identified distribution type.
    
    Examples:
    ---------
    >>> # Basic distribution identification
    >>> dist_types = identify_distribution_types(df, ['age', 'salary', 'experience'])
    
    >>> # With manual specification of distribution types
    >>> dist_types = identify_distribution_types(
    ...     df, 
    ...     ['age', 'salary', 'rating', 'score'],
    ...     uniform_cols=['rating'],
    ...     multimodal_cols=['score']
    ... )
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

## Countplot analysis
def plot_dynamic_countplot(df, col_series, ncols=6, figsize=(26, 18), stat='count', hue=None, order=None):
    """
    Plots a dynamic grid of countplot for a list of categorical columns from a DataFrame.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.
    col_series : list of str
        List of column names to include in the countplots.
    ncols : int, optional, default=6
        Number of columns in the subplot grid. Adjust this value to change grid width.
    figsize : tuple, optional, default=(26, 18)
        Size of the figure to control plot dimensions.
    stat : str, optional, default='count'
        The statistic to compute for the bars
    hue : str, optional, default=None
        Column name to use for color encoding
    order : list, optional, default=None
        Specific ordering for the categorical variables

    Returns:
    -------
    None
        Displays a grid of countplots.
    
    Examples:
    ---------
    >>> # Basic countplot
    >>> plot_dynamic_countplot(df, ['gender', 'department', 'job_title'])
    
    >>> # Countplot with hue grouping
    >>> plot_dynamic_countplot(df, ['department', 'job_title'], hue='gender')
    
    >>> # With custom order
    >>> order_list = [['A', 'B', 'C'], ['Small', 'Medium', 'Large']]
    >>> plot_dynamic_countplot(df, ['category', 'size'], order=order_list, ncols=2)
    
    >>> # Show top 5 categories for each column (most frequent)
    >>> cats_cols = ['Category', 'Province', 'City']
    >>> plot_dynamic_countplot(
    ...     df=df,
    ...     col_series=cats_cols,
    ...     ncols=1,
    ...     figsize=(12, 10),
    ...     order=[df[cat].value_counts().iloc[:5].index.tolist() for cat in cats_cols]
    ... )
    
    >>> # Show bottom 5 categories for each column (least frequent)
    >>> plot_dynamic_countplot(
    ...     df=df,
    ...     col_series=cats_cols,
    ...     ncols=1,
    ...     figsize=(12, 10),
    ...     order=[df[cat].value_counts().iloc[-5:].index.tolist() for cat in cats_cols]
    ... )
    """

    # Calculate required number of rows based on number of plots and specified columns
    num_plots = len(col_series)
    nrows = math.ceil(num_plots / ncols)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    # Convert ax to array if it's a single subplot
    if num_plots == 1:
        ax = np.array([ax])
    else: 
        ax = ax.flatten()  # Flatten the axes array for easy indexing

    # Generate countplot for each column
    for i, col in enumerate(col_series):
        # Get the specific order for this column if provided
        if isinstance(order, list):
            current_order = order[i].tolist() if hasattr(order[i], 'tolist') else order[i]
        else:
            current_order = None

        sns.countplot(data=df,
                     ax=ax[i],
                     x=col,
                     stat=stat,
                     hue=hue,
                     order=current_order)
        if hue is not None:
            ax[i].set_title(f'Countplot sof {col} by {hue}s')
        else:
            ax[i].set_title(f'Countplot of {col}')
        ax[i].set_ylabel(f'Count of {col}')
        ax[i].set_xlabel(f'{col}')

        # Add value labels on top of bars
        for p in ax[i].patches:
            ax[i].annotate(
                f'{int(p.get_height())}',  # The text to display
                (p.get_x() + p.get_width() / 2., p.get_height()),  # The position
                ha='center',  # Horizontal alignment
                va='bottom'   # Vertical alignment
            )

    # Remove any unused subplots if total subplots exceed columns in cols
    for j in range(num_plots, len(ax)):
        fig.delaxes(ax[j])

    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.show()

## Correlation heatmap analysis on numerical features and target
def plot_correlation_heatmap(df, col_series, corr_method='pearson', figsize=(8, 6), cmap='coolwarm'):
    """
    Plots a correlation heatmap for the specified columns in a dataframe.

    Parameters:
    ----------
    df : pd.DataFrame
        The input dataframe containing the data.
    col_series : list
        List of column names to include in the correlation matrix.
    corr_method : str 
        Correlation method ('pearson', 'spearman', or 'kendall'). Default is 'pearson'.
    figsize : tuple
        Size of the heatmap figure (width, height). Default is (8, 6).
    cmap : str
        Color map for the heatmap. Default is 'coolwarm'.

    Returns:
    -------
    None
        Displays the correlation heatmap.
    
    Examples:
    --------
    >>> included_col = ['Income', 'Total_Spending', 'Age', 'CVR']
    >>> plot_correlation_heatmap(df_filtered_outliers, col_series=included_col, corr_method='pearson')
    >>> plot_correlation_heatmap(df_filtered_outliers, col_series=included_col, corr_method='spearman')
    """
    
    # Compute correlation matrix
    correlation_matrix = df[col_series].corr(method=corr_method)
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(data=correlation_matrix, cmap=cmap, annot=True, fmt='.3f', vmin=-1, vmax=1)
    plt.title(f'{corr_method.capitalize()} Correlation')
    
    plt.tight_layout()
    plt.show()