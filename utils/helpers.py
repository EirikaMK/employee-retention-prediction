"""
Utility Functions for Employee Intention To Stay
Helper functions for data processing, visualization, and analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_dataset(filepath):
    """
    Load and validate dataset
    
    Args:
        filepath (str): Path to CSV file
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(filepath)
        df.columns = [c.strip() for c in df.columns]
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")


def validate_columns(df, required_cols):
    """
    Validate that required columns exist in dataframe
    
    Args:
        df (pd.DataFrame): Dataset to validate
        required_cols (list): List of required column names
    
    Returns:
        tuple: (bool, list) - (is_valid, missing_columns)
    """
    missing = [col for col in required_cols if col not in df.columns]
    return len(missing) == 0, missing


def get_variable_groups(df):
    """
    Extract variable groups from dataframe
    
    Args:
        df (pd.DataFrame): Dataset
    
    Returns:
        dict: Dictionary of variable groups
    """
    groups = {
        'Q': [col for col in df.columns if col.startswith('Q') and col[1:].isdigit()],
        'POS': [col for col in df.columns if col.startswith('POS')],
        'JS': [col for col in df.columns if col.startswith('JS')],
        'WLB': [col for col in df.columns if col.startswith('WLB')],
        'ITS': [col for col in df.columns if col.startswith('ITS')]
    }
    return groups


def calculate_summary_stats(df, columns=None):
    """
    Calculate summary statistics for specified columns
    
    Args:
        df (pd.DataFrame): Dataset
        columns (list, optional): Columns to summarize. If None, use all numeric columns
    
    Returns:
        pd.DataFrame: Summary statistics
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    summary = pd.DataFrame({
        'Count': df[columns].count(),
        'Mean': df[columns].mean(),
        'Std': df[columns].std(),
        'Min': df[columns].min(),
        '25%': df[columns].quantile(0.25),
        'Median': df[columns].median(),
        '75%': df[columns].quantile(0.75),
        'Max': df[columns].max(),
        'Missing': df[columns].isnull().sum()
    })
    
    return summary.round(2)


def create_output_directory(base_path, subdirs=None):
    """
    Create output directory structure
    
    Args:
        base_path (str): Base output path
        subdirs (list, optional): List of subdirectories to create
    
    Returns:
        dict: Dictionary of created paths
    """
    base = Path(base_path)
    base.mkdir(exist_ok=True, parents=True)
    
    paths = {'base': base}
    
    if subdirs:
        for subdir in subdirs:
            path = base / subdir
            path.mkdir(exist_ok=True)
            paths[subdir] = path
    
    return paths


def export_dataframe(df, filepath, index=False):
    """
    Export dataframe to CSV with error handling
    
    Args:
        df (pd.DataFrame): Dataframe to export
        filepath (str): Output filepath
        index (bool): Whether to include index
    
    Returns:
        bool: Success status
    """
    try:
        df.to_csv(filepath, index=index)
        return True
    except Exception as e:
        print(f"Error exporting dataframe: {str(e)}")
        return False


def format_percentage(value, decimals=2):
    """
    Format value as percentage
    
    Args:
        value (float): Value to format
        decimals (int): Number of decimal places
    
    Returns:
        str: Formatted percentage string
    """
    return f"{value*100:.{decimals}f}%"


def create_comparison_table(models_dict, metric='F1 Score'):
    """
    Create comparison table from model results
    
    Args:
        models_dict (dict): Dictionary of model names and scores
        metric (str): Metric name
    
    Returns:
        pd.DataFrame: Formatted comparison table
    """
    df = pd.DataFrame(list(models_dict.items()), columns=['Model', metric])
    df = df.sort_values(metric, ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    return df[['Rank', 'Model', metric]]


def get_top_features(feature_importance_dict, n=10):
    """
    Get top N most important features
    
    Args:
        feature_importance_dict (dict): Dictionary of features and importance scores
        n (int): Number of top features to return
    
    Returns:
        pd.DataFrame: Top features dataframe
    """
    df = pd.DataFrame(list(feature_importance_dict.items()), 
                     columns=['Feature', 'Importance'])
    return df.nlargest(n, 'Importance').reset_index(drop=True)


def calculate_class_distribution(df, target_columns):
    """
    Calculate class distribution for target variables
    
    Args:
        df (pd.DataFrame): Dataset
        target_columns (list): List of target column names
    
    Returns:
        pd.DataFrame: Class distribution summary
    """
    distributions = {}
    
    for col in target_columns:
        dist = df[col].value_counts().sort_index()
        distributions[col] = dist
    
    return pd.DataFrame(distributions).fillna(0).astype(int)


def check_missing_values(df):
    """
    Comprehensive missing value analysis
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    
    Returns:
        pd.DataFrame: Missing value report
    """
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': missing_count.values,
        'Missing_Percent': missing_percent.values
    })
    
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
        'Missing_Count', ascending=False
    ).reset_index(drop=True)
    
    return missing_df


def encode_likert_scale(df, columns, scale_range=(1, 5)):
    """
    Ensure Likert scale columns are properly encoded
    
    Args:
        df (pd.DataFrame): Dataset
        columns (list): Columns to encode
        scale_range (tuple): Valid scale range (min, max)
    
    Returns:
        pd.DataFrame: Dataset with encoded columns
    """
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            # Clip values to valid range
            df_copy[col] = df_copy[col].clip(scale_range[0], scale_range[1])
    
    return df_copy


def generate_correlation_summary(df, threshold=0.7):
    """
    Find highly correlated variable pairs
    
    Args:
        df (pd.DataFrame): Dataset
        threshold (float): Correlation threshold
    
    Returns:
        pd.DataFrame: Highly correlated pairs
    """
    # Convert to numeric
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = df_numeric.corr()
    
    # Find pairs above threshold
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                pairs.append({
                    'Variable_1': corr_matrix.columns[i],
                    'Variable_2': corr_matrix.columns[j],
                    'Correlation': round(corr_matrix.iloc[i, j], 3)
                })
    
    return pd.DataFrame(pairs).sort_values('Correlation', 
                                          key=abs, 
                                          ascending=False)


def prepare_prediction_output(original_df, predictions, confidence_scores=None):
    """
    Combine original data with predictions
    
    Args:
        original_df (pd.DataFrame): Original dataset
        predictions (np.ndarray): Predicted values
        confidence_scores (np.ndarray, optional): Confidence scores
    
    Returns:
        pd.DataFrame: Combined dataset with predictions
    """
    result_df = original_df.copy()
    
    # Add predictions
    pred_cols = [f'ITS{i}_Predicted' for i in range(1, predictions.shape[1] + 1)]
    pred_df = pd.DataFrame(predictions, columns=pred_cols, index=result_df.index)
    result_df = pd.concat([result_df, pred_df], axis=1)
    
    # Add confidence scores if provided
    if confidence_scores is not None:
        conf_cols = [f'ITS{i}_Confidence' for i in range(1, confidence_scores.shape[1] + 1)]
        conf_df = pd.DataFrame(confidence_scores, columns=conf_cols, index=result_df.index)
        result_df = pd.concat([result_df, conf_df], axis=1)
    
    return result_df


def format_model_name(name):
    """
    Format model name for display
    
    Args:
        name (str): Raw model name
    
    Returns:
        str: Formatted name
    """
    name_map = {
        'RandomForest': 'Random Forest',
        'GradientBoosting': 'Gradient Boosting',
        'LogisticRegression': 'Logistic Regression',
        'DecisionTree': 'Decision Tree',
        'XGBoost': 'XGBoost',
        'SVM': 'Support Vector Machine',
        'KNN': 'K-Nearest Neighbors'
    }
    return name_map.get(name, name)


def calculate_confidence_interval(scores, confidence=0.95):
    """
    Calculate confidence interval for scores
    
    Args:
        scores (array-like): Array of scores
        confidence (float): Confidence level
    
    Returns:
        tuple: (mean, lower_bound, upper_bound)
    """
    from scipy import stats
    
    scores = np.array(scores)
    mean = np.mean(scores)
    std_err = stats.sem(scores)
    interval = std_err * stats.t.ppf((1 + confidence) / 2., len(scores)-1)
    
    return mean, mean - interval, mean + interval


# Visualization utilities

def set_plot_style():
    """Set consistent plot styling"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11


def save_figure(fig, filepath, dpi=150):
    """
    Save figure with consistent settings
    
    Args:
        fig: Matplotlib figure
        filepath (str): Output path
        dpi (int): Resolution
    """
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        return True
    except Exception as e:
        print(f"Error saving figure: {str(e)}")
        return False


def create_metric_card_data(accuracy, precision, recall, f1):
    """
    Create formatted data for metric cards
    
    Args:
        accuracy, precision, recall, f1 (float): Metric values
    
    Returns:
        dict: Formatted metrics
    """
    return {
        'Accuracy': format_percentage(accuracy),
        'Precision': format_percentage(precision),
        'Recall': format_percentage(recall),
        'F1 Score': format_percentage(f1)
    }
