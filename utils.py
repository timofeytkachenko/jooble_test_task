import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm

# Features
FEATURE_CODE = 0
ID_JOB = 'id_job'
MAX_FEATURE_2_INDEX = 'max_feature_2_index'
MAX_FEATURE_2_ABS_MEAN_DIFF = 'max_feature_2_abs_mean_diff'
FEATURE_NAME_PREFIX = 'feature_2_stand_'
MEAN, AVERAGE, STD, MIN, MAX = 'mean', 'average', 'std', 'min', 'max'
EVAL_STATS = [MEAN, AVERAGE, STD, MIN, MAX]


def standardize(test_df, stats_dict):
    """Standardize function

    Parameters
    ----------
    test_df: pd.DataFrame
        Test dataFrame
    stats_dict: dict
        Dictionary with training data statistics(mean, average, std, min, max)

    Returns
    -------
    pd.DataFrame
        Standardized dataframe
    """
    return (test_df - stats_dict[MEAN]) / stats_dict[STD]


def train_preprocessing_feature_code_2(train_df_path, chunksize):
    """Train dataframe preprocessing

    Parameters
    ----------
    train_df_path: str
        Path to training data(tsv format)
    chunksize: int
        Preprocessing batch size

    Returns
    -------
    stats_dict: dict
        Dictionary with training data statistics(mean, average, std, min, max)
   """

    # Read train_df
    train_df = pd.read_csv(train_df_path, sep='\t', chunksize=chunksize)

    # Gathering train_df statistics by chunks
    stats_list = list()
    pbar = tqdm(train_df)
    pbar.set_description("Training dataframe processing")
    for chunk in pbar:
        # Split string features by separator and remove first column(feature code)
        chunk = chunk.features.str.split(',', expand=True).drop(columns=[FEATURE_CODE]).astype(int)
        stats_list.append(chunk.agg(EVAL_STATS))

    # Evaluate main statistics
    stats_dict = {stat: pd.concat([item.loc[stat] for item in stats_list], axis=1).mean(axis=1) for stat in
                  [MEAN, AVERAGE, STD]}

    # Evaluate min,max
    min_max_dict = {stat: pd.concat([item.loc[stat] for item in stats_list], axis=1).agg(stat, axis=1) for stat in
                    [MIN, MAX]}

    # Dataframe statistics dict
    stats_dict.update(min_max_dict)

    return stats_dict


def test_preprocessing_feature_code_2(test_df_path, test_proc_save_path, stats_dict, chunksize, norm_func):
    """Test dataframe preprocessing

    Parameters
    ----------
    test_df_path: str
        Path to training data(tsv format)
    test_proc_save_path: str
        Path to save preprocessed test data(tsv format)
    stats_dict: dict
        Dictionary with training data statistics(mean, average, std, min, max)
    chunksize: int
        Preprocessing batch size
    norm_func: callable
        Custom normalization function.
   """

    # Read test_df
    test_df = pd.read_csv(test_df_path, sep='\t', chunksize=chunksize)

    pbar = tqdm(enumerate(test_df))
    pbar.set_description("Test dataframe processing")
    for chunk_ind, chunk in pbar:
        # Split string features by separator and remove first column(feature code)
        chunk = pd.concat(
            [chunk.id_job, chunk.features.str.split(',', expand=True).drop(columns=[FEATURE_CODE]).astype(int)], axis=1)

        # Job feature columns
        job_features = list(
            set(chunk.columns.tolist()).difference([ID_JOB, MAX_FEATURE_2_INDEX, MAX_FEATURE_2_ABS_MEAN_DIFF]))

        # Get max id from feature vector
        max_id = chunk[job_features].idxmax(axis=1)

        # Make max id column
        chunk[MAX_FEATURE_2_INDEX] = max_id

        # Abs Mean Diff evaluation
        chunk[MAX_FEATURE_2_ABS_MEAN_DIFF] = chunk.lookup(max_id.index, max_id) - stats_dict[MEAN][max_id].values

        # Dataframe with stanfardize features
        chunk = pd.concat([chunk[[ID_JOB, MAX_FEATURE_2_INDEX, MAX_FEATURE_2_ABS_MEAN_DIFF]],
                           norm_func(chunk[job_features], stats_dict)], axis=1)

        # Rename columns
        column_names = {name: FEATURE_NAME_PREFIX + str(name) for name in job_features}
        chunk.rename(columns=column_names, inplace=True)

        # Write csv
        header = True if chunk_ind == 0 else False
        chunk.to_csv(test_proc_save_path, sep='\t', index=False, header=header, columns=chunk.columns, mode='a')


def preprocessing(train_df_path, test_df_path, test_proc_save_path, chunksize=1000, norm_func=None):
    """Dataframe preprocessing. Test data normalization and preprocessing.

    Parameters
    ----------
    train_df_path: str
        Path to training data(tsv format)
    test_df_path: str
        Path to test data(tsv format)
    test_proc_save_path: str
        Preprocessed data save path
    chunksize: int
        Preprocessing batch size
    norm_func: callable or None
        Custom normalization function. The default is None which implies standardizing(Z-Score).
   """

    # Read first row
    feature_code = None
    train_df = pd.read_csv(train_df_path, sep='\t', chunksize=1)
    for chunk in train_df:
        # Get feature code
        feature_code = int(chunk.features.str.split(',', expand=True).iloc[0, FEATURE_CODE])
        break

    # Normalization function(standardizing by default)
    if norm_func is None:
        norm_func = standardize

    if feature_code == 2:
        stats_dict = train_preprocessing_feature_code_2(test_df_path, chunksize)
        test_preprocessing_feature_code_2(test_df_path, test_proc_save_path, stats_dict, chunksize=chunksize,
                                          norm_func=norm_func)
    else:
        raise Exception('Unknown feature code')
