import pandas as pd
import numpy as np
from tqdm import tqdm
import re

tqdm.pandas()

data_src = 'all_variants.csv'
data = pd.read_csv(data_src)  # read the .csv files with all covid submissions


def only_substs(s):
    """
    Leaves only substitutions out of all the mutations of the covid variant.
    Excludes deletions and insertions. Uses regular expressions.
    Please note that mutations become sorted in a string.
    :param s: string representing mutations in the original file (separeted by a comma)
    :return: the same string without the deletions and insertions
    """
    return ','.join(sorted(re.findall(r'\w+\d?_[ACDEFGHIKLMNPQRSTVWY]\d+[ACDEFGHIKLMNPQRSTVWY]', s)))


def extract_substs(df):
    """
    Extracts substitutions out of dataset as a part of pipeline.
    :param df: dataframe with covid submissions
    :return: same dataframe only with substitutions in .combineMuts column
    """
    df['combineMuts'] = df['combineMuts'].progress_apply(only_substs)
    return df


def assign_frequencies_le(df):
    """
    Gets frequencies of each UNIQUE submission .
    Dataframe MUST possess column .var_le - labels encodings for unique variants (submissions).
    :param df: dataframe with covid submissions
    :return: same dataframe with frequencies for unique submissions as a new column
    """
    unique, counts = np.unique(df.var_le, return_counts=True)
    freq_dict = {unique[i]: counts[i] for i in range(len(unique))}
    df['frequency_le'] = df.var_le.progress_apply(lambda x: freq_dict[x])
    return df


def assign_frequencies_clade(df):
    """
    Gets frequencies of each unique clade .
    :param df: dataframe with covid submissions
    :return: same dataframe with frequencies for unique clades as a new column
    """
    unique, counts = np.unique(df.currCovClade, return_counts=True)
    freq_dict = {unique[i]: counts[i] for i in range(len(unique))}
    df['frequency_clade'] = df.currCovClade.progress_apply(lambda x: freq_dict[x])
    return df


def assign_frequencies_lineage(df):
    """
    Gets frequencies of each unique lineage .
    :param df: dataframe with covid submissions
    :return: same dataframe with frequencies for unique lineages as a new column
    """
    unique, counts = np.unique(df.currLineage, return_counts=True)
    freq_dict = {unique[i]: counts[i] for i in range(len(unique))}
    df['frequency_lineage'] = df.currLineage.progress_apply(lambda x: freq_dict[x])
    return df


def drop_unassigned_lineages(df):
    """
    Drops submissions with unassigned lineages.
    :param df: dataframe with covid submissions
    :return: same dataframe without submissions with unassigned lineages
    """
    return df[df.currLineage != 'Unassigned']


df = (data
      [['drop', 'combineMuts', 'currCollectiondate', 'currCovClade', 'currLineage']]  # choose specific columns
      .dropna()  # prop rows with nan values
      # .loc[:1000,:]#[:8000000,0]
      .pipe(drop_unassigned_lineages)
      .sort_values(by='currCollectiondate')  # sort df by collection date
      .pipe(assign_frequencies_clade)
      .pipe(assign_frequencies_lineage)
      .assign(n_muts=lambda df_: df_.combineMuts.apply(lambda x: x.count(',')))  # count number of mutations for each
      # submission. Please note that for deletions and insertion actual number of mutations could be grater
      .pipe(extract_substs)
      .assign(n_substs=lambda df_: df_.combineMuts.apply(lambda x: x.count(',')))  # count number of substitutions
      # for each submission
      .assign(var_le=lambda df_: pd.factorize(df_.combineMuts)[0])  # assign label encodings for unique variants
      .pipe(assign_frequencies_le)
      )

df.to_csv('score_date_substs_freq.csv', index=False)  # save new .csv
