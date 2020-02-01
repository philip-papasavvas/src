"""
Created on: 15 Feb 2020
Analysis of data on CrossFit Open scores from Kaggle:
https://www.kaggle.com/jeanmidev/crossfit-games
"""
import os
import pandas as pd
import numpy as np
import re
from collections import Counter

# plotting
import plotly.graph_objects as go
import seaborn as sns

from utils import find, change_dict_keys

data_dir = "/Users/philip_p/Documents/cf_open_data"
pd.set_option("display.max_columns", 10)

scores_df_drop_cols = ["breakdown", "judge", "score", "is_scaled", "scoreidentifier"]

def return_columns_dict(input_df, col1, col2):
    return dict(zip(input_df[col1], input_df[col2]))


def cleanse_scores_df(input_df, percentile=True, division=None, scaled=False, workout=None,
                      cols_to_drop=None):
    """Function to cleanse the athelete scores df input dataframe

    Args:
        input_df (dataframe): Input dataframe of CrossFit Open scores (can be from any year)
        percentile (bool, default True): Whether to add a percentile to each score for each workout
        division (str, list): Divisions interested in doing analysis for:
            ['Men', Men (35-39)', 'Men (40-44)', 'Men (45-49)', 'Men (50-54)', 'Men (55-59)', 'Men (60+)',
            'Women', 'Women (35-39)', 'Women (40-44)', 'Women (50-54)', 'Women (45-49)', 'Women (55-59)', 'Women (60+)']
        scaled (bool, default False): Return the scaled athlete scores
        workout (int): Which open workout it was: 1 to 5 (2020)
        cols_to_drop (str,list): Columns to drop from dataframe

    Returns:
        final_df (pd.DataFrame): Final dataframe after filtering etc
    """
    df_minus_cols = input_df.drop(labels=cols_to_drop, axis=1)

    if division is None:
        division = []

    assert "division" in df_minus_cols.columns, "Division has been dropped from columns," \
                                                "therefore cannot filter for this"
    if division is None:
        division = df_minus_cols["divison"].unique()

    drop_divs_df = df_minus_cols.loc[df_minus_cols['division'].isin(division)].copy(True)

    assert "scaled" in drop_divs_df.columns, "Scaled has been dropped from columns, so " \
                                             "cannot filter for this"

    post_scaled_df = drop_divs_df.loc[drop_divs_df['scaled'] == scaled].copy(True)
    post_scaled_df.rename(columns={'ordinal': 'workout'}, inplace=True)

    assert "workout" in post_scaled_df.columns, "The workout is column is not present and" \
                                                "cannot be filtered"

    workout_df_full = post_scaled_df.copy(True)
    if workout is None:
        workout_df = workout_df_full.loc[workout_df_full["workout"].ge(1)]
        print("No specific workout was selected, hence all chosen")
    else:
        workout_df = workout_df_full.loc[workout_df_full["workout"] == workout]

    workout_df.fillna({'affiliate': 'unaffiliated'}, inplace=True)

    if percentile:
        pass

    final_df = workout_df[pd.notnull(workout_df['scoredisplay'])].copy(True)

    return final_df


if __name__ == "__main__":
    # read the data
    score_df = pd.read_csv(find(folder_path=data_dir, pattern=["2020", "score"], full_path=True))
    athlete_df = pd.read_csv(find(folder_path=data_dir, pattern=["2020", "athletes"], full_path=True))

    country_dict = return_columns_dict(input_df=athlete_df, col1="countryoforigincode",
                                       col2="countryoforiginname")
    athlete_dict = return_columns_dict(input_df=athlete_df, col1="competitorid",
                                       col2="competitorname")

    all_df = cleanse_scores_df(input_df=score_df, division=None, workout=None,
                               cols_to_drop=scores_df_drop_cols)
    # all_df = cleanse_scores_df(input_df=score_df, division=["Men", "Women"], workout=None)
    men_per_workout = dict(Counter(all_df[all_df['division'] == "Men"]['workout']))
    women_per_workout = dict(Counter(all_df[all_df['division'] == "Women"]['workout']))

    aldgate_df = all_df.loc[all_df['affiliate'].str.contains("aldgate", case=False)].copy(True)
    aldgate_df['name'] = aldgate_df['competitorid'].map(athlete_dict)


    new_men_count_dict = change_dict_keys(in_dict=men_per_workout, text='Men')
    new_women_count_dict = change_dict_keys(in_dict=women_per_workout, text='Women')
    joint_count_dict = {**new_men_count_dict, **new_women_count_dict}

    all_df['gender_workout'] = all_df['division'] + "_" + all_df['workout'].map(str)
    all_df['percentile'] = 100 * all_df['rank'] / all_df['gender_workout'].map(joint_count_dict)
    all_df['name'] = all_df['competitorid'].map(athlete_dict)

    all_df[all_df['name'].str.contains("Papasavvas", case=False)]

    b_plot = sns.boxplot(x='workout', y='percentile', hue="division",
                         orient="v",
                         data=all_df.loc[all_df['affiliate'].str.contains("aldgate", case=False)],
                         ).set(xlabel="2020 Open Workout",
                               ylabel="Percentile for Workout",
                               title="CrossFit Aldgate - The Open 2020 - Results"
                               )

    sns.violinplot(x='workout', y='rank', hue="division", data=all_df, split=True,
                   inner="quart")
    sns.despine(left=True)
