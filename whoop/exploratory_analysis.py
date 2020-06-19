# Created by Philip P on 23 May 2020
# Exploratory data analysis into csv downloaded from habitdash.com

import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 600)


def clean_input_whoop_data(input_data: pd.DataFrame) -> pd.DataFrame:
    """Method to clean input Whoop dataframe from HabitDash.com

    Args:
        input_data: Input csv titled "YYYY-MM-DD Habit Dash (flat file).csv"

    Returns:
        input_date: Cleaned dataframe with columns ['date', 'field', 'value']
    """
    assert 'date' in input_data.columns, "Input data does not have a valid 'date' column"

    print(f"Reading input data, {input_data['date'].nunique()} "
          f"days worth of data...")

    # rename the field column
    new_field_name_map = \
        {k: k.replace("whoop_", "") for k in input_data['field'].unique()}

    input_data['field'] = input_data['field'].map(new_field_name_map)
    input_data.drop('source', axis=1, inplace=True)
    input_data['date'] = pd.to_datetime(input_data['date'])

    return input_data

if __name__ == "__main__":

    # --- READ WHOOP DATA
    data_dir = "/Users/philip_p/Documents/whoop/"
    whoop_file = [x for x in os.listdir(data_dir) if re.search("flat file", x)][0]
    habit_dash_df = pd.read_csv(os.path.join(data_dir, whoop_file))

    cleaned_df = clean_input_whoop_data(input_data=habit_dash_df)

    cleaned_df['field'].nunique() # 31 measures

    macro_fields = \
        set([x.split("_")[0] for x in cleaned_df['field'].unique()])

    # recovery score plotting
    recovery = cleaned_df.loc[cleaned_df['field'] == 'recovery_score'].copy(True)
    recovery['colour'] = pd.cut(x=recovery['value'],
                                bins=[0, 33, 67, 100],
                                labels=['red', 'yellow', 'green'])
    sns.set_style('darkgrid')
    recover_plot = sns.lineplot(x='date', y='value',
                                data=recovery)
    recover_plot.axes.axhline(y=33, color='yellow')
    recover_plot.axes.axhline(y=67, color='green')

    recover_plot = sns.scatterplot(x='date', y='value',
                                   data=recovery)

    # principal component analysis
    # recovery (for next day) is response variable, inputs are HRV, sleep score, RHR
    cleaned_df.head(3)

    summary_df = cleaned_df.loc[cleaned_df['field'].isin(
        ['recovery_score', 'recovery_rhr', 'recovery_hrv', 'sleep_score_total'])
    ]

    pca_df = pd.pivot(
        data=summary_df,
        index='date',
        columns='field'
    )
    pca_df.columns = list(pca_df.columns.get_level_values(1))

    from sklearn.preprocessing import StandardScaler

    features = ['recovery_hrv', 'recovery_rhr', 'sleep_score_total']
    x = pca_df.loc[:, features].values
    y = pca_df.loc[:, 'recovery_score'].values

    x = StandardScaler().fit_transform(x)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2) # fit to two main principal components

    principal_components = pca.fit_transform(x)
    principal_df = pd.DataFrame(
        data=principal_components,
        columns=['pc_one', 'pc_two']
    )

    final_df = pd.concat([principal_df, pca_df['recovery_score']], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    targets = ['']

    # look at correlation between HRV, sleep performance and RHR and Recovery score
    pca_df.columns
    from scipy.stats import spearmanr, pearsonr

    hrv_corr, _ = spearmanr(pca_df['recovery_hrv'], pca_df['recovery_score'])
    hrv_corr_pearson, _ = pearsonr(pca_df['recovery_hrv'], pca_df['recovery_score'])
    rhr_corr, _ = spearmanr(pca_df['recovery_rhr'], pca_df['recovery_score'])
    sleep_performance_corr, _ = spearmanr(pca_df['sleep_score_total'], pca_df['recovery_score'])

    pca_df_detailed = pca_df.copy(True)
    pca_df_detailed['status'] = pd.cut(x=pca_df_detailed['recovery_score'],
                                       bins=[0, 33, 67, 100],
                                       labels=['red', 'yellow', 'green'])
    hrv_ax = sns.scatterplot(
        x='recovery_hrv',
        y='recovery_score',
        data=pca_df_detailed,
        hue='status',
        palette=['red', 'orange', 'green']
    )
    hrv_ax.set_title("WHOOP Correlation: HRV v Recovery Score")

    # correlation
    pca_df.corr()

    rhr_ax = sns.scatterplot(
        x='recovery_rhr',
        y='recovery_score',
        data=pca_df_detailed,
        hue='status',
        palette=['red', 'orange', 'green']
    )
    rhr_ax.set_title("WHOOP Correlation: RHR v Recovery Score")

    sleep_ax = sns.scatterplot(
        x='sleep_score_total',
        y='recovery_score',
        data=pca_df_detailed,
        hue='status',
        palette=['red', 'orange', 'green']
    )
    sleep_ax.set_title("WHOOP Correlation: Sleep Score v Recovery Score")
