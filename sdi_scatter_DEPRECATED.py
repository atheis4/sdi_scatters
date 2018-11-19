# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 10:20:24 2016

@author: atheis
"""

import argparse
import time
import warnings

import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import ColorConverter
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer

from db_tools.ezfuncs import query
from db_queries import get_model_results
from db_queries import get_covariate_estimates
from db_queries import get_location_metadata


def get_age_weights():
    """."""
    sql_query = """
        SELECT
            age_group_id,
            age_group_weight_value
        FROM
            shared.age_group_weight agw
        JOIN
            shared.gbd_round USING (gbd_round_id)
        WHERE
            gbd_round = 2016
        AND
            age_group_weight_description = 'IHME standard age weight';
          """
    age_standard_data = query(sql_query, conn_def='cod')
    return age_standard_data


def generate_asr(data, index_columns, data_columns, age_weights):
    """
    Takes in a dataframe in count space, calculates age-std, and adds it in

    Returns:
        The original dataframe with age-std rates appended
    """
    temp = data.copy(deep=True)
    temp = pd.merge(data, age_weights, on=['age_group_id'])
    for c in data_columns:
        temp[c] = temp[c] * temp['age_group_weight_value']
    temp['age_group_id'] = 27
    temp = temp.drop('age_group_weight_value', axis=1)
    temp = temp.groupby(index_columns).sum().reset_index()
    return temp


def get_unique_values(data, column):
    """
    Takes a dataframe and column name.

    Returns:
    List of unique column values
    """
    return list(np.unique(data.loc[:, column]))


def generate_cmap():
    """
    Return custom cmap with IHME region colors
    """
    return ListedColormap(_return_rgb_colors(), 'region_color_code', N=21)


def _return_rgb_colors():
    """."""
    colors = ['#9E0142', '#B31C42', '#C93742', '#DE5242', '#F46D43', '#F68955', '#F9A667',
              '#FBC379', '#FEE08B', '#F8E58E', '#F2EA91', '#ECEF94', '#E6F598', '#C6E89B',
              '#A6DB9E', '#86CEA1', '#66C2A5', '#64A5A4', '#6288A3', '#606BA2', '#5E4FA2']
    cc = ColorConverter()
    return [cc.to_rgb(i) for i in colors]


def create_legend_patches():
    """."""
    regions = ['East Asia', 'Southeast Asia', 'Oceania', 'Central Asia',
               'Central Europe', 'Eastern Europe', 'High-income Asia Pacific',
               'Australasia', 'Western Europe', 'Southern Latin America',
               'High-income North America', 'Caribbean',
               'Andean Latin America', 'Central Latin America',
               'Tropical Latin America', 'North Africa and Middle East',
               'South Asia', 'Central Sub-Saharan Africa',
               'Eastern Sub-Saharan Africa', 'Southern Sub-Saharan Africa',
               'Western Sub-Saharan Africa']
    return [mpatches.Patch(color=_return_rgb_colors()[i], label=regions[i]) for i in range(21)]


def merge_dataframes(dataframe_1, dataframe_2, merge_column='location_id'):
    """."""
    return pd.merge(dataframe_1, dataframe_2, on=[merge_column])


def generate_covariate_data():
    """."""
    cov_df = get_covariate_estimates(covariate_id=881)
    cov_df = cov_df[(cov_df.age_group_id == 22) & (cov_df.year_id == 2016)]
    cov_df.rename(columns={'mean_value': 'sdi_value'}, inplace=True)
    return cov_df[['location_id', 'sdi_value']].copy(deep=True)


def generate_location_data():
    df = get_location_metadata(location_set_id=35, gbd_round_id=4)
    df = df[df.level == 3]
    id_list = get_unique_values(df, 'location_id')
    df.region_id = df.region_id.astype(int)
    df = df[['location_id', 'region_id', 'ihme_loc_id']].copy(deep=True)
    return df, id_list


def query_risk_data(mvid):
    q = """
    SELECT
        shared.rei_name
    FROM
        epi.model_estimate_final mef
    JOIN model_version mv
        ON mef.model_version_id=mv.model_version_id
    JOIN modelable_entity_rei mer
        ON mv.modelable_entity_id=mer.modelable_entity_id
    JOIN shared.rei shared
        ON mer.rei_id=shared.rei_id
    WHERE mef.model_version_id={}
    LIMIT 1;
        """.format(mvid)
    return query(q, conn_def='epi')


def query_cause_data(model_ver_id, gbd_team):
    """
    Takes a model_version_id and returns a dataframe with cause information

    Returns:
        dataframe with cause_id, cause_name, cause_medium, and cause_short
    """
    if gbd_team == 'cod':
        q = """
        SELECT
            shared.cause_id, cause_name
        FROM
            cod.model AS cod_db
        JOIN model_version AS mod_ver
            ON cod_db.model_version_id = mod_ver.model_version_id
        JOIN shared.cause AS shared
            ON mod_ver.cause_id = shared.cause_id
        WHERE cod_db.model_version_id = {}
        LIMIT 1;
            """.format(model_ver_id)
        return query(q, conn_def=gbd_team)
    else:
        q = """
        SELECT
            shared.cause_id, cause_name
        FROM
            epi.model_estimate_final AS epi_mod_est
        JOIN model_version AS mod_ver
            ON epi_mod_est.model_version_id = mod_ver.model_version_id
        JOIN modelable_entity_cause AS mod_ent
            ON mod_ver.modelable_entity_id = mod_ent.modelable_entity_id
        JOIN shared.cause AS shared
            ON mod_ent.cause_id = shared.cause_id
        WHERE epi_mod_est.model_version_id = {}
        LIMIT 1;
            """.format(model_ver_id)
        df = query(q, conn_def=gbd_team)
        if len(df) == 0:
            cause = query_risk_data(model_ver_id)
            return cause
        else:
            return df


def prune_epi_dataframe(epi_data):
    """."""
    return epi_data.iloc[:, [1, 2, 3, 10]].copy(deep=True)


def add_normalized_y_to_df(df):
    """."""
    norm = Normalizer().fit(df.iloc[:, 3])
    df['normalized_y'] = norm.transform(df.iloc[:, 3]).reshape(-1, 1)
    return df


def graph_data(df, cause, sex, measure):
    """."""
    # Set up plot and input data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(df.iloc[:, 2], df.iloc[:, 3], c=df.iloc[:, 0], alpha=0.4,
               cmap=generate_cmap(), linewidth=0, s=50)

    # Axes
    max_val = max(df.iloc[:, 3])

    ax.set_xlim(0, 1)
    ax.set_ylim((max_val * 0.99) - max_val, max_val + (max_val * 0.05))

    # Apply ISO3 code to samples without a 2% neighbor
    nearest_neighbors_labels(df, ax)
    label_graph(cause, sex, measure, ax)


def label_graph(cause, sex, measure, ax):
    """."""
    if measure == 'deaths':
        extra_args = ' (ASDR per 100,000)'
    elif measure == 'prevalence':
        extra_args = ' Prevalence per capita'
    elif measure == 'incidence':
        extra_args = ' Incidence per capita'
    else:
        extra_args = ' Mortality Proportion'

    ax.set_title(cause.title() + ' ' + measure.title() + ' in '\
             + sex.title() + '\nto Socio-demographic Index by Country',
             fontsize=18)
    ax.set_xlabel('Socio-demographic Index', fontsize=12)
    ax.set_ylabel(cause.title() + '\n' + extra_args, fontsize=12)


def nearest_neighbors_labels(df, ax):
    """."""
    df = add_normalized_y_to_df(df)

    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(df.iloc[:, [2, -1]])
    radius_nbrs = nbrs.radius_neighbors(df.iloc[:, [2, -1]], return_distance=False, radius=0.015)

    labeled_samples = []
    for i in range(len(radius_nbrs)):
        if len(radius_nbrs[i]) == 1:
            labeled_samples.append(radius_nbrs[i][0])

    for i in labeled_samples:
        ax.annotate(df.iloc[i, 1], alpha=0.5, fontsize=7, xy=df.iloc[i, [2, 3]],
                    xytext=(df.iloc[i, [2]] - 0.0125, df.iloc[i, [3]] - (max(df.iloc[:, 3]) * 0.03)))


def output_pdf(mvid, cause, sex, measure, age_group_id, path):
    """."""
    if age_group_id not in [-1, 27]:
        name_str = str(mvid) + '_' + '2016' + '_' + cause.lower().replace(' ', '_') +'_' + measure + '_' + sex + '_age_group_' + str(age_group_id) + '_sdi_scatter.pdf'
    else:
        name_str = str(mvid) + '_' + '2016' + '_' + cause.lower().replace(' ', '_') + '_' + measure + '_' + sex + '_sdi_scatter.pdf'
    output_str = path + name_str
    plt.savefig(output_str, format='pdf', dpi=300)
    return name_str


def generate_legend(path):
    """."""
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis('off')
    plt.tight_layout()
    ax.legend(handles=create_legend_patches(), loc='center', fontsize=7)

    plt.savefig(path + 'legend.pdf', dpi=300)


def parse_arguments():
    """."""
    parser = argparse.ArgumentParser()
    parser.add_argument('gbd_team', type=str, help='\'cod\' or \'epi\'')
    parser.add_argument('model_ver_id', type=int, help='five digit int')
    parser.add_argument('-measure', '--measure_id', default=None, type=int,
                    help='an integer: 5=prevalence, 6=incidence, 18=proportion')
    parser.add_argument('-by_age', '--by_age', default=None, type=int,
                    help='specify an individual age_group_id')
    parser.add_argument('filepath', type=str, help='location to output your scatterplot')

    args = parser.parse_args()
    gbd_team = args.gbd_team
    model_ver_id = args.model_ver_id
    measure_id = args.measure_id
    if measure_id:
        assert measure_id in [5, 6, 18], 'Invalid measure_id passed: 5 = prevalence, 6 = incidence, 18 = proportion'
    age_group_id = args.by_age
    path = args.filepath

    return gbd_team, model_ver_id, measure_id, age_group_id, path


def main():
    gbd_team, mvid, measure_id, age_group_id, path = parse_arguments()
    if path[-1] != '/':
        path += '/'

    location_df, location_id_list = generate_location_data()
    covariate_df = generate_covariate_data()

    if not age_group_id:
        if gbd_team == 'cod':
            age_group_id = -1
        else:
            age_group_id = 27

    model_results = get_model_results(gbd_team,
                                      model_version_id=mvid,
                                      age_group_id=age_group_id,
                                      measure_id=measure_id,
                                      gbd_round_id=4,
                                      year_id=2016,
                                      location_id=location_id_list)

    # Check dataframe for information
    assert not model_results.iloc[:, 0].empty, 'No gbd round 4 data found for this model version or age group id'

    label_df = query_cause_data(mvid, gbd_team)

    loc_cov_df = merge_dataframes(location_df, covariate_df)

    if gbd_team == 'cod':
        model_results = generate_asr(
            model_results[['location_id', 'year_id', 'age_group_id', 'sex_id', 'mean_death_rate']],
            ['location_id', 'year_id', 'age_group_id', 'sex_id'], ['mean_death_rate'],
            get_age_weights()
        )

        sex = get_unique_values(model_results, 'sex_id')[0]
        sex = 'males' if sex == 1 else 'females'
        measure = 'deaths'
        cause = label_df.iloc[0, 1]

        cod_df = merge_dataframes(loc_cov_df, model_results)
        cod_df = cod_df.iloc[:, [1, 2, 3, 7]].copy(deep=True)
        cod_df[['death_rate_x_100000']] = cod_df[['mean_death_rate']] * 100000
        cod_df.drop(labels='mean_death_rate', axis=1, inplace=True)

        len_error = 'Model version id {} does not return results for all 195 \
                     countries'.format(mvid)
        if len(cod_df) != 195:
            assert len_error

        graph_data(cod_df, cause, sex, measure)
        name_str = output_pdf(mvid, cause, sex, measure, age_group_id, path)

    else:
        cause = label_df.iloc[0, 1]
        sex = ['males', 'females']

        male_epi_results = model_results[model_results.sex_id == 1]
        epi_xy_df = merge_dataframes(loc_cov_df, male_epi_results)
        epi_xy_df = prune_epi_dataframe(epi_xy_df)

        female_epi_results = model_results[model_results.sex_id == 2]
        epi_xx_df = merge_dataframes(loc_cov_df, female_epi_results)
        epi_xx_df = prune_epi_dataframe(epi_xx_df)

        if measure_id == 5:
            measure = 'prevalence'
        elif measure_id == 6:
            measure = 'incidence'
        elif measure_id == 18:
            measure = 'proportion'

        if len(epi_xx_df) != 195 or len(epi_xy_df) != 195:
            assert len_error

        graph_data(epi_xy_df, cause, sex[0], measure)
        epi_xy_name_str = output_pdf(mvid, cause, sex[0], measure, age_group_id, path)

        graph_data(epi_xx_df, cause, sex[1], measure)
        epi_xx_name_str = output_pdf(mvid, cause, sex[1], measure, age_group_id, path)

    plt.clf()
    generate_legend(path)

    print 'Success!\n-------------'
    if gbd_team == 'epi':
        print 'File name: {}'.format(epi_xy_name_str)
        print 'File name: {}'.format(epi_xx_name_str)
    else:
        print 'File name: {}'.format(name_str)
    print 'File location: {}'.format(path)
    print '-------------'
    print 'legend.pdf also saved to {}'.format(path)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    start_time = time.time()
    main()
    print 'Runtime: {} seconds'.format(time.time() - start_time)
