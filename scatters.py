#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:18:03 2017

@author: atheis
"""

import argparse
import os
import time
import warnings

import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import ColorConverter
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from db_tools.ezfuncs import query
from db_queries import get_location_metadata
from db_queries import get_covariate_estimates
from db_queries import get_model_results

from stata_wrapper_functions import get_outputs_temp as go


def parse_args():
    """Assing and validate command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('risk_type', type=str,
                        help='\'exposure\' or \'sev\'')
    parser.add_argument('-mvid', type=int, default=None,
                        help='five digit int')
    parser.add_argument('-measure', '--measure_id', default=None, type=int,
                        help='an int representing the measure_id to query')
    parser.add_argument('-rei', '--rei_id', default=None, type=int,
                        help='rei_id of the risk factor for SEV scatters.')
    parser.add_argument('-by_age', '--age_group_id', default=None, type=int,
                        help='specify an individual age_group_id')
    parser.add_argument('-by_sex', '--sex_id', default=-1, type=int,
                        help='1: males, 2: females, 3: both (default)')
    parser.add_argument('filepath', type=str,
                        help='scatterplot destination location')
    args = parser.parse_args()

    # separate args into variables and test for dependency parameters
    rf_type = args.risk_type
    mvid = args.mvid
    measure_id = args.measure_id
    rei_id = args.rei_id
    age_group_id = args.age_group_id
    sex_id = args.sex_id
    outpath = args.filepath

    # define error messages for inputs
    rf_e = 'first parameter must be \'exposure\' or \'sev\''
    mvid_e = 'You must provide a model version id for an exposure scatter'
    measure_e = 'You must provide a measure_id value for an exposure scatter'
    sev_e = 'must enter an rei_id for risk factor SEV scatters'

    # validate inputs
    if rf_type == 'exposure':
        assert mvid, mvid_e
        assert measure_id, measure_e
    elif rf_type == 'sev':
        assert rei_id, sev_e
    else:
        raise ValueError, rf_e

    # add backslash onto filepath if missing
    if outpath[-1] != '/':
        outpath += '/'

    return rf_type, mvid, measure_id, rei_id, age_group_id, sex_id, outpath


def generate_location_data(gbd_round_id=4):
    df = get_location_metadata(location_set_id=35, gbd_round_id=gbd_round_id)
    df = df[df.level == 3]
    id_list = list(np.unique(df.location_id))
    df.region_id = df.region_id.astype(int)
    df = df[['location_id', 'region_id', 'ihme_loc_id']].copy(deep=True)
    return df, id_list


def generate_covariate_data(gbd_round_id=4):
    round_map = {3: 2015, 4: 2016}
    cov_df = get_covariate_estimates(covariate_id=881)
    cov_df = cov_df[(cov_df.age_group_id == 22) &
                    (cov_df.year_id == round_map[gbd_round_id])]
    cov_df.rename(columns={'mean_value': 'sdi_value'}, inplace=True)
    return cov_df[['location_id', 'sdi_value']].copy(deep=True)


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


def query_risk_name_from_id(rei_id):
    q = """
    SELECT
        shared.rei_name
    FROM
        shared.rei shared
    WHERE
        shared.rei_id={}
    LIMIT 1;
    """.format(rei_id)
    return query(q, conn_def='epi')


def rgb():
    colors = ['#9E0142', '#B31C42', '#C93742', '#DE5242', '#F46D43', '#F68955',
              '#F9A667', '#FBC379', '#FEE08B', '#F8E58E', '#F2EA91', '#ECEF94',
              '#E6F598', '#C6E89B', '#A6DB9E', '#86CEA1', '#66C2A5', '#64A5A4',
              '#6288A3', '#606BA2', '#5E4FA2']
    cc = ColorConverter()
    return [cc.to_rgb(i) for i in colors]


def create_legend_patches():
    regions = ['East Asia', 'Southeast Asia', 'Oceania', 'Central Asia',
               'Central Europe', 'Eastern Europe', 'High-income Asia Pacific',
               'Australasia', 'Western Europe', 'Southern Latin America',
               'High-income North America', 'Caribbean',
               'Andean Latin America', 'Central Latin America',
               'Tropical Latin America', 'North Africa and Middle East',
               'South Asia', 'Central Sub-Saharan Africa',
               'Eastern Sub-Saharan Africa', 'Southern Sub-Saharan Africa',
               'Western Sub-Saharan Africa']
    return [mpatch.Patch(color=rgb()[i], label=regions[i]) for i in range(21)]


def generate_cmap():
    return ListedColormap(rgb(), 'region_color_code', N=21)


def add_normalized_y(df):
    df['norm_y'] = normalize(df.iloc[:, 2]).reshape(-1, 1)
    return df


def graph_data(df, rei_name, measure):
    # Set up plot and input data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(df.iloc[:, 3], df.iloc[:, 5], c=df.iloc[:, 1], alpha=0.4,
               cmap=generate_cmap(), linewidth=0, s=50)

    # Axes
    max_val = max(df.iloc[:, 5])
    min_val = min(df.iloc[:, 5])

    # set limits
    ax.set_xlim(0, 1)
    ax.set_ylim((min_val * .99) - min_val, max_val + (max_val * .05))

    # NearestNeighbors labeling
    nearest_neighbors_labels(df, ax)

    # Label
    sex_map = {1: 'Males', 2: 'Females'}
    sex = df.iloc[0, 4]
    rei_name = str(rei_name)
    ax.set_xlabel('Socio-Demographic Index', fontsize=12)
    measure = measure.title() if measure != 'SEV' else measure
    ax.set_ylabel('{} in {} (per capita)'.format(
                  measure, sex_map[sex]),
                  fontsize=12)
    ax.set_title('{} {}\nin {} to Socio-Demographic Index'.format(
                 rei_name.title(),
                 measure,
                 sex_map[sex]),
                 fontsize=18)


def nearest_neighbors_labels(df, ax):
    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(df.iloc[:, [3, 6]])
    radius_nbrs = nbrs.radius_neighbors(df.iloc[:, [3, 6]],
                                        return_distance=False,
                                        radius=0.015)

    labeled_samples = []
    for i in range(len(radius_nbrs)):
        if len(radius_nbrs[i]) == 1:
            labeled_samples.append(radius_nbrs[i][0])

    for i in labeled_samples:
        ax.annotate(df.iloc[i, 2],
                    alpha=0.5,
                    fontsize=7,
                    xy=df.iloc[i, [3, 5]],
                    xytext=(df.iloc[i, [3]] - 0.015,
                            df.iloc[i, [5]] - (max(df.iloc[:, 5]) * 0.04)))


def create_filename(df, rei_name, mvid, rei_id, measure, year):
    sex_map = {1: 'males', 2: 'females'}
    sex = df.iloc[0, 4]
    if mvid:
        name_str = '{}_{}_{}_{}_{}_sdi_scatter.pdf'.format(str(mvid),
            rei_name.lower().replace(' ', '_'),
            measure.lower().replace(' ', '_'),
            sex_map[sex],
            str(year))
    else:
        name_str = '{}_{}_{}_{}_sev_sdi_scatter.pdf'.format(str(rei_id),
            rei_name.lower().replace(' ', '_'),
            sex_map[sex],
            str(year))
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


def main():
    # unpack command line args
    rf_type, mvid, measure_id, rei_id, age_group_id, \
        sex_id, outpath = parse_args()

    # generate location and covariate data from db
    loc_df, loc_ids = generate_location_data()
    sdi_df = generate_covariate_data()

    # set default age_group_id
    if not age_group_id:
        age_group_id = 27

    # query dbs for data
    if rf_type == 'exposure':
        print('Checking for 2016 results...')
        df = get_model_results('epi',
                               model_version_id=mvid,
                               age_group_id=age_group_id,
                               measure_id=measure_id,
                               gbd_round_id=4,
                               year_id=2016,
                               location_id=loc_ids)
        year = 2016
        # if gbd_round and year_id don't generate results, query 4 and 2016
        if len(df) == 0:
            print('2016 results not found, defaulting to 2015 results...')
            # reset loc and cov dfs for round 4 and 2016 data
            loc_df, loc_ids = generate_location_data(3)
            sdi_df = generate_covariate_data(3)
            df = get_model_results('epi',
                                   model_version_id=mvid,
                                   age_group_id=age_group_id,
                                   measure_id=measure_id,
                                   gbd_round_id=3,
                                   year_id=2015,
                                   location_id=loc_ids)
            year = 2015
        # isolate REI name
        rei_name = str(query_risk_data(mvid).iloc[0, 0])
        # isolate measure value
        measure = str(df.iloc[0, 6])
        df = df.iloc[:, [1, 4, 7]]
    else:
        print('Checking for 2016 results...')
        inpath = '/share/central_comp/sev/181/summary/to_upload/{}.csv'.format(
            rei_id)
        if os.path.isfile(inpath):
            # read csv
            df = pd.read_csv(inpath)
            # limit to 2016 and age-standardized results
            df = df[(df.age_group_id == 27) & (df.year_id == 2016)]
            # merge on location data
            df = pd.merge(df,
                          pd.DataFrame(loc_ids, columns=['location_id']),
                          on='location_id')
            year = 2016
            rei_name = query_risk_name_from_id(rei_id).iloc[0, 0]
            measure = 'SEV'
            df = df.iloc[:, [2, 3, 7]]
        else:
            print('2016 results not found, defaulting to 2015 results...')
            print('Generating Stata query:')
            df = go('rei',
                    rei_id=rei_id,
                    age_group_id=27,
                    measure_id=29,
                    metric_id=3,
                    gbd_round=2015,
                    year_id=2015,
                    location_id=loc_ids,
                    sex_id=[1, 2])
            year = 2015
            # isolate REI name
            rei_name = df.iloc[0, 4]
            # isolate measure value
            measure = 'SEV'
            df = df.iloc[:, [5, 10, 14]]

    # if rei name includes gender, remove gender from name.
    if 'males' in rei_name or 'females' in rei_name:
        rei_list = rei_name.split()
        rei_list = rei_list[:-2]
        rei_name = ' '.join(rei_list)

    # add normalized y value to end of df
    df = add_normalized_y(df)

    # merge db data with loc_sdi data
    loc_sdi_df = pd.merge(loc_df, sdi_df, on='location_id')
    df = pd.merge(loc_sdi_df, df, on='location_id')

    # split data on sex
    male_df = df[df.sex_id == 1]
    female_df = df[df.sex_id == 2]

    if len(male_df) > 0:
        # graph and save male data
        graph_data(male_df, rei_name, measure)
        filename = create_filename(male_df,
                                   rei_name,
                                   mvid,
                                   rei_id,
                                   measure,
                                   year)
        plt.savefig(outpath + filename, format='pdf', dpi=300)
        print
        print('File: {}\nSaved to: {}'.format(filename, outpath))

    if len(female_df) > 0:
        # graph and save female data
        graph_data(female_df, rei_name, measure)
        filename = create_filename(female_df,
                                   rei_name,
                                   mvid,
                                   rei_id,
                                   measure,
                                   year)
        plt.savefig(outpath + filename, format='pdf', dpi=300)
        print
        print('File: {}\nSaved to: {}'.format(filename, outpath))

    # create legend
    generate_legend(outpath)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    start_time = time.time()
    main()
    print('Runtime: {}'.format(time.time() - start_time))
