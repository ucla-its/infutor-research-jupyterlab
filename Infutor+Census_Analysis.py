from yaml import safe_load
from glob import glob
from pprint import pprint

import pandas as pd


### Configuration ###

with open('./Infutor+Census_Analysis.yaml') as f:
    config = safe_load(f)

verbose = config['verbose']
lines_per_chunk = config['lines per chunk']

periods = config['periods']


### Load data ###

if verbose:
    print("Loading infutor data...")

df_moves = pd.concat(
    [
        pd.read_pickle(chunk_file)
        for chunk_file in glob(
            f'./data/infutor+census/moves-{lines_per_chunk}/*.pkl'
        )
    ],
    ignore_index=True
)

if verbose:
    print("Loading census data...")

arg_census = {
    'filepath_or_buffer': './Neighborhood_ridership_Oct_22.dta',
    'columns': [
        'tractid',
        'absloss10',
        'totalpop10',
        'totalpop15',
        'popdens10',
        'popdens15',
        'boardings10',
        'boardings15',
    ]
}

df_census = pd.read_stata(**arg_census)

df_high_loss_areas = pd.read_pickle(
    './data/infutor+census/df_high_loss_areas.pkl'
)


### Analysis ###

if verbose:
    print("Starting analysis...")

# FIXME: Both first records and missing data represented by NaN

def year_to_effdate(year):
    return 100 * year + 1

for period in periods:
    infutor_start, infutor_end = period['infutor interval']
    census_year = period['census year']

    df_period_moves = df_moves[
        df_moves['date_left'].between(
            year_to_effdate(infutor_start), year_to_effdate(infutor_end)
        )
    ]

    df_actual_moves = df_period_moves.dropna(subset=['date_arrived'])
    

    # FIXME: Everything
    area_results = {}

    area_results[
        "Total population at beginning of the period"
    ] = (df_actual_moves.groupby('orig_fips')
                        .size()
                        .reindex(df_high_loss_areas['tractid'], fill_value=0))

    with(pd.option_context('display.max_rows', None)):
        print(area_results["Total population at beginning of the period"])


    entire_sample_results = {}

    entire_sample_results[
        "Total population at beginning of the period"
    ] = df_census[f'totalpop{census_year}'].sum()


    # results[
    #     "Number of total moves that began in high-loss tracts"
    # ] = df_period_moves['orig_fips'].isin(df_high_loss_areas['tractid']).sum()

    entire_sample_results[
        "Number of total moves"
    ] = df_actual_moves.shape[0]

    # results[
    #     "Number of total moves that began in high-loss tracts and ended in the "
    #     "same tract"
    # ] = (
    #         df_period_moves['orig_fips'].isin(df_high_loss_areas['tractid'])
    #         & (df_period_moves['orig_fips'] == df_period_moves['dest_fips'])
    # ).sum()

    entire_sample_results[
        "Number of total moves that began and ended in the same tract"
    ] = (df_actual_moves['orig_fips'] == df_actual_moves['dest_fips']).sum()

    # results[
    #     "Number of total moves that began in high-loss tracts and ended in a "
    #     "different high-loss tract"
    # ] = (
    #     df_period_moves['orig_fips'].isin(df_high_loss_areas['tractid'])
    #     & df_period_moves['dest_fips'].isin(df_high_loss_areas['tractid'])
    #     & (df_period_moves['orig_fips'] != df_period_moves['dest_fips'])
    # ).sum()

    # results[
    #     "Number of total moves that began in the tract and ended outside LA or "
    #     "Orange Counties"
    # ] = (
    #     df_period_moves['orig_fips'].isin(df_high_loss_areas['tractid'])
    #     & ~(df_period_moves['dest_fips'].astype('string')
    #                                     .str[1:4]
    #                                     .isin(['037', '059']))
    # ).sum()

    entire_sample_results[
        "Number of total moves that ended outside LA or Orange Counties"
    ] = (
        ~df_actual_moves['dest_fips'].astype('string')
                                     .str[1:4]
                                     .isin(['037', '059'])
    ).sum()

    # results[
    #     "Number of total moves that began in the tract and ended outside the "
    #     "high-loss deciles"
    # ] = (
    #     df_period_moves['orig_fips'].isin(df_high_loss_areas['tractid'])
    #     & ~df_period_moves['dest_fips'].isin(df_high_loss_areas['tractid'])
    # ).sum()

    
    # dist_moves_out = df_period_moves[
    #     df_period_moves['orig_fips'].isin(df_high_loss_areas['tractid'])
    #     & ~df_period_moves['dest_fips'].isin(df_high_loss_areas['tractid'])
    # ]['dist']

    dist_moves_out = df_actual_moves[
        ~df_period_moves['dest_fips'].isin(df_high_loss_areas['tractid'])
    ]['dist']

    dist_q3, dist_q1 = dist_moves_out.quantile(
        [0.75, 0.25], interpolation='midpoint'
    )
    dist_iqr = dist_q3 - dist_q1
    entire_sample_results[
        "Interquartile range of move distances out of high-loss tracts"
    ] = dist_iqr

    entire_sample_results[
        "Mean distance of moves out"
    ] = dist_moves_out.mean()


    # moves_that_stay_within_tract = (
    #     df_period_moves['orig_fips'].isin(df_high_loss_areas['tractid'])
    #     & (df_period_moves['orig_fips'] == df_period_moves['dest_fips'])
    # ).sum()
    # results[
    #     "Share of moves that stay within tract"
    # ] = moves_that_stay_within_tract / total_moves

    moves_that_stay_within_tract = (
        df_actual_moves['orig_fips'] == df_actual_moves['dest_fips']
    ).sum()
    entire_sample_results[
        "Share of moves that stay within tract"
    ] = (moves_that_stay_within_tract
            / entire_sample_results['Number of total moves'])

    # moves_that_stay_within_decile = (
    #     df_period_moves['orig_fips'].isin(df_high_loss_areas['tractid'])
    #     & df_period_moves['dest_fips'].isin(df_high_loss_areas['tractid'])
    # ).sum()
    # results[
    #     "Share of moves that stay within high-loss decile"
    # ] = moves_that_stay_within_decile / total_moves


    # TODO: weighted averages and median


    # TODO: density stuff percentages


    # FIXME: Assumes any na is from missing first records
    # df_actual_moves = df_period_moves.dropna(subset=['date_arrived'])
    # results[
    #     "Number of total moves into the high-loss tracts"
    # ] = df_actual_moves['dest_fips'].isin(df_high_loss_areas['tractid']).sum()
    

    # TODO: calculate more density stuff



    # FIXME: Assumes any na is from missing first records
    # df_actual_moves = df_period_moves.dropna(subset=['date_arrived'])
    # dist_moves_in = df_actual_moves[
    #     ~df_actual_moves['orig_fips'].isin(df_high_loss_areas['tractid'])
    #     & df_actual_moves['dest_fips'].isin(df_high_loss_areas['tractid'])
    # ]['dist']

    # dist_q3, dist_q1 = dist_moves_in.quantile(
    #     [0.75, 0.25], interpolation='midpoint'
    # )
    # dist_iqr = dist_q3 - dist_q1
    # results[
    #     "Interquartile range of move distance for moves that end in high-loss "
    #     "tracts"
    # ] = dist_iqr

    # results[
    #     "Mean move distance for the above"
    # ] = dist_moves_in.mean()


    # TODO: calculate migration rates


    print(f"\nperiod {period} results:\n")
    pprint(entire_sample_results)
