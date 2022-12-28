from glob import glob
import warnings
from pprint import pprint

from yaml import safe_load
import pandas as pd
import numpy as np


### Configuration ###

with open('./Infutor+Census_Analysis.yaml') as f:
    config = safe_load(f)

verbose = config['verbose']
lines_per_chunk = config['lines per chunk']
export = config['write to file']

periods = config['periods']


### Load data ###

if verbose:
    print("Loading infutor data...")

df_all_moves = pd.concat(
    [
        pd.read_pickle(chunk_file)
        for chunk_file in glob(
            f'./data/infutor+census/moves-{lines_per_chunk}/*.pkl'
        )
    ]
).sort_index()

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

se_high_loss_areas = pd.read_pickle(
    './data/infutor+census/se_high_loss_areas.pkl'
)


### Analysis ###

if verbose:
    print("Starting analysis...")

def years_to_effdate(beginning, end):
    return (100 * beginning + 1), (100 * end + 12)

def census_col_by_area(name, df=df_census):
    se = df[['tractid', name]].set_index('tractid').squeeze()
    return se

def calculate_iqr(se):
    q3, q1 = se.quantile([0.75, 0.25], interpolation='midpoint')
    iqr = q3 - q1
    return iqr

def filter_moves_using_fips(
    moves, test, orig_mapping=lambda x: x, dest_mapping=lambda x: x
):
    moves_orig_data = moves['orig_fips'].map(orig_mapping)
    moves_dest_data = moves['dest_fips'].map(dest_mapping)
    filtered_moves = moves[test(moves_orig_data, moves_dest_data)]
    return filtered_moves

for period in periods:
    infutor_start, infutor_end = period['infutor interval']
    census_year = period['census year']


    df_moves = df_all_moves[
        df_all_moves['date_left'].between(
            *years_to_effdate(infutor_start, infutor_end)
        )
    ]


    area_results = {}

    area_results[
        "Total population at beginning of the period"
    ] = df_high_loss_areas[f'totalpop{census_year}'].sum()


    area_results[
        "Number of total moves that began in high-loss tracts"
    ] = df_moves.loc[:, 'high-loss', :].shape[0]

    area_results[
        "Number of total moves that began in high-loss tracts and ended in the "
        "same tract"
    ] = df_moves.loc[True, 'high-loss', :].shape[0]

    area_results[
        "Number of total moves that began in high-loss tracts and ended in a "
        "different high-loss tract"
    ] = df_moves.loc[False, 'high-loss', 'high-loss'].shape[0]

    area_results[
        "Number of total moves that began in the tract and ended outside LA or "
        "Orange Counties"
    ] = df_moves.loc[:, 'high-loss', 'outside'].shape[0]

    area_results[
        "Number of total moves that began in the tract and ended outside the "
        "high-loss deciles"
    ] = df_moves.loc[:, 'high-loss', ['LA/OC', 'outside']].shape[0]


    area_results[
        "Interquartile range of all move distances out of high-loss tracts"
    ] = calculate_iqr(df_moves.loc[False, 'high-loss', :]['dist'])

    area_results[
        "Interquartile range of move distances out of high-loss tracts that "
        "end in LA or the OC"
    ] = calculate_iqr(
        df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']]['dist']
    )


    area_results[
        "Mean distance of all moves out"
    ] = df_moves.loc[False, 'high-loss', :]['dist'].mean()

    area_results[
        "Mean distance of moves out that end in LA or the OC"
    ] = df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']]['dist'].mean()


    area_results[
        "Share of moves that stay within tract"
    ] = df_moves.loc[True, 'high-loss', :].shape[0] \
        / df_moves.loc[:, 'high-loss', :].shape[0]

    area_results[
        "Share of moves that stay within high-loss decile"
    ] = df_moves.loc[:, 'high-loss', 'high-loss'].shape[0] \
        / df_moves.loc[:, 'high-loss', :].shape[0]


    area_results[
        "Weighted average density of destination tracts for moves that end in "
        "LA and Orange County but are not in high-loss decile"
    ] = df_moves.loc[:, 'high-loss', 'LA/OC'] \
                .groupby('dest_fips') \
                .size() \
                .mul(census_col_by_area(f'popdens{census_year}')) \
                .sum() \
        / df_moves.loc[:, 'high-loss', 'LA/OC'].shape[0]

    area_results[
        "Median density of destination tracts for moves that end in LA and "
        "Orange County but are not in high-loss decile"
    ] = df_moves.loc[:, 'high-loss', 'LA/OC']['dest_fips'] \
                .map(census_col_by_area(f'popdens{census_year}')) \
                .median()

    area_results[
        "Weighted average ridership of destination tracts for moves that end "
        "in LA and Orange County but are not in high-loss decile"
    ] = df_moves.loc[:, 'high-loss', 'LA/OC'] \
                .groupby('dest_fips') \
                .size() \
                .mul(census_col_by_area(f'boardings{census_year}')) \
                .sum() \
        / df_moves.loc[:, 'high-loss', 'LA/OC'].shape[0]

    area_results[
        "Median ridership of destination tracts for moves that end in LA "
        "and Orange County but are not in high-loss decile"
    ] = df_moves.loc[:, 'high-loss', 'LA/OC']['dest_fips'] \
                .map(census_col_by_area(f'boardings{census_year}')) \
                .median()


    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to tracts below a density of 16,000 out of high loss tracts that "
        "end anywhere"
    ] = 100 * filter_moves_using_fips(
        df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']],
        lambda orig, dest: dest < orig,
        lambda _: 16_000,
        census_col_by_area(f'popdens{census_year}')
    ).shape[0] / df_moves.loc[False, 'high-loss', :].shape[0]

    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to tracts below a density of 16,000 out of high loss tracts that "
        "stay in LA and OC"
    ] = 100 * filter_moves_using_fips(
        df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']],
        lambda orig, dest: dest < orig,
        lambda _: 16_000,
        census_col_by_area(f'popdens{census_year}')
    ).shape[0] \
    / df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']].shape[0]


    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to tracts below a density of 20k out of high loss tracts that "
        "end anywhere"
    ] = 100 * filter_moves_using_fips(
        df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']],
        lambda orig, dest: dest < orig,
        lambda _: 20_000,
        census_col_by_area(f'popdens{census_year}')
    ).shape[0] / df_moves.loc[False, 'high-loss', :].shape[0]

    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to tracts below a density of 20k out of high loss tracts that "
        "stay in LA and OC"
    ] = 100 * filter_moves_using_fips(
        df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']],
        lambda orig, dest: dest < orig,
        lambda _: 20_000,
        census_col_by_area(f'popdens{census_year}')
    ).shape[0] \
    / df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']].shape[0]


    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to a lower-density tract out of high loss tracts that end "
        "anywhere"
    ] = 100 * filter_moves_using_fips(
        df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']],
        lambda orig, dest: dest < orig,
        census_col_by_area(f'popdens{census_year}'),
        census_col_by_area(f'popdens{census_year}')
    ).shape[0] / df_moves.loc[False, 'high-loss', :].shape[0]

    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to a lower-density tract out of high loss tracts that stay in LA "
        "and OC"
    ] = 100 * filter_moves_using_fips(
        df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']],
        lambda orig, dest: dest < orig,
        census_col_by_area(f'popdens{census_year}'),
        census_col_by_area(f'popdens{census_year}')
    ).shape[0] \
    / df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']].shape[0]


    area_results[
        "The average change in population density of a move out of a high-loss "
        "tract that end in LA or OC"
    ] = (
        df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']]['dest_fips'] \
                .map(census_col_by_area(f'popdens{census_year}')) \
      - df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']]['orig_fips'] \
                  .map(census_col_by_area(f'popdens{census_year}'))
    ).mean()


    area_results[
        "Number of total moves into the high-loss tracts"
    ] = df_moves.loc[False, :, 'high-loss'].shape[0]

    area_results[
        "Number of total moves into the high-loss tracts for all in-moves that "
        "aren't from another high-loss tract"
    ] = df_moves.loc[:, ['LA/OC', 'outside'], 'high-loss'].shape[0]

    area_results[
        "Number of total moves into the high-loss tracts for all in-moves that "
        "don't start outside LA or Orange Counties"
    ] = df_moves.loc[False, ['high-loss', 'LA/OC'], 'high-loss'].shape[0]


    area_results[
        "Weighted average density of tracts where in-moves originated for all "
        "moves that aren't within the exact same tract"
    ] = df_moves.loc[False, :, 'high-loss'] \
                .groupby('orig_fips') \
                .size() \
                .mul(census_col_by_area(f'popdens{census_year}')) \
                .sum() \
        / df_moves.loc[False, :, 'high-loss'].shape[0]

    area_results[
        "Weighted average density of tracts where in-moves originated for all "
        "moves that originate outside the high-loss tracts but that don't "
        "originate outside LA or Orange County"
    ] = df_moves.loc[:, 'LA/OC', 'high-loss'] \
                .groupby('orig_fips') \
                .size() \
                .mul(census_col_by_area(f'popdens{census_year}')) \
                .sum() \
        / df_moves.loc[:, 'LA/OC', 'high-loss'].shape[0]

    area_results[
        "Median density of tracts where in-moves originated for all moves that "
        "aren't within the exact same tract"
    ] = df_moves.loc[False, :, 'high-loss']['orig_fips'] \
                .map(census_col_by_area(f'popdens{census_year}')) \
                .median()

    area_results[
        "Median density of tracts where in-moves originated for all moves that "
        "originate outside the high-loss tracts but that don't originate "
        "outside LA or Orange County"
    ] = df_moves.loc[:, 'LA/OC', 'high-loss']['orig_fips'] \
                .map(census_col_by_area(f'popdens{census_year}')) \
                .median()


    area_results[
        "Percent of moves in that came from lower-density places"
    ] = 100 * filter_moves_using_fips(
        df_moves.loc[False, :, 'high-loss'],
        lambda orig, dest: orig < dest,
        census_col_by_area(f'popdens{census_year}'),
        census_col_by_area(f'popdens{census_year}')
    ).shape[0] / df_moves.loc[False, :, 'high-loss'].shape[0]


    area_results[
        "Average change in density that resulted from a move into a high-loss "
        "tract"
    ] = (
        df_moves.loc[False, :, 'high-loss']['dest_fips'] \
                .map(census_col_by_area(f'popdens{census_year}')) \
        - df_moves.loc[False, :, 'high-loss']['orig_fips'] \
                  .map(census_col_by_area(f'popdens{census_year}'))
    ).mean()


    area_results[
        "Interquartile range of move distance for moves that end in high-loss "
        "tracts for all moves not originating in same tract"
    ] = calculate_iqr(df_moves.loc[False, :, 'high-loss']['dist'])

    area_results[
        "Interquartile range of move distance for moves that end in high-loss "
        "tracts for all moves except those that start in another high-loss "
        "tract"
    ] = calculate_iqr(
        df_moves.loc[:, ['LA/OC', 'outside'], 'high-loss']['dist']
    )

    area_results[
        "Interquartile range of move distance for moves that end in high-loss "
        "tracts for all moves that don't start outside LA or Orange County"
    ] = calculate_iqr(
        df_moves.loc[False, ['high-loss', 'LA/OC'], 'high-loss']['dist']
    )

    area_results[
        "Mean move distance for moves that end in high-loss tracts for all "
        "moves not originating in same tract"
    ] = df_moves.loc[False, :, 'high-loss']['dist'].mean()

    area_results[
        "Mean move distance for moves that end in high-loss tracts for all "
        "moves except those that start in another high-loss tract"
    ] = df_moves.loc[:, ['LA/OC', 'outside'], 'high-loss']['dist'].mean()

    area_results[
        "Mean move distance for moves that end in high-loss tracts for all "
        "moves that don't start outside LA or Orange County"
    ] = df_moves.loc[False, ['high-loss', 'LA/OC'], 'high-loss']['dist'].mean()


    area_results[
        "An overall rate of outmigration from high-loss tracts"
    ] = df_moves.loc[False, 'high-loss', :].shape[0] \
        / df_high_loss_areas[f'totalpop{census_year}'].sum()
    
    area_results[
        "An overall rate of in-migration to high-loss tracts"
    ] = df_moves.loc[False, :, 'high-loss'].shape[0] \
        / df_high_loss_areas[f'totalpop{census_year}'].sum()


    if export:
        prompt_width = max(map(len, area_results.keys())) + 2

        with open(
            f'./data/infutor+census/For_High-Loss_Areas_{infutor_start}-'
            f'{infutor_end}.txt',
            'w'
        ) as f:
            for k, v in area_results.items():
                prompt = f'{k}: '
                print(f'{prompt:<{prompt_width}}{v}', file=f)

    if verbose:
        with pd.option_context('display.max_columns', None):
            print(f"\nperiod {period} results:\n")
            pprint(area_results)
