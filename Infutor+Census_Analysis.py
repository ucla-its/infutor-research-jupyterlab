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
to_print = config['analysis to print']
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

def count_moves(moves):
    return moves.shape[0]

def agg_moves_by_area(
    moves, by, func='size', subset=[], fill_value=0, areas=se_high_loss_areas
):
    se_result = moves.groupby(by)[subset] \
                     .agg(func) \
                     .reindex(areas, fill_value=fill_value) \
                     .squeeze()
    return se_result

def calculate_iqr(se):
    q3, q1 = se.quantile([0.75, 0.25], interpolation='midpoint')
    iqr = q3 - q1
    return iqr

def weighted_average_by_area(
    moves, by, other, census_col, areas=se_high_loss_areas
):
    move_totals_by_area = moves.groupby(by).size()

    se_result = moves.groupby([by, other]) \
                     .size() \
                     .unstack(fill_value=0) \
                     .mul(census_col) \
                     .sum(axis=1) \
                     .div(move_totals_by_area) \
                     .reindex(areas)
    return se_result

def median_by_area(moves, by, other, census_col, areas=se_high_loss_areas):
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        se_result = moves.dropna(subset=[by]) \
                         .set_index(by)[other] \
                         .map(census_col) \
                         .groupby(by) \
                         .median() \
                         .reindex(areas)

    return se_result

def filter_moves_using_fips(
    moves, test, orig_mapping=lambda x: x, dest_mapping=lambda x: x
):
    moves_orig_data = moves['orig_fips'].map(orig_mapping)
    moves_dest_data = moves['dest_fips'].map(dest_mapping)
    filtered_moves = moves[test(moves_orig_data, moves_dest_data)]
    return filtered_moves

def average_change_by_area(moves, by, census_col, areas=se_high_loss_areas):
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        moves_by_area = moves.set_index(by, drop=False)

    moves_dest_data_by_area = moves_by_area['dest_fips'].map(census_col)
    moves_orig_data_by_area = moves_by_area['orig_fips'].map(census_col)

    result = (moves_dest_data_by_area - moves_orig_data_by_area).groupby(by) \
                                                                .mean() \
                                                                .reindex(areas)
    return result

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
    ] = census_col_by_area(f'totalpop{census_year}', df_high_loss_areas)


    area_results[
        "Number of total moves that began in high-loss tracts"
    ] = agg_moves_by_area(df_moves.loc[:, 'high-loss', :], 'orig_fips')

    area_results[
        "Number of total moves that began in high-loss tracts and ended in the "
        "same tract"
    ] = agg_moves_by_area(df_moves.loc[True, 'high-loss', :], 'orig_fips')

    area_results[
        "Number of total moves that began in high-loss tracts and ended in a "
        "different high-loss tract"
    ] = agg_moves_by_area(
        df_moves.loc[False, 'high-loss', 'high-loss'], 'orig_fips'
    )

    area_results[
        "Number of total moves that began in the tract and ended outside LA or "
        "Orange Counties"
    ] = agg_moves_by_area(df_moves.loc[:, 'high-loss', 'outside'], 'orig_fips')

    area_results[
        "Number of total moves that began in the tract and ended outside the "
        "high-loss deciles"
    ] = agg_moves_by_area(
        df_moves.loc[:, 'high-loss', ['LA/OC', 'outside']], 'orig_fips'
    )


    area_results[
        "Interquartile range of all move distances out of high-loss tracts"
    ] = agg_moves_by_area(
        df_moves.loc[False, 'high-loss', :],
        'orig_fips',
        calculate_iqr,
        ['dist'],
        np.NaN
    )

    area_results[
        "Interquartile range of move distances out of high-loss tracts that "
        "end in LA or the OC"
    ] = agg_moves_by_area(
        df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']],
        'orig_fips',
        calculate_iqr,
        ['dist'],
        np.NaN
    )


    area_results[
        "Mean distance of all moves out"
    ] = agg_moves_by_area(
        df_moves.loc[False, 'high-loss', :],
        'orig_fips',
        'mean',
        ['dist'],
        np.NaN
    )

    area_results[
        "Mean distance of moves out that end in LA or the OC"
    ] = agg_moves_by_area(
        df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']],
        'orig_fips',
        'mean',
        ['dist'],
        np.NaN
    )


    area_results[
        "Share of moves that stay within tract"
    ] = agg_moves_by_area(df_moves.loc[True, 'high-loss', :], 'orig_fips') \
        / agg_moves_by_area(df_moves.loc[:, 'high-loss', :], 'orig_fips')

    area_results[
        "Share of moves that stay within high-loss decile"
    ] = agg_moves_by_area(
        df_moves.loc[:, 'high-loss', 'high-loss'], 'orig_fips'
    ) / agg_moves_by_area(df_moves.loc[:, 'high-loss', :], 'orig_fips')


    area_results[
        "Weighted average density of destination tracts for moves that end in "
        "LA and Orange County but are not in high-loss decile"
    ] = weighted_average_by_area(
        df_moves.loc[:, 'high-loss', 'LA/OC'],
        'orig_fips',
        'dest_fips',
        census_col_by_area(f'popdens{census_year}')
    )

    area_results[
        "Median density of destination tracts for moves that end in LA and "
        "Orange County but are not in high-loss decile"
    ] = median_by_area(
        df_moves.loc[:, 'high-loss', 'LA/OC'],
        'orig_fips',
        'dest_fips',
        census_col_by_area(f'popdens{census_year}')
    )

    area_results[
        "Weighted average ridership of destination tracts for moves that end "
        "in LA and Orange County but are not in high-loss decile"
    ] = weighted_average_by_area(
        df_moves.loc[:, 'high-loss', 'LA/OC'],
        'orig_fips',
        'dest_fips',
        census_col_by_area(f'boardings{census_year}')
    )

    area_results[
        "Median ridership of destination tracts for moves that end in LA "
        "and Orange County but are not in high-loss decile"
    ] = median_by_area(
        df_moves.loc[:, 'high-loss', 'LA/OC'],
        'orig_fips',
        'dest_fips',
        census_col_by_area(f'boardings{census_year}')
    )


    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to tracts below a density of 16,000 out of high loss tracts that "
        "end anywhere"
    ] = 100 * agg_moves_by_area(
        filter_moves_using_fips(
            df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']],
            lambda orig, dest: dest < orig,
            lambda _: 16_000,
            census_col_by_area(f'popdens{census_year}')
        ),
        'orig_fips'
    ) / agg_moves_by_area(df_moves.loc[False, 'high-loss', :], 'orig_fips')

    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to tracts below a density of 16,000 out of high loss tracts that "
        "stay in LA and OC"
    ] = 100 * agg_moves_by_area(
        filter_moves_using_fips(
            df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']],
            lambda orig, dest: dest < orig,
            lambda _: 16_000,
            census_col_by_area(f'popdens{census_year}')
        ),
        'orig_fips'
    ) / agg_moves_by_area(
        df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']], 'orig_fips'
    )


    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to tracts below a density of 20k out of high loss tracts that "
        "end anywhere"
    ] = 100 * agg_moves_by_area(
        filter_moves_using_fips(
            df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']],
            lambda orig, dest: dest < orig,
            lambda _: 20_000,
            census_col_by_area(f'popdens{census_year}')
        ),
        'orig_fips'
    ) / agg_moves_by_area(df_moves.loc[False, 'high-loss', :], 'orig_fips')

    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to tracts below a density of 20k out of high loss tracts that "
        "stay in LA and OC"
    ] = 100 * agg_moves_by_area(
        filter_moves_using_fips(
            df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']],
            lambda orig, dest: dest < orig,
            lambda _: 20_000,
            census_col_by_area(f'popdens{census_year}')
        ),
        'orig_fips'
    ) / agg_moves_by_area(
        df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']], 'orig_fips'
    )


    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to a lower-density tract out of high loss tracts that end "
        "anywhere"
    ] = 100 * agg_moves_by_area(
        filter_moves_using_fips(
            df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']],
            lambda orig, dest: dest < orig,
            census_col_by_area(f'popdens{census_year}'),
            census_col_by_area(f'popdens{census_year}')
        ),
        'orig_fips'
    ) / agg_moves_by_area(df_moves.loc[False, 'high-loss', :], 'orig_fips')

    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to a lower-density tract out of high loss tracts that stay in LA "
        "and OC"
    ] = 100 * agg_moves_by_area(
        filter_moves_using_fips(
            df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']],
            lambda orig, dest: dest < orig,
            census_col_by_area(f'popdens{census_year}'),
            census_col_by_area(f'popdens{census_year}')
        ),
        'orig_fips'
    ) / agg_moves_by_area(
        df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']], 'orig_fips'
    )


    area_results[
        "The average change in population density of a move out of a high-loss "
        "tract that end in LA or OC"
    ] = average_change_by_area(
        df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']],
        'orig_fips',
        census_col_by_area(f'popdens{census_year}')
    )


    area_results[
        "Number of total moves into the high-loss tracts"
    ] = agg_moves_by_area(df_moves.loc[False, :, 'high-loss'], 'dest_fips')

    area_results[
        "Number of total moves into the high-loss tracts for all in-moves that "
        "aren't from another high-loss tract"
    ] = agg_moves_by_area(
        df_moves.loc[:, ['LA/OC', 'outside'], 'high-loss'], 'dest_fips'
    )

    area_results[
        "Number of total moves into the high-loss tracts for all in-moves that "
        "don't start outside LA or Orange Counties"
    ] = agg_moves_by_area(
        df_moves.loc[False, ['high-loss', 'LA/OC'], 'high-loss'], 'dest_fips'
    )


    area_results[
        "Weighted average density of tracts where in-moves originated for all "
        "moves that aren't within the exact same tract"
    ] = weighted_average_by_area(
        df_moves.loc[False, :, 'high-loss'],
        'dest_fips',
        'orig_fips',
        census_col_by_area(f'popdens{census_year}')
    )

    area_results[
        "Weighted average density of tracts where in-moves originated for all "
        "moves that originate outside the high-loss tracts but that don't "
        "originate outside LA or Orange County"
    ] = weighted_average_by_area(
        df_moves.loc[:, 'LA/OC', 'high-loss'],
        'dest_fips',
        'orig_fips',
        census_col_by_area(f'popdens{census_year}')
    )

    area_results[
        "Median density of tracts where in-moves originated for all moves that "
        "aren't within the exact same tract"
    ] = median_by_area(
        df_moves.loc[False, :, 'high-loss'],
        'dest_fips',
        'orig_fips',
        census_col_by_area(f'popdens{census_year}')
    )

    area_results[
        "Median density of tracts where in-moves originated for all moves that "
        "originate outside the high-loss tracts but that don't originate "
        "outside LA or Orange County"
    ] = median_by_area(
        df_moves.loc[:, 'LA/OC', 'high-loss'],
        'dest_fips',
        'orig_fips',
        census_col_by_area(f'popdens{census_year}')
    )


    area_results[
        "Percent of moves in that came from lower-density places"
    ] = 100 * agg_moves_by_area(
        filter_moves_using_fips(
            df_moves.loc[False, :, 'high-loss'],
            lambda orig, dest: orig < dest,
            census_col_by_area(f'popdens{census_year}'),
            census_col_by_area(f'popdens{census_year}')
        ),
        'dest_fips'
    ) / agg_moves_by_area(
        df_moves.loc[False, :, 'high-loss'], 'dest_fips'
    )


    area_results[
        "Average change in density that resulted from a move into a high-loss "
        "tract"
    ] = average_change_by_area(
        df_moves.loc[False, :, 'high-loss'],
        'dest_fips',
        census_col_by_area(f'popdens{census_year}')
    )


    area_results[
        "Interquartile range of move distance for moves that end in high-loss "
        "tracts for all moves not originating in same tract"
    ] = agg_moves_by_area(
        df_moves.loc[False, :, 'high-loss'],
        'dest_fips',
        calculate_iqr,
        ['dist'],
        np.NaN
    )

    area_results[
        "Interquartile range of move distance for moves that end in high-loss "
        "tracts for all moves except those that start in another high-loss "
        "tract"
    ] = agg_moves_by_area(
        df_moves.loc[:, ['LA/OC', 'outside'], 'high-loss'],
        'dest_fips',
        calculate_iqr,
        ['dist'],
        np.NaN
    )

    area_results[
        "Interquartile range of move distance for moves that end in high-loss "
        "tracts for all moves that don't start outside LA or Orange County"
    ] = agg_moves_by_area(
        df_moves.loc[False, ['high-loss', 'LA/OC'], 'high-loss'],
        'dest_fips',
        calculate_iqr,
        ['dist'],
        np.NaN
    )

    area_results[
        "Mean move distance for moves that end in high-loss tracts for all "
        "moves not originating in same tract"
    ] = agg_moves_by_area(
        df_moves.loc[False, :, 'high-loss'],
        'dest_fips',
        'mean',
        ['dist'],
        np.NaN
    )

    area_results[
        "Mean move distance for moves that end in high-loss tracts for all "
        "moves except those that start in another high-loss tract"
    ] = agg_moves_by_area(
        df_moves.loc[:, ['LA/OC', 'outside'], 'high-loss'],
        'dest_fips',
        'mean',
        ['dist'],
        np.NaN
    )

    area_results[
        "Mean move distance for moves that end in high-loss tracts for all "
        "moves that don't start outside LA or Orange County"
    ] = agg_moves_by_area(
        df_moves.loc[False, ['high-loss', 'LA/OC'], 'high-loss'],
        'dest_fips',
        'mean',
        ['dist'],
        np.NaN
    )


    area_results[
        "An overall rate of outmigration from high-loss tracts"
    ] = agg_moves_by_area(df_moves.loc[False, 'high-loss', :], 'orig_fips') \
        / census_col_by_area(f'totalpop{census_year}', df_high_loss_areas)
    
    area_results[
        "An overall rate of in-migration to high-loss tracts"
    ] = agg_moves_by_area(df_moves.loc[False, :, 'high-loss'], 'dest_fips') \
        / census_col_by_area(f'totalpop{census_year}', df_high_loss_areas)


    entire_sample_results = {}

    entire_sample_results[
        "Total population at beginning of the period"
    ] = df_census[f'totalpop{census_year}'].sum()


    entire_sample_results[
        "Number of total moves"
    ] = count_moves(df_moves)

    entire_sample_results[
        "Number of total moves that began and ended in the same tract"
    ] = count_moves(df_moves.loc[True, 'high-loss', 'high-loss'])

    entire_sample_results[
        "Number of total moves that ended outside LA or Orange Counties"
    ] = count_moves(df_moves.loc[:, :, 'outside'])


    entire_sample_results[
        "Interquartile range of move distances out of high-loss tracts"
    ] = calculate_iqr(
        df_moves.loc[:, 'high-loss', ['LA/OC', 'outside']]['dist']
    )

    entire_sample_results[
        "Mean distance of moves out"
    ] = df_moves.loc[:, 'high-loss', ['LA/OC', 'outside']]['dist'].mean()


    entire_sample_results[
        "Share of moves that stay within tract"
    ] = count_moves(df_moves.loc[True, 'high-loss', :]) \
        / count_moves(df_moves.loc[:, 'high-loss', :])


    entire_sample_results[
        "Average change in density of a move"
    ] = (
        df_moves['dest_fips'].map(census_col_by_area(f'popdens{census_year}')) \
        - df_moves['orig_fips'].map(census_col_by_area(f'popdens{census_year}'))
    ).mean()


    df_results = pd.concat(
        [
            result.rename(description)
            for description, result in area_results.items()
        ],
        axis='columns'
    )

    if export:
        prompt_width = max(map(len, entire_sample_results.keys())) + 2

        with open(
            f'./data/infutor+census/For_The_Entire_LA-OC_Sample_{infutor_start}'
            f'-{infutor_end}.txt',
            'w'
        ) as f:
            for k, v in entire_sample_results.items():
                prompt = f'{k}: '
                print(f'{prompt:<{prompt_width}}{v}', file=f)

        df_results.to_csv(
            f'./data/infutor+census/For_High-Loss_Areas_{infutor_start}-'
            f'{infutor_end}.csv'
        )

    if verbose:
        with pd.option_context('display.max_columns', None):
            print(f"\nperiod {period} results:\n")
            pprint(entire_sample_results)
            print(df_results[to_print])
