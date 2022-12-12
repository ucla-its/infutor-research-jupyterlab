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

# TODO: Remove
def in_counties(fips, county_codes=['037', '059']):
    return fips.astype('string').str[1:4].isin(county_codes)

for period in periods:
    infutor_start, infutor_end = period['infutor interval']
    census_year = period['census year']


    totalpop_by_area = df_high_loss_areas[
        ['tractid', f'totalpop{census_year}']
    ].set_index('tractid').squeeze()

    totalpop = df_census[f'totalpop{census_year}'].sum()

    popdens_by_area = df_census[
        ['tractid', f'popdens{census_year}']
    ].set_index('tractid').squeeze()
    boardings_by_area = df_census[
        ['tractid', f'boardings{census_year}']
    ].set_index('tractid').squeeze()


    df_moves = df_all_moves[
        df_all_moves['date_left'].between(
            *years_to_effdate(infutor_start, infutor_end)
        )
    ]

    # TODO: Remove
    df_actual_moves = df_moves.dropna(subset=['date_arrived'])


    # TODO: Rewrite to use new indices
    type4578_10_11_12_13_moves = df_actual_moves[
        ~df_actual_moves['dest_fips'].isin(se_high_loss_areas)
    ]
    type1258_moves = df_moves[
        df_moves['orig_fips'].isin(se_high_loss_areas)
    ]
    type1369_moves = df_actual_moves[
        df_actual_moves['dest_fips'].isin(se_high_loss_areas)
    ]

    type258_moves = type1258_moves[
        type1258_moves['orig_fips'] != type1258_moves['dest_fips']
    ]

    type12_moves = type1258_moves[
        type1258_moves['dest_fips'].isin(se_high_loss_areas)
    ]
    type25_moves = type258_moves[in_counties(type258_moves['dest_fips'])]
    type58_moves = type1258_moves[
        ~type1258_moves['dest_fips'].isin(se_high_loss_areas)
    ]

    type1_moves = type1258_moves[
        type1258_moves['orig_fips'] == type1258_moves['dest_fips']
    ]
    type2_moves = type1258_moves[
        type1258_moves['dest_fips'].isin(se_high_loss_areas)
        & (type1258_moves['orig_fips'] != type1258_moves['dest_fips'])
    ]
    type5_moves = type1258_moves[
        in_counties(type1258_moves['dest_fips'])
        & ~type1258_moves['dest_fips'].isin(se_high_loss_areas)
    ]
    type8_moves = type1258_moves[~in_counties(type1258_moves['dest_fips'])]


    type25_orig_popdens = type25_moves['orig_fips'].map(popdens_by_area)
    type25_dest_popdens = type25_moves['dest_fips'].map(popdens_by_area)

    type25_16k_moves = type25_moves[type25_dest_popdens < 16000]
    type25_20k_moves = type25_moves[type25_dest_popdens < 20000]
    type25_lower_moves = type25_moves[
        type25_dest_popdens < type25_orig_popdens
    ]


    type1_totals = agg_moves_by_area(type1_moves, 'orig_fips')
    type2_totals = agg_moves_by_area(type2_moves, 'orig_fips')
    type8_totals = agg_moves_by_area(type8_moves,'orig_fips')
    type12_totals = agg_moves_by_area(type12_moves,'orig_fips')
    type25_totals = agg_moves_by_area(type25_moves, 'orig_fips')
    type25_16k_totals = agg_moves_by_area(type25_16k_moves, 'orig_fips')
    type25_20k_totals = agg_moves_by_area(type25_20k_moves,'orig_fips')
    type25_lower_totals = agg_moves_by_area(type25_lower_moves, 'orig_fips')
    type58_totals = agg_moves_by_area(type58_moves, 'orig_fips')
    type258_totals = agg_moves_by_area(type258_moves, 'orig_fips')
    type1258_totals = agg_moves_by_area(type1258_moves, 'orig_fips')
    type1369_totals = agg_moves_by_area(type1369_moves, 'dest_fips')

    type78_10_13_total = (~in_counties(df_actual_moves['dest_fips'])).sum()
    type147_total = (
        df_actual_moves['orig_fips'] == df_actual_moves['dest_fips']
    ).sum()
    moves_total = df_actual_moves.shape[0]

    type5_totals_by_orig = type5_moves.groupby('orig_fips').size()
    type1369_totals_by_dest = type1369_moves.groupby('dest_fips').size()

    type25_dist_iqrs = agg_moves_by_area(
        type25_moves, 'orig_fips', calculate_iqr, ['dist'], np.NaN
    )
    type258_dist_iqrs = agg_moves_by_area(
        type258_moves, 'orig_fips', calculate_iqr, ['dist'], np.NaN
    )
    type1369_dist_iqrs = agg_moves_by_area(
        type1369_moves, 'dest_fips', calculate_iqr, ['dist'], np.NaN
    )

    type4578_10_11_12_13_dist_iqr = calculate_iqr(
        type4578_10_11_12_13_moves['dist']
    )

    type25_dist_means = agg_moves_by_area(
        type25_moves, 'orig_fips', 'mean', ['dist'], np.NaN
    )
    type258_dist_means = agg_moves_by_area(
        type258_moves, 'orig_fips', 'mean', ['dist'], np.NaN
    )
    type1369_dist_means = agg_moves_by_area(
        type1369_moves, 'dest_fips', 'mean', ['dist'], np.NaN
    )

    type4578_10_11_12_13_dist_mean = type4578_10_11_12_13_moves['dist'].mean()

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        type25_moves_by_orig = type25_moves.set_index('orig_fips', drop=False)
        type1369_moves_by_dest = type1369_moves.set_index(
            'dest_fips', drop=False
        )

    type25_orig_popdens_by_orig = type25_moves_by_orig[
        'orig_fips'
    ].map(popdens_by_area)
    type1369_orig_popdens_by_dest = type1369_moves_by_dest[
        'orig_fips'
    ].map(popdens_by_area)

    orig_popdens = df_actual_moves['orig_fips'].map(popdens_by_area)

    type25_dest_popdens_by_orig = type25_moves_by_orig[
        'dest_fips'
    ].map(popdens_by_area)
    type1369_dest_popdens_by_dest = type1369_moves_by_dest[
        'dest_fips'
    ].map(popdens_by_area)

    dest_popdens = df_actual_moves['dest_fips'].map(popdens_by_area)


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
    ] = df_moves.shape[0]

    # FIXME -> type1_total
    entire_sample_results[
        "Number of total moves that began and ended in the same tract"
    ] = df_moves.loc[True, :, :].shape[0]

    entire_sample_results[
        "Number of total moves that ended outside LA or Orange Counties"
    ] = df_moves.loc[:, :, 'outside'].shape[0]


    entire_sample_results[
        "Interquartile range of move distances out of high-loss tracts"
    ] = calculate_iqr(df_moves.loc[:, 'high-loss', ['LA/OC', 'outside']]['dist'])

    entire_sample_results[
        "Mean distance of moves out"
    ] = df_moves.loc[:, 'high-loss', ['LA/OC', 'outside']]['dist'].mean()


    entire_sample_results[
        "Share of moves that stay within tract"
    ] = df_moves.loc[True, 'high-loss', :].shape[0] \
        / df_moves.loc[:, 'high-loss', :].shape[0]


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

    if verbose:
        with pd.option_context('display.max_columns', None):
            print(f"\nperiod {period} results:\n")
            pprint(entire_sample_results)
            print(df_results[to_print])
