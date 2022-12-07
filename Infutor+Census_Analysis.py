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

df_moves = pd.concat(
    [
        pd.read_pickle(chunk_file)
        for chunk_file in glob(
            f'./data/infutor+census/moves-{lines_per_chunk}/*.pkl'
        )
    ]
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

se_high_loss_areas = pd.read_pickle(
    './data/infutor+census/se_high_loss_areas.pkl'
)


### Analysis ###

if verbose:
    print("Starting analysis...")

def years_to_effdate(beginning, end):
    return (100 * beginning + 1), (100 * end + 12)

# TODO: Cache if analysis too slow
def agg_moves_by_area(
    moves,
    by,
    func='size',
    subset=[],
    fill_value=0,
    areas=se_high_loss_areas
):
    se_result = (moves.groupby(by)[subset]
                      .agg(func)
                      .reindex(areas, fill_value=fill_value)
                      .squeeze())
    return se_result

def calculate_iqr(se):
    q3, q1 = se.quantile([0.75, 0.25], interpolation='midpoint')
    iqr = q3 - q1
    return iqr

def in_counties(fips, county_codes=['037', '059']):
    return fips.astype('string').str[1:4].isin(county_codes)

for period in periods:
    infutor_start, infutor_end = period['infutor interval']
    census_year = period['census year']


    totalpop_by_area = df_high_loss_areas[['tractid', f'totalpop{census_year}']]

    totalpop = df_census[f'totalpop{census_year}'].sum()

    popdens_by_area = df_census[
        ['tractid', f'popdens{census_year}']
    ].set_index('tractid').squeeze()
    boardings_by_area = df_census[
        ['tractid', f'boardings{census_year}']
    ].set_index('tractid').squeeze()


    df_period_moves = df_moves[
        df_moves['date_left'].between(
            *years_to_effdate(infutor_start, infutor_end)
        )
    ]

    # TODO: Remove
    df_actual_moves = df_period_moves.dropna(subset=['date_arrived'])


    # TODO: Rewrite to use new indices
    type4578_10_11_12_13_moves = df_actual_moves[
        ~df_actual_moves['dest_fips'].isin(se_high_loss_areas)
    ]
    type1258_moves = df_period_moves[
        df_period_moves['orig_fips'].isin(se_high_loss_areas)
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

        type25_moves_by_orig = type25_moves.set_index('orig_fips',drop=False)
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
    ] = totalpop_by_area


    area_results[
        "Number of total moves that began in high-loss tracts"
    ] = type1258_totals # [:, 'high-loss', :]

    area_results[
        "Number of total moves that began in high-loss tracts and ended in the "
        "same tract"
    ] = type1_totals # [True, 'high-loss', :]

    area_results[
        "Number of total moves that began in high-loss tracts and ended in a "
        "different high-loss tract"
    ] = type2_totals # [False, 'high-loss', 'high-loss']

    area_results[
        "Number of total moves that began in the tract and ended outside LA or "
        "Orange Counties"
    ] = type8_totals # [:, 'high-loss', 'outside']

    area_results[
        "Number of total moves that began in the tract and ended outside the "
        "high-loss deciles"
    ] = type58_totals # [:, 'high-loss', ['LA/OC', 'outside']]


    area_results[
        "Interquartile range of all move distances out of high-loss tracts"
    ] = type258_dist_iqrs # [False, 'high-loss', :]

    area_results[
        "Interquartile range of move distances out of high-loss tracts that "
        "end in LA or the OC"
    ] = type25_dist_iqrs # [False, 'high-loss', ['high-loss', 'LA/OC']]


    area_results[
        "Mean distance of all moves out"
    ] = type258_dist_means # [False, 'high-loss', :]

    area_results[
        "Mean distance of moves out that end in LA or the OC"
    ] = type25_dist_means # [False, 'high-loss', ['high-loss', 'LA/OC']]


    area_results[
        "Share of moves that stay within tract"
    ] = type1_totals / type1258_totals # [True, 'high-loss', :] / [:, 'high-loss', :]

    area_results[
        "Share of moves that stay within high-loss decile"
    ] = type12_totals / type1258_totals # [:, 'high-loss', 'high-loss'] / [:, 'high-loss', :]


    area_results[
        "Weighted average density of destination tracts for moves that end in "
        "LA and Orange County but are not in high-loss decile"
    ] = (type5_moves.groupby(['orig_fips', 'dest_fips']) # [:, 'high-loss', 'LA/OC']
                    .size()
                    .unstack(fill_value=0)
                    .mul(popdens_by_area)
                    .sum(axis=1)
                    .div(type5_totals_by_orig)
                    .reindex(se_high_loss_areas))

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        area_results[
            "Median density of destination tracts for moves that end in LA and "
            "Orange County but are not in high-loss decile"
        ] = (type5_moves.dropna(subset=['orig_fips']) # [:, 'high-loss', 'LA/OC']
                        .set_index('orig_fips')['dest_fips']
                        .map(popdens_by_area)
                        .groupby('orig_fips')
                        .median()
                        .reindex(se_high_loss_areas))

    area_results[
        "Weighted average ridership of destination tracts for moves that end "
        "in LA and Orange County but are not in high-loss decile"
    ] = (type5_moves.groupby(['orig_fips', 'dest_fips']) # [:, 'high-loss', 'LA/OC']
                    .size()
                    .unstack(fill_value=0)
                    .mul(boardings_by_area)
                    .sum(axis=1)
                    .div(type5_totals_by_orig)
                    .reindex(se_high_loss_areas))

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        area_results[
            "Median ridership of destination tracts for moves that end in LA "
            "and Orange County but are not in high-loss decile"
        ] = (type5_moves.dropna(subset=['orig_fips']) # [:, 'high-loss', 'LA/OC']
                        .set_index('orig_fips')['dest_fips']
                        .map(boardings_by_area)
                        .groupby('orig_fips')
                        .median()
                        .reindex(se_high_loss_areas))


    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to tracts below a density of 16,000 out of high loss tracts that "
        "end anywhere"
    ] = 100 * type25_16k_totals / type258_totals # [False, 'high-loss', ['high-loss', 'LA/OC']] / [False, 'high-loss', :]

    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to tracts below a density of 16,000 out of high loss tracts that "
        "stay in LA and OC"
    ] = 100 * type25_16k_totals / type25_totals # [False, 'high-loss', ['high-loss', 'LA/OC']]


    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to tracts below a density of 20k out of high loss tracts that "
        "end anywhere"
    ] = 100 * type25_20k_totals / type258_totals # [False, 'high-loss', ['high-loss', 'LA/OC']] / [False, 'high-loss', :]

    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to tracts below a density of 20k out of high loss tracts that "
        "stay in LA and OC"
    ] = 100 * type25_20k_totals / type25_totals # [False, 'high-loss', ['high-loss', 'LA/OC']]


    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to a lower-density tract out of high loss tracts that end "
        "anywhere"
    ] = 100 * type25_lower_totals / type258_totals # [False, 'high-loss', ['high-loss', 'LA/OC']] / [False, 'high-loss', :]

    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to a lower-density tract out of high loss tracts that stay in LA "
        "and OC"
    ] = 100 * type25_lower_totals / type25_totals # [False, 'high-loss', ['high-loss', 'LA/OC']]


    area_results[
        "The average change in population density of a move out of a high-loss "
        "tract that end in LA or OC"
    ] = (
        type25_dest_popdens_by_orig - type25_orig_popdens_by_orig # [False, 'high-loss', ['high-loss', 'LA/OC']]
    ).groupby('orig_fips').mean().reindex(se_high_loss_areas)


    # FIXME -> type369_totals [False, :, 'high-loss']
    #       -> type69_totals [:, ['LA/OC', 'outside'], 'high-loss']
    #       -> type36_totals [False, ['high-loss', 'LA/OC'], 'high-loss']
    area_results[
        "Number of total moves into the high-loss tracts"
    ] = type1369_totals


    # FIXME -> type369_moves [False, :, 'high-loss']
    #       -> type6_moves [:, 'LA/OC', 'high-loss']
    area_results[
        "Weighted average density of tracts where in-moves originated"
    ] = (type1369_moves.groupby(['dest_fips', 'orig_fips'])
                       .size()
                       .unstack(fill_value=0)
                       .mul(popdens_by_area)
                       .sum(axis=1)
                       .div(type1369_totals_by_dest)
                       .reindex(se_high_loss_areas))

    # FIXME -> type369_moves [False, :, 'high-loss']
    #       -> type6_moves [:, 'LA/OC', 'high-loss']
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        area_results[
            "Median density of tracts where in-moves originated"
        ] = (type1369_moves.dropna(subset=['dest_fips'])
                           .set_index('dest_fips')['orig_fips']
                           .map(popdens_by_area)
                           .groupby('dest_fips')
                           .median()
                           .reindex(se_high_loss_areas))


    # TODO: Percent of moves in that came from lower-density places


    # FIXME -> type369 [False, :, 'high-loss']
    area_results[
        "Average change in density that resulted from a move into a high-loss "
        "tract"
    ] = (
        type1369_dest_popdens_by_dest - type1369_orig_popdens_by_dest
    ).groupby('dest_fips').mean().reindex(se_high_loss_areas)


    # FIXME -> type369_totals [False, :, 'high-loss']
    #       -> type69_totals [:, ['LA/OC', 'outside'], 'high-loss']
    #       -> type36_totals [False, ['high-loss', 'LA/OC'], 'high-loss']
    area_results[
        "Interquartile range of move distance for moves that end in high-loss "
        "tracts"
    ] = type1369_dist_iqrs

    # FIXME -> type369_totals [False, :, 'high-loss']
    #       -> type69_totals [:, ['LA/OC', 'outside'], 'high-loss']
    #       -> type36_totals [False, ['high-loss', 'LA/OC'], 'high-loss']
    area_results[
        "Mean move distance for the above"
    ] = type1369_dist_means


    # TODO: calculate migration rates

    num_months = 12 * (infutor_end - infutor_start + 1)

    # area_results[
    #     "Moves out per month"
    # ] = type258_totals / num_months [False, 'high-loss', :]
    # type258_totals / totalpop_by_area [False, 'high-loss', :]

    # type369_totals / num_months [False, :, 'high-loss']
    # type369_totals / totalpop_by_area [False, :, 'high-loss']

    entire_sample_results = {}

    entire_sample_results[
        "Total population at beginning of the period"
    ] = totalpop


    entire_sample_results[
        "Number of total moves"
    ] = moves_total

    # FIXME -> type1_total
    entire_sample_results[
        "Number of total moves that began and ended in the same tract"
    ] = type147_total # [True, :, :]

    entire_sample_results[
        "Number of total moves that ended outside LA or Orange Counties"
    ] = type78_10_13_total # [:, :, 'outside']


    # FIXME -> type58_dist_iqr [:, 'high-loss', ['LA/OC', 'outside']]
    entire_sample_results[
        "Interquartile range of move distances out of high-loss tracts"
    ] = type4578_10_11_12_13_dist_iqr

    # FIXME -> type58_dist_iqr [:, 'high-loss', ['LA/OC', 'outside']]
    entire_sample_results[
        "Mean distance of moves out"
    ] = type4578_10_11_12_13_dist_mean


    # FIXME -> type1_total [True, 'high-loss', :] / type1258_total [:, 'high-loss', :]
    entire_sample_results[
        "Share of moves that stay within tract"
    ] = type147_total / moves_total


    entire_sample_results[
        "Average change in density of a move"
    ] = (dest_popdens - orig_popdens).mean()


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
