from yaml import safe_load
from glob import glob
import warnings
from pprint import pprint

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

se_high_loss_areas = df_high_loss_areas['tractid']


### Analysis ###

if verbose:
    print("Starting analysis...")

# FIXME: Both first records and missing data represented by NaN
# NOTE: Assuming date_arrived=NaN are first records, since data filtered by
#       effdate in preprocessing

def years_to_effdate(beginning, end):
    return (100 * beginning + 1), (100 * end + 12)

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

    df_period_moves = df_moves[
        df_moves['date_left'].between(
            *years_to_effdate(infutor_start, infutor_end)
        )
    ]

    df_actual_moves = df_period_moves.dropna(subset=['date_arrived'])


    area_results = {}

    area_results[
        "Total population at beginning of the period"
    ] = (
        df_high_loss_areas[
            ['tractid', f'totalpop{census_year}']
        ].set_index('tractid')
         .squeeze()
    )


    moves_from_high = df_actual_moves[
        df_actual_moves['orig_fips'].isin(se_high_loss_areas)
    ]

    area_results[
        "Number of total moves that began in high-loss tracts"
    ] = agg_moves_by_area(moves_from_high, 'orig_fips')

    area_results[
        "Number of total moves that began in high-loss tracts and ended in the "
        "same tract"
    ] = agg_moves_by_area(
        moves_from_high[
            moves_from_high['orig_fips'] == moves_from_high['dest_fips']
        ],
        'orig_fips'
    )

    area_results[
        "Number of total moves that began in high-loss tracts and ended in a "
        "different high-loss tract"
    ] = agg_moves_by_area(
        moves_from_high[
            moves_from_high['dest_fips'].isin(se_high_loss_areas)
            & (moves_from_high['orig_fips'] != moves_from_high['dest_fips'])
        ],
        'orig_fips'
    )

    area_results[
        "Number of total moves that began in the tract and ended outside LA or "
        "Orange Counties"
    ] = agg_moves_by_area(
        moves_from_high[~in_counties(moves_from_high['dest_fips'])],
        'orig_fips'
    )


    moves_out = moves_from_high[
        ~moves_from_high['dest_fips'].isin(se_high_loss_areas)
    ]

    area_results[
        "Number of total moves that began in the tract and ended outside the "
        "high-loss deciles"
    ] = agg_moves_by_area(moves_out, 'orig_fips')

    all_moves_out = moves_from_high[
        moves_from_high['orig_fips'] != moves_from_high['dest_fips']
    ]

    area_results[
        "Interquartile range of all move distances out of high-loss tracts"
    ] = agg_moves_by_area(
        all_moves_out,
        'orig_fips',
        calculate_iqr,
        ['dist'],
        np.NaN
    )

    area_results[
        "Mean distance of all moves out"
    ] = agg_moves_by_area(all_moves_out, 'orig_fips', 'mean', ['dist'], np.NaN)

    moves_end_in_LA_OC = all_moves_out[
        all_moves_out['orig_fips'] != all_moves_out['dest_fips']
    ]

    area_results[
        "Interquartile range of move distances out of high-loss tracts that "
        "end in LA or the OC"
    ] = agg_moves_by_area(
        moves_end_in_LA_OC,
        'orig_fips',
        calculate_iqr,
        ['dist'],
        np.NaN
    )

    area_results[
        "Mean distance of moves out that end in LA or the OC"
    ] = agg_moves_by_area(
        moves_end_in_LA_OC,
        'orig_fips',
        'mean',
        ['dist'],
        np.NaN
    )


    area_results[
        "Share of moves that stay within tract"
    ] = (
        area_results[
            'Number of total moves that began in high-loss tracts and ended in '
            'the same tract'
        ]
        / area_results['Number of total moves that began in high-loss tracts']
    )

    area_results[
        "Share of moves that stay within high-loss decile"
    ] = (
        agg_moves_by_area(
            moves_from_high[
                moves_from_high['dest_fips'].isin(se_high_loss_areas)
            ],
            'orig_fips'
        )
        / area_results['Number of total moves that began in high-loss tracts']
    )


    moves_to_LA_OC_not_in_high = moves_from_high[
        in_counties(moves_from_high['dest_fips'])
        & ~moves_from_high['dest_fips'].isin(se_high_loss_areas)
    ]

    popdens_by_area = df_census[
        ['tractid', f'popdens{census_year}']
    ].set_index('tractid').squeeze()

    move_totals1 = moves_to_LA_OC_not_in_high.groupby('orig_fips').size()

    area_results[
        "Weighted average density of destination tracts for moves that end in "
        "LA and Orange County but are not in high-loss decile"
    ] = (moves_to_LA_OC_not_in_high.groupby(['orig_fips', 'dest_fips'])
                                   .size()
                                   .unstack(fill_value=0)
                                   .mul(popdens_by_area)
                                   .sum(axis=1)
                                   .div(move_totals1)
                                   .reindex(se_high_loss_areas))

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        area_results[
            "Median density of destination tracts for moves that end in LA and "
            "Orange County but are not in high-loss decile"
        ] = (moves_to_LA_OC_not_in_high.dropna(subset=['orig_fips'])
                                       .set_index('orig_fips')['dest_fips']
                                       .map(popdens_by_area)
                                       .groupby('orig_fips')
                                       .median()
                                       .reindex(se_high_loss_areas))

    boardings_by_area = df_census[
        ['tractid', f'boardings{census_year}']
    ].set_index('tractid').squeeze()

    area_results[
        "Weighted average ridership of destination tracts for moves that end "
        "in LA and Orange County but are not in high-loss decile"
    ] = (moves_to_LA_OC_not_in_high.groupby(['orig_fips', 'dest_fips'])
                                   .size()
                                   .unstack(fill_value=0)
                                   .mul(boardings_by_area)
                                   .sum(axis=1)
                                   .div(move_totals1)
                                   .reindex(se_high_loss_areas))

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        area_results[
            "Median ridership of destination tracts for moves that end in LA and "
            "Orange County but are not in high-loss decile"
        ] = (moves_to_LA_OC_not_in_high.dropna(subset=['orig_fips'])
                                       .set_index('orig_fips')['dest_fips']
                                       .map(boardings_by_area)
                                       .groupby('orig_fips')
                                       .median()
                                       .reindex(se_high_loss_areas))


    dest_popdens_end_LA_OC = moves_end_in_LA_OC[
        'dest_fips'
    ].map(popdens_by_area)

    num_moves_16000 = agg_moves_by_area(
        moves_end_in_LA_OC[
            dest_popdens_end_LA_OC < 16000
        ],
        'orig_fips'
    )

    num_all = agg_moves_by_area(all_moves_out, 'orig_fips')
    num_LA_OC = agg_moves_by_area(moves_end_in_LA_OC, 'orig_fips')

    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to tracts below a density of 16,000 out of high loss tracts that "
        "end anywhere"
    ] = 100 * num_moves_16000 / num_all

    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to tracts below a density of 16,000 out of high loss tracts that "
        "stay in LA and OC"
    ] = 100 * num_moves_16000 / num_LA_OC

    num_moves_20k = agg_moves_by_area(
        moves_end_in_LA_OC[
            dest_popdens_end_LA_OC < 20000
        ],
        'orig_fips'
    )

    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to tracts below a density of 20k out of high loss tracts that "
        "end anywhere"
    ] = 100 * num_moves_20k / num_all

    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to tracts below a density of 20k out of high loss tracts that "
        "stay in LA and OC"
    ] = 100 * num_moves_20k / num_LA_OC

    orig_popdens_end_LA_OC = moves_end_in_LA_OC[
        'orig_fips'
    ].map(popdens_by_area)

    num_moves_lower = agg_moves_by_area(
        moves_end_in_LA_OC[
            dest_popdens_end_LA_OC < orig_popdens_end_LA_OC
        ],
        'orig_fips'
    )

    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to a lower-density tract out of high loss tracts that end "
        "anywhere"
    ] = 100 * num_moves_lower / num_all

    area_results[
        "Percent of moves out of high-loss tracts that end in LA or OC and "
        "went to a lower-density tract out of high loss tracts that stay in LA "
        "and OC"
    ] = 100 * num_moves_lower / num_LA_OC

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        orig_indexed_end_LA_OC = moves_end_in_LA_OC.set_index(
            'orig_fips',
            drop=False
        )

    orig_indexed_dest_popdens_end_LA_OC = orig_indexed_end_LA_OC[
        'dest_fips'
    ].map(popdens_by_area)
    orig_indexed_orig_popdens_end_LA_OC = orig_indexed_end_LA_OC[
        'orig_fips'
    ].map(popdens_by_area)

    area_results[
        "The average change in population density of a move out of a high-loss "
        "tract that end in LA or OC"
    ] = (
        orig_indexed_dest_popdens_end_LA_OC
        - orig_indexed_orig_popdens_end_LA_OC
    ).groupby('orig_fips').mean().reindex(se_high_loss_areas)


    moves_in = df_actual_moves[
        df_actual_moves['dest_fips'].isin(se_high_loss_areas)
    ]

    area_results[
        "Number of total moves into the high-loss tracts"
    ] = agg_moves_by_area(moves_in, 'dest_fips')


    # TODO: calculate more density stuff

    move_totals2 = moves_in.groupby('dest_fips').size()

    area_results[
        "Weighted average density of tracts where in-moves originated"
    ] = (moves_in.groupby(['dest_fips', 'orig_fips'])
                 .size()
                 .unstack(fill_value=0)
                 .mul(popdens_by_area)
                 .sum(axis=1)
                 .div(move_totals2)
                 .reindex(se_high_loss_areas))

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        area_results[
            "Median density of tracts where in-moves originated"
        ] = (moves_in.dropna(subset=['dest_fips'])
                     .set_index('dest_fips')['orig_fips']
                     .map(popdens_by_area)
                     .groupby('dest_fips')
                     .median()
                     .reindex(se_high_loss_areas))

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        dest_indexed_moves_in = moves_in.set_index(
            'dest_fips',
            drop=False
        )

    dest_indexed_dest_popdens_moves_in = dest_indexed_moves_in[
        'dest_fips'
    ].map(popdens_by_area)
    dest_indexed_orig_popdens_moves_in = dest_indexed_moves_in[
        'orig_fips'
    ].map(popdens_by_area)

    area_results[
        "Average change in density that resulted from a move into a high-loss "
        "tract"
    ] = (
        dest_indexed_dest_popdens_moves_in
        - dest_indexed_orig_popdens_moves_in
    ).groupby('dest_fips').mean().reindex(se_high_loss_areas)


    area_results[
        "Interquartile range of move distance for moves that end in high-loss "
        "tracts"
    ] = agg_moves_by_area(
        moves_in,
        'dest_fips',
        calculate_iqr,
        ['dist'],
        np.NaN
    )

    area_results[
        "Mean move distance for the above"
    ] = agg_moves_by_area(moves_in, 'dest_fips', 'mean', ['dist'], np.NaN)


    # TODO: calculate migration rates

    num_months = 12 * (infutor_end - infutor_start + 1)

    # area_results[
    #     "Moves out per month"
    # ] = 


    entire_sample_results = {}

    entire_sample_results[
        "Total population at beginning of the period"
    ] = df_census[f'totalpop{census_year}'].sum()


    entire_sample_results[
        "Number of total moves"
    ] = df_actual_moves.shape[0]

    entire_sample_results[
        "Number of total moves that began and ended in the same tract"
    ] = (df_actual_moves['orig_fips'] == df_actual_moves['dest_fips']).sum()

    entire_sample_results[
        "Number of total moves that ended outside LA or Orange Counties"
    ] = (~in_counties(df_actual_moves['dest_fips'])).sum()


    dist_moves_out = df_actual_moves[
        ~df_actual_moves['dest_fips'].isin(se_high_loss_areas)
    ]['dist']

    entire_sample_results[
        "Interquartile range of move distances out of high-loss tracts"
    ] = calculate_iqr(dist_moves_out)

    entire_sample_results[
        "Mean distance of moves out"
    ] = dist_moves_out.mean()


    moves_that_stay_within_tract = (
        df_actual_moves['orig_fips'] == df_actual_moves['dest_fips']
    ).sum()
    entire_sample_results[
        "Share of moves that stay within tract"
    ] = (moves_that_stay_within_tract
            / entire_sample_results['Number of total moves'])


    entire_sample_results[
        "Average change in density of a move"
    ] = (
        df_actual_moves['dest_fips'].map(popdens_by_area)
        - df_actual_moves['orig_fips'].map(popdens_by_area)
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
