from yaml import safe_load
from glob import glob
from pprint import pprint

import pandas as pd
import numpy as np


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

    area_results[
        "Interquartile range of move distances out of high-loss tracts"
    ] = agg_moves_by_area(
        moves_out,
        'orig_fips',
        calculate_iqr,
        ['dist'],
        np.NaN
    )

    area_results[
        "Mean distance of moves out"
    ] = agg_moves_by_area(moves_out, 'orig_fips', 'mean', ['dist'], np.NaN)


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
    

    moves_to_LA_OC_not_in_high = df_actual_moves[
        in_counties(df_actual_moves['dest_fips'])
        & ~df_actual_moves['dest_fips'].isin(se_high_loss_areas)
    ]

    # TODO: weighted averages and median
    # [X]: moves <- filter(all_moves)
    # [X]: num_moves <- size(group(moves, (orig, dest)))
    # [X]: move_totals <- size(group(moves, (orig)))
    # [X]: weighted_densities <- num_moves * popdens
    # [X]: result <- sum(weighted_densities) / move_totals

    popdens_by_area = df_census[
        ['tractid', f'popdens{census_year}']
    ].set_index('tractid').squeeze()

    move_totals = moves_to_LA_OC_not_in_high.groupby('orig_fips').size()

    area_results[
        "Weighted average density of destination tracts for moves that end in "
        "LA and Orange County but are not in high-loss decile"
    ] = (moves_to_LA_OC_not_in_high.groupby(['orig_fips', 'dest_fips'])
                                   .size()
                                   .unstack(fill_value=0)
                                   .mul(popdens_by_area)
                                   .sum(axis=1)
                                   .div(move_totals)
                                   .reindex(se_high_loss_areas))

    # area_results[
    #     "Median density of destination tracts for moves that end in LA and "
    #     "Orange County but are not in high-loss decile"
    # ]

    # TODO: density stuff percentages


    moves_in = df_actual_moves[
        df_actual_moves['dest_fips'].isin(se_high_loss_areas)
    ]

    area_results[
        "Number of total moves into the high-loss tracts"
    ] = agg_moves_by_area(moves_in, 'dest_fips')


    # TODO: calculate more density stuff


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
    ] = "Not yet implemented" # TODO


    df_results = pd.concat(
        [
            result.rename(description)
            for description, result in area_results.items()
        ],
        axis='columns'
    )

    with pd.option_context('display.max_columns', None):
        print(f"\nperiod {period} results:\n")
        pprint(entire_sample_results)
        print(df_results)
