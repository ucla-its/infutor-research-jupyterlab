import numpy as np
import pandas as pd

from infutor_census_analysis import *


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
    ] = count_moves(df_moves.loc[True, :, :])

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
    ] = count_moves(df_moves.loc[True, :, :]) / count_moves(df_moves)


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
        export_agg_results(
            entire_sample_results,
            create_txt_filename(False, infutor_start, infutor_end)
        )
        
        df_results.to_csv(
            f'./data/infutor+census/For_High-Loss_Areas_{infutor_start}-'
            f'{infutor_end}.csv'
        )

    if verbose:
        with pd.option_context('display.max_columns', None):
            print(f"\nperiod {period} results:\n")
            pprint(entire_sample_results)
            print(df_results[to_print])
