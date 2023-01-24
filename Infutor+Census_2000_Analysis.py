from infutor_census_analysis import *


infutor_start = 2000
infutor_end = 2000


df_moves = df_all_moves[
    df_all_moves['date_left'].between(
        *years_to_effdate(infutor_start, infutor_end)
    )
]


area_results = {}

area_results[
    "Total population at beginning of the period"
] = df_2000_high_loss_areas['SE_T001_001'].sum()


area_results[
    "Number of total moves that began in high-loss tracts"
] = count_moves(df_moves.loc[:, 'high-loss', :])

area_results[
    "Number of total moves that began in high-loss tracts and ended in the "
    "same tract"
] = count_moves(df_moves.loc[True, 'high-loss', :])

area_results[
    "Number of total moves that began in high-loss tracts and ended in a "
    "different high-loss tract"
] = count_moves(df_moves.loc[False, 'high-loss', 'high-loss'])

area_results[
    "Number of total moves that began in the tract and ended outside LA or "
    "Orange Counties"
] = count_moves(df_moves.loc[:, 'high-loss', 'outside'])

area_results[
    "Number of total moves that began in the tract and ended outside the "
    "high-loss deciles"
] = count_moves(df_moves.loc[:, 'high-loss', ['LA/OC', 'outside']])


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
] = count_moves(df_moves.loc[True, 'high-loss', :]) \
    / count_moves(df_moves.loc[:, 'high-loss', :])

area_results[
    "Share of moves that stay within high-loss decile"
] = count_moves(df_moves.loc[:, 'high-loss', 'high-loss']) \
    / count_moves(df_moves.loc[:, 'high-loss', :])


area_results[
    "Weighted average density of destination tracts for moves that end in "
    "LA and Orange County but are not in high-loss decile"
] = df_moves.loc[:, 'high-loss', 'LA/OC'] \
            .groupby('dest_fips') \
            .size() \
            .mul(census_col_by_area_2000('SE_T002_002')) \
            .sum() \
    / count_moves(df_moves.loc[:, 'high-loss', 'LA/OC'])

area_results[
    "Median density of destination tracts for moves that end in LA and "
    "Orange County but are not in high-loss decile"
] = df_moves.loc[:, 'high-loss', 'LA/OC']['dest_fips'] \
            .map(census_col_by_area_2000('SE_T002_002')) \
            .median()

# area_results[
#     "Weighted average ridership of destination tracts for moves that end "
#     "in LA and Orange County but are not in high-loss decile"
# ] = df_moves.loc[:, 'high-loss', 'LA/OC'] \
#             .groupby('dest_fips') \
#             .size() \
#             .mul(census_col_by_area(f'boardings{census_year}')) \
#             .sum() \
#     / count_moves(df_moves.loc[:, 'high-loss', 'LA/OC'])

# area_results[
#     "Median ridership of destination tracts for moves that end in LA "
#     "and Orange County but are not in high-loss decile"
# ] = df_moves.loc[:, 'high-loss', 'LA/OC']['dest_fips'] \
#             .map(census_col_by_area(f'boardings{census_year}')) \
#             .median()


area_results[
    "Percent of moves out of high-loss tracts that end in LA or OC and "
    "went to tracts below a density of 9,360 out of high loss tracts that "
    "end anywhere"
] = 100 * count_moves(
    filter_moves_using_fips(
        df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']],
        lambda orig, dest: dest < orig,
        lambda _: 9360,
        census_col_by_area_2000('SE_T002_002')
    )
) / count_moves(df_moves.loc[False, 'high-loss', :])

area_results[
    "Percent of moves out of high-loss tracts that end in LA or OC and "
    "went to tracts below a density of 9,360 out of high loss tracts that "
    "stay in LA and OC"
] = 100 * count_moves(
    filter_moves_using_fips(
        df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']],
        lambda orig, dest: dest < orig,
        lambda _: 9360,
        census_col_by_area_2000('SE_T002_002')
    )
) / count_moves(df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']])


area_results[
    "Percent of moves out of high-loss tracts that end in LA or OC and "
    "went to tracts below a density of 16,700 out of high loss tracts that "
    "end anywhere"
] = 100 * count_moves(
    filter_moves_using_fips(
        df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']],
        lambda orig, dest: dest < orig,
        lambda _: 16_700,
        census_col_by_area_2000('SE_T002_002')
    )
) / count_moves(df_moves.loc[False, 'high-loss', :])

area_results[
    "Percent of moves out of high-loss tracts that end in LA or OC and "
    "went to tracts below a density of 16,700 out of high loss tracts that "
    "stay in LA and OC"
] = 100 * count_moves(
    filter_moves_using_fips(
        df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']],
        lambda orig, dest: dest < orig,
        lambda _: 16_700,
        census_col_by_area_2000('SE_T002_002')
    )
) / count_moves(df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']])


area_results[
    "Percent of moves out of high-loss tracts that end in LA or OC and "
    "went to a lower-density tract out of high loss tracts that end "
    "anywhere"
] = 100 * count_moves(
    filter_moves_using_fips(
        df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']],
        lambda orig, dest: dest < orig,
        census_col_by_area_2000('SE_T002_002'),
        census_col_by_area_2000('SE_T002_002')
    )
 ) / count_moves(df_moves.loc[False, 'high-loss', :])

area_results[
    "Percent of moves out of high-loss tracts that end in LA or OC and "
    "went to a lower-density tract out of high loss tracts that stay in LA "
    "and OC"
] = 100 * count_moves(
    filter_moves_using_fips(
        df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']],
        lambda orig, dest: dest < orig,
        census_col_by_area_2000('SE_T002_002'),
        census_col_by_area_2000('SE_T002_002')
    )
) / count_moves(df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']])


area_results[
    "The average change in population density of a move out of a high-loss "
    "tract that end in LA or OC"
] = (
    df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']]['dest_fips'] \
            .map(census_col_by_area_2000('SE_T002_002')) \
    - df_moves.loc[False, 'high-loss', ['high-loss', 'LA/OC']]['orig_fips'] \
              .map(census_col_by_area_2000('SE_T002_002'))
).mean()


area_results[
    "Number of total moves into the high-loss tracts"
] = count_moves(df_moves.loc[False, :, 'high-loss'])

area_results[
    "Number of total moves into the high-loss tracts for all in-moves that "
    "aren't from another high-loss tract"
] = count_moves(df_moves.loc[:, ['LA/OC', 'outside'], 'high-loss'])

area_results[
    "Number of total moves into the high-loss tracts for all in-moves that "
    "don't start outside LA or Orange Counties"
] = count_moves(df_moves.loc[False, ['high-loss', 'LA/OC'], 'high-loss'])


area_results[
    "Weighted average density of tracts where in-moves originated for all "
    "moves that aren't within the exact same tract"
] = df_moves.loc[False, :, 'high-loss'] \
            .groupby('orig_fips') \
            .size() \
            .mul(census_col_by_area_2000('SE_T002_002')) \
            .sum() \
    / count_moves(df_moves.loc[False, :, 'high-loss'])

area_results[
    "Weighted average density of tracts where in-moves originated for all "
    "moves that originate outside the high-loss tracts but that don't "
    "originate outside LA or Orange County"
] = df_moves.loc[:, 'LA/OC', 'high-loss'] \
            .groupby('orig_fips') \
            .size() \
            .mul(census_col_by_area_2000('SE_T002_002')) \
            .sum() \
    / count_moves(df_moves.loc[:, 'LA/OC', 'high-loss'])

area_results[
    "Median density of tracts where in-moves originated for all moves that "
    "aren't within the exact same tract"
] = df_moves.loc[False, :, 'high-loss']['orig_fips'] \
            .map(census_col_by_area_2000('SE_T002_002')) \
            .median()

area_results[
    "Median density of tracts where in-moves originated for all moves that "
    "originate outside the high-loss tracts but that don't originate "
    "outside LA or Orange County"
] = df_moves.loc[:, 'LA/OC', 'high-loss']['orig_fips'] \
            .map(census_col_by_area_2000('SE_T002_002')) \
            .median()


area_results[
    "Percent of moves in that came from lower-density places"
] = 100 * count_moves(
    filter_moves_using_fips(
        df_moves.loc[False, :, 'high-loss'],
        lambda orig, dest: orig < dest,
        census_col_by_area_2000('SE_T002_002'),
        census_col_by_area_2000('SE_T002_002')
    )
) / count_moves(df_moves.loc[False, :, 'high-loss'])


area_results[
    "Average change in density that resulted from a move into a high-loss "
    "tract"
] = (
    df_moves.loc[False, :, 'high-loss']['dest_fips'] \
            .map(census_col_by_area_2000('SE_T002_002')) \
    - df_moves.loc[False, :, 'high-loss']['orig_fips'] \
              .map(census_col_by_area_2000('SE_T002_002'))
).mean()


area_results[
    "Interquartile range of move distance for moves that end in high-loss "
    "tracts for all moves not originating in same tract"
] = calculate_iqr(df_moves.loc[False, :, 'high-loss']['dist'])

area_results[
    "Interquartile range of move distance for moves that end in high-loss "
    "tracts for all moves except those that start in another high-loss "
    "tract"
] = calculate_iqr(df_moves.loc[:, ['LA/OC', 'outside'], 'high-loss']['dist'])

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
] = count_moves(df_moves.loc[False, 'high-loss', :]) \
    / df_2000_high_loss_areas['SE_T001_001'].sum()

area_results[
    "An overall rate of in-migration to high-loss tracts"
] = count_moves(df_moves.loc[False, :, 'high-loss']) \
    / df_2000_high_loss_areas['SE_T001_001'].sum()

area_results[
    "Average median gross rent"
] = df_2000_high_loss_areas['SE_T167_001'].mean()

area_results[
    "Average median renter income"
] = df_2000_high_loss_areas['SE_T093_003'].mean()


entire_sample_results = {}

entire_sample_results[
    "Total population at beginning of the period"
] = df_2000_census['SE_T001_001'].sum()


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
    df_moves['dest_fips'].map(census_col_by_area_2000('SE_T002_002')) \
    - df_moves['orig_fips'].map(census_col_by_area_2000('SE_T002_002'))
).mean()


entire_sample_results[
    "Average median gross rent"
] = df_2000_census['SE_T167_001'].mean()

entire_sample_results[
    "Average median renter income"
] = df_2000_census['SE_T093_003'].mean()


if export:
    export_agg_results(
        area_results, create_txt_filename(True, infutor_start, infutor_end)
    )

    export_agg_results(
        entire_sample_results,
        create_txt_filename(False, infutor_start, infutor_end)
    )

if verbose:
    print_agg_results(area_results, "Area")
    print_agg_results(entire_sample_results, "Entire sample")
