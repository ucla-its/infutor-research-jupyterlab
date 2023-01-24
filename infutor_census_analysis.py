from glob import glob
import warnings
from pprint import pprint
from datetime import datetime

from yaml import safe_load
import pandas as pd


### Configuration ###

with open('./Infutor+Census_Analysis.yaml') as f:
    config = safe_load(f)

verbose = config['verbose']
lines_per_chunk = config['lines per chunk']
export = config['write to file']

periods = config['periods']

to_print = config['analysis to print']
timestampped = config['include timestamp']


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


df_2000_census = pd.read_excel(
    './year2000_census_tracts.xlsx',
    usecols=[
        'Geo_FIPS',     # fips
        'SE_T001_001',  # total population
        'SE_T002_001',  # total population (again)
        'SE_T002_002',  # population density
        'SE_T167_001',  # median gross rent
        'SE_T093_003',  # median renter incomes
    ],
    skiprows=1
)

df_2000_high_loss_areas = df_2000_census[
    df_2000_census['Geo_FIPS'].isin(se_high_loss_areas)
]


### Analysis Functions ###

if verbose:
    print("Starting analysis...")

def years_to_effdate(beginning, end):
    return (100 * beginning + 1), (100 * end + 12)

def census_col_by_area(name, df=df_census):
    se = df[['tractid', name]].set_index('tractid').squeeze()
    return se

def census_col_by_area_2000(name, df=df_2000_census):
    se = df[['Geo_FIPS', name]].set_index('Geo_FIPS').squeeze()
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

def print_agg_results(results, name=None):
    if name:
        print(f"{name} results:")
    pprint(results)

def create_txt_filename(
    is_high_loss, infutor_start, infutor_end, timestampped=timestampped
):
    if is_high_loss:
        name = 'For_High-Loss_Areas'
    else:
        name = 'For_The_Entire_LA-OC_Sample'

    if timestampped:
        timestamp = datetime.today().strftime(' (created %Y-%m-%d_%H%M%S)')
    else:
        timestamp = ''

    return f'./data/infutor+census/{name}_{infutor_start}-{infutor_end}' \
           f'{timestamp}.txt'

def export_agg_results(results, filename):
    prompt_width = max(map(len, results.keys())) + 2

    with open(filename, 'w') as f:
        for k, v in results.items():
            prompt = f'{k}: '
            print(f'{prompt:<{prompt_width}}{v}', file=f)
