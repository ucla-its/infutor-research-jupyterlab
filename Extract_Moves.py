from os import remove
from pathlib import Path

from yaml import safe_load
import pandas as pd
import numpy as np


### Configuration ###

with open('./Extract_Moves.yaml') as f:
    config = safe_load(f)

project_name = config['project name']

verbose = config['verbose']
lines_per_chunk = config['lines per chunk']
force = config['force re-extraction']
create_csv = config['create csv']

move_creation_method = config['move creation method']
z4types = config['zip+4 types']
county_codes = config['county codes']


### Set up environment ###

if verbose:
    print("Creating status file and output directories...")

# Create status file
if force:
    try:
        remove(f"./{project_name}_status-{lines_per_chunk}.txt")
    except FileNotFoundError:
        pass

try:
    f = open(f"./{project_name}_status-{lines_per_chunk}.txt")
except FileNotFoundError:
    f = open(f"./{project_name}_status-{lines_per_chunk}.txt", 'a')
    finished_chunks = set()
else:
    finished_chunks = {int(line) for line in f.readlines()}
finally:
    f.close()

# Create chunks folder
Path(f'./data/{project_name}/moves-{lines_per_chunk}').mkdir(
    parents=True, exist_ok=True
)


### Process all_states.csv ###

if verbose:
    if force:
        print("Force processing all_states.csv...")
    else:
        print("Processing all_states.csv...")

# Load high-loss areas
se_high_loss_areas = pd.read_pickle(
    './data/infutor+census/se_high_loss_areas.pkl'
)

# Settings
usecols_all_states = ['z4type', 'effdate']
for i in range(2, 11):
    usecols_all_states.append(f'z4type{i}')
    usecols_all_states.append(f'effdate{i}')
for i in range(1, 11):
    usecols_all_states.append(f'lat{i}')
    usecols_all_states.append(f'lon{i}')
    usecols_all_states.append(f'fips{i}')

dtype_all_states = {
    'z4type': 'string',
    'effdate': 'float',
    'lat1': 'float',
    'lon1': 'float',
    'fips1': 'float',
}
for i in range(2, 11):
    dtype_all_states[f'z4type{i}'] = 'string'
    dtype_all_states[f'effdate{i}'] = 'float'
    dtype_all_states[f'lat{i}'] = 'float'
    dtype_all_states[f'lon{i}'] = 'float'
    dtype_all_states[f'fips{i}'] = 'float'

list_dict_col_names = [
    {
        'z4type': 'z4type',
        'effdate': 'effdate',
        'lat1': 'lat',
        'lon1': 'lon',
        'fips1': 'fips',
    }
]
for i in range(2, 11):
    list_dict_col_names.append(
        {
            f'z4type{i}': 'z4type',
            f'effdate{i}': 'effdate',
            f'lat{i}': 'lat',
            f'lon{i}': 'lon',
            f'fips{i}': 'fips',
        }
    )

# FIXME: Maybe handle "Not in California" differently

na_values_all_states = 'Not in California'

# Load all_states
it_all_states = pd.read_csv(
    "./data/all_states.csv",
    usecols=usecols_all_states,
    dtype=dtype_all_states,
    chunksize=lines_per_chunk,
    na_values=na_values_all_states
)

# Define move creation method
def dont_dropna(df_areas):
    df_moves = df_areas[['effdate', 'lat', 'lon', 'fips']]
    df_moves = df_moves.rename(
        columns={
            'effdate': 'date_left',
            'lat': 'dest_lat',
            'lon': 'dest_lon',
            'fips': 'dest_fips',
        }
    )
    df_moves[['date_arrived', 'orig_lat', 'orig_lon', 'orig_fips']] = (
        df_moves.groupby(df_moves.index)[
            ['date_left', 'dest_lat', 'dest_lon', 'dest_fips']
        ].shift()
    )
    return df_moves

def preshift_dropna(df_areas):
    df_areas = df_areas.dropna(subset=['effdate', 'lat', 'lon', 'fips'])
    df_moves = dont_dropna(df_areas)
    return df_moves

def postshift_dropna(df_areas):
    df_moves = dont_dropna(df_areas)
    df_moves = df_moves.dropna()
    return df_moves

def drop_first_origin(df_areas):
    df_moves = dont_dropna(df_areas)
    df_moves = df_moves.dropna(subset=['date_arrived'])
    return df_moves

if move_creation_method == 0:
    create_moves = dont_dropna
elif move_creation_method == 1:
    create_moves = preshift_dropna
elif move_creation_method == 2:
    create_moves = postshift_dropna
elif move_creation_method == 3:
    create_moves = drop_first_origin

# vectorized haversine function: https://stackoverflow.com/a/40453439
def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    slightly modified version: of http://stackoverflow.com/a/29546836/2901002

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    """
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a))

# Categorization functions
def in_areas(fips, areas=se_high_loss_areas):
    return fips.isin(areas)

def in_counties(fips, county_codes=['037', '059']):
    return fips.astype('string').str[1:4].isin(county_codes)

def which_region(fips):
    return (fips.mask(in_areas, 'high-loss')
                .mask(in_counties, 'LA/OC')
                .where(lambda se: se.isin(['high-loss', 'LA/OC']), 'outside'))

def is_to_self(moves):
    return moves['orig_fips'] == moves['dest_fips']

# Process all chunks
for index, chunk in enumerate(it_all_states):
    if (index in finished_chunks) and (not force):
        if verbose:
            print(f"Chunk {index} done")
        continue

    # Split rows into individual areas
    list_df_all_areas = [
        chunk[list_dict_col_names[i].keys()].rename(
            columns=list_dict_col_names[i]
        ) for i in range(10)
    ]

    # Recombine all areas + sort by effdate
    df_all_areas = (
        pd.concat(list_df_all_areas)
            .dropna(subset=['effdate'])
            .sort_values('effdate', kind='stable')
    )

    # All effdates that do not have a z4type in z4types
    bi_all_dropped = ~(
        df_all_areas['z4type'].isin(z4types)
            .groupby([df_all_areas.index, df_all_areas['effdate']])
            .transform('any')
    )

    # Change values so selected effdates are not removed
    df_all_areas.loc[bi_all_dropped, 'z4type'] = 'empty'
    df_all_areas.loc[bi_all_dropped, 'fips'] = np.NaN

    # Filter by Zip+4 type
    z4types_mask = (*z4types, 'empty')

    df_filtered_areas = df_all_areas[df_all_areas['z4type'].isin(z4types_mask)]

    # Choose leftmost fips of each effdate
    df_filtered_areas = (
        df_filtered_areas.groupby([df_filtered_areas.index, 'effdate']).first()
    )
    df_filtered_areas = df_filtered_areas.reset_index('effdate')

    # Link previous & next areas as moves
    df_all_moves = create_moves(df_filtered_areas)

    # Filter by county code
    df_filtered_moves = df_all_moves[
        df_all_moves['orig_fips'].astype('string').str[1:4].isin(county_codes)
        | df_all_moves['dest_fips'].astype('string').str[1:4].isin(county_codes)
    ]

    # Calculate move distance in km
    with pd.option_context('mode.chained_assignment', None):
        df_filtered_moves['dist'] = haversine(
            df_filtered_moves['orig_lat'],
            df_filtered_moves['orig_lon'],
            df_filtered_moves['dest_lat'],
            df_filtered_moves['dest_lon']
        )

    df_final_moves = df_filtered_moves[
        ['date_arrived', 'orig_fips', 'date_left', 'dest_fips', 'dist']
    ]

    # Categorize/index moves by orig and dest
    with pd.option_context('mode.chained_assignment', None):
        df_final_moves['to_self'] = is_to_self(df_final_moves)
        df_final_moves['orig_region'] = which_region(
            df_final_moves['orig_fips']
        )
        df_final_moves['dest_region'] = which_region(
            df_final_moves['dest_fips']
        )

    df_final_moves = df_final_moves.set_index(
        ['to_self', 'orig_region', 'dest_region']
    )

    # Write to file
    if create_csv:
        df_final_moves.to_csv(
            f"./data/{project_name}/moves-{lines_per_chunk}/{index}.csv"
        )

    df_final_moves.to_pickle(
        f'./data/{project_name}/moves-{lines_per_chunk}/{index}.pkl'
    )

    # Update status
    with open(
        f"./{project_name}_status-{lines_per_chunk}.txt", 'a'
    ) as status_file:
        status_file.write(f"{index}\n")

    if verbose:
        print(f"Chunk {index} done")
