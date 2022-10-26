from pandas import read_stata

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

df_census = read_stata(**arg_census)

df_high_loss_areas = (
    df_census[df_census['absloss10'] == 1].drop('absloss10', axis=1))

# Save pickle to use later
df_high_loss_areas.to_pickle('./data/infutor+census/df_high_loss_areas.pkl')

# Save csv for manual inspection
df_high_loss_areas['tractid'] = (
    df_high_loss_areas['tractid'].map('{:.0f}'.format))

arg_csv = {
    'path_or_buf': './data/infutor+census/high-loss_areas.csv',
    'index': False
}

df_high_loss_areas.to_csv(**arg_csv)
