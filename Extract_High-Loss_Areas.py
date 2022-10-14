from pandas import read_csv

df_census = read_csv(
    './neighborhood_change_theo.csv', dtype={'tractid': 'string'}
)

df_high_loss_areas = df_census[df_census['absloss10'] == 1]['tractid']

df_high_loss_areas.to_csv('./data/infutor+census/high-loss_areas.csv')
