"""
Module for generating sample data from dcfireems for modeling building.
"""

import calendar
from csv import DictWriter
from datetime import datetime

# libraries
import numpy as np
import numpy.random
import pandas as pd

DISPATCH_CALLS_AGG_DATA = {
    'date': [
        '08/2014', '09/2014', '10/2014', '11/2014',
        '12/2014', '01/2015', '02/2015', '03/2015',
        '4/2015', '05/2015', '06/2015', '07/2015',
        '08/2015'
    ],
    'type': {
        'ems': [
            13323, 13392, 13548, 11833,
            12684, 12844, 11280, 12790,
            12940, 14632, 14624, 14740,
            15044
        ],
        'fire': [
            2624, 2513, 2755, 2706,
            2425, 2888, 3694, 2751,
            2908, 2910, 3106, 3166,
            2893
        ]
    },
    'priority': {
        'ems': {
            'high': [
                6679, 6762, 6839, 5840,
                6479, 6534, 5737, 6538,
                6656, 7360, 7748, 7730,
                7775
            ],
            'low': [
                6644, 6630, 6709, 5993,
                6205, 6310, 5543, 6252,
                6284, 7272, 6876, 7010,
                7269
            ]
        },
        'fire': {
            'high': [
                1352, 1240, 1364, 1378,
                1138, 1397, 1832, 1327,
                1399, 1410, 1404, 1420,
                1361
            ],
            'low': [
                1272, 1273, 1391, 1328,
                1287, 1491, 1862, 1424,
                1509, 1500, 1702, 1746,
                1532
            ]
        }
    }
}
TWIT_HANDLE = 'https://twitter.com/dcfireems/status/'
FIELD_NAMES = [
    'date', 'total_calls', 'critical',
    'non_critical', 'fire', 'source'
]
SAMPLE_HIST_WEATHER_CSV = './Washington_2014-08-01_2015-08-31.csv'


def _get_days() -> list:
    """Returns the number of days in the month for date timestamp."""
    days = []
    for d in DISPATCH_CALLS_AGG_DATA['date']:
        month, year = d.split('/')
        # handle leap
        days.append(calendar.monthrange(int(year), int(month))[1])

    return days


def _get_total_calls() -> dict:
    tot_calls = {}
    for i, d in enumerate(DISPATCH_CALLS_AGG_DATA['date']):
        ems = DISPATCH_CALLS_AGG_DATA['type']['ems'][i]
        fire = DISPATCH_CALLS_AGG_DATA['type']['fire'][i]
        tot_calls[d] = ems + fire

    return tot_calls


def _calculate_means() -> dict:
    """
    Get the mean critical ems, non-critical ems, and fire calls for each month.
    """
    daily_means = {}
    days = np.asarray(_get_days())
    crit_ems = np.asarray(
        DISPATCH_CALLS_AGG_DATA['priority']['ems']['high']
    ) / days
    daily_means['crit_ems'] = crit_ems
    non_crit_ems = np.asarray(
        DISPATCH_CALLS_AGG_DATA['priority']['ems']['low']
    ) / days
    daily_means['non_crit_ems'] = non_crit_ems
    fire_other = np.asarray(
        DISPATCH_CALLS_AGG_DATA['type']['fire']
    ) / days
    daily_means['fire'] = fire_other
    return daily_means


def get_dummy_csv(std_dev=2, to_csv=False) -> list:
    """Generate sample csv data using metrics from YTD August 2015.

    Data source: https://fems.dc.gov/publication/total-dispatched-calls-type
    """
    sample_rows = []
    daily_means = _calculate_means()
    for i, n in enumerate(_get_days()):
        # simulate normal distributed daily sample data for the month
        crit_ems = np.rint(numpy.random.normal(
            daily_means['crit_ems'][i], std_dev, n
        ))
        non_crit_ems = np.rint(numpy.random.normal(
            daily_means['non_crit_ems'][i], std_dev, n
        ))
        fire_other = np.rint(numpy.random.normal(
            daily_means['fire'][i], std_dev, n
        ))
        tot_calls = crit_ems + non_crit_ems + fire_other

        # create simulated sample row
        for idx in range(n):
            month, year = DISPATCH_CALLS_AGG_DATA["date"][i].split('/')
            date = '/'.join([month, str(idx + 1), year])
            row = {
                'date': date,
                'total_calls': tot_calls[idx],
                'critical': crit_ems[idx],
                'non_critical': non_crit_ems[idx],
                'fire': fire_other[idx],
                'source': TWIT_HANDLE + '-'.join([month, str(idx + 1), year])
            }
            sample_rows.append(row)

    if to_csv:
        path = './sample_twitter_data.csv'
        with open(path, mode='w', newline='') as csv:
            writer = DictWriter(csv, fieldnames=FIELD_NAMES)
            writer.writeheader()
            writer.writerows(sample_rows)
            print('Saved sample to {}'.format(path))

    return sample_rows


def get_season(date: datetime) -> str:
    """Return season for given date."""
    year = date.year
    seasons = {
        'Summer': (datetime(year, 6, 21), datetime(year, 9, 22)),
        'Autumn': (datetime(year, 9, 23), datetime(year, 12, 20)),
        'Spring': (datetime(year, 3, 21), datetime(year, 6, 20))
    }
    for season, (season_start, season_end) in seasons.items():
        if season_start <= date <= season_end:
            return season
    else:
        return 'Winter'


def summarize_weather_data(
        hist_csv=SAMPLE_HIST_WEATHER_CSV, to_csv=True
) -> pd.DataFrame:
    """Engineer features from historical weather data for model building."""
    hist_df = pd.read_csv(hist_csv)
    hist_df.dt_iso = pd.to_datetime(hist_df.dt_iso)
    hist_df['date'] = pd.to_datetime(hist_df.dt_iso.dt.date)

    # drop useless columns: timezone, city_name, lat, lon, sea_level, grnd_level,
    # rain_3h (will use cumulative sum for rain_1h for rain total), snow_3h,
    # weather_id in lieu of weather_description, and weather_icon
    hist_df = hist_df.drop(columns=[
        'city_name', 'lat', 'lon', 'sea_level', 'grnd_level',
        'rain_3h', 'snow_3h', 'weather_id', 'weather_icon', 'timezone'
    ])

    # let's get mean values per day for quantitative columns and replace NaNs with zero
    grp = hist_df.groupby(by='date').mean()
    mean_df = grp.loc[:, ['temp', 'feels_like', 'pressure', 'humidity',
                          'wind_speed', 'wind_deg', 'rain_1h', 'snow_1h', 'clouds_all']].reset_index()
    mean_df.rename(columns={
        'rain_1h': 'avg_rain', 'snow_1h': 'avg_snow'
    }, inplace=True)
    mean_df.fillna(0, inplace=True)

    # let's get sums for rain_1h and snow_1h so we have total rainfall and snowfall in mm
    grp = hist_df.groupby(by='date').sum()
    sum_df = grp.loc[:, ['rain_1h', 'snow_1h']].rename(columns={
        'rain_1h': 'total_rain', 'snow_1h': 'total_snow'
    }).reset_index()

    # let's update temp_min/max columns with min/max across the whole day
    grp = hist_df.groupby(by='date').min()
    min_df = grp.loc[:, 'temp_min'].reset_index()
    grp = hist_df.groupby(by='date').min()
    max_df = grp.loc[:, 'temp_max'].reset_index()

    # let's one-hot encode weather_main and weather_description columns
    one_hot_df = pd.get_dummies(
        hist_df[['date', 'weather_main', 'weather_description']],
        prefix='weather'
    )
    one_hot_df = one_hot_df.groupby(by='date').max().reset_index()

    # merge engineered features into single dataframe
    weather_df = pd.merge(mean_df, sum_df, how='left', on='date')
    weather_df = pd.merge(weather_df, min_df, how='left', on='date')
    weather_df = pd.merge(weather_df, max_df, how='left', on='date')
    weather_df = pd.merge(weather_df, one_hot_df, how='left', on='date')

    if to_csv:
        file_name = f'Munged_{hist_csv.split("/")[1]}'
        weather_df.to_csv(file_name, index=False)

    return weather_df


# print(get_dummy_csv(to_csv=True))
# print(get_season(datetime(2015, 8, 1)))
# print(summarize_weather_data().head())
