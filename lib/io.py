import pandas as pd
import datetime as dt
import geopandas as gpd
from functools import reduce


class Load(object):
    def __init__(self, data_path: str = 'data/kc_houses_data.csv'):
        self.data_path = data_path

        return None

    def load_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.path_load)

        return data

    def get_geodata(self, url: str = None):
        if url is None:
            url = str(
                'https://opendata.arcgis.com/datasets/'
                '83fc2e72903343aabff6de8cb445b81c_2.geojson')

        geofile = gpd.read_file(url)

        return geofile


class ETL(object):
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

        return self

    def subset_data(self, f_zipcode, f_attrubutes):
        if (f_zipcode != []) & (f_attrubutes != []):
            df = self.data.loc[
                self.data['zipcode'].isin(f_zipcode),
                f_attrubutes]

        elif (f_zipcode != []) & (f_attrubutes == []):
            df = self.data.loc[self.data['zipcode'].isin(f_zipcode), :]

        elif (f_zipcode == []) & (f_attrubutes != []):
            df = self.data.loc[:, f_attrubutes]

        else:
            df = self.data.copy()

        return df

    def default_metrics(data: pd.DataFrame) -> pd.DataFrame:
        # Average metrics
        df1 = data[['id', 'zipcode']].groupby(
            'zipcode').count().reset_index()

        df2 = data[['price', 'zipcode']].groupby(
            'zipcode').mean().reset_index()

        df3 = data[['sqft_living', 'zipcode']].groupby(
            'zipcode').mean().reset_index()

        df4 = data[['price_m2', 'zipcode']].groupby(
            'zipcode').mean().reset_index()

        data_frames = [df1, df2, df3, df4]
        df_merged = reduce(lambda left, right: pd.merge(
            left, right, on=['zipcode'], how='inner'), data_frames)

        df_merged.columns = [
            'zipcode', 'total_houses', 'mean_price', 'sqft_living', 'price_m2']

        return df_merged

    def describe(
            data: pd.DataFrame,
            stats: list = ['median', 'skew', 'mad', 'kurt']) -> pd.DataFrame:

        d = data.describe()

        return d.append(data.reindex(d.columns, axis=1).agg(stats)).T

    def datetime_format(x):
        return dt.datetime.strptime(x, '%Y-%m-%d').date()


if __name__ == '__main__':
    pass
