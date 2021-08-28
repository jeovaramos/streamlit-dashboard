import pandas as pd
import datetime as dt
import streamlit as st
import geopandas as gpd
from functools import reduce


class ETL(object):
    def __init__(self, data_path: str = 'data/kc_house_data.csv'):
        self.data_path = data_path

        return None

    @st.cache(allow_output_mutation=True)
    def load_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.data_path)

        return data

    @st.cache()
    def get_geodata(self, url: str = None):
        if url is None:
            url = str(
                'https://opendata.arcgis.com/datasets/'
                '83fc2e72903343aabff6de8cb445b81c_2.geojson')

        geofile = gpd.read_file(url)

        return geofile

    def price_m2(self, data: pd.DataFrame):
        data['price_m2'] = data['price'] / (data['sqft_lot'] * 0.092903)

        return None

    def datetime_format(self, data: pd.DataFrame):

        data['date'] = pd.to_datetime(
            data['date']).dt.strftime('%Y-%m-%d')

        return None

    def date_format(x):
        return dt.datetime.strptime(x, '%Y-%m-%d').date()

    def feature_engineering(self, data: pd.DataFrame):
        self.price_m2(data)
        self.datetime_format(data)

        return None

    def subset_data(self, data, f_zipcode, f_columns):
        if (f_zipcode != []) & (f_columns != []):
            df = data.loc[data['zipcode'].isin(f_zipcode), f_columns]

        elif (f_zipcode != []) & (f_columns == []):
            df = data.loc[data['zipcode'].isin(f_zipcode), :]

        elif (f_zipcode == []) & (f_columns != []):
            df = data.loc[:, ['zipcode'] + f_columns]

        else:
            df = data.copy()

        return df

    def default_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
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

    def extended_describe(self, data: pd.DataFrame) -> pd.DataFrame:
        stats = ['median', 'skew', 'mad', 'kurt']
        d = data.describe()

        return d.append(data.reindex(d.columns, axis=1).agg(stats)).T


if __name__ == '__main__':
    pass
