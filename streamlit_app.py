import numpy as np
import pandas as pd
import streamlit as st
from functools import reduce


def subset_data(data, f_zipcode, f_attrubutes):
    if (f_zipcode != []) & (f_attrubutes != []):
        df = data.loc[data['zipcode'].isin(f_zipcode),
                      f_attrubutes]

    elif (f_zipcode != []) & (f_attrubutes == []):
        df = data.loc[data['zipcode'].isin(f_zipcode), :]

    elif (f_zipcode == []) & (f_attrubutes != []):
        df = data.loc[:, f_attrubutes]

    else:
        df = data.copy()
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


st.set_page_config(
    page_title="KC Houses - Jeová Ramos",
    page_icon=":house:",
    layout="wide")


@st.cache(allow_output_mutation=True)
def get_data(path: str):
    data = pd.read_csv(path)

    return data


# Read the data
data = get_data('data/kc_house_data.csv')

# Adding features
data['price_m2'] = data['price'] / (data['sqft_lot'] * 0.092903)

# Showing five lines of data
st.title("Raw data sample")
st.write(data.sample(5))

# Data Overview
st.title("Data Overview")

f_attrubutes = st.sidebar.multiselect(
    'Enter Columns', data.columns)

f_zipcode = st.sidebar.multiselect(
    label='Enter zipcode', options=data['zipcode'].unique())

# Subset data
data = subset_data(
        data, f_zipcode, f_attrubutes)

st.write(
    data.sort_values('price')
)

# Create two columns
c1, c2 = st.columns(2)

# First Column
c1.title("Resume averages")

# Calculating metrics
c1.dataframe(
    default_metrics(data).sort_values('price_m2'),
    height=500, width=500)

# Second Column
c2.title("Statistical descriptive")

# Subset numerical columns
num_attributes = data.select_dtypes(include=['int64', 'float64'])
num_attributes.drop('id', axis=1, inplace=True)

# Calculating descriptive statistics
c2.dataframe(describe(num_attributes),
             height=500, width=500)


if __name__ == '__main__':
    pass
