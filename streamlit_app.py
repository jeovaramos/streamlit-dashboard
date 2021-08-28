import pandas as pd
import streamlit as st
import geopandas as gpd
import plotly.express as px
from functools import reduce
import datetime as dt
from lib.io import ETL
from lib.sections import Sections


# @st.cache(allow_output_mutation=True)
# def get_data(path: str):
#     data = pd.read_csv(path)

#     return data


def get_geofile(url: str = None) -> gpd.GeoDataFrame:
    if url is None:
        url = str(
            'https://opendata.arcgis.com/datasets/'
            '83fc2e72903343aabff6de8cb445b81c_2.geojson')

    geofile = gpd.read_file(url)

    return geofile


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


# def describe(
#         data: pd.DataFrame,
#         stats: list = ['median', 'skew', 'mad', 'kurt']) -> pd.DataFrame:

#     d = data.describe()

#     return d.append(data.reindex(d.columns, axis=1).agg(stats)).T


def datetime_format(x):
    return dt.datetime.strptime(x, '%Y-%m-%d').date()


st.set_page_config(
    page_title="KC Houses - Jeová Ramos",
    page_icon=":house:",
    layout="wide")


# Read the data
data = ETL().load_data()

# Adding features
ETL().feature_engineering(data)


#################################
# Interactive Data Overview
#################################
Sections.data_overview(data)

#################################
# Resume average metrics
#################################
Sections.resume_metrics(data)


#################################
# Mapping
#################################
Sections.maps(data)

#################################
# Interactive charts - Price evolution
#################################
st.sidebar.title('Commercial Options')
st.title('Commercial Attributes')

# Filters
df = data[['yr_built', 'price']].groupby('yr_built').mean().reset_index()
date_min = data['yr_built'].min()
date_max = data['yr_built'].max()
select_range = st.sidebar.select_slider(
    label='Year built range',
    options=range(date_min, date_max + 1),
    value=[date_min, date_max]
    )

df = df.loc[
    (df['yr_built'] >= select_range[0]) &
    (df['yr_built'] <= select_range[1])]
fig = px.line(df, x='yr_built', y='price')
st.plotly_chart(fig, use_container_width=True)

# Filters
df = data[['date', 'price']].groupby('date').mean().reset_index()
date_min = datetime_format(
    data['date'].min())
date_max = datetime_format(
    data['date'].max())
days_interval = range((date_max + dt.timedelta(days=1) - date_min).days)


select_range = st.sidebar.select_slider(
    label='Date range',
    options=[date_min + dt.timedelta(days=x) for x in days_interval],
    value=[date_min, date_max]
    )

df['date'] = df['date'].apply(datetime_format)

df = df[
    (df['date'] >= select_range[0]) &
    (df['date'] <= select_range[1])]
fig = px.line(df, x='date', y='price')
st.plotly_chart(fig, use_container_width=True)

#################################
# Histogram
#################################
st.header('Price distribution')
st.subheader('Select max price')

f_price = st.slider(
    'Price',
    int(data['price'].min()),
    int(data['price'].max()),
    int(data['price'].mean())
    )

df = data.loc[data['price'] < f_price]
fig = px.histogram(df, x='price', nbins=50)
st.plotly_chart(fig, use_container_width=True)

#################################
# Houses distribution by features
#################################
st.header('Multivariate analysis')

# Filters
f_bathroomns = st.sidebar.selectbox(
    'Max number of bathrooms',
    sorted(
        set(
            data['bathrooms'].unique()
        )
    ),
    index=data['bathrooms'].unique().shape[0] - 1
)

f_floor = st.sidebar.selectbox(
    'Max number of floors',
    sorted(
        set(
            data['floors'].unique()
        )
    ),
    index=data['floors'].unique().shape[0] - 1
)

# Charts
c1, c2, c3 = st.columns(3)

c1.header('Feature 1')
f1 = c1.selectbox(
    'Feature 1',
    options=[
        'price', 'sqft_living', 'sqft_lot', 'sqft_above',
        'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode'])

st.sidebar.title('Attributes Options')
f_feature1 = st.sidebar.selectbox(
    f'Max number of {f1}',
    sorted(
        set(
            data['bedrooms'].unique()
        )
    ),
    index=data['bedrooms'].unique().shape[0] - 1
)

st.subheader('Select Attribute')
feature = st.selectbox(
    'Features',
    options=[
        'price', 'sqft_living', 'sqft_lot', 'sqft_above',
        'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode'])

df = data.loc[(data[f1] <= f_feature1) &
              (data['bathrooms'] <= f_bathroomns) &
              (data['floors'] <= f_floor)]

fig = px.histogram(df, x=feature, nbins=50)
st.plotly_chart(fig, use_container_width=True)

fig = px.histogram(df, x=f1, nbins=20)
c1.plotly_chart(fig, use_container_width=True)

c2.header('Houses per bathrooms')
fig = px.histogram(df, x='bathrooms', nbins=20)
c2.plotly_chart(fig, use_container_width=True)

c3.header('Houses per floor')
fig = px.histogram(df, x='floors', nbins=20)
c3.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    pass
