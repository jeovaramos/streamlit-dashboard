import folium
import pandas as pd
import streamlit as st
import geopandas as gpd
import plotly.express as px
from functools import reduce
import datetime as dt
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster


@st.cache(allow_output_mutation=True)
def get_data(path: str):
    data = pd.read_csv(path)

    return data


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


def describe(
        data: pd.DataFrame,
        stats: list = ['median', 'skew', 'mad', 'kurt']) -> pd.DataFrame:

    d = data.describe()

    return d.append(data.reindex(d.columns, axis=1).agg(stats)).T


def datetime_format(x):
    return dt.datetime.strptime(x, '%Y-%m-%d').date()


st.set_page_config(
    page_title="KC Houses - Jeová Ramos",
    page_icon=":house:",
    layout="wide")


# Read the data
data_raw = get_data('data/kc_house_data.csv')

# Adding features
data_raw['price_m2'] = data_raw['price'] / (data_raw['sqft_lot'] * 0.092903)
data_raw['date'] = pd.to_datetime(data_raw['date']).dt.strftime('%Y-%m-%d')


#################################
# Interactive Data Overview
#################################
st.title("Data Overview")

st.sidebar.title('Data Overview Options')
f_attrubutes = st.sidebar.multiselect(
    'Enter Columns', data_raw.columns)

f_zipcode = st.sidebar.multiselect(
    label='Enter zipcode', options=data_raw['zipcode'].unique())

# Subset data
st.write(
    subset_data(
        data_raw,
        f_zipcode,
        f_attrubutes
    )
)

#################################
# Resume average metrics
#################################

# Create two columns
c1, c2 = st.columns(2)

# First Column
c1.title("Resume averages")

# Calculating metrics
c1.dataframe(
    default_metrics(data_raw).sort_values('price_m2'),
    height=500, width=500)

# Second Column
c2.title("Statistical descriptive")

# Subset numerical columns
num_attributes = data_raw.select_dtypes(include=['int64', 'float64'])
num_attributes.drop('id', axis=1, inplace=True)

# Calculating descriptive statistics
c2.dataframe(describe(num_attributes).sort_index(axis=0),
             height=500, width=500)

#################################
# Mapping
#################################

# st.title('Region Overview')
# c1, c2 = st.columns(2)
# c1.header('Portfolio Density')

# df = data_raw.sample(50)
# # Base map
# density_map = folium.Map(
#     location=[data_raw['lat'].mean(), data_raw['long'].mean()],
#     default_zoom_start=15)
# make_cluster = MarkerCluster().add_to(density_map)

# for name, row in df.iterrows():
#     folium.Marker(
#         location=[row['lat'], row['long']],
#         popup=folium.Popup(
#             html=folium.IFrame(
#                 str(
#                     '<h4><b>Price:</b> R$ {:,.2f}</h4><h4><b>Features:</b></h4><h6>{} sqft</h6><h6>{} bedrooms</h6> <h6>{} bathrooms</h6><h6>{} year built</h6>'.format(
#                         row['price'],
#                         row['sqft_living'],
#                         row['bedrooms'],
#                         row['bathrooms'],
#                         row['yr_built']
#                     )
#                 )
#             ),
#             min_width=200,
#             max_width=300,
#             min_height=500,
#             max_height=1000,
#         ),
#         icon=folium.Icon(color='red', prefix='fa', icon='fas fa-home')
#     ).add_to(make_cluster)

# with c1:
#     folium_static(density_map)

# # Region Price Map
# c2.header('Price density')
# df = data_raw[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
# df.columns = ['ZIP', 'PRICE']
# df = df.sample(5)

# region_price_map = folium.Map(
#     location=[data_raw['lat'].mean(), data_raw['long'].mean()],
#     default_zoom_start=15)

# geofile = get_geofile()
# geofile = geofile[geofile['ZIP'].isin(df['ZIP'].unique())]

# folium.Choropleth(
#     data=df, columns=df.columns,
#     geo_data=geofile,
#     key_on='feature.properties.ZIP',
#     fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.2,
#     legend_name='Average price').add_to(region_price_map)

# with c2:
#     folium_static(region_price_map)

#################################
# Interactive charts - Price evolution
#################################
st.sidebar.title('Commercial Options')
st.title('Commercial Attributes')

# Filters
df = data_raw[['yr_built', 'price']].groupby('yr_built').mean().reset_index()
date_min = data_raw['yr_built'].min()
date_max = data_raw['yr_built'].max()
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
df = data_raw[['date', 'price']].groupby('date').mean().reset_index()
date_min = datetime_format(
    data_raw['date'].min())
date_max = datetime_format(
    data_raw['date'].max())
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
    int(data_raw['price'].min()),
    int(data_raw['price'].max()),
    int(data_raw['price'].mean())
    )

df = data_raw.loc[data_raw['price'] < f_price]
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
            data_raw['bathrooms'].unique()
        )
    ),
    index=data_raw['bathrooms'].unique().shape[0] - 1
)

f_floor = st.sidebar.selectbox(
    'Max number of floors',
    sorted(
        set(
            data_raw['floors'].unique()
        )
    ),
    index=data_raw['floors'].unique().shape[0] - 1
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
            data_raw['bedrooms'].unique()
        )
    ),
    index=data_raw['bedrooms'].unique().shape[0] - 1
)

st.subheader('Select Attribute')
feature = st.selectbox(
    'Features',
    options=[
        'price', 'sqft_living', 'sqft_lot', 'sqft_above',
        'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode'])

df = data_raw.loc[(data_raw[f1] <= f_feature1) &
                  (data_raw['bathrooms'] <= f_bathroomns) &
                  (data_raw['floors'] <= f_floor)]

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
