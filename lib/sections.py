from lib.io import ETL
import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import datetime as dt


class Sections():
    def __init__(self) -> None:
        pass

    def data_overview(data: pd.DataFrame) -> None:

        st.title("Data Overview")
        st.sidebar.title('Data Overview Options')
        f_columns = st.sidebar.multiselect(
            'Enter Columns', data.columns)

        f_zipcode = st.sidebar.multiselect(
            label='Enter zipcode', options=data['zipcode'].unique())

        # Subset data
        st.write(
            ETL().subset_data(data, f_zipcode, f_columns)
            )

        return None

    def resume_metrics(data: pd.DataFrame) -> None:
        # Create two columns
        c1, c2 = st.columns(2)

        # First Column
        c1.title("Resume averages")

        # Calculating metrics
        c1.dataframe(
            ETL().default_metrics(data).sort_values('price_m2'),
            height=500, width=500)

        # Second Column
        c2.title("Statistical descriptive")

        # Subset numerical columns
        num_attributes = data.select_dtypes(include=['int64', 'float64'])
        num_attributes.drop('id', axis=1, inplace=True)

        # Calculating descriptive statistics
        c2.dataframe(
            ETL().extended_describe(data=num_attributes).sort_index(axis=0),
            height=500, width=500)

        return None

    def maps(data: pd.DataFrame) -> None:
        st.title('Region Overview')
        c1, c2 = st.columns(2)
        c1.header('Portfolio Density')

        df = data.sample(1000)
        # Base map
        density_map = folium.Map(
            location=[data['lat'].mean(), data['long'].mean()],
            default_zoom_start=15)
        make_cluster = MarkerCluster().add_to(density_map)

        for _, row in df.iterrows():
            folium.Marker(
                location=[row['lat'], row['long']],
                popup=folium.Popup(
                    html=folium.IFrame(
                        str(
                            '<h4><b>Price:</b> R$ {:,.2f}</h4><h4><b>Features:</b></h4><h6>{} sqft</h6><h6>{} bedrooms</h6> <h6>{} bathrooms</h6><h6>{} year built</h6>'.format(
                                row['price'],
                                row['sqft_living'],
                                row['bedrooms'],
                                row['bathrooms'],
                                row['yr_built']
                            )
                        )
                    ),
                    min_width=200,
                    max_width=300,
                    min_height=500,
                    max_height=1000,
                ),
                icon=folium.Icon(color='red', prefix='fa', icon='fas fa-home')
            ).add_to(make_cluster)

        with c1:
            folium_static(density_map)

        # Region Price Map
        c2.header('Price average by zipcode')

        df = data.copy()
        df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
        df.columns = ['ZIP', 'PRICE']

        region_price_map = folium.Map(
            location=[data['lat'].mean(), data['long'].mean()],
            default_zoom_start=15)

        geofile = ETL().get_geodata()
        geofile = geofile[geofile['ZIP'].isin(df['ZIP'].unique())]

        folium.Choropleth(
            data=df, columns=df.columns,
            geo_data=geofile,
            key_on='feature.properties.ZIP',
            fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.2,
            legend_name='Average price').add_to(region_price_map)

        with c2:
            folium_static(region_price_map)

        return None

    def _slider_max(self, data, feature: str) -> None:
        slider_filter = st.sidebar.selectbox(
            f'Max number of {feature}',
            sorted(
                set(
                    data[feature].unique()
                )
            ),
            index=data[feature].unique().shape[0] - 1
        )

        return slider_filter

    def price_evolution(data: pd.DataFrame) -> None:
        st.sidebar.title('Commercial Options')
        st.title('Commercial Attributes')

        # Filters
        df = data[['yr_built', 'price']].groupby(
            'yr_built').mean().reset_index()
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
        date_min = ETL.date_format(
            data['date'].min())
        date_max = ETL.date_format(
            data['date'].max())
        days_interval = range(
            (date_max + dt.timedelta(days=1) - date_min).days)

        select_range = st.sidebar.select_slider(
            label='Date range',
            options=[date_min + dt.timedelta(days=x) for x in days_interval],
            value=[date_min, date_max]
            )

        df['date'] = df['date'].apply(ETL.date_format)

        df = df[
            (df['date'] >= select_range[0]) &
            (df['date'] <= select_range[1])]
        fig = px.line(df, x='date', y='price')
        st.plotly_chart(fig, use_container_width=True)

        return None

    def histogram(data: pd.DataFrame) -> None:

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

        return None

    def multivariate_analysis(self, data: pd.DataFrame) -> None:
        st.title('Multivariate Analysis')
        st.sidebar.title('Multivariate Options')

        options = [
            'price', 'bathrooms', 'bedrooms', 'floors', 'sqft_living',
            'sqft_lot', 'sqft_above', 'sqft_basement', 'yr_built',
            'yr_renovated', 'zipcode'
            ]

        c1, c2, c3, c4 = st.columns((2, 1, 1, 1))
        # c3, c4 = st.columns(2)

        # Feature 1
        c1.header('Feature 1')
        feature_1 = c1.selectbox('Feature 1', options=options, index=0)
        filter_1 = self._slider_max(data, feature_1)

        # Feature 2
        c2.header('Feature 2')
        feature_2 = c2.selectbox('Feature 2', options=options, index=2)
        filter_2 = self._slider_max(data, feature_2)

        # Feature 3
        c3.header('Feature 3')
        feature_3 = c3.selectbox('Feature 3', options=options, index=3)
        filter_3 = self._slider_max(data, feature_3)

        # Feature 4
        c4.header('Feature 4')
        feature_4 = c4.selectbox('Feature 4', options=options, index=4)
        filter_4 = self._slider_max(data, feature_4)

        df = data.loc[(data[feature_1] <= filter_1) &
                      (data[feature_2] <= filter_2) &
                      (data[feature_3] <= filter_3) &
                      (data[feature_4] <= filter_4)]

        fig1 = px.histogram(df, x=feature_1, nbins=10)
        c1.plotly_chart(fig1, use_container_width=True)

        fig2 = px.histogram(df, x=feature_2, nbins=10)
        c2.plotly_chart(fig2, use_container_width=True)

        fig3 = px.histogram(df, x=feature_3, nbins=10)
        c3.plotly_chart(fig3, use_container_width=True)

        fig4 = px.histogram(df, x=feature_4, nbins=10)
        c4.plotly_chart(fig4, use_container_width=True)

        return None


if __name__ == '__main__':
    pass
