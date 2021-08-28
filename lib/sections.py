from lib.io import ETL
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster


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

        df = data.sample(50)
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
        c2.header('Price density')
        df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
        df.columns = ['ZIP', 'PRICE']
        df = df.sample(5)

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


if __name__ == '__main__':
    pass
