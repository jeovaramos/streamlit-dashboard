import numpy as np
import pandas as pd
import streamlit as st


def default_selection(data, f_zipcode, f_attrubutes):
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


def default_metrics(data):
    # Average metrics
    df1 = data[['id', 'zipcode']].groupby(
        'zipcode').count().reset_index()

    df2 = data[['id', 'zipcode']].groupby(
        'zipcode').mean().reset_index()

    df3 = data[['sqft_living', 'zipcode']].groupby(
        'zipcode').mean().reset_index()

    df4 = data[['price_m2', 'zipcode']].groupby(
        'zipcode').mean().reset_index()

    # Merge
    mtc = pd.merge(df1, df2, on='zipcode', how='inner')
    mtc = pd.merge(mtc, df3, on='zipcode', how='inner')
    mtc = pd.merge(mtc, df4, on='zipcode', how='inner')

    mtc.columns = [
        'zipcode', 'total_houses', 'price', 'sqft_living', 'price_m2']

    return mtc


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

# ======================
# Data Overview
# ======================

st.title("Data Overview")

f_attrubutes = st.sidebar.multiselect(
    'Enter Columns', data.columns)

f_zipcode = st.sidebar.multiselect(
    label='Enter zipcode', options=data['zipcode'].unique())

st.write(
    default_selection(
        data,
        f_zipcode,
        f_attrubutes
    ).sort_values('price')
)

st.title("Resume averages")
st.dataframe(
    default_metrics(data).sort_values('price_m2'),
    height=600)

st.title("Statistical descriptive")
num_attributes = data.select_dtypes(include=['int64', 'float64'])
num_attributes.drop('id', axis=1, inplace=True)
average = pd.DataFrame(num_attributes).apply(np.mean)
median = pd.DataFrame(num_attributes).apply(np.median)
std = pd.DataFrame(num_attributes).apply(np.std)
max_ = pd.DataFrame(num_attributes).apply(np.max)
min_ = pd.DataFrame(num_attributes).apply(np.min)

df1 = pd.concat([average, median, std, max_, min_], axis=1).reset_index()
df1.columns = ['attributes', 'average', 'median', 'std', 'max', 'min']
st.dataframe(df1.sort_values('attributes'))


if __name__ == '__main__':
    pass
