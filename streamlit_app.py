import pandas as pd
import streamlit as st


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
data['price_m2'] = data['price'] / data['sqft_lot']

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


st.write(
    default_selection(
        data,
        f_zipcode,
        f_attrubutes
    )
)

if __name__ == '__main__':
    pass
