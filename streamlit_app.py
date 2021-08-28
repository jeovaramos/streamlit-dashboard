from lib.sections import Sections
from lib.io import ETL
import streamlit as st


st.set_page_config(
    page_title="KC Houses - Jeová Ramos",
    page_icon=":house:",
    layout="wide")


if __name__ == '__main__':
    data = ETL().load_data()

    ETL().feature_engineering(data)

    Sections.maps(data)

    Sections().multivariate_analysis(data)

    Sections.histogram(data)

    Sections.price_evolution(data)

    Sections.resume_metrics(data)

    Sections.data_overview(data)
