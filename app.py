import streamlit as st
from apps.spatial_domain_filters import SpatialDomainFilter
from apps.frequency_domain_filters import FrequencyDomainFilters
from apps.histogram_technique import HistogramTechnique
from apps.portfolio import Portfolio


st.set_page_config(page_title="Image Filters App")

st.sidebar.image("./images/logo.png")

if 'active_app' not in st.session_state:
    st.session_state.active_app = 'Portfolio'

apps = {
    "Spatial Domain Filters": "🪐 Spatial Domain Filters",
    "Frequency Domain Filters": "🪐 Frequency Domain Filters",
    "Histogram Technique":"🪐 Histogram Technique",
    "Portfolio":"😎 Portfolio",
}

for app_key in apps:
    if app_key != "spatial_filters" and app_key != "frequency_filters" and st.sidebar.button(apps[app_key], key=app_key):
        st.session_state.active_app = app_key

if st.session_state.active_app == "Spatial Domain Filters":
    SpatialDomainFilter(st)

if st.session_state.active_app == "Frequency Domain Filters":
    FrequencyDomainFilters(st)

if st.session_state.active_app == "Histogram Technique":
    HistogramTechnique(st)

if st.session_state.active_app == "Portfolio":
    Portfolio(st)

st.markdown(
    """
    <style>
        .css-1d391kg {
            padding: 10px;
            background-color: #fff;
        }
        
        .stButton > button {
            width: 100%;
            border-radius: 5px;
            border: 2px solid #ccc;
            margin: 2px 0;
            color: black;
            background-color: #FFF0F0;
        }

         .stButton > button > div > p{
            color:#ff4b4b;
         }
        
        .stButton > button:hover,.stButton > button:hover > div > p {
            background-color: #ff4b4b;
            outline: none;
            color: white;
        }

        .stButton > button:focus,.stButton > button:focus > div > p{
            background-color: #ff4b4b;
            outline: none;
            color: white;
        }

        .stButton > button {
            display: flex;
            align-items: center;
            justify-content: start;
            padding-left: 10px;
            border: none;
        }

        .st-emotion-cache-6qob1r.eczjsme3{
            background-color: #FF885D;
        }

        .st-emotion-cache-1nm2qww.eczjsme2 svg{
            color: white;
        }

        .st-emotion-cache-13ln4jf{
            max-width: 60rem;
        }

        .st-emotion-cache-1b0udgb.e115fcil0{
            color: #ff4b4b;
        }

        # .st-emotion-cache-1v0mbdj.e115fcil1{
        #     height: 90px;
        #     width: 90px;
        #     border-radius: 50%;
        #     background-color: white;
        # }

        .st-emotion-cache-1kyxreq.e115fcil2{
            display: flex;
            justify-content: center;
            align-items: center;
        }

    </style>
    """,
    unsafe_allow_html=True
)