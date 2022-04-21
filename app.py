import streamlit as st
import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
from operator import itemgetter
from typing import Callable, Dict, List, Set, Tuple
from scipy.optimize import curve_fit
import csv
import json
from collections import defaultdict

import math 
import warnings
warnings.simplefilter('ignore', np.RankWarning)
from pprint import pprint
import os
import sys
warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
# sys.path.insert(1, '../util')
# sys.path.insert(1, '../methods')
from func_for_st import upload_N_preview,user_input_parameters,preprocess_data,coarse_filter,detect_using_deltaTangent,FFOD_filter,plot_task1_N_task2,LoadNPreprocessData, PlotNSave,download_button



colum_names   ={"pressure":{"time":"Elapsed time(hr)",
                             "measure":"Pressure(psia)",
                             "first_order_derivative":"first_order_derivative",
                             "second_order_derivative":"second_order_derivative"},
                "rate":{"time":"Elapsed time(hr)",
                        "measure":"Liquid rate(STB/D)"}}

# fontsize_css = f""" 
# <style>
#     div.stDataFrame.css-1287gwp.e11ks2pe8 {{
#   font-size: 0.8rem;
# }}
# section.css-po3vlj.exg6vvm15 {{
#   font-size: 0.8rem;
# }}

# small.css-1aehpvj.euu6i2w0 {{
#   font-size: 0.8rem;
# }}

# div.stDataFrame.css-1gzbsyi.e11ks2pe8 {{
#   font-size: 0.8rem;
# }}
# </style> """
     

st.set_page_config(
    page_title="Transients Identification App", 
    page_icon="📌", 
    initial_sidebar_state="expanded",
    layout="centered"
)

# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# st.markdown(fontsize_css, unsafe_allow_html=True)

def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

# st.title("📌 Transient Identification App")
st.markdown("### 📌 Transient Identification App")
# st.header("")

with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
-   The *Transient Identification* app is an easy-to-use interface to detect the transients based on the pressure measurements from Permament Downhole Gauge!
-   The outcomes include:
    - The detected *shut-in periods* & *flowing periods*.
    - The detected *break points* in *flowing periods*.
	    """
    )

    st.markdown("")
st.markdown("")
st.markdown("")

input_df_pressure, input_df_rate=upload_N_preview()

st.markdown("")
st.markdown("")
if len(input_df_pressure)>0 and len(input_df_rate)>0:
         
    st.markdown("### 🔑 Select Method & Parameters")
    with st.form(key="my_form"):
        c1, c2 = st.columns(2)
        with c1:
            denoise = st.radio(
                f"Denoise",
                options=["Yes", "No"],
                help="Select yes if you want to denoise the pressure measurements.",
            )
        # methods = st.multiselect(
        #     "Methods",
        #     options=["DeltaTangent","DeltaFOD","PatternRecognition"],
        #     help="Select the methods you want to use for transients identification.",
        #     default=["DeltaTangent"],
        # )
        with c2:
            methods = st.radio(
                "Methods",
                options=["DeltaTangent","DeltaFOD","PatternRecognition"],
                help="Select the methods you want to use for transients identification. Recommend using *DeltaTangent*",
            )
     
        print("methods",methods)
        with st.expander("Adjust parameters",expanded=True):
            st.markdown("##### Parameters")
            parameters1 = user_input_parameters()
            
        
            
        submit_button = st.form_submit_button(label="Submit")
        print("submit_button",submit_button)
        parameters={"Methods":methods, "Denoise": denoise}
        parameters.update(parameters1)    
        # st.write(pd.DataFrame(parameters, index=[0]))
    if not submit_button:
        st.stop()
    pressure_df,rate_df=preprocess_data(input_df_pressure,input_df_rate,denoise)
    
    st.write("")
    st.write("")
    st.markdown("### 🎈 Results")
    st.write("")
    
    points=coarse_filter(pressure_df,colum_names)
    print("after coarse filter",len(points))
    buildup,drawdown=detect_using_deltaTangent(points, parameters,pressure_df,colum_names)
    print("after detect_using_deltaTangent",len(buildup),len(drawdown))
    buildup,drawdown=FFOD_filter(buildup,drawdown,pressure_df)
    buildup=[int(point) for point in buildup]
    drawdown=[int(point) for point in drawdown]
    print("after FFOD_filter",len(buildup),len(drawdown))
    print("--------type",type(buildup[0]))
    plot_task1_N_task2(colum_names,parameters,buildup,drawdown,pressure_df,rate_df)
    
    
    
    
    
            