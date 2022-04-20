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
sys.path.insert(1, '../util')
sys.path.insert(1, '../methods')
from util import SelectRows,calculate_derivative,pointInterval_to_pressure,point_dt_to_pressure,print_tuning_parameters,timeInterval_to_sub_df, save_json, load_json
# from plot import PlotNSave
# from plot2 import PlotNSave, plot_histogram
# from data_load_N_preprocess import LoadNPreprocessData
from base_classes import CurveParametersCalc
# from patternRecognition_method import PatternRecognitionMethod
from tangent_method import TangentMethod
# from advanced_method import detect_max_FOD
from derivative_method import DerivativeMethod
from store_transients import StoreTransients
# from store_transients2 import StoreTransients
from func_for_st import PlotNSave, LoadNPreprocessData, download_button

colum_names   ={"pressure":{"time":"Elapsed time(hr)",
                             "measure":"Pressure(psia)",
                             "first_order_derivative":"first_order_derivative",
                             "second_order_derivative":"second_order_derivative"},
                "rate":{"time":"Elapsed time(hr)",
                        "measure":"Liquid rate(STB/D)"}}

def upload_N_preview():
    input_df_pressure=pd.DataFrame()
    input_df_rate=pd.DataFrame()
    st.markdown("## ✨ Upload & Preview ")
    with st.expander("""Upload pressure & flow rate data or only pressure data."""):
        # ce, c1, ce, c2, ce = st.columns([0.01, 3, 0.07, 3, 0.07])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Pressure Data")
            uploaded_file_pressure = st.file_uploader("Upload your pressure file", type=["txt"]) 
            if uploaded_file_pressure is not None:
                input_df_pressure = pd.read_csv(uploaded_file_pressure, 
                                            delimiter=" ",
                                            skiprows=2, 
                                            names=[colum_names["pressure"]["time"], 
                                                    colum_names["pressure"]["measure"]],
                                            skipinitialspace = True)   
            if uploaded_file_pressure!=None:
                st.dataframe(input_df_pressure.head()) 

        with c2:
            st.markdown("##### Flow Rate Data")
            uploaded_file_rate = st.file_uploader("Upload your rate file", type=["txt"])
            if uploaded_file_rate is not None:
                input_df_rate = pd.read_csv(uploaded_file_rate, 
                                            delimiter=" ",
                                            skiprows=2, 
                                            names=[colum_names["rate"]["time"], 
                                                    colum_names["rate"]["measure"]], 
                                            skipinitialspace = True)   
        
    
        
            if uploaded_file_rate!=None:
                st.dataframe(input_df_rate.head())
        
    return input_df_pressure, input_df_rate


def user_input_parameters():
    # denoise_checkBox = st.checkbox(
    #         "Denoise",
    #         value=True,
    #         help="Denoise the input pressure measurements",
    #     )

    # data_inOneRow= st.number_input(
    #             "Data in One Row",
    #             value=1200,
    #             min_value=100,
    #             max_value=3000,
    #             step=100,
    #             help="""Number of data points in every row in a detail plot.""",
    #             # help="Minimum value for the keyphrase_ngram_range. keyphrase_ngram_range sets the length of the resulting keywords/keyphrases. To extract keyphrases, simply set keyphrase_ngram_range to (1, # 2) or higher depending on the number of words you would like in the resulting keyphrases.",
    #         )
    
    
    # c1, c2 = st.columns([0.01, 3, 0.07, 3, 0.07])
    c1, c2 = st.columns(2)
    with c1:
        rows_detailPlot= st.number_input(
                    "The number of rows for a detail plot",
                    value=12,
                    min_value=1,
                    max_value=300,
                    step=1,
                    help="""The whole datasets will be plotted in *rows_plot* rows.""",
                )

        point_halfWindow = st.number_input(
                    "Point Window",
                    value=10,
                    min_value=5,
                    max_value=100,
                    help="""The number of points for observation at the left side or right side. Smaller number preferred when the distribution of the data points is very sparse.""",
                )
        
        polynomial_order = st.number_input(
                    "Polynomial Order",
                    value=1,
                    min_value=1,
                    max_value=5,
                    help="""Recommend using 1 for most cases. More minor transients will be detected with larger number.""",
                )
        
    with c2:
    
        deltaTangent_criterion = st.number_input(
                    "Delta Tangent Threshold",
                    value=20.0,
                    min_value=1.0,
                    max_value=1000.0,
                    step=1.0,
                    format="%.1f",
                    help=""" *Delta Tangent*: Substraction of left tangent and right tangent for a certain point.""",
                )
        
        minor_threshold_shutIn = st.number_input(
                    "Minor Shut-in Threshold",
                    value=0.020,
                    min_value=0.,
                    max_value=10.,
                    step=0.001,
                    format="%.3f",
                    help="""The value to tune the threshold for removing minor *Shut-in Periods*. Set the value to be zero, if you want to keep all the transients that have been screened out.""",
                )
        
        minor_threshold_Flowing = st.number_input(
                    "Minor Flowing Threshold",
                    value=0.020,
                    min_value=0.,
                    max_value=10.,
                    step=0.001,
                    format="%.3f",
                    help="""The value to tune the threshold for removing minor transients in *Flowing Periods*. Set the value to be zero, if you want to keep all the transients that have been screened out.""",
                )
    
    parameters={
        # "denoise_checkBox":denoise_checkBox,
                # "data_inOneRow":int(data_inOneRow),
                "rows_detailPlot": int(rows_detailPlot),
                "Point Window":int(point_halfWindow),
                "Polynomial Order": int(polynomial_order),
                "DeltaTangent Threshold":deltaTangent_criterion,
                "Minor Shut-in Threshold":minor_threshold_shutIn,
                "Minor Flowing Threshold":minor_threshold_Flowing}
    return  parameters

def preprocess_data(input_df_pressure,input_df_rate,denoise):
        processed_data_denoised=LoadNPreprocessData(pressure_df=input_df_pressure.copy(), 
                 rate_df=input_df_rate.copy(),  
                 colum_names=colum_names, 
                 use_SG_smoothing=(denoise=="Yes"))
        pressure_df=processed_data_denoised.pressure_df
        rate_df=processed_data_denoised.rate_df
        pressure_df=pressure_df[0:3000]
        # pressure_df=pressure_df[0:5000]
       
        return pressure_df,rate_df

def coarse_filter(pressure_df,colum_names):
    derivativeMethod=DerivativeMethod(pressure_df,colum_names)
    points=derivativeMethod.percentile_FOD(50,10)
    return points

def detect_using_deltaTangent(points, parameters,pressure_df,colum_names):
    pressure_measure=list(pressure_df[colum_names["pressure"]["measure"]])
    pressure_time=list(pressure_df[colum_names["pressure"]["time"]])
    time_halfWindow=None
    point_halfWindow=parameters["Point Window"]
    polynomial_order=parameters["Polynomial Order"]
    # tangent_type="single_point"
    tangent_type="average"
    deltaTangent_criterion=parameters["DeltaTangent Threshold"]
    identify_useDeltaTangent=TangentMethod(time_halfWindow,point_halfWindow,tangent_type=tangent_type,polynomial_order=polynomial_order)
    buildup,drawdown=identify_useDeltaTangent.predict_useDeltaTangent(pressure_measure,pressure_time,points,deltaTangent_criterion)
    return buildup,drawdown

def FFOD_filter(buildup,drawdown,pressure_df):
    time_step=0.2
    First_FOD=[]
    for buildupOrDrawdown in [buildup,drawdown]:      
        buildupOrDrawdown_df=SelectRows(pressure_df).select_byIndexValueList(buildupOrDrawdown)
        buildupOrDrawdown_first_FOD=DerivativeMethod(buildupOrDrawdown_df,colum_names).detect_first_FOD(time_step)
        First_FOD.append(buildupOrDrawdown_first_FOD)
    buildup=First_FOD[0]
    drawdown=First_FOD[1]
    return buildup,drawdown

def plot_task1_N_task2(colum_names,parameters,buildup,drawdown,pressure_df,rate_df):
    minor_threshold_shutIn=parameters["Minor Shut-in Threshold"]
    minor_threshold_Flowing=parameters["Minor Flowing Threshold"]
    mode="Derivative"
    transients=StoreTransients(pressure_df,
                            minor_threshold_shutIn,
                            minor_threshold_Flowing,
                            buildup,
                            drawdown,
                            colum_names,
                            mode)
    all_flowing=[]
    flowingTransient_objects=transients.flowingTransient_objects
    for flowingTransient_object in flowingTransient_objects:
        all_flowing.append({"Flowing Period":flowingTransient_object.flowing_period,
                            "Breakpoints in Flowing Period":flowingTransient_object.points_inFlowTransient})

    output=parameters.copy()
    del output["rows_detailPlot"]
    output.update({"Number of Shut-in":len(transients.major_drawDown),
                            "Number of All Build-up":len(transients.allPointsStored["buildUp"]), 
                            "Number of All Draw-down":len(transients.allPointsStored["drawDown"]), 
                            "Shut-in Periods":transients.shutInperiods,
                            "Flowing Period & Breakpoints in Flowing":all_flowing})
    
    # output_df=pd.DataFrame(output, index=[0])
    output_df=pd.DataFrame()
    output_df=output_df.append(output,ignore_index=True)
    # output_df=output_df.append({"Shut-in Periods":transients.shutInperiods},ignore_index=True)
    # output_df=output_df.append({"Flowing Period & Breakpoints in Flowing":all_flowing},ignore_index=True)
    print("output_df",output_df)
    # display(output_df)
    
    
    st.markdown("##### 1. Parameters & Detected results")
    # cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])
    c1, c2, c3 = st.columns(3)

    with c1:
        CSVButton2 = download_button(output_df, "Data.csv", "📥 Download (.csv)")
    with c2:
        CSVButton2 = download_button(output, "Data.txt", "📥 Download (.txt)")
    with c3:
        CSVButton2 = download_button(output, "Data.json", "📥 Download (.json)")

    st.write(output_df)
    st.header("")
    
    plot_whole=True
    plot_details=True
    plot_statistics=False
    txt=""
    data_inOneRow=int(len(pressure_df)/parameters["rows_detailPlot"])+1
    #plot task 1
    st.markdown("##### 2. Split shut-in & flowing periods")
    file_name="split shut-in & flowing periods"
    filename_toSave_whole=file_name+"_whole.pdf"
    filename_toSave_details=file_name+"_details.pdf"
    plot_name=file_name

    detect_points_dict={"buildUp":transients.major_buildUp,
            "drawDown":transients.major_drawDown}
    
    PlotNSave(pressure_df,
            rate_df,
            detect_points_dict,
            [],
            data_inOneRow,
            plot_name,
            txt,
            filename_toSave_whole,
            filename_toSave_details,
            plot_statistics,
            plot_whole,
            plot_details,
            colum_names)
    
    st.header("")
    
    #plot task 2
    st.markdown("##### 3. All detected break points")
    file_name="all detected break points"
    filename_toSave_whole=file_name+"_whole.pdf"
    filename_toSave_details=file_name+"_details.pdf"
    plot_name=file_name

    detect_points_dict=transients.allPointsStored
    
    PlotNSave(pressure_df,
            rate_df,
            detect_points_dict,
            [],
            data_inOneRow,
            plot_name,
            txt,
            filename_toSave_whole,
            filename_toSave_details,
            plot_statistics,
            plot_whole,
            plot_details,
            colum_names)
    

    
    

    

st.set_page_config(
    page_title="Transients Identification App", page_icon="📌", initial_sidebar_state="expanded"
)

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

# c30, c31, c32 = st.columns([2.5, 1, 3])
# with c30:
    # st.write(
    #     """
    # # 📌 Transient Identification App
    # """
    # )
st.title("📌 Transient Identification App")
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

input_df_pressure, input_df_rate=upload_N_preview()


if len(input_df_pressure)>0 and len(input_df_rate)>0:
         
    st.markdown("### 🔑 Select Methods for Identification")
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
        with st.expander("Adjust parameters"):
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
    st.markdown("## 🎈 Results for Transient Identification ")
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
    
    
    
    
    
            