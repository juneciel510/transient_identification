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
from func_for_st import PlotNSave, LoadNPreprocessData


# def user_input_features():
#     island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
#     sex = st.sidebar.selectbox('Sex',('male','female'))
#     bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
#     bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
#     flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
#     body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
#     data = {'island': island,
#             'bill_length_mm': bill_length_mm,
#             'bill_depth_mm': bill_depth_mm,
#             'flipper_length_mm': flipper_length_mm,
#             'body_mass_g': body_mass_g,
#             'sex': sex}
#     features = pd.DataFrame(data, index=[0])
#     return features

def user_input_parameters():
    denoise_checkBox = st.checkbox(
            "Denoise",
            value=True,
            help="Denoise the input pressure measurements",
        )
    data_inOneRow= st.number_input(
                "Data in One Row",
                value=1200,
                min_value=100,
                max_value=3000,
                step=100,
                help="""Number of data points in every row in a detail plot.""",
                # help="Minimum value for the keyphrase_ngram_range. keyphrase_ngram_range sets the length of the resulting keywords/keyphrases. To extract keyphrases, simply set keyphrase_ngram_range to (1, # 2) or higher depending on the number of words you would like in the resulting keyphrases.",
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
    
    parameters={"denoise_checkBox":denoise_checkBox,
                "data_inOneRow":data_inOneRow,
                "point_halfWindow":point_halfWindow,
                "polynomial_order": polynomial_order,
                "deltaTangent_criterion":deltaTangent_criterion,
                "minor_threshold_shutIn":minor_threshold_shutIn,
                "minor_threshold_Flowing":minor_threshold_Flowing}
    return  parameters,pd.DataFrame(parameters, index=[0])




st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

colum_names   ={"pressure":{"time":"Elapsed time(hr)",
                             "measure":"Pressure(psia)",
                             "first_order_derivative":"first_order_derivative",
                             "second_order_derivative":"second_order_derivative"},
                "rate":{"time":"Elapsed time(hr)",
                        "measure":"Liquid rate(STB/D)"}}




# Collects user input features into dataframe
uploaded_file_pressure = st.sidebar.file_uploader("Upload your input pressure file", type=["txt"])

if uploaded_file_pressure is not None:
    input_df_pressure = pd.read_csv(uploaded_file_pressure, 
                                  delimiter=" ",
                                  skiprows=2, 
                                  names=[colum_names["pressure"]["time"], 
                                         colum_names["pressure"]["measure"]],
                                  skipinitialspace = True)    
else:
    input_df_pressure=None

uploaded_file_rate = st.sidebar.file_uploader("Upload your input rate file", type=["txt"])
if uploaded_file_rate is not None:
    input_df_rate = pd.read_csv(uploaded_file_rate, 
                                delimiter=" ",
                                skiprows=2, 
                                names=[colum_names["rate"]["time"], 
                                        colum_names["rate"]["measure"]], 
                                skipinitialspace = True)   
else:
    input_df_rate=None
    
if uploaded_file_pressure!=None and uploaded_file_rate!=None:
    print("-----------uploaded_file_pressure:",uploaded_file_pressure)  
    processed_data_denoised=LoadNPreprocessData(pressure_df=input_df_pressure.copy(), 
                 rate_df=input_df_rate.copy(),  
                 colum_names=colum_names, 
                 use_SG_smoothing=True)
    pressure_df=processed_data_denoised.pressure_df
    rate_df=processed_data_denoised.rate_df
    pressure_df=pressure_df[0:3000]
    # pressure_df=pressure_df[0:5000]
    pressure_measure=list(pressure_df[colum_names["pressure"]["measure"]])
    pressure_time=list(pressure_df[colum_names["pressure"]["time"]])

parameters,parameters_df = user_input_parameters()

data_inOneRow=parameters["data_inOneRow"]

# Displays the user input features
st.subheader('User Parameters')
st.write(parameters_df)

# st.subheader('Input dataset preview')
st.markdown("### Input dataset preview")

if uploaded_file_pressure!=None and uploaded_file_rate!=None:
    
    st.dataframe(input_df_pressure.head())
    st.dataframe(input_df_rate.head())
    # st.dataframe(pressure_df.head())
    # st.dataframe(rate_df.head())
    derivativeMethod=DerivativeMethod(pressure_df,colum_names)
    points=derivativeMethod.percentile_FOD(50,10)
    
    time_halfWindow=None
    point_halfWindow=parameters["point_halfWindow"]
    polynomial_order=parameters["polynomial_order"]
    # tangent_type="single_point"
    tangent_type="average"
    deltaTangent_criterion=parameters["deltaTangent_criterion"]
    identify_useDeltaTangent=TangentMethod(time_halfWindow,point_halfWindow,tangent_type=tangent_type,polynomial_order=polynomial_order)
    buildup_DT,drawdown_DT=identify_useDeltaTangent.predict_useDeltaTangent(pressure_measure,pressure_time,points,deltaTangent_criterion)
    print("===============output==================")
    # pprint(f"buildup_DT: {len(buildup_DT)},drawdown_DT: {len(drawdown_DT)}")
    points=[buildup_DT,drawdown_DT]
    time_step=0.2
    First_FOD=[]
    for buildupOrDrawdown,point_type in zip(points,["buildUp","drawDown"]):      
        buildupOrDrawdown_df=SelectRows(pressure_df).select_byIndexValueList(buildupOrDrawdown)
        buildupOrDrawdown_first_FOD=DerivativeMethod(buildupOrDrawdown_df,colum_names).detect_first_FOD(time_step)
        First_FOD.append(buildupOrDrawdown_first_FOD)
        # pprint(f"len({point_type}_df): {len(buildupOrDrawdown_df)},len({point_type}_first_FOD): {len(buildupOrDrawdown_first_FOD)}")
    print("===============output==================")
    print("First_FOD")
    
    points_buildUp=First_FOD[0]
    points_drawDown=First_FOD[1]
    minor_threshold_shutIn=parameters["minor_threshold_shutIn"]
    minor_threshold_Flowing=parameters["minor_threshold_Flowing"]
    mode="Derivative"
    transients=StoreTransients(pressure_df,
                            minor_threshold_shutIn,
                            minor_threshold_Flowing,
                            points_buildUp,
                            points_drawDown,
                            colum_names,
                            mode)
    
    print(f"buildup:{len(transients.major_buildUp)}, drawdown:{len(transients.major_drawDown)}")
    points_type="majorTransients"
# parameters={}
    parameters={"Order":polynomial_order,
                "TanThre":deltaTangent_criterion,
                "shutTr":minor_threshold_shutIn,
            "flowTr":minor_threshold_Flowing}
    detect_points_dict={"buildUp":transients.major_buildUp,
                    "drawDown":transients.major_drawDown}

    buildup=detect_points_dict["buildUp"]
    drawdown=detect_points_dict["drawDown"]
    txt=f"buildup:{len(buildup)}, drawdown:{len(drawdown)}"
    pprint(txt)
    plot_whole=True
    plot_details=True
    plot_statistics=False
    save=True
    # folder_name=f"{method}/{points_type}/pointHalfWin_{point_halfWindow}_timeStep_{time_step}"
    folder_name=""
    file_name=""
    for index,name in enumerate(parameters):
        file_name+=f"{name}_{parameters[name]}"
        if index!=len(parameters)-1:
            file_name+="_"
    if save:
        # filename_toSave_whole="../data_output/"+folder_name+"/"+file_name+"_whole.pdf"
        # os.makedirs(os.path.dirname(filename_toSave_whole), exist_ok=True)
        # filename_toSave_details="../data_output/"+folder_name+"/"+file_name+"_details.pdf"
        filename_toSave_whole=file_name+"_whole.pdf"
        filename_toSave_details=file_name+"_details.pdf"
    else:
        filename_toSave_whole=""
        filename_toSave_details=""
    plot_name=file_name

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
else:
    st.write('Awaiting CSV file to be uploaded.')


