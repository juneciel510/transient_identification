from re import T
import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from matplotlib import rcParams
from operator import itemgetter
from typing import Callable, Dict, List, Set, Tuple


def detect_breakpoints(first_order_derivative):
    breakpoints=[]
    std=statistics.stdev(first_order_derivative)
    print(std)
    for i in range(len(first_order_derivative)-1):
        if (first_order_derivative[i]>0)^(first_order_derivative[i+1]>0) and abs(first_order_derivative[i+1])>2*std :
            breakpoints.append(i+1)

    return breakpoints




def detect_breakpoints_2(first_order_derivative,second_order_derivative):
    breakpoints=[]
    std_1=statistics.stdev(first_order_derivative)
    std_2=statistics.stdev(second_order_derivative)
    print(f"---the std of first_order_derivative: {std_1}\n---the std of second_order_derivative: {std_2}")
    compare_point=first_order_derivative[0]
    for i in range(len(first_order_derivative)-1):
        if (compare_point>0)^(first_order_derivative[i+1]>0):
            if (abs(second_order_derivative[i+1])>2.4*std_2) and (abs(first_order_derivative[i+1])>2.4*std_1) :
                breakpoints.append(i+1)
                compare_point=first_order_derivative[i+1]
        else:
            compare_point=first_order_derivative[i+1]
            
 
    return breakpoints


def detect_breakpoints_3(first_order_derivative,second_order_derivative, noise_threshold):
    breakpoints=[]
#     denoised_first_order_derivative=[]
    std_1=statistics.stdev(first_order_derivative)
    std_2=statistics.stdev(second_order_derivative)
    print(std_1,std_2)
    compare_point=first_order_derivative[0]
    for i in range(len(first_order_derivative)-1):
#         denoised_first_order_derivative.append(compare_point)
        if (first_order_derivative[i]>0)^(first_order_derivative[i+1]>0) and abs(first_order_derivative[i+1])>noise_threshold*std_1:
            breakpoints.append(i+1)  
        elif (compare_point>0)^(first_order_derivative[i+1]>0):
            if (abs(second_order_derivative[i+1])>noise_threshold*std_2) and (abs(first_order_derivative[i+1])>noise_threshold*std_1) :
                breakpoints.append(i+1)
                compare_point=first_order_derivative[i+1]
        else:
            compare_point=first_order_derivative[i+1]
  
 
    return breakpoints

def detect_breakpoints_4(pressure_df,colum_names:Dict[str,List[str]], noise_threshold:float,window_time_duration:float,close_zero_threshold:float)->List[int]:
    """
    Args:
        window_time_duration: in hours
    
    """
    pressure_time=pressure_df[colum_names["pressure"][0]]
    first_order_derivative=pressure_df[colum_names["pressure"][2]]
    second_order_derivative=pressure_df[colum_names["pressure"][3]]
    breakpoints=[]

    std_1=statistics.stdev(first_order_derivative)
    std_2=statistics.stdev(second_order_derivative)
    print(std_1,std_2)
    compare_point=first_order_derivative[0]
    for i in range(len(first_order_derivative)-1):
    
        window_end_time=pressure_time[i]
        window_start_time=pressure_time[i]-window_time_duration
        # print("first_order_derivative",first_order_derivative)
        if window_start_time>=0 and (abs(second_order_derivative[i+1])>noise_threshold*2*std_2) and (abs(first_order_derivative[i+1])>noise_threshold*2*std_1):
            pressure_df_inWindow=pressure_df.loc[(pressure_time >= window_start_time) & (pressure_time <= window_end_time)]
      
            #dp/dt|tau close to zero
            close_zero_inWindow=True
            for value in pressure_df_inWindow[colum_names["pressure"][2]]:
                if abs(value)>close_zero_threshold*std_1:
                    close_zero_inWindow=False
                    break
            if close_zero_inWindow:
                breakpoints.append(i+1)
             
        if (first_order_derivative[i]>0)^(first_order_derivative[i+1]>0) and abs(first_order_derivative[i+1])>noise_threshold*std_1:
            breakpoints.append(i+1)  

        elif (compare_point>0)^(first_order_derivative[i+1]>0):
            if (abs(second_order_derivative[i+1])>noise_threshold*std_2) and (abs(first_order_derivative[i+1])>noise_threshold*std_1) :
                breakpoints.append(i+1)
                compare_point=first_order_derivative[i+1]
    

        else:
            compare_point=first_order_derivative[i+1]
  
 
    return breakpoints


def detect_breakpoints_startPoint(first_order_derivative,noise_threshold=2.5):
    breakpoints=[]
    std=statistics.stdev(first_order_derivative)
    for i in range(len(first_order_derivative)):
        if abs(first_order_derivative[i])>noise_threshold*std:
            breakpoints.append(i)
    return breakpoints

def detect_breakpoints_startPoint2(first_order_derivative,second_order_derivative,noise_threshold):
    breakpoints=[]
    std_1=statistics.stdev(first_order_derivative)
    std_2=statistics.stdev(second_order_derivative)
    for i in range(len(first_order_derivative)-1):
        if (abs(second_order_derivative[i+1])>noise_threshold*std_2) and (abs(first_order_derivative[i+1])>noise_threshold*std_1) :
                breakpoints.append(i+1)
    return breakpoints

