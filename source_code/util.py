import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from matplotlib import rcParams
from operator import itemgetter
from typing import Callable, Dict, List, Set, Tuple




def load_data_from_txt(pressure_file:str, rate_file:str, colum_names:Dict[str,List[str]]
                   ={"pressure":["Elapsed time","Data"],
                    "rate":["Elapsed time","Liquid rate"]})->(pd.DataFrame,pd.DataFrame,pd.DataFrame): 
    pressure_time, pressure_measure=colum_names["pressure"]
    rate_time, rate_measure = colum_names["rate"]
    pressure_df = pd.read_csv(pressure_file, delimiter=" ",skiprows=2, names=[pressure_time, pressure_measure],skipinitialspace = True) 
    rate_df = pd.read_csv(rate_file, delimiter=" ",skiprows=2, names=[rate_time, rate_measure],skipinitialspace = True) 
    pressure_rate = pd.concat([pressure_df, rate_df]).sort_values(by=pressure_time) 
    return pressure_df,rate_df,pressure_rate



def calculate_derivative(x_coordinate:List[float],y_coordinate:List[float])->List[float]:
    """
    calculate forward derivative, the last point use the backforward derivative
    Args:
            x_coordinate: the value of x coordinate
            y_coordinate: the value of y coordinate

        Returns:
            derivative.
    """
    if len(x_coordinate)!=len(y_coordinate):
        print(f"the length of x_coordinate '{len(x_coordinate)}' is not equal to the length of y_coordinate '{len(y_coordinate)}'")
        return None
    
    length=len(y_coordinate)
    
    derivative=[0.0]*length
    for i in range(length-1):
        derivative[i]=(y_coordinate[i+1]-y_coordinate[i])/(x_coordinate[i+1]-x_coordinate[i])

    #calculate for the last point
    derivative[-1]=(y_coordinate[length-1]-y_coordinate[length-2])/(x_coordinate[length-1]-x_coordinate[length-2])
    return derivative


def convert_timestamp2hour(start_timestamp:pd._libs.tslibs.timestamps.Timestamp,timestamps:List[pd._libs.tslibs.timestamps.Timestamp])->List[float]:
    hours=[0.0]*len(timestamps)
    for i in range(len(timestamps)-1):
        hours[i+1]=(timestamps[i+1]-start_timestamp).total_seconds()/3600
    return hours

def produce_pressure_4metrics(pressure_df:pd.DataFrame,colum_names:Dict[str,List[str]])->pd.DataFrame:
    pressure_time, pressure_measure,first_order_derivative,second_order_derivative=colum_names["pressure"]
    x_coordinate=pressure_df[pressure_time]
    first_order_derivative=calculate_derivative(x_coordinate,pressure_df[pressure_measure])
    second_order_derivative=calculate_derivative(x_coordinate,first_order_derivative)

    #add first and second derivative to pressure_df dataframe
    pressure_df["first_order_derivative"]=first_order_derivative
    pressure_df["second_order_derivative"]=second_order_derivative
    # pd.set_option('display.max_rows', pressure_df.shape[0]+1)
    return pressure_df
    

# class data_preprocessing:
    