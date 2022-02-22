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
    

def pointsInterval_to_timeInterval(start_point:int, 
                              end_point:int, 
                              pressure_df:pd.DataFrame, 
                              colum_names:Dict[str,Dict[str,str]]):
    '''
    Args:
    start_point, end_point:the index of the point in the pressure time
    Return:
    time interval between the two points in hours, can be negative
    '''
    pressure_time=pressure_df[colum_names["pressure"]["time"]]
    return pressure_time[end_point]-pressure_time[start_point]

# def timeInterval_to_pointsInterval(start_time,end_time,pressure_df,rate_df,colum_names):
#     rate_time=rate_df[colum_names["rate"]["time"]]
#     sub_rate_df=rate_df.loc[(rate_time >= start_time) & (rate_time <= end_time)]
#     # pd.set_option('display.max_rows', sub_rate_df.shape[0]+1)
#     pressure_time=pressure_df[colum_names["pressure"]["time"]]
#     sub_pressure_df=pressure_df.loc[(pressure_time >= start_time) & (pressure_time <= end_time)]
#     return sub_rate_df,sub_pressure_df

def timeInterval_to_sub_df(start_time,end_time,pressure_df,rate_df,colum_names):
    rate_time=rate_df[colum_names["rate"]["time"]]
    sub_rate_df=rate_df.loc[(rate_time >= start_time) & (rate_time <= end_time)]
    # pd.set_option('display.max_rows', sub_rate_df.shape[0]+1)
    pressure_time=pressure_df[colum_names["pressure"]["time"]]
    sub_pressure_df=pressure_df.loc[(pressure_time >= start_time) & (pressure_time <= end_time)]
    return sub_rate_df,sub_pressure_df
# def timeInterval_to_indexOfMaxValue():

def point_dt_to_pressure(point_index:int,
                               pressure_df, 
                               delta_t:float=1/12,
                               colum_names:Dict={"pressure":{"time":"Date",
                                "measure":"Pressure (psia)",
                                "first_order_derivative":"first_order_derivative",
                                "second_order_derivative":"second_order_derivative"},
                                 "rate":{"time":"Time@end",
                                         "measure":"Liquid rate (STB/D)"}})->List[float]:
    """
    Args:
        delta_t:hour
    """
    pressure_time=pressure_df[colum_names["pressure"]["time"]]
    start_time=pressure_time.iloc[point_index]
    end_time=start_time+delta_t
    if end_time>start_time:   
        sub_pressure_df=pressure_df.loc[(pressure_time >= start_time) & (pressure_time <= end_time)]
        return sub_pressure_df[colum_names["pressure"]["measure"]]
    
    sub_pressure_df=pressure_df.loc[(pressure_time >= end_time) & (pressure_time <= start_time)]
    display(sub_pressure_df)
    return sub_pressure_df[colum_names["pressure"]["measure"]]

def pointInterval_to_pressure(point_index:int,
                               pressure_df, 
                               delta_point:int=10,
                               colum_names:Dict={"pressure":{"time":"Date",
                                "measure":"Pressure (psia)",
                                "first_order_derivative":"first_order_derivative",
                                "second_order_derivative":"second_order_derivative"},
                                 "rate":{"time":"Time@end",
                                         "measure":"Liquid rate (STB/D)"}})->List[float]:
    """
    extract pressure measurements between point_index and point_index+delta_point
    Args:
    """
    if delta_point>0:
        return pressure_df[colum_names["pressure"]["measure"]].iloc[point_index:point_index+delta_point]
  
    return pressure_df[colum_names["pressure"]["measure"]].iloc[point_index+delta_point:point_index]
    
    
    