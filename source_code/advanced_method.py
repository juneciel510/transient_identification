from typing import Callable, Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import math

def detect_max_FOD(pressure_df:pd.DataFrame,
                   time_step:float=2,
                   colum_names:Dict[str,Dict[str,str]]
                    ={"pressure":{"time":"Elapsed time",
                                    "measure":"Data",
                                    "first_order_derivative":"first_order_derivative",
                                    "second_order_derivative":"second_order_derivative"},
                        "rate":{"time":"Elapsed time","measure":"Liquid rate"}}
                    )->List[int]:
    pressure_df["abs(first_order_derivative)"]=pressure_df[colum_names["pressure"]["first_order_derivative"]].abs()
    max_time=list(pressure_df[colum_names["pressure"]["time"]])[-1]
    group_number=math.ceil(max_time/time_step)
    pressure_time=pressure_df[colum_names["pressure"]["time"]]
    #devide the pressure_df into multiple sub df according to time step
    sub_pressure_dfs=[pressure_df.loc[(pressure_time >= i*time_step) & (pressure_time  <= (i+1)*time_step)] for i in range(group_number)]
    #get the index of max absolute value of first order derivative
    index_max_FOD=[sub_pressure_df["abs(first_order_derivative)"].idxmax() for sub_pressure_df in sub_pressure_dfs if len(sub_pressure_df)>0]
    return index_max_FOD