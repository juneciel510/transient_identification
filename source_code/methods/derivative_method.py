from typing import Callable, Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import math
import statistics

class DerivativeMethod:
    def __init__(self,
                 pressure_df:pd.DataFrame,
                colum_names:Dict[str,Dict[str,str]]
                ={"pressure":{"time":"Elapsed time",
                            "measure":"Data",
                            "first_order_derivative":"first_order_derivative",
                            "second_order_derivative":"second_order_derivative"},
                "rate":{"time":"Elapsed time",
                        "measure":"Liquid rate"}}
                ):
        self.pressure_df=pressure_df
        self.pressure_measure=pressure_df[colum_names["pressure"]["measure"]]
        self.pressure_time=pressure_df[colum_names["pressure"]["time"]]
        self.first_order_derivative=pressure_df[colum_names["pressure"]["first_order_derivative"]]
        self.second_order_derivative=pressure_df[colum_names["pressure"]["second_order_derivative"]]
        self.std_1=statistics.stdev(self.first_order_derivative)
        
    def detect_max_FOD(self,
                   time_step:float=2
                    )->List[int]:
        """
        get the indices of the points with maximum first_order_derivative in every time step
        """
        
        self.pressure_df["abs(first_order_derivative)"]=self.first_order_derivative.abs()
        max_time=list(self.pressure_time)[-1]
        group_number=math.ceil(max_time/time_step)
        #devide the pressure_df into multiple sub df according to time step
        sub_pressure_dfs=[self.pressure_df.loc[(self.pressure_time >= i*time_step) & (self.pressure_time  <= (i+1)*time_step)] for i in range(group_number)]
        #get the index of max absolute value of first order derivative
        index_max_FOD=[sub_pressure_df["abs(first_order_derivative)"].idxmax() for sub_pressure_df in sub_pressure_dfs if len(sub_pressure_df)>0]
        
        # filtered_points=[point_index for point_index in index_max_FOD if self.first_order_derivative[point_index]>0.02*self.std_1 ]
        return index_max_FOD
        