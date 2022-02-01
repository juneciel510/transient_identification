import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from matplotlib import rcParams
from operator import itemgetter
from typing import Callable, Dict, List, Set, Tuple

class LoadNPreprocessData:
    def __init__(self,
                 pressure_filePath:str, 
                 rate_filePath:str, 
                 colum_names:Dict[str,Dict[str,str]], 
                 skip_rows:int,
                 noise_threshold:float):
        self.pressure_filePath=pressure_filePath
        self.rate_filePath=rate_filePath
        self.colum_names=colum_names
        self.skip_rows=skip_rows
        self.pressure_df=pd.DataFrame()
        self.rate_df=pd.DataFrame()
        self.pressureNrate_df=pd.DataFrame()
        
    def load_data_from_txt(self): 
    
        # pressure_time, pressure_measure=colum_names["pressure"]
        # rate_time, rate_measure = colum_names["rate"]
        self.pressure_df = pd.read_csv(self.pressure_filePath, 
                                  delimiter=" ",
                                  skiprows=self.skip_rows, 
                                  names=[self.colum_names["pressure"]["time"], 
                                         self.colum_names["pressure"]["measure"]],
                                  skipinitialspace = True) 
        self.rate_df = pd.read_csv(self.rate_filePath, 
                              delimiter=" ",
                              skiprows=self.skip_rows, 
                              names=[self.colum_names["pressure"]["time"], 
                                     self.colum_names["pressure"]["measure"]], 
                              skipinitialspace = True) 
        self.pressureNrate_df = pd.concat([self.pressure_df, self.rate_df]).sort_values(by=self.colum_names["pressure"]["time"]) 
        return None
    
    def calculate_derivative(self,x_coordinate:List[float],y_coordinate:List[float])->List[float]:
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
        
    