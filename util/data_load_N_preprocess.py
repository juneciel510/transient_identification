import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from matplotlib import rcParams
from operator import itemgetter
from typing import Callable, Dict, List, Set, Tuple
from scipy.signal import savgol_filter

class LoadNPreprocessData:
    """
    Read pressure and flow rate data from txt or xlsx file and produce pressure dataframe and flow rate dataframe
    Preprocessing includes derivative calculation and denoising.
    Args:
        pressure_filePath: file path to load pressure data
        rate_filePath: file path to load flow rate data
        colum_names: the column names for the produced pressure & rate dataframe
        skip_rows: the number of rows to skip when reading from txt file 
        sheet_name: sheet names for xlxs file if read from xlsx file
        use_SG_smoothing: set False if denoising is not needed
        window_length: window length for SG smoothing
        polyorder: polynomial order for SG smoothing
    """
    def __init__(self,
                 pressure_filePath:str, 
                 rate_filePath:str, 
                 colum_names:Dict[str,Dict[str,str]]={"pressure":{"time":"Elapsed time(hr)",
                                                        "measure":"Pressure(psia)",
                                                        "first_order_derivative":"first_order_derivative",
                                                        "second_order_derivative":"second_order_derivative"},
                                            "rate":{"time":"Elapsed time(hr)",
                                                    "measure":"Liquid rate(STB/D)"}}, 
                 skip_rows:int=2,
                 sheet_name:Dict[str,str]={"pressure":"Pressure",
                                           "rate":"Rate"},
                 use_SG_smoothing:bool=True,
                 window_length:int=199,
                 polyorder:int=3)->None:
        self.pressure_filePath=pressure_filePath
        self.rate_filePath=rate_filePath
        self.colum_names=colum_names
        self.skip_rows=skip_rows
        self.sheet_name=sheet_name
        
        self.use_SG_smoothing=use_SG_smoothing
        self.window_length=window_length
        self.polyorder=polyorder
        
        self.pressure_df=pd.DataFrame()
        self.rate_df=pd.DataFrame()
        self.pressureNrate_df=pd.DataFrame()
        
        print("---load data from 'txt' or 'xlsx' files...")
        self.load_data()
        
        if self.use_SG_smoothing:
            print("---denoising data using S-G smoothing...")
            self.SG_smoothing()
        self.produce_pressure_4metrics()
        print("---The first & second order derivative has been calculated and appended to pressure dataframe")
        self.concatenate_pressureNRate()
    
    def load_data(self)->None:
        '''
        load data from 'txt' or 'xlsx' files
        convert time to hours in float if it is "timestamp" format
        '''
        #check if file is txt or xlsx
        if self.pressure_filePath.split(".")[-1]=="txt" and self.rate_filePath.split(".")[-1]=="txt":
            self.pressure_df = pd.read_csv(self.pressure_filePath, 
                                  delimiter=" ",
                                  skiprows=self.skip_rows, 
                                  names=[self.colum_names["pressure"]["time"], 
                                         self.colum_names["pressure"]["measure"]],
                                  skipinitialspace = True) 
            self.rate_df = pd.read_csv(self.rate_filePath, 
                                delimiter=" ",
                                skiprows=self.skip_rows, 
                                names=[self.colum_names["rate"]["time"], 
                                        self.colum_names["rate"]["measure"]], 
                                skipinitialspace = True) 
        elif self.pressure_filePath.split(".")[-1]=="xlsx" and self.rate_filePath.split(".")[-1]=="xlsx":
            self.pressure_df = pd.DataFrame(pd.read_excel(self.pressure_filePath, sheet_name=self.sheet_name["pressure"]))
            self.rate_df = pd.DataFrame(pd.read_excel(self.rate_filePath, sheet_name=self.sheet_name["rate"]))
        else:
            print("only can load data from 'txt' or 'xlsx' files")
        
        #if time is timestamp, convert to float representing hours
        pressure_time_type= type(self.pressure_df[self.colum_names["pressure"]["time"]][0])  
        rate_time_type= type(self.rate_df[self.colum_names["rate"]["time"]][0])  
        
        if  (pressure_time_type is float) and  (rate_time_type is float):
            return None
        
        if (pressure_time_type is pd.Timestamp) and (rate_time_type is pd.Timestamp):
            timestamps=self.pressure_df[self.colum_names["pressure"]["time"]]
            start_timestamp=timestamps[0]
            self.pressure_df[self.colum_names["pressure"]["time"]]=self.convert_timestamp2hour(timestamps,start_timestamp)
            
            timestamps=self.rate_df[self.colum_names["rate"]["time"]]
            self.rate_df[self.colum_names["rate"]["time"]]=self.convert_timestamp2hour(timestamps,start_timestamp)
        else:
            print("check the time type")
        return None

    
    def concatenate_pressureNRate(self)->None:
        self.pressureNrate_df = pd.concat([self.pressure_df, self.rate_df]).sort_values(by=self.colum_names["pressure"]["time"]) 
        return None
    
    def convert_timestamp2hour(self,timestamps,start_timestamp)->List[float]:
        hours=[0.0]*len(timestamps)
        for i in range(len(timestamps)-1):
            hours[i+1]=(timestamps[i+1]-start_timestamp).total_seconds()/3600        
        return hours
    
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
    
    def produce_pressure_4metrics(self)->None:
        """
        Calculate first & second order derivative and add them to pressure dataframe
        """
        x_coordinate=self.pressure_df[self.colum_names["pressure"]["time"]]
        first_order_derivative=self.calculate_derivative(x_coordinate,self.pressure_df[self.colum_names["pressure"]["measure"]])
        second_order_derivative=self.calculate_derivative(x_coordinate,first_order_derivative)

        #add first and second derivative to pressure_df dataframe
        self.pressure_df["first_order_derivative"]=first_order_derivative
        self.pressure_df["second_order_derivative"]=second_order_derivative
        # pd.set_option('display.max_rows', pressure_df.shape[0]+1)
        return None
        
    def SG_smoothing(self)->None:
        """
        Denoise pressure measurements
        """
        denoised_pressure_measures=savgol_filter(self.pressure_df[self.colum_names["pressure"]["measure"]],
                                             self.window_length,
                                             self.polyorder)
        self.pressure_df[self.colum_names["pressure"]["measure"]]=denoised_pressure_measures
        return None