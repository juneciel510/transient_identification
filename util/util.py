import numpy as np
import pandas as pd
import statistics
import json
import matplotlib.pyplot as plt
from matplotlib import rcParams
from operator import itemgetter
from typing import Callable, Dict, List, Set, Tuple
import sys,os.path
methods_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
+ '/methods/')
sys.path.append(methods_dir)
from tangent_method import TangentMethod


def save_json(data,filePath):
    with open(filePath, 'w') as fp:
        json.dump(data, fp)
        
def load_json(filePath:str):  
    with open(filePath) as infile:
        data = json.load(infile)
    return data

def load_data_from_txt(pressure_file:str, 
                       rate_file:str, 
                       colum_names:Dict[str,List[str]]
                   ={"pressure":["Elapsed time","Data"],
                    "rate":["Elapsed time","Liquid rate"]}
                   )->(pd.DataFrame,pd.DataFrame,pd.DataFrame): 
    pressure_time, pressure_measure=colum_names["pressure"]
    rate_time, rate_measure = colum_names["rate"]
    pressure_df = pd.read_csv(pressure_file, delimiter=" ",skiprows=2, names=[pressure_time, pressure_measure],skipinitialspace = True) 
    rate_df = pd.read_csv(rate_file, delimiter=" ",skiprows=2, names=[rate_time, rate_measure],skipinitialspace = True) 
    pressure_rate = pd.concat([pressure_df, rate_df]).sort_values(by=pressure_time) 
    return pressure_df,rate_df,pressure_rate



def calculate_derivative_forward(x_coordinate:List[float],
                         y_coordinate:List[float]
                         )->List[float]:
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

def calculate_derivative(x_coordinate:List[float],
                         y_coordinate:List[float]
                         )->List[float]:
    """
    calculate backforward derivative, the last point use the forward derivative
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
    for i in range(length):
        if i==0:     
            derivative[i]=(y_coordinate[i+1]-y_coordinate[i])/(x_coordinate[i+1]-x_coordinate[i])
        else:
            derivative[i]=(y_coordinate[i]-y_coordinate[i-1])/(x_coordinate[i]-x_coordinate[i-1])

    return derivative


def convert_timestamp2hour(start_timestamp:pd._libs.tslibs.timestamps.Timestamp,
                           timestamps:List[pd._libs.tslibs.timestamps.Timestamp]
                           )->List[float]:
    hours=[0.0]*len(timestamps)
    for i in range(len(timestamps)-1):
        hours[i+1]=(timestamps[i+1]-start_timestamp).total_seconds()/3600
    return hours

def produce_pressure_4metrics(pressure_df:pd.DataFrame,
                              colum_names:Dict[str,List[str]]
                              )->pd.DataFrame:
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
                              colum_names:Dict[str,Dict[str,str]]
                              )->float:
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

def timeInterval_to_sub_df(start_time:float,
                           end_time:float,
                           pressure_df:pd.DataFrame,
                           rate_df:pd.DataFrame,
                           colum_names:Dict[str,Dict[str,str]]
                            ={"pressure":{"time":"Elapsed time",
                                            "measure":"Data",
                                            "first_order_derivative":"first_order_derivative",
                                            "second_order_derivative":"second_order_derivative"},
                                "rate":{"time":"Elapsed time","measure":"Liquid rate"}}
                            )->pd.DataFrame:
    rate_time=rate_df[colum_names["rate"]["time"]]
    sub_rate_df=rate_df.loc[(rate_time >= start_time) & (rate_time <= end_time)]
    # pd.set_option('display.max_rows', sub_rate_df.shape[0]+1)
    pressure_time=pressure_df[colum_names["pressure"]["time"]]
    sub_pressure_df=pressure_df.loc[(pressure_time >= start_time) & (pressure_time <= end_time)]
    return sub_rate_df,sub_pressure_df




def point_dt_to_pressure(point_index:int,
                        pressure_df, 
                        delta_t:float=1/12,
                        colum_names:Dict={"pressure":{"time":"Date",
                        "measure":"Pressure (psia)",
                        "first_order_derivative":"first_order_derivative",
                        "second_order_derivative":"second_order_derivative"},
                            "rate":{"time":"Time@end",
                                    "measure":"Liquid rate (STB/D)"}}
                        )->List[float]:
    """
    extract pressure measurements 
    between the time interval [time_point_index, time_point_index+delta_t]
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
                                         "measure":"Liquid rate (STB/D)"}}
                               )->List[float]:
    """
    extract pressure measurements between point_index and point_index+delta_point
    """
    if delta_point>0:
        return pressure_df[colum_names["pressure"]["measure"]].iloc[point_index:point_index+delta_point]
  
    return pressure_df[colum_names["pressure"]["measure"]].iloc[point_index+delta_point:point_index]


def print_tuning_parameters(percentile_tuning,fine_tuning):
    txt=f"percentile_tuning======\n"
    for key,value in percentile_tuning.items():
        txt=txt+f"{key}:\n{value}\n"

    txt+="\nfine_tuning===========\n"
    for key,value in fine_tuning.items():
        txt=txt+f"{key}:\n{value}\n"        
    return txt


class SelectRows:
    def __init__(self,df):
        self.df = df
    def select_byColValue(self,col_name1,value):
        """
        Select Rows where Column is Equal to Specific Value
        """
        return self.df.loc[self.df[col_name1] == value]
    def select_byColValueList(self,col_name1,value_list):
        """
        Select Rows where Column Value is in List of Values
        """
        return self.df.loc[self.df[col_name1].isin(value_list)]
    def select_byMultipleColValue(self,col_name1,col_name2,value1,value2):
        """
        Select Rows where Column Value is in List of Values
        """
        return self.df.loc[(self.df[col_name1] == value1) & (self.df[col_name2] < value2)]
    
    
    def select_byIndexValueList(self,value_list):
        """
        Select Rows where index Value is in List of Values
        """
        return self.df.iloc[value_list]
    
    
    


def non_uniform_savgol(x, y, window, polynom):
    """
    Applies a Savitzky-Golay filter to y with non-uniform spacing
    as defined in x

    This is based on https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do

    Parameters
    ----------
    x : array_like
        List of floats representing the x values of the data
    y : array_like
        List of floats representing the y values. Must have same length
        as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size

    Returns
    -------
    np.array of float
        The smoothed y values
    """
    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))     # Matrix
    tA = np.empty((polynom, window))    # Transposed matrix
    t = np.empty(window)                # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed


class analyze_FOD_tangent(TangentMethod):
    def __init__(self,
                 pressure_df,
                 polynomial_order,
                 point_halfWindow,
                colum_names:Dict[str,Dict[str,str]]
                   ={"pressure":{"time":"Elapsed time",
                                 "measure":"Data",
                                 "first_order_derivative":"first_order_derivative",
                                 "second_order_derivative":"second_order_derivative"},
                    "rate":{"time":"Elapsed time","measure":"Liquid rate"}}):
        super().__init__(polynomial_order=polynomial_order,point_halfWindow=point_halfWindow)
        self.pressure_df=pressure_df
        self.colum_names=colum_names
        self.pressure_measure=list(pressure_df[self.colum_names["pressure"]["measure"]])
        self.pressure_time=list(pressure_df[self.colum_names["pressure"]["time"]])
    def get_FOD_tangent(self,points):
        """
        Returns:
        dataframe with column name:
        ["point_index","tangent_left","tangent_right","deltaTangent","first_order_derivative","Elapsed time"]
        """
        tangent_df=self.produce_tangent_inWindow(self.pressure_measure,
                                                 self.pressure_time,
                                                 points,
                                                 polynomial_order=self.polynomial_order,
                                                 point_halfWindow=self.point_halfWindow)
        fod=self.pressure_df.iloc[points,:]["first_order_derivative"]
        time=self.pressure_df.iloc[points,:][self.colum_names["pressure"]["time"]]
        tangent_df["deltaTangent"]=tangent_df["tangent_right"]-tangent_df["tangent_left"]
        tangent_df["first_order_derivative"]=list(fod)
        tangent_df[self.colum_names["pressure"]["time"]]=list(time)
        return tangent_df
    
    def plot_tangent(self,points):
        tangent_plot=self.produce_tangent_inWindow(self.pressure_measure,
                                                 self.pressure_time,
                                                 points,
                                                   data_type="for_plot",
                                                   point_halfWindow=self.point_halfWindow,
                                                 polynomial_order=self.polynomial_order,
                                                 point_halfWindow_tagentPlot=self.point_halfWindow)
        #         display(tangent_plot)
        for point_index in points:
            self.plot_tangent_inPointWindow(tangent_plot,point_index)