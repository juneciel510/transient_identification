#class CurveFitCalc:
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
from scipy.signal import savgol_filter
import math 
import bisect

from plot import plot_histogram


def polyval_func_wrapper(x, *parameters):
    y = np.polyval(parameters,x)
    return y
def linear_func_wrapper(x, a,b):
    y = a*x+b
    return y
def log_func_wrapper(x,a,b,c,d):
    y=a*np.log(x)+d+b*x+c*x*x
    return y


class CurveParametersCalc:
    """
    A class to learn pattern from ground_truth of PDG pressure data.
    And predict the buildUp and drawDown points for a given dataset.
    Args:
    Returns:
    """
    def __init__(self):
        # self.buildUp_or_drawDown=""
        self.pattern_names=["buildUp","drawDown"]
        self.breakpoints_forLearn=defaultdict(list)
        
    
    def extract_points_inTimeWindow(self,
                                    pressure_measure:List[float],
                                    pressure_time:List[float],
                                    points:List[int],
                                    time_halfWindow:float,
                                    min_pointsNumber:int=8)->pd.DataFrame: 
        """
        extract pressure measure & time data for 'points' 
        in 'timewindow' 
        if the number of points in the half timewindow is less than 'min_pointsNumber'
        then we extract 'min_pointsNumber' points
        
        Args:
            pressure_measure: pressure measure for the whole dataset
            pressure_time: pressure time for the whole dataset
            points: a list contains index of points 
            time_halfWindow: half timewindow
            
        Returns:
            a dataframe containing five columns, each row for a point
            -------------
            columns=['point_index',
                    'pressure_time_left', 
                    'pressure_measure_left',
                    'pressure_time_right', 
                    'pressure_measure_right']
            -------------
        """
        # print("-------start to extract points data inTimeWindow")
        data_inWindow=pd.DataFrame(columns=['point_index',
                                'pressure_time_left', 
                                'pressure_measure_left',
                                'pressure_time_right', 
                                'pressure_measure_right'])
        
        for point_index in points:
            if point_index>(len(pressure_measure)-min_pointsNumber) or point_index<min_pointsNumber:
                continue
            #convert timewindow to point window 
            time_leftStart=pressure_time[point_index]-time_halfWindow
            time_rightEnd=pressure_time[point_index]+time_halfWindow
            if time_leftStart>=0 and time_rightEnd<=pressure_time[-1]:
                halfWinow_left=point_index-bisect.bisect_left(pressure_time, time_leftStart) 
                
                if halfWinow_left<min_pointsNumber:
                    halfWinow_left=min_pointsNumber
                
                halfWinow_right=bisect.bisect_left(pressure_time, time_rightEnd)-1-point_index
                if halfWinow_right<min_pointsNumber:
                    halfWinow_right=min_pointsNumber
            
                data=self.extract_singlePoint_inPointWindow(pressure_measure,pressure_time,point_index,halfWinow_left,halfWinow_right)
            
                data_inWindow=data_inWindow.append(data,ignore_index=True)
        return data_inWindow
    
    def extract_points_inPointWindow(self,
                                    pressure_measure:List[float],
                                    pressure_time:List[float],
                                    points:List[int],
                                    point_halfWindow:int)->pd.DataFrame: 
        """
        extract pressure measure & time data for 'points' 
        in 'pointwindow' 
        e.g. 8 points at the left & 8 points at the right
        
        Args:
            pressure_measure: pressure measure for the whole dataset
            pressure_time: pressure time for the whole dataset
            points: a list contains index of points 
            time_halfWindow: half timewindow
            
        Returns:
            a dataframe containing five columns, each row for a point
            -------------
            columns=['point_index',
                    'pressure_time_left', 
                    'pressure_measure_left',
                    'pressure_time_right', 
                    'pressure_measure_right']
            -------------
        """
        # print("-------start to extract points data inTimeWindow")
        data_inWindow=pd.DataFrame(columns=['point_index',
                                'pressure_time_left', 
                                'pressure_measure_left',
                                'pressure_time_right', 
                                'pressure_measure_right'])
        
        points_valid=[point for point in points if point-point_halfWindow>=0 and point+point_halfWindow<len(pressure_measure) ]
        for point_index in points_valid: 
                data=self.extract_singlePoint_inPointWindow(pressure_measure,pressure_time,point_index,point_halfWindow,point_halfWindow)
            
                data_inWindow=data_inWindow.append(data,ignore_index=True)
        return data_inWindow
                
    def extract_singlePoint_inPointWindow(self,
                                          pressure_measure:List[float],
                                          pressure_time:List[float],
                                          point_index:int,
                                          halfWinow_left:int,
                                          halfWinow_right:int
                                          )->Dict[str,List[float]]: 

        """
        extract pressure measure & time data for a single point
        in point window  [point_index-point_halfWindow,point_index+point_halfWindow]
        
        Args:
            pressure_measure: pressure measure for the whole dataset
            pressure_time: pressure time for the whole dataset
            point_index: index of points 
            halfWinow_left: the number of points to be extracted on the left side of the point_index
            halfWinow_right: the number of points to be extracted on the left side of the point_index
            
        Returns:
            a dictionary, the keys() are as follows:
            -------------
            ['point_index',
            'pressure_time_left', 
            'pressure_measure_left',
            'pressure_time_right', 
            'pressure_measure_right']
            -------------
        """
        data={"point_index":int(point_index)}
        
        #left side
        sub_measure=pressure_measure[point_index+1-halfWinow_left:point_index+1]
        sub_time=pressure_time[point_index+1-halfWinow_left:point_index+1]
        curve_pressure=[round(measure-sub_measure[-1],6) for measure in sub_measure]
        curve_time=[round(time-sub_time[-1],6) for time in sub_time]
        data.update({"pressure_time_left":curve_time,
                        "pressure_measure_left":curve_pressure})
        
        #right side
        sub_measure=pressure_measure[point_index:point_index+halfWinow_right]
        sub_time=pressure_time[point_index:point_index+halfWinow_right]
        curve_pressure=[round(measure-sub_measure[0],6) for measure in sub_measure]
        curve_time=[round(time-sub_time[0],6) for time in sub_time]
        data.update({"pressure_time_right":curve_time,
                    "pressure_measure_right":curve_pressure})
        return data
                             
    def fit_curve(self,
                  xdata:List[float],
                  ydata:List[float],   
                  fitting_type:str,
                  polynomial_order:int,
                  plot_color:str,
                  show_plot:bool=True)->List[float]:
        """
        curve fit and plot xdata, ydata and their fitted funtion
        Args: 
            xdata: x coordinate data
            ydata: y coordinate data
            fitting_type: the type of fitting function
            plot_color: plot color
            polynomial_order: if the fitting type is "polynomial", you can specify the order by setting this argument
        Returns:
            the parameters of the fitted curve
        """
       
        x = np.asarray(xdata)
        y = np.asarray(ydata)
        
        fitting_func=self.choose_fittingFunction(fitting_type)
        
        if fitting_type=="polynomial":
            parameters=np.polyfit(x,y,polynomial_order)
        if fitting_type=="linear" or fitting_type=="log":
            parameters, covariance = curve_fit(fitting_func, x, y)
        if show_plot:
            self.plot_fittingCurve(x, y,fitting_type, plot_color, *parameters)
 
        return parameters
    
    def plot_fittingCurve(self,
                          x:np.array, 
                          y:np.array,
                          fitting_type:str, 
                          plot_color:str, 
                          *parameters)->None:
        """
        plot x, y and their fitted funtion
        Args: 
            x: x coordinate data
            y: y coordinate data
            fitting_type: the type of fitting function
            plot_color: plot color
            parameters: the parameters of fitting function
        Returns:
            None
        """
        fitting_func=self.choose_fittingFunction(fitting_type)
        y_fit=fitting_func(np.asarray(x), *parameters)
        # if fitting_type=="linear":
        #     plt.figure(figsize = (10, 10))
        #     plt.axis([-1, 1, -10, 10])
        # plt.plot(x, y,color=plot_color,  marker='o')
        
        #when detect_breakpoint_type invoke this funtion,
        # set the title to be the breakpoint index
        # if self.buildUp_or_drawDown!="":
        #     if fitting_type=="linear":
        #         plot_title=f"{self.ground_truth[i]}"      
        #     if fitting_type=="polynomial" or fitting_type=="log":
        #         plot_title=f"{self.buildUp_or_drawDown}---{self.ground_truth[i]}"   
        # self.plot_fittingCurve(x, y,fitting_type, plot_color, plot_title, *parameters)
        
        # print("parameters",parameters)
        # plt.scatter(x=x,y=y,color=plot_color)
        plt.scatter(x=x,y=y,label="data point",color="green")
        plt.plot(x, y_fit, label="fitted curve",color=plot_color,linestyle='-')
        plt.title("")
        # if self.buildUp_or_drawDown!="":
        #     plt.scatter(x=x,y=y,color=plot_color)
        #     plt.plot(x, y_fit, color=plot_color,linestyle='-')
        #     plt.title(self.buildUp_or_drawDown)
        # if fitting_type=="log" or fitting_type=="polynomial":
        #     plt.show()
    
    def calculate_Parameters_allCurve(self,
                                      data_inWindow:pd.DataFrame,
                                      fitting_type:str,
                                      polynomial_order:int =3,
                                      show_plot:bool=True)->pd.DataFrame:
        """
        for data in window (all given points), calculate the parameters of all fitted curves
        Args: 
            data_inWindow: data in window
            fitting_type: the type of fitting function
        Returns:
            dataframe containing 3 columns
            -------
            ["point_index", 
            "left_curves_parameters",
            "right_curves_parameters"]
            -------
        """
        print(f"-------calculate_Parameters_allCurve using '{fitting_type}' fitting")
        curveLeft_parameters=[]
        curveRight_parameters=[]
        parameters_allCurves=pd.DataFrame()
        
        plt.figure(figsize = (20, 10))

        for i in range(len(data_inWindow)):
            #left side
            xdata=data_inWindow["pressure_time_left"][i]
            ydata=data_inWindow["pressure_measure_left"][i]
            plot_color="yellow"  
            curveLeft_parameters.append(self.fit_curve(xdata,ydata,fitting_type,polynomial_order,plot_color,show_plot))

            
            #right side
            xdata=data_inWindow["pressure_time_right"][i]
            ydata=data_inWindow["pressure_measure_right"][i]           
            plot_color="blue" 
            curveRight_parameters.append(self.fit_curve(xdata,ydata,fitting_type,polynomial_order,plot_color,show_plot))
            
        parameters_allCurves=pd.DataFrame({"point_index":data_inWindow["point_index"].values,
                                           "left_curves_parameters":curveLeft_parameters,
                                           "right_curves_parameters":curveRight_parameters})
        return parameters_allCurves
        
    def calculate_Parameters_allCurve_groundTruth(self,
                                                  pressure_measure:List[float],
                                                  pressure_time:List[float],
                                                  ground_truth:List[int],
                                                  time_halfWindow_forLearn:float,
                                                  min_pointsNumber:int,
                                                  fitting_type:str
                                                  )->Dict[str,pd.DataFrame]:
        """ 
        calculate parameters of fitting curves for ground truth
        Args:
            pressure_measure: pressure measure for the whole dataset
            pressure_time: pressure time for the whole dataset
            ground_truth: a list contains index of break points 
            fitting_type: the type for the fitting function            
        Returns:
            None
        """
        print("==================")
        print(f"start to learn..., using {fitting_type} curve fitting")
        ground_truth_buildUp,ground_truth_drawDown=self.detect_breakpoint_type(pressure_measure,
                                                                           pressure_time,
                                                                           ground_truth, 
                                                                           time_halfWindow_forLearn,
                                                                           min_pointsNumber)
        print(f"------In the input ground truth: {len(ground_truth_buildUp)} points are detected as buildup, {len(ground_truth_drawDown)} points are detected as drawDown")
        self.breakpoints_forLearn={"buildUp":ground_truth_buildUp,
                                   "drawDown":ground_truth_drawDown}
        # self.breakpoints_forLearn_multipleLearn.append(self.breakpoints_forLearn)
        # print("----ground_truth_buildUp,ground_truth_drawDown",ground_truth_buildUp,ground_truth_drawDown)

        parameters_allCurves_groundTruth={"buildUp":pd.DataFrame(columns=["point_index",
                                                                               "left_curves_parameters",
                                                                               "right_curves_parameters"]), 
                                               "drawDown":pd.DataFrame(columns=["point_index",
                                                                                "left_curves_parameters",
                                                                                "right_curves_parameters"])}

        allPoints=[ground_truth_buildUp,ground_truth_drawDown]

        for points,buildUp_or_drawDown in zip(allPoints, self.pattern_names):
            # self.buildUp_or_drawDown=buildUp_or_drawDown
            data_inWindow=self.extract_points_inTimeWindow(pressure_measure,pressure_time,points,time_halfWindow_forLearn,min_pointsNumber)
            parameters_allCurves_groundTruth[buildUp_or_drawDown]=self.calculate_Parameters_allCurve(data_inWindow,fitting_type=fitting_type)
        return parameters_allCurves_groundTruth
 
    def detect_breakpoint_type(self,
                               pressure_measure:List[float],
                               pressure_time:List[float],
                               points:List[int],
                                time_halfWindow:float,
                                min_pointsNumber:int
                              )->(List[int],List[int]):
        """
        detect the break points are buildup or drawdown
        do the linear fitting,
        if slope_left>slope_right: drawdown
        else: buildup
        Args: 
            pressure_measure: pressure measure for the whole dataset
            pressure_time: pressure time for the whole dataset
            points: list of point indices for detection
        Returns:
            two lists: 
            one for "buildUp" point index, 
            one for "drawDown" point index
        """
        print("=======================")
        print("detect breakpoint type.....")
        # self.buildUp_or_drawDown=""
        data_inWindow=self.extract_points_inTimeWindow(pressure_measure,pressure_time,points,time_halfWindow,min_pointsNumber)
        parameters_allCurves=self.calculate_Parameters_allCurve(data_inWindow,fitting_type="linear",show_plot=False)  

        breakpoint_buildUp=[]
        breakpoint_drawDown=[]
        for index,parameter in parameters_allCurves.iterrows():
            #compare slope and convert index in the parameters_allCurves to in pressure_df
            if parameter["left_curves_parameters"][0]<parameter["right_curves_parameters"][0]:
                if parameter["left_curves_parameters"][0]<0 and parameter["right_curves_parameters"][0]<0:
                    pass
                    # print(f"===============\n"
                    #       f"point '{points[index]}' is going to be classified as buildup,\n"
                    #       f"however the ground truth should be drawdown,\n"
                    #       f"there must be a bump in window thus should not be included into breakpoints for learning\n"
                    #       f"===============")
                else:
                    breakpoint_buildUp.append(points[index])
                
                
            if parameter["left_curves_parameters"][0]>parameter["right_curves_parameters"][0]:
                if parameter["left_curves_parameters"][0]>0 and parameter["right_curves_parameters"][0]>0:
                    pass
                    # print(f"===============\n"
                    #       f"point '{points[index]}' is going to be classified as drawdown,\n"
                    #       f"however the ground truth should be biuldup,\n"
                    #       f"there must be a cavity in window thus should not be included into breakpoints for learning\n"
                    #       f"===============")
                else:
                    breakpoint_drawDown.append(points[index])
                    
        return breakpoint_buildUp,breakpoint_drawDown    
    
    def choose_fittingFunction(self,
                               fitting_type:str
                               )->Callable[..., List[float]]:
        """
        choose fitting function
        """
        if fitting_type == "linear":
            fitting_func=linear_func_wrapper
        elif fitting_type == "polynomial":
            fitting_func=polyval_func_wrapper
        elif fitting_type == "log":
            fitting_func=log_func_wrapper
        else:
            print('fitting type must be "linear", "polynomial" or "log"')
        return fitting_func
    
    # def produce_parameters_givenPoints(self,
    #                                    pressure_measure:List[float],
    #                                     pressure_time:List[float],
    #                                     points:List[int],
    #                                     time_halfWindow:float,
    #                                     fitting_type:str)->pd.DataFrame:
    #     """
    #     extract the data of the given points in the timewindow
    #     and
    #     calculate the parameter for all curves fitting these points
    #     Args:
    #         pressure_measure: pressure measure for the whole dataset
    #         pressure_time: pressure time for the whole dataset
    #         points: a list contains index of points 
    #         time_halfWindow: half timewindow
    #         fitting_type: the type for the fitting function
                
    #     Returns:
    #         a dataframe containing 3 columns, each row for a point
    #         -------------
    #         ["point_index", 
    #         "left_curves_parameters",
    #         "right_curves_parameters"]
    #         -------------
    #     """
    #     if self.buildUp_or_drawDown!="":
    #         print("=============")
    #         print(f"start to learn '{self.buildUp_or_drawDown}' pattern...")
    #     data_inWindow=self.extract_points_inTimeWindow(pressure_measure,pressure_time,points,time_halfWindow)
    #     parameters_allCurves=self.calculate_Parameters_allCurve(data_inWindow,fitting_type=fitting_type)
    #     return parameters_allCurves
 
        
    
class SaveNLoad:
    def load_pattern(self,filePath_loadPattern:str):
        """
        load the saved pattern
        """
        if filePath_loadPattern==None:
            parameters_twoPatterns={"buildUp":{},
                                    "drawDown":{}}
            return parameters_twoPatterns
        
        with open(filePath_loadPattern) as infile:
            parameters_twoPatterns = json.load(infile)
        print(f"The pattern parameters {parameters_twoPatterns} are loaded")
        return parameters_twoPatterns 
    
    def sav_pattern(self,parameters_twoPatterns:Dict[str,Dict],
                    filePath_savedPattern:str
                    )->None:
        with open(filePath_savedPattern, 'w') as fp:
            json.dump(parameters_twoPatterns, fp)