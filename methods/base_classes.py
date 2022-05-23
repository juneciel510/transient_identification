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

from extract_points import ExtractPoints_inWindow






def polyval_func_wrapper(x, *parameters):
    y = np.polyval(parameters,x)
    return y
def linear_func_wrapper(x, a,b):
    y = a*x+b
    return y
def log_func_wrapper(x,a,b,c,d):
    y=a*np.log(x)+d+b*x+c*x*x
    return y


class CurveParametersCalc(ExtractPoints_inWindow):
    """
    A class to fit curves and calculate parameters for the fitted curves.
    Args:
        pattern_names: buildup pattern or drawdown pattern
        breakpoints_forLearn: the points for fitting curves
    
    """
    def __init__(self,mode:str="forPatternRecognition")->None:
        ExtractPoints_inWindow.__init__(self,
                                     coordinate_names=["pressure_time","pressure_measure"],
                                     mode=mode)
        self.pattern_names=["buildUp","drawDown"]
        self.breakpoints_forLearn=defaultdict(list)
    
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
 
        return np.round_(parameters,5)
    
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
        
        # plt.ylim(-400, 200)
        plt.scatter(x=x,y=y,label="data point",color="green")
        plt.plot(x, y_fit, label="fitted curve",color=plot_color,linestyle='-')
        plt.title("")
    
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
            polynomial_order: the order for polynomial fitting
            show_plot: set to True if you want to plot all these fitted curves.
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
            time_halfWindow_forLearn: time half window for these ground truth
            min_pointsNumber: minimum number of points if the time window  contains less points than 'min_pointsNumber'  
            fitting_type: fitting type for the curve fitting
        Returns:
            None
        """
        print("==================")
        print(f"start to learn..., using {fitting_type} curve fitting")
        ground_truth_buildUp,ground_truth_drawDown=self.detect_breakpoint_type(pressure_measure,
                                                                           pressure_time,
                                                                           ground_truth, 
                                                                           time_halfWindow_forLearn,
                                                                           None,
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
            data_inWindow=self.extract_points_inWindow(pressure_measure,
                                                           pressure_time,
                                                           points,
                                                           time_halfWindow_forLearn,
                                                           None,
                                                           min_pointsNumber)
            parameters_allCurves_groundTruth[buildUp_or_drawDown]=self.calculate_Parameters_allCurve(data_inWindow,fitting_type=fitting_type)
        return parameters_allCurves_groundTruth
 
    def detect_breakpoint_type(self,
                               pressure_measure:List[float],
                               pressure_time:List[float],
                               points:List[int],
                                time_halfWindow:float=None,
                                point_halfWindow:int=None,
                                min_pointsNumber:int=8
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
            time_halfWindow: set to be None if use point window
            point_halfWindow: set to be None if use time window
            min_pointsNumber: minimum number of points if the time window  contains less points than 'min_pointsNumber'  
        Returns:
            two lists: 
            one for "buildUp" point index, 
            one for "drawDown" point index
        """
        print("=======================")
        print("detect breakpoint type.....")
        data_inWindow=self.extract_points_inWindow(pressure_measure,pressure_time,points,time_halfWindow,point_halfWindow,min_pointsNumber)
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
            
            