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


def test_func1(x, a,b,c,d):
    y = a+b*x-c*np.exp(-d*x)
    return y
def test_func2(x, a,b,c,d):
    y = a+b*x-c*np.exp(d*x)
    return y
def test_func3(x, a,b,c,d):
    y = a+c*np.exp(d*x)
    return y

def polyval_func_wrapper(x, *parameters):
    y = np.polyval(parameters,x)
    return y
def linear_func_wrapper(x, a,b):
    y = a*x+b
    return y
def log_func_wrapper(x,a,b,c,d):
    y=a*np.log(x)+d+b*x+c*x*x
    return y


class PatternRecognition:
    """
    A class to learn pattern from ground_truth of PDG pressure data.
    And predict the buildUp and drawDown points for a given dataset.
    Args:
    Returns:
    """
    def __init__(self, 
                 point_halfWindow:int=8,
                 time_halfWindow_forPredict:float=0.5,
                 time_halfWindow_forLearn:float=1,
                percentile_tuning={
                                    "buildUp":{"left":[90,10],
                                              "right":[90,10]},
                                   "drawDown":{"left":[90,10],
                                              "right":[90,10]}
                                   },
                fine_tuning={"buildUp":{"left_top":1,
                                    "left_bottom":1,
                                    "right_top":1,
                                    "right_bottom":1},
                            "drawDown":{"left_top":1,
                                        "left_bottom":1,
                                        "right_top":1,
                                        "right_bottom":1}},

                 filePath_learnedPattern="../data_output/Learned_Pattern.jason",
                ):
        
        #to store the points for learn, classified as buildup & drawdown
        self.breakpoints_forLearn=defaultdict(list)
        # self.breakpoints_forLearn_multipleLearn=[]
        #to store the points predicted 
        # self.detectedpoints=defaultdict(list)
        self.point_halfWindow=point_halfWindow
        #time window for learn
        self.time_halfWindow_forLearn=time_halfWindow_forLearn
        #time window for predict
        self.time_halfWindow_forPredict=time_halfWindow_forPredict
        self.percentile_tuning=percentile_tuning
        self.fine_tuning=fine_tuning
        self.filePath_learnedPattern=filePath_learnedPattern

        
        self.buildUp_or_drawDown=""
        self.pattern_names=["buildUp","drawDown"]
        self.border_names=["left_top","left_bottom","right_top","right_bottom"]

        self.data_forPredict=pd.DataFrame(columns=["point_index",
                                                   "pressure_time_left",
                                                    "pressure_measure_left",
                                                    "pressure_time_right",
                                                    "pressure_measure_right",
                                                    "buildUp",
                                                    "drawDown"])

        
        self.parameters_allCurves_groundTruth={"buildUp":pd.DataFrame(columns=["point_index",
                                                                               "left_curves_parameters",
                                                                               "right_curves_parameters"]), 
                                               "drawDown":pd.DataFrame(columns=["point_index",
                                                                                "left_curves_parameters",
                                                                                "right_curves_parameters"])}

        #to store patterns for buildup or drawdown
        self.parameters_twoPatterns={"buildUp":{"left_top":[],
                                               "left_bottom":[],
                                               "right_top":[],
                                               "right_bottom":[]},
                                    "drawDown":{"left_top":[],
                                               "left_bottom":[],
                                               "right_top":[],
                                               "right_bottom":[]}}
        
        #to store tangents for buildup or drawdown
        self.tangents_twoPatterns={"buildUp":{"left_top":None,
                                               "left_bottom":None,
                                               "right_top":None,
                                               "right_bottom":None},
                                    "drawDown":{"left_top":None,
                                               "left_bottom":None,
                                               "right_top":None,
                                               "right_bottom":None}}
        
        self.tangent_forPredict=pd.DataFrame(columns=['point_index',
                                                'tangent_left', 
                                                'tangent_right'])
            
        self.for_tuning={"buildUp":{"left_top":[],
                                    "left_bottom":[],
                                    "right_top":[],
                                    "right_bottom":[]},
                        "drawDown":{"left_top":[],
                                    "left_bottom":[],
                                    "right_top":[],
                                    "right_bottom":[]}}
    
    
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
    

    def extract_points_inTimeWindow(self,
                                    pressure_measure:List[float],
                                    pressure_time:List[float],
                                    points:List[int],
                                    time_halfWindow:float)->pd.DataFrame: 
        """
        extract pressure measure & time data for 'points' 
        in 'timewindow' 
        if the number of points in the half timewindow is less than 'self.point_halfWindow'
        then we extract 'self.point_halfWindow' points
        
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
            if point_index>(len(pressure_measure)-self.point_halfWindow) or point_index<self.point_halfWindow:
                continue
            #convert timewindow to point window 
            time_leftStart=pressure_time[point_index]-time_halfWindow
            time_rightEnd=pressure_time[point_index]+time_halfWindow
            if time_leftStart>=0 and time_rightEnd<=pressure_time[-1]:
                halfWinow_left=point_index-bisect.bisect_left(pressure_time, time_leftStart) 
                
                if halfWinow_left<self.point_halfWindow:
                    halfWinow_left=self.point_halfWindow
                
                halfWinow_right=bisect.bisect_left(pressure_time, time_rightEnd)-1-point_index
                if halfWinow_right<self.point_halfWindow:
                    halfWinow_right=self.point_halfWindow
            
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
        in 'timewindow' 
        if the number of points in the half timewindow is less than 'self.point_halfWindow'
        then we extract 'self.point_halfWindow' points
        
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
                  plot_color:str,
                  polynomial_order:int)->List[float]:
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
        if self.buildUp_or_drawDown!="":
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
        
        print("parameters",parameters)
        # plt.scatter(x=x,y=y,color=plot_color)
        plt.scatter(x=x,y=y,label="data point",color="green")
        plt.plot(x, y_fit, label="fitted curve",color=plot_color,linestyle='-')
        plt.title(self.buildUp_or_drawDown)
        # if self.buildUp_or_drawDown!="":
        #     plt.scatter(x=x,y=y,color=plot_color)
        #     plt.plot(x, y_fit, color=plot_color,linestyle='-')
        #     plt.title(self.buildUp_or_drawDown)
        # if fitting_type=="log" or fitting_type=="polynomial":
        #     plt.show()
    
    def calculate_Parameters_allCurve(self,data_inWindow:pd.DataFrame,fitting_type:str,polynomial_order:int =3)->pd.DataFrame:
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
            curveLeft_parameters.append(self.fit_curve(xdata,ydata,fitting_type,plot_color,polynomial_order))

            
            #right side
            xdata=data_inWindow["pressure_time_right"][i]
            ydata=data_inWindow["pressure_measure_right"][i]           
            plot_color="blue" 
            curveRight_parameters.append(self.fit_curve(xdata,ydata,fitting_type,plot_color,polynomial_order))
            
        parameters_allCurves=pd.DataFrame({"point_index":data_inWindow["point_index"].values,
                                           "left_curves_parameters":curveLeft_parameters,
                                           "right_curves_parameters":curveRight_parameters})
        return parameters_allCurves
        
            
    def calculate_parameters_pattern(self,
                                     fitting_type:str,
                                     loadedParameters_pattern)->Dict[str,List[float]]:
        """
        for buildup or drawdown pattern
        calculate the parameters of four borders 
        to contain all fitted curves of ground truth points
        Args: 
            fitting_type: the type of fitting function
        Returns:
            dictionary
            keys:
            -------
            "left_top"
            "left_bottom"
            "right_top"
            "right_bottom"
            -------
            values:
            parameter list of four borders of buildUp or drawDown
        
        """

        parameters_pattern={}
        fitting_func=self.choose_fittingFunction(fitting_type)
        parameters_allCurves=self.parameters_allCurves_groundTruth[self.buildUp_or_drawDown]
            
        number=8
        y_left_allCurve = np.empty((0, number), float)
        y_right_allCurve = np.empty((0, number), float)
        x_left=np.linspace(start = -self.time_halfWindow_forLearn, stop = 0, num = number)
        x_right=np.linspace(start = 0, stop = self.time_halfWindow_forLearn, num = number)
    
        fig=plt.figure(figsize = (20, 10))
        plt.xlim([-self.time_halfWindow_forPredict,self.time_halfWindow_forPredict])
        if self.buildUp_or_drawDown=="buildUp":
            plt.ylim([-50, 280])
        else:
            plt.ylim([-400, 100])
            
        curve_number=len(parameters_allCurves)
        
        print(f"-----start to calculate'{self.buildUp_or_drawDown}' pattern parameter, there are {curve_number} for calculation" )
        for i in range(curve_number):
            y_left=fitting_func(x_left, *parameters_allCurves["left_curves_parameters"][i])
            y_right=fitting_func(x_right, *parameters_allCurves["right_curves_parameters"][i])
            
            y_left_allCurve=np.append(y_left_allCurve,np.array([y_left]), axis=0)
            y_right_allCurve=np.append(y_right_allCurve,np.array([y_right]), axis=0)
                  
           
        for left_or_right, x,y_allCurve in zip(["left","right"],
                                                [x_left,x_right],
                                                [y_left_allCurve,y_right_allCurve]):
            percentile_upperBound,percentile_lowerBound=self.percentile_tuning[self.buildUp_or_drawDown][left_or_right]
            fine_tuning_max=self.fine_tuning[self.buildUp_or_drawDown][left_or_right+"_top"]
            fine_tuning_min=self.fine_tuning[self.buildUp_or_drawDown][left_or_right+"_bottom"]
            left_parameters_pattern=self.find_border(x,y_allCurve,fitting_type,percentile_upperBound,percentile_lowerBound,fine_tuning_max,fine_tuning_min)
        
            parameters_pattern[left_or_right+"_top"]=left_parameters_pattern["top"]
            parameters_pattern[left_or_right+"_bottom"]=left_parameters_pattern["bottom"]
            
        
        self.legend_unique_labels(fig)
        return parameters_pattern
        
    def legend_unique_labels(self,figure):
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        figure.legend(by_label.values(), by_label.keys(),shadow=True, fontsize='large')


    def truncate_byPercentile(self,
                              y_allCurve:np.array,
                              percentile_upperBound:float,
                              percentile_lowerBound:float)->(List[float],List[float]):  
        """
        remove the data out of upper and lower bounds 
        of every column of a two dimensional array,
        then get maximum and minimum values of each column.
        Args: 
            y_allCurve: two dimensional array, every row represent a curve
            percentile_upperBound
            percentile_lowerBound
        Returns:
            a list of maximun value for each column
            a list of minimum value for each column
        """     
        y_allCurve_min=[]
        y_allCurve_max=[]
        
        for column_index in range(y_allCurve.shape[1]):
            list_max,list_min=self.truncateList_byPercentile(y_allCurve[:,column_index],
                                                             percentile_upperBound,
                                                             percentile_lowerBound)
            y_allCurve_min.append(list_min)
            y_allCurve_max.append(list_max)
        return y_allCurve_min,y_allCurve_max
    
    def find_border(self,
                    x:List[float],
                    y_allCurve:List[float],
                    fitting_type:str,
                    percentile_upperBound:float,
                    percentile_lowerBound:float,
                    fine_tuning_max:float,
                    fine_tuning_min:float,
                    plot_title="",
                    polynomial_order:int =3)->Dict[str,List[float]]:
            
        """
        find top border and bottom border for all curves
        Args: 
            y_allCurve: two dimensional array, every row represent a curve
            percentile_upperBound
            percentile_lowerBound
        Returns:
            a list of maximun value for each column
            a list of minimum value for each column
        """
        for y in y_allCurve:
            plt.plot(x, y, '-', label='New_fit',
                color="orange",linewidth=1)
           
        parameters_half_pattern={}

        y_allCurve_min,y_allCurve_max=self.truncate_byPercentile(y_allCurve,percentile_upperBound,percentile_lowerBound)
        y_allCurve_min=np.asarray(y_allCurve_min)*fine_tuning_min
        y_allCurve_max=np.asarray(y_allCurve_max)*fine_tuning_max
        # plt.figure(figsize = (20, 10))
        plot_color="green"
        parameters_half_pattern["top"]=self.fit_curve(x,y_allCurve_max,fitting_type,plot_color,polynomial_order)
        parameters_half_pattern["bottom"]=self.fit_curve(x,y_allCurve_min,fitting_type,plot_color,polynomial_order)
        
        return parameters_half_pattern
    
         
            
    def plot_in_pattern(self, ax, x_axis, y_axis,y_borders,pattern_name,point_index):
        ax.set(title=f"{pattern_name}, point_index:{point_index}")
        ax.plot(x_axis, y_borders[0],"k")
        ax.plot(x_axis, y_borders[1], "k")
        ax.scatter(x_axis,y_axis,s=2**2)
        print(f"--in {pattern_name} pattern")
        points_aboveTop=sum(y_axis>y_borders[0])
        points_belowBottom=sum(y_axis<y_borders[1])     
        print(f"--{len(x_axis)} points for comparison, {points_aboveTop} points are above top, {points_belowBottom} points under bottom")
        
        
    def check_in_pattern(self,
                             data_forPredict:pd.DataFrame,
                             point_index:int,
                             plot=False):  
        """
        check if a point is buildUp or drawDown, or not in any pattern
        plot if plot is True
        Args: 
            data_forPredict: include data in window for points and data in borders
            point_index: index of a point
            plot: True/False
        Returns:
            "buildUp", "drawDown" or None
        """
        data_plotPoint=data_forPredict.loc[data_forPredict['point_index'] == point_index]
        in_pattern=None
        axs=[None,None]
        if plot:
            print("=============")
            axs = (plt.figure(constrained_layout=True).subplots(1, 2, sharex=True))
        
        x_axis=[data_plotPoint["pressure_time_left"].values[0],
                data_plotPoint["pressure_time_right"].values[0]]
        y_axis=[data_plotPoint["pressure_measure_left"].values[0],
                data_plotPoint["pressure_measure_right"].values[0]]

        for ax, pattern_name in zip(axs, self.pattern_names):
            
            
            #left
            y_borders_left=[data_plotPoint[pattern_name].values[0]['left_top'],
                            data_plotPoint[pattern_name].values[0]['left_bottom']]
             #right
            y_borders_right=[data_plotPoint[pattern_name].values[0]['right_top'],
                            data_plotPoint[pattern_name].values[0]['right_bottom']]
            
            if plot:
                print("------left")
                self.plot_in_pattern(ax, x_axis[0], y_axis[0],y_borders_left,pattern_name,point_index)
                
                print("------right") 
                self.plot_in_pattern(ax, x_axis[1], y_axis[1],y_borders_right,pattern_name,point_index)
            
            criterion=(sum(np.array(y_axis[0][0:-1])>=np.array(y_borders_left[1][0:-1]))>0.8*len(y_axis[0][0:-1]) and 
                sum(np.array(y_axis[0][0:-1])<=np.array(y_borders_left[0][0:-1]))>0.8*len(y_axis[0][0:-1]) and
                sum(np.array(y_axis[1][0:-1])>=np.array(y_borders_right[1][0:-1]))>0.8*len(y_axis[1][0:-1]) and 
                sum(np.array(y_axis[1][0:-1])<=np.array(y_borders_right[0][0:-1]))>0.8*len(y_axis[1][0:-1]))
            if criterion:
                in_pattern=pattern_name
                break
        if plot:
            plt.show()
        return in_pattern
                  
        
    
    def calculate_tuningParameters(self,point_index,pattern_name):
        data_plotPoint=self.data_forPredict.loc[self.data_forPredict['point_index'] == point_index]
           
        x_axis=[data_plotPoint["pressure_time_left"].values[0],
                data_plotPoint["pressure_time_right"].values[0]]
        y_axis=[data_plotPoint["pressure_measure_left"].values[0],
                data_plotPoint["pressure_measure_right"].values[0]]


            
        #left
        y_borders_left=[data_plotPoint[pattern_name].values[0]['left_top'],
                        data_plotPoint[pattern_name].values[0]['left_bottom']]
    
        for index in range(len(x_axis[0])):
            if y_axis[0][index]>y_borders_left[0][index]:
                self.for_tuning[pattern_name]['left_top'].append(round(y_axis[0][index]/y_borders_left[0][index],5))
            if y_axis[0][index]<y_borders_left[1][index]:
                self.for_tuning[pattern_name]['left_bottom'].append(round(y_axis[0][index]/y_borders_left[1][index],5))
                
            
    
        #right
        y_borders_right=[data_plotPoint[pattern_name].values[0]['right_top'],
                        data_plotPoint[pattern_name].values[0]['right_bottom']]
        
        for index in range(len(x_axis[1])):
            if y_axis[1][index]>y_borders_right[0][index]:
                self.for_tuning[pattern_name]['right_top'].append(round(y_axis[1][index]/y_borders_right[0][index],5))
            if y_axis[1][index]<y_borders_right[1][index]:
                self.for_tuning[pattern_name]['right_bottom'].append(round(y_axis[1][index]/y_borders_right[1][index],5))

    
    
    def tuning(self,
               detected_buildUp:List[int],
               detected_drawDown:List[int]):
        groundTruth_buildUp,groundTruth_drawDown=self.breakpoints_forLearn.values()
        buildUp_notDetected=[buildUp for buildUp in groundTruth_buildUp if buildUp not in detected_buildUp]
        drawDown_notDetected=[drawDown for drawDown in groundTruth_drawDown if drawDown not in detected_drawDown]
        
        for point_index in buildUp_notDetected:
            self.calculate_tuningParameters(point_index,"buildUp")
            
        for point_index in drawDown_notDetected:
            self.calculate_tuningParameters(point_index,"drawDown")
        # print("----self.for_tuning",self.for_tuning)
        for pattern_name, border_tuning_parameters in self.for_tuning.items():
            for border_name, tuning_parameters in border_tuning_parameters.items():
                if len(tuning_parameters)==0:
                    self.fine_tuning[pattern_name][border_name]
                else:
                    self.fine_tuning[pattern_name][border_name]=max(tuning_parameters)
        
        
    def detect_breakpoint_type(self,
                               pressure_measure:List[float],
                               pressure_time:List[float],
                               points:List[int]
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
        self.buildUp_or_drawDown=""
        data_inWindow=self.extract_points_inTimeWindow(pressure_measure,pressure_time,points,self.time_halfWindow_forPredict)
        parameters_allCurves=self.calculate_Parameters_allCurve(data_inWindow,fitting_type="linear")  

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
                               fitting_type:str)->Callable[..., List[float]]:
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
 
        
    def get_pattern(self,
                    fitting_type:str,
                    loadedParameters_pattern=None):
        """
        get parameters of pattern borders for buildUp and drawDown,
        assign to self.parameters_twoPatterns,
        Args:
            fitting_type: the type for the fitting function
        """
        print("==================")
        print("start to get pattern..., using '{fitting_type}' fitting")
        for pattern_name in self.pattern_names:  
            self.buildUp_or_drawDown=pattern_name     
            parameters_pattern=self.calculate_parameters_pattern(fitting_type,loadedParameters_pattern)
            self.parameters_twoPatterns[pattern_name]=parameters_pattern
    
    def learn(self,
              pressure_measure:List[float],
              pressure_time:List[float],
              ground_truth:List[int],
              fitting_type:str):
        """ 
        calculate parameters of fitting curves for ground truth
        assign to self.parameters_allCurves_groundTruth
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
                                                                           ground_truth)
        print(f"------In the input ground truth: {len(ground_truth_buildUp)} points are detected as buildup, {len(ground_truth_drawDown)} points are detected as drawDown")
        self.breakpoints_forLearn={"buildUp":ground_truth_buildUp,
                                   "drawDown":ground_truth_drawDown}
        # self.breakpoints_forLearn_multipleLearn.append(self.breakpoints_forLearn)
        # print("----ground_truth_buildUp,ground_truth_drawDown",ground_truth_buildUp,ground_truth_drawDown)

        allPoints=[ground_truth_buildUp,ground_truth_drawDown]

        for points,buildUp_or_drawDown in zip(allPoints, self.pattern_names):
            self.buildUp_or_drawDown=buildUp_or_drawDown
            data_inWindow=self.extract_points_inTimeWindow(pressure_measure,pressure_time,points,self.time_halfWindow_forLearn)
            self.parameters_allCurves_groundTruth[buildUp_or_drawDown]=self.calculate_Parameters_allCurve(data_inWindow,fitting_type=fitting_type)
    
        

    def sav_pattern(self):
        data=self.parameters_twoPatterns
        with open(self.filePath_learnedPattern, 'w') as fp:
            json.dump(data, fp)

    def calculate_y_onBorders(self,
                              x_left:List[float],
                              x_right:List[float],
                              fitting_type:str)->Dict[str,Dict[str,List[float]]]:
        """
        calculate y coordinates on borders 
        coresponding to x coordinates[x_left,x_right]
        
        Args:
            x_left:x coordinates of points in the left window of a certain point,
            x_right:x coordinates of points in the right window of a certain point
            fitting_type: the type for the fitting function            
        Returns:
            y coordinates on every borders
            -------
            {"buildUp":{"left_top":[],
                        "left_bottom":[],
                        "right_top":[],
                        "right_bottom":[]},
            "drawDown":{"left_top":[],
                        "left_bottom":[],
                        "right_top":[],
                        "right_bottom":[]}}
            -------
        """               
        y_borders_twoPattern=defaultdict(dict)
        x_left=np.array(x_left)
        x_right=np.array(x_right)
        xzip=[x_left,
            x_left,
            x_right,
            x_right]
        
        fitting_func=self.choose_fittingFunction(fitting_type)
        for pattern_name,parameters_pattern in self.parameters_twoPatterns.items():     
            for border_name, x, parameters in zip(self.border_names,xzip,parameters_pattern.values()):
                y_borders_twoPattern[pattern_name][border_name]=fitting_func(x, *parameters)
        
        return y_borders_twoPattern

        
    def predict_usePatternRecognition(self,
                pressure_measure:List[float],
                pressure_time:List[float],
                points:List[int],
                fitting_type="polynomial")->(List[float],List[float]):
        """ 
        identify the breakpoints for the given dataset.
        Using the pattern recognition method.
        # if the mode is "whole_dataset", check every point in the dataset
        # if the mode is "refine_detection", check the points which already detected by the previous detection.
        Args:
            pressure_measure: pressure measure for the whole dataset
            pressure_time: pressure time for the whole dataset
            points: indices of points to be identify
            # mode: "whole_dataset"/ "refine_detection"
            fitting_type: the type for the fitting function            
        Returns:
            two lists for buildUp and drawDown break points indices
        """     
        print("==================")
        print("start to predict using pattern...")
        # self.std_2=statistics.stdev(second_order_derivative)
        self.buildUp_or_drawDown=""
        self.data_forPredict=pd.DataFrame(columns=["point_index",
                                                   "pressure_time_left",
                                                    "pressure_measure_left",
                                                    "pressure_time_right",
                                                    "pressure_measure_right",
                                                    "buildUp",
                                                    "drawDown"])
        
        # if mode=="whole_dataset":   
        #     points=[point_index for point_index in range(len(pressure_measure))]
        # elif mode=="refine_detection":
        #     points=[]
        #     for pattern_name, detected_points in self.detectedpoints.items():
        #         points.extend(detected_points)
        #         print(f"there are {len(detected_points)} '{pattern_name}' inputted for second prediction")
        #         # self.points_Detectedin1=[point for point in self.breakpoints_forLearn[pattern_name] if point in detected_points]
        #         # print(f"---------there are {len(self.points_Detectedin1)} points_Detectedin1")   
        # else:
        #     print("check the mode, it must be 'whole_dataset' or 'refine_detection'...")
            
        borderData=pd.DataFrame()
        data_inWindow=self.extract_points_inTimeWindow(pressure_measure,pressure_time,points,self.time_halfWindow_forPredict)
        for index,curveData_singlePoint in data_inWindow.iterrows():     
            curveData_singlePoint["pressure_time_right"].reverse()
            curveData_singlePoint["pressure_measure_right"].reverse()
        
            y_borders_twoPattern=self.calculate_y_onBorders(curveData_singlePoint["pressure_time_left"],curveData_singlePoint["pressure_time_right"],fitting_type)
            borderData=borderData.append(y_borders_twoPattern,ignore_index=True)
         
        self.data_forPredict=pd.concat([data_inWindow, borderData], axis=1)  
   
        detectedpoints={"buildUp":[],
                        "drawDown":[]}
        for point_index in self.data_forPredict["point_index"]:
            breakpoints_type=self.check_in_pattern(self.data_forPredict,point_index)  
            if breakpoints_type==None:
                continue
            else:
                detectedpoints[breakpoints_type].append(point_index)
        
        points_buildUp,points_drawDown=detectedpoints.values()
        print(f"after second prediction, there are {len(points_buildUp)} detected buildup points, {len(points_drawDown)} drawdown detected")
  
        
        points=points_buildUp+points_drawDown
        points_buildUp,points_drawDown=self.detect_breakpoint_type(pressure_measure,pressure_time,points)
        print(f"after second prediction, the results are filtered further, there are {len(points_buildUp)} detected buildup points, {len(points_drawDown)} drawdown detected")
        return points_buildUp,points_drawDown
            
    def calculte_tangent_nDegreePolynomial(self,x,parameters):
        """
        calculate the tangent of x coordinate
        for the polynomial curve with arbitrary degree
        Args:
            x: value of x coordinate    
            parameters: the parameters of the curve       
        Returns:
            tangent
        """
        slope=0.0
        degree=len(parameters)-1
        for i in range(degree):
            slope=slope+(degree-i)*parameters[i]*x**(degree-i-1)
        return slope
        
    def produce_tangent_inWindow(self,
                                   pressure_measure:List[float],
                                    pressure_time:List[float],
                                    points:List[int],
                                    fitting_type:str,
                                    data_type:str="single_point",
                                    time_halfWindow:float=None,
                                    point_halfWindow:int=None,
                                    polynomial_order:int=3,
                                    point_halfWindow_tagentPlot:int=5
                                    )->pd.DataFrame:
        
        if time_halfWindow!=None and point_halfWindow!=None:
            print("if you want to use time window, please set 'point_halfWindow' to be None, vice versa")
            return None
        if time_halfWindow!=None:
            data_inWindow=self.extract_points_inTimeWindow(pressure_measure,
                                        pressure_time,
                                        points,
                                        time_halfWindow)
        if point_halfWindow!=None:
            data_inWindow=self.extract_points_inPointWindow(pressure_measure,
                                                            pressure_time,
                                                            points,
                                                            point_halfWindow)
        # display(data_inWindow)
        parameters_allCurves=self.calculate_Parameters_allCurve(data_inWindow,fitting_type,polynomial_order)
        # display(parameters_allCurves)
        
        tangent_inWindow=pd.DataFrame()
        
        for index,parameters in parameters_allCurves.iterrows():
            pressure_time_left=data_inWindow.iloc[index]["pressure_time_left"]
            pressure_time_right=data_inWindow.iloc[index]["pressure_time_right"]
            pressure_measure_left=data_inWindow.iloc[index]["pressure_measure_left"]
            pressure_measure_right=data_inWindow.iloc[index]["pressure_measure_right"]
            tangent_left=self.calculte_tangent_nDegreePolynomial(np.asarray(pressure_time_left),parameters["left_curves_parameters"])
            tangent_right=self.calculte_tangent_nDegreePolynomial(np.asarray(pressure_time_right),parameters["right_curves_parameters"])
            
            if data_type=="for_plot":
                data={"point_index":parameters["point_index"],
                    "pressure_time_left":pressure_time_left[-point_halfWindow_tagentPlot:],
                    "pressure_time_right":pressure_time_right[0:point_halfWindow_tagentPlot],
                    "pressure_measure_left":pressure_measure_left[-point_halfWindow_tagentPlot:],
                    "pressure_measure_right":pressure_measure_right[0:point_halfWindow_tagentPlot],
                    "left_curves_parameters":parameters["left_curves_parameters"],
                    "right_curves_parameters":parameters["right_curves_parameters"],
                    "tangentFamily_left":tangent_left[-point_halfWindow_tagentPlot:],
                    "tangentFamily_right":tangent_right[0:point_halfWindow_tagentPlot]}
                tangent_inWindow=tangent_inWindow.append(data,ignore_index=True)
            elif data_type=="average": 
                tangent_left_average=sum(tangent_left)/len(tangent_left)
                tangent_right_average=sum(tangent_right)/len(tangent_right)
                data={"point_index":parameters["point_index"],
                    "tangent_left":tangent_left_average,
                    "tangent_right":tangent_right_average}
                tangent_inWindow=tangent_inWindow.append(data,ignore_index=True)        
            elif data_type=="single_point":              
                data={"point_index":parameters["point_index"],
                    "tangent_left":tangent_left[-1],
                    "tangent_right":tangent_right[0]}   
                tangent_inWindow=tangent_inWindow.append(data,ignore_index=True)      
            else:
                print("Invalid data_type, must be 'for_plot', 'average' or 'single_point'...")
                return None 
        
            tangent_inWindow["point_index"] = tangent_inWindow["point_index"].astype(int)
            
        return tangent_inWindow
    
    def tangentLine_func(self,x,tangent_slope,x0,y0):
        return tangent_slope*x-tangent_slope*x0+y0
    
    def plot_tangent_inPointWindow(self,data_forTangentPlot,point_index):
        plt.figure(figsize = (20, 10))
        data_plotPoint=data_forTangentPlot.loc[data_forTangentPlot['point_index'] == point_index]
        pressure_time=[data_plotPoint["pressure_time_left"].values[0],data_plotPoint["pressure_time_right"].values[0]]
        pressure_measure=[data_plotPoint["pressure_measure_left"].values[0],data_plotPoint["pressure_measure_right"].values[0]]
        parameters=[data_plotPoint["left_curves_parameters"].values[0],data_plotPoint["right_curves_parameters"].values[0]]
        tangents=[data_plotPoint["tangentFamily_left"].values[0],data_plotPoint["tangentFamily_right"].values[0]]
        fitting_type="polynomial"
        point_colors=["green","orange"]
        tangentLine_colors=["red","blue"]
        print("=======point_index===========",point_index)
        for x,y,parameter,tangent,point_color,tangentLine_color in zip(pressure_time,pressure_measure,parameters,tangents,point_colors,tangentLine_colors):   
            plt.scatter(x,y,color=point_color)
            self.plot_fittingCurve(x,y,fitting_type,"yellow",*parameter)
            for i in range(len(x)):
                plot_step=max([abs(item) for item in x])/(len(x)*2)
                x_tangent=[x[i]-plot_step,x[i]+plot_step]
                print(f"tangent[{i}]:{tangent[i]}")
                y_tangent=self.tangentLine_func(np.asarray(x_tangent),tangent[i],x[i],y[i])
                plt.plot(x_tangent, y_tangent, label="tangent line",color=tangentLine_color,linestyle='-')
        plt.show()
                
        
            
    
    def get_tangent(self,
                    parameters_allCurves:pd.DataFrame,
                    fitting_type:str)->pd.DataFrame:
        """ 
        from the parameters of all fitted curves,
        get the tangent values of all curves
        Args:
            parameters_allCurves: pd.DataFrame in which each row 
            represents the parameters of left and right curves of a point
            fitting_type: the type for the fitting function            
        Returns:
            pd.DataFrame in which each row represents the left and right tangent of a point
        """  
        #for polynomial fit the third parameter is the tangent
        if fitting_type=="polynomial":
            n=len(parameters_allCurves["left_curves_parameters"][0])-2
        elif fitting_type=="linear":
            n=0 
        else:  
            print("please check fitting type, need to be 'polynomial' or 'linear'")
        tangent_left=[parameters[n] for parameters in parameters_allCurves["left_curves_parameters"]] 
        tangent_right=[parameters[n] for parameters in parameters_allCurves["right_curves_parameters"]] 
        tangent_df=pd.DataFrame({"point_index":parameters_allCurves["point_index"],"tangent_left":tangent_left,"tangent_right":tangent_right})
        return tangent_df
    
    def truncateList_byPercentile(self,
                                  tempList:List[float],
                                  percentile_upperBound:float,
                                  percentile_lowerBound:float)->(float,float):
        """ 
        removes the items of a list, which is 
        not between the percentile_upperBound, and percentile_lowerBound,
        then get the max and min value of the trancated list
        Args:
            tempList: a list for processing.
            percentile_upperBound: upper bound of the np.percentile function
            percentile_lowerBound: lower bound of the np.percentile function       
        Returns:
            the max and min value of the trancated list by percentile upper&lower bound.
        """  
        upper_bound=np.percentile(tempList, percentile_upperBound, axis=0,method="normal_unbiased")
        lower_bound=np.percentile(tempList, percentile_lowerBound, axis=0,method="normal_unbiased")
        list_truncated=[item for item in tempList if item>=lower_bound and item<=upper_bound]
        list_max=max(list_truncated)
        list_min=min(list_truncated)
        return list_max,list_min
    
    def get_deltaTangent_criterion(self,fitting_type:str):
        print("==================")
        print(f"start to get get deltaTangent..., using '{fitting_type}' fitting")
        delta_tangent=[]
        for pattern_name in self.pattern_names:
            tangent_groundTruth_buildUpOrDrawDown=self.get_tangent(self.parameters_allCurves_groundTruth[pattern_name],fitting_type)
            tangent_left=tangent_groundTruth_buildUpOrDrawDown["tangent_left"]
            tangent_right=tangent_groundTruth_buildUpOrDrawDown["tangent_right"]  
            # display(tangent_left-tangent_right)  
            delta_tangent.append(min(abs(tangent_left-tangent_right))) 
        # print("delta_tangent",delta_tangent)
        return min(delta_tangent)
 

    
    def get_tangents_twoPatterns(self,fitting_type:str):
        """ 
        from the groundtruth, get upperBound lowerBound of the 'buildUp' and 'drawDown'
        Args:
            fitting_type: the type for the fitting function        
        Returns:
            dictionary.
            ----------
            self.tangents_twoPatterns={"buildUp":{"left_top":float,
                                               "left_bottom":float,
                                               "right_top":float,
                                               "right_bottom":float},
                                    "drawDown":{"left_top":float,
                                               "left_bottom":float,
                                               "right_top":float,
                                               "right_bottom":float}}
            ----------
        """  
        print("==================")
        print(f"start to get tangent pattern..., using '{fitting_type}' fitting")
        for pattern_name in self.pattern_names:
            tangent_groundTruth_buildUpOrDrawDown=self.get_tangent(self.parameters_allCurves_groundTruth[pattern_name],fitting_type)
            
            #left side
            self.tangents_twoPatterns[pattern_name]["left_top"]=max(tangent_groundTruth_buildUpOrDrawDown["tangent_left"])
            self.tangents_twoPatterns[pattern_name]["left_bottom"]=min(tangent_groundTruth_buildUpOrDrawDown["tangent_left"])
                    
            #right side
            self.tangents_twoPatterns[pattern_name]["right_top"]=max(tangent_groundTruth_buildUpOrDrawDown["tangent_right"])
            self.tangents_twoPatterns[pattern_name]["right_bottom"]=min(tangent_groundTruth_buildUpOrDrawDown["tangent_right"])

    def check_in_deltaTangent(self,deltaTangent_criterion:float,tangent_forPredict:pd.DataFrame):
        buildup=[]
        drawdown=[]
        for index,row in tangent_forPredict.iterrows():
            if abs(row["tangent_left"]-row["tangent_right"])>deltaTangent_criterion:
                if row["tangent_right"]>0:
                    buildup.append(int(row["point_index"]))
                else:
                    drawdown.append(int(row["point_index"]))
                    
        return buildup,drawdown
        
    
    def predict_useDeltaTangent(self,
                             pressure_measure:List[float],
                             pressure_time:List[float],
                             first_order_derivative:List[float],
                             points:List[int],
                             time_halfWindow:float=None,
                             point_halfWindow:int=None,
                             deltaTangent_criterion:float=40,
                             polynomial_order:int =3,
                             tangent_type:str="single_point",
                             fitting_type:str="polynomial" 
                             )->(List[int],List[int]):
        """
        predict the breakpoint use the difference between left tangent and right tangent
        """
        print("==================")
        print("start to predict using tangent...")
        self.buildUp_or_drawDown=""
        std_1=statistics.stdev(first_order_derivative)
        # points=[point_index for point_index in range(len(second_order_derivative)) if second_order_derivative[point_index]>0.05*std_2 ]
        filtered_points=[point_index for point_index in points if first_order_derivative[point_index]>0.02*std_1 ]
        print("len(filtered_points)",len(filtered_points))
        # deltaTangent_criterion=self.get_deltadeltaTangent_criterion(fitting_type)
        # data_inWindow=self.extract_points_inPointWindow(pressure_measure,pressure_time,points,point_halfWindow)
        # parameters_allCurves=self.calculate_Parameters_allCurve(data_inWindow,fitting_type,polynomial_order)     
        # tangent_forPredict_get=self.get_tangent(parameters_allCurves,fitting_type)
        # display(tangent_forPredict_get)
        tangent_forPredict=self.produce_tangent_inWindow(pressure_measure,
                                                        pressure_time,
                                                        points,
                                                        fitting_type,
                                                        tangent_type,
                                                        time_halfWindow,
                                                        point_halfWindow,
                                                        polynomial_order)
        self.tangent_forPredict=tangent_forPredict
        # display(tangent_forPredict)
        points_buildUp,points_drawDown=self.check_in_deltaTangent(deltaTangent_criterion,
                                               tangent_forPredict)
        print(f"detect {len(points_buildUp)} buildups and {len(points_drawDown)} drawdowns " )
        return points_buildUp,points_drawDown
        
    
    def predict_usingTangentRange(self,
                             pressure_measure:List[float],
                             pressure_time:List[float],
                             points:List[int],
                             fitting_type="polynomial"):
        """ 
        identify the breakpoints if the left tangent and right tangent are in some coresponding range.
        # if the mode is "whole_dataset", check every point in the dataset
        # if the mode is "refine_detection", check the points which already detected by the previous detection.
        Args:
            pressure_measure: pressure measure for the whole dataset
            pressure_time: pressure time for the whole dataset
            points: indices of points to be identify
            # mode: "whole_dataset"/ "refine_detection"
            fitting_type: the type for the fitting function            
        Returns:
            two lists for buildUp and drawDown break points indices
        """  
        print("==================")
        print("start to predict using tangent...")
        self.buildUp_or_drawDown=""
        # if mode=="whole_dataset":   
        #     points=[point_index for point_index in range(len(pressure_measure))]
        # elif mode=="refine_detection":
        #     points=[]
        #     for detected_points in self.detectedpoints.values():
        #         points.extend(detected_points)
        # else:
        #     print("check the mode, it must be 'whole_dataset' or 'refine_detection'...")
        
        data_inWindow=self.extract_points_inTimeWindow(pressure_measure,pressure_time,points,self.time_halfWindow_forPredict)
        parameters_allCurves=self.calculate_Parameters_allCurve(data_inWindow,fitting_type)        
        tangent_forPredict=self.get_tangent(parameters_allCurves,fitting_type)
        
        self.tangent_forPredict=tangent_forPredict
        
        points_buildUp,points_drawDown=self.check_in_tangentRange(tangent_forPredict)     
            
        points=points_buildUp+points_drawDown
        print(f"before filter, the length of buildup {points_buildUp}, the length of drawdown {points_drawDown}")
        points_buildUp,points_drawDown=self.detect_breakpoint_type(pressure_measure,pressure_time,points)
        return points_buildUp,points_drawDown
    
    
    def check_in_tangentRange(self,tangent_forPredict:pd.DataFrame)->Dict[str,List[float]]:
        """ 
        from the tangents of points, identify the types of points
        Args:
            pressure_measure: pressure measure for the whole dataset
            pressure_time: pressure time for the whole dataset
            mode: "whole_dataset"/ "refine_detection"
            fitting_type: the type for the fitting function            
        Returns:
            two lists for buildUp and drawDown break points indices
        """  
        
        detectedpoints={}
        for pattern_name in self.pattern_names:
    
            sub_df=tangent_forPredict.loc[(tangent_forPredict["tangent_left"]>=self.tangents_twoPatterns[pattern_name]["left_bottom"]) &
                                    (tangent_forPredict["tangent_left"]<=self.tangents_twoPatterns[pattern_name]["left_top"]) &
                                    (tangent_forPredict["tangent_right"]>=self.tangents_twoPatterns[pattern_name]["right_bottom"]) &
                                    (tangent_forPredict["tangent_right"]<=self.tangents_twoPatterns[pattern_name]["right_top"])]

            detectedpoints[pattern_name]=list(sub_df["point_index"])
        return detectedpoints.values()
    

        
            
        
            
        
      