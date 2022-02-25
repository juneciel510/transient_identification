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

import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from matplotlib import rcParams
from operator import itemgetter
from typing import Callable, Dict, List, Set, Tuple
from scipy.signal import savgol_filter
import math 
import bisect

from plot2 import plot_histogram

#pattern recognition method
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
        
        #to store the input groundtruth, including buildup & drawdown together
        self.ground_truth =[]
        #to store the points for learn, classified as buildup & drawdown
        self.breakpoints_forLearn=defaultdict(list)
        # self.breakpoints_forLearn_multipleLearn=[]
        #to store the points predicted 
        self.detectedpoints=defaultdict(list)
        self.point_halfWindow=point_halfWindow
        #time window for learn
        self.time_halfWindow_forLearn=time_halfWindow_forLearn
        #time window for predict
        self.time_halfWindow_forPredict=time_halfWindow_forPredict
        self.percentile_tuning=percentile_tuning
        self.fine_tuning=fine_tuning
        self.filePath_learnedPattern=filePath_learnedPattern

        
        self.std_2=None
        self.buildUp_or_drawDown=""
        self.pattern_names=["buildUp","drawDown"]
        self.border_names=["left_top","left_bottom","right_top","right_bottom"]
        # self.data_inWindow=pd.DataFrame(columns=['point_index',
        #                                      'pressure_time_left', 
        #                                      'pressure_measure_left',
        #                                      'pressure_time_right', 
        #                                      'pressure_measure_right'])
        # self.borderData=pd.DataFrame(columns=self.border_names)
        self.data_forPredict=pd.DataFrame(columns=["pressure_time_left",
                                                  "pressure_measure_left",
                                                  "pressure_time_right",
                                                  "pressure_measure_right",
                                                  "left_top",
                                                  "left_bottom",
                                                  "right_top",
                                                  "right_bottom"])
        # self.parameters_allCurves=pd.DataFrame(columns=["point_index",
        #                                                 "left_curves_parameters",
        #                                                 "right_curves_parameters"])
        #key is pattern name, value is self.parameters_allCurves
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
                                                'left_tangent', 
                                                'right_tangent'])
            
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
            print("-------start to extract points data inTimeWindow")
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
                  plot_color:str)->List[float]:
        """
        curve fit and plot xdata, ydata and their fitted funtion
        Args: 
            xdata: x coordinate data
            ydata: y coordinate data
            plot_color: plot color
            fitting_type: the type of fitting function
            plot_title: the title of the plot
        Returns:
            the parameters of the fitted curve
        """
       
        x = np.asarray(xdata)
        y = np.asarray(ydata)
        
        fitting_func=self.choose_fittingFunction(fitting_type)
        
        if fitting_type=="polynomial":
            parameters=np.polyfit(x,y,3)
        if fitting_type=="linear" or fitting_type=="log":
            parameters, covariance = curve_fit(fitting_func, x, y)
        self.plot_fittingCurve(x, y,fitting_type, plot_color, *parameters)
 
        return parameters
    
    def plot_fittingCurve(self,
                          x:np.array, 
                          y:np.array,
                          fitting_type:str, 
                          plot_color:str, 
                          *parameters)->None:
        """
        plot x, yand their fitted funtion
        Args: 
            xdata: x coordinate data
            ydata: y coordinate data
            plot_color: plot color
            fitting_type: the type of fitting function
            plot_title: the title of the plot
        Returns:
            the parameters of the fitted curve
        """
        fitting_func=self.choose_fittingFunction(fitting_type)
        y_fit=fitting_func(x, *parameters)
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
        
        if self.buildUp_or_drawDown!="":
            plt.scatter(x=x,y=y,color=plot_color)
            plt.plot(x, y_fit, color=plot_color,linestyle='-')
            plt.title(self.buildUp_or_drawDown)
        # if fitting_type=="log" or fitting_type=="polynomial":
        #     plt.show()
    
    def calculate_Parameters_allCurve(self,data_inWindow:pd.DataFrame,fitting_type:str)->pd.DataFrame:
        """
        plot x, yand their fitted funtion
        Args: 
            data_inWindow: data in window
            fitting_type: the type of fitting function
            plot_title: the title of the plot
        Returns:
            dataframe containing 3 columns
            -------
            ["point_index", 
            "left_curves_parameters",
            "right_curves_parameters"]
            -------
        """
        print("---------------calculate_Parameters_allCurve",fitting_type)
        curveLeft_parameters=[]
        curveRight_parameters=[]
        parameters_allCurves=pd.DataFrame()
        
        plt.figure(figsize = (20, 10))

        for i in range(len(data_inWindow)):
            #left side
            xdata=data_inWindow["pressure_time_left"][i]
            ydata=data_inWindow["pressure_measure_left"][i]
            plot_color="yellow"  
            curveLeft_parameters.append(self.fit_curve(xdata,ydata,fitting_type,plot_color))

            
            #right side
            xdata=data_inWindow["pressure_time_right"][i]
            ydata=data_inWindow["pressure_measure_right"][i]           
            plot_color="blue" 
            curveRight_parameters.append(self.fit_curve(xdata,ydata,fitting_type,plot_color))
            
        parameters_allCurves=pd.DataFrame({"point_index":data_inWindow["point_index"].values,
                                           "left_curves_parameters":curveLeft_parameters,
                                           "right_curves_parameters":curveRight_parameters})
        return parameters_allCurves
        
            
    def calculate_parameters_pattern(self,fitting_type,loadedParameters_pattern):
        """
        from the parameters of allCurves
        calculate the parameters of four borders 
        to include all these curves
        -------------
        "left_top"
        "left_bottom"
        "right_top"
        "right_bottom"
        -------------
        of buildup/drawdown pattern
        """

        parameters_pattern={}
        fitting_func=self.choose_fittingFunction(fitting_type)
        parameters_allCurves=self.parameters_allCurves_groundTruth[self.buildUp_or_drawDown]

        # #if has old patterns, load old patterns and compare with new fitting curves
        # if len(loadedParameters_pattern)>0:
        #     old_pattern={"left_curves_parameters":[loadedParameters_pattern["left_bottom"],
        #                                            loadedParameters_pattern["left_top"]],
        #                 "right_curves_parameters":[loadedParameters_pattern["right_bottom"],
        #                                            loadedParameters_pattern["right_top"]]}
        #     parameters_allCurves.append(pd.DataFrame(old_pattern), ignore_index = True)
            
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
            

            # if len(loadedParameters_pattern)>0 and (i==0 or i==1):
            #     plt.plot(x_right, y_right, 'r+', 
            #              label='Old_pattern',
            #              color="red",
            #              linewidth=4)
            #     plt.plot(x_left, y_left, 'r+', 
            #              label='Old_patern',
            #              color="red",
            #              linewidth=4)
            # else:
            plt.plot(x_right, y_right, '-', label='New_fit',
                    color="orange",linewidth=1)
            plt.plot(x_left, y_left, '-', label='New_fit',
                        color="orange",linewidth=1)        
        percentile_upperBound,percentile_lowerBound=self.percentile_tuning[self.buildUp_or_drawDown]["left"]
        fine_tuning_max=self.fine_tuning[self.buildUp_or_drawDown]["left_top"]
        fine_tuning_min=self.fine_tuning[self.buildUp_or_drawDown]["left_bottom"]
        left_parameters_pattern=self.find_border(x_left,y_left_allCurve,fitting_type,percentile_upperBound,percentile_lowerBound,fine_tuning_max,fine_tuning_min)
        
        percentile_upperBound,percentile_lowerBound=self.percentile_tuning[self.buildUp_or_drawDown]["right"]
        fine_tuning_max=self.fine_tuning[self.buildUp_or_drawDown]["right_top"]
        fine_tuning_min=self.fine_tuning[self.buildUp_or_drawDown]["right_bottom"]
        right_parameters_pattern=self.find_border(x_right,y_right_allCurve,fitting_type,percentile_upperBound,percentile_lowerBound,fine_tuning_max,fine_tuning_min)
        
        parameters_pattern["left_top"]=left_parameters_pattern["top"]
        parameters_pattern["left_bottom"]=left_parameters_pattern["bottom"]
        parameters_pattern["right_top"]=right_parameters_pattern["top"]
        parameters_pattern["right_bottom"]=right_parameters_pattern["bottom"]
        
        self.legend_without_duplicate_labels(fig)
        return parameters_pattern
        
    def legend_without_duplicate_labels(self,figure):
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        figure.legend(by_label.values(), by_label.keys(),shadow=True, fontsize='large')


    def truncate_byPercentile(self,y_allCurve:np.array,percentile_upperBound:float,percentile_lowerBound:float):  
        """
        y_allCurve: two dimensional array
        """     
        y_allCurve_min=[]
        y_allCurve_max=[]
        upper_bound=np.percentile(y_allCurve, percentile_upperBound, axis=0,method="normal_unbiased")
        lower_bound=np.percentile(y_allCurve, percentile_lowerBound, axis=0,method="normal_unbiased")
        for column_index in range(y_allCurve.shape[1]):
            # plot_histogram(y_allCurve[:,column_index],title=str(column_index))
            # print("y_allCurve[:,column]",y_allCurve[:,column_index])
            y_allCurve_columnTruncated=[y for y in y_allCurve[:,column_index] if y>=lower_bound[column_index] and y<=upper_bound[column_index]]
            y_allCurve_min.append(min(y_allCurve_columnTruncated))
            y_allCurve_max.append(max(y_allCurve_columnTruncated))
            # plot_histogram(y_allCurve_columnTruncated,title=str(column_index)+"percentiled")
        return np.asarray(y_allCurve_min),np.asarray(y_allCurve_max)
    
    def find_border(self,x:List[float],y_allCurve:List[float],fitting_type,percentile_upperBound,percentile_lowerBound,fine_tuning_max,fine_tuning_min,plot_title=""):
            
        """

        calculate parameters of top curve and buttom curve 
        for left or right side of buildup or drawndown pattern
        
        Args:
            y_allCurve: two dimensional nparray
        """
        parameters_half_pattern={}

        plot_color="green"
        
        # # y_allCurve_max=y_allCurve.max(axis=0)
        # y_allCurve_max=np.percentile(y_allCurve, 90, axis=0,method="normal_unbiased")
        # # y_allCurve_min=y_allCurve.min(axis=0)
        # y_allCurve_min=np.percentile(y_allCurve, 10, axis=0,method="normal_unbiased")
        
        y_allCurve_min,y_allCurve_max=self.truncate_byPercentile(y_allCurve,percentile_upperBound,percentile_lowerBound)
        y_allCurve_min=y_allCurve_min*fine_tuning_min
        y_allCurve_max=y_allCurve_max*fine_tuning_max
        # plt.figure(figsize = (20, 10))
        parameters_half_pattern["top"]=self.fit_curve(x,y_allCurve_max,fitting_type,plot_color)
     

        parameters_half_pattern["bottom"]=self.fit_curve(x,y_allCurve_min,fitting_type,plot_color)
        
        return parameters_half_pattern
    
    def check_in_pattern(self, data_forPredict,point_index):

        data_plotPoint=data_forPredict.loc[data_forPredict['point_index'] == point_index]
    
        
        # pattern_names=["buildUp","drawDown"]
        x_axis=[data_plotPoint["pressure_time_left"].values[0],
                data_plotPoint["pressure_time_right"].values[0]]
        y_axis=[data_plotPoint["pressure_measure_left"].values[0],
                data_plotPoint["pressure_measure_right"].values[0]]

        for pattern_name in self.pattern_names:
            
            
            #left
            y_borders_left=[data_plotPoint[pattern_name].values[0]['left_top'],
                            data_plotPoint[pattern_name].values[0]['left_bottom']]
            
            y_borders_right=[data_plotPoint[pattern_name].values[0]['right_top'],
                            data_plotPoint[pattern_name].values[0]['right_bottom']]
            
            criterion=(sum(np.array(y_axis[0][0:-1])>=np.array(y_borders_left[1][0:-1]))>0.8*len(y_axis[0][0:-1]) and 
                sum(np.array(y_axis[0][0:-1])<=np.array(y_borders_left[0][0:-1]))>0.8*len(y_axis[0][0:-1]) and
                sum(np.array(y_axis[1][0:-1])>=np.array(y_borders_right[1][0:-1]))>0.8*len(y_axis[1][0:-1]) and 
                sum(np.array(y_axis[1][0:-1])<=np.array(y_borders_right[0][0:-1]))>0.8*len(y_axis[1][0:-1]))
            
            # if point_index in self.points_Detectedin1:       
            # # if pattern_name=="drawDown" and point_index in self.breakpoints_forLearn[pattern_name]:
            #     print("--------point_index:",point_index)
            #     print(f"in {pattern_name} pattern")
            #     print("*****-----------------criterion:",criterion)
            #     print("sum(np.array(y_axis[0][0:-1])>=np.array(y_borders_left[1][0:-1])),len(y_axis[0][0:-1]",sum(np.array(y_axis[0][0:-1])>=np.array(y_borders_left[1][0:-1])),len(y_axis[0][0:-1]))
            #     print("sum(np.array(y_axis[0][0:-1])<=np.array(y_borders_left[0][0:-1])),len(y_axis[0][0:-1])",sum(np.array(y_axis[0][0:-1])<=np.array(y_borders_left[0][0:-1])),len(y_axis[0][0:-1]))
            #     print("sum(np.array(y_axis[1][0:-1])>=np.array(y_borders_right[1][0:-1])),len(y_axis[1][0:-1])",sum(np.array(y_axis[1][0:-1])>=np.array(y_borders_right[1][0:-1])),len(y_axis[1][0:-1]))
            #     print("sum(np.array(y_axis[1][0:-1])<=np.array(y_borders_right[0][0:-1])),len(y_axis[1][0:-1])",sum(np.array(y_axis[1][0:-1])<=np.array(y_borders_right[0][0:-1])),len(y_axis[1][0:-1]))
        
                
            #     # print(len(x_axis[0]),len(y_axis[0]),len(y_borders_left[0]),len(y_borders_left[1]))
            #     points_aboveTop=sum(y_axis[0]>y_borders_left[0])
            #     points_belowBottom=sum(y_axis[0]<y_borders_left[1])
            #     print("-------------left")
            #     print(f"{len(x_axis[0])} points for comparison, {points_aboveTop} points are above top, {points_belowBottom} points under bottom")
                
            #     points_aboveTop=sum(y_axis[1]>y_borders_right[0])
            #     points_belowBottom=sum(y_axis[1]<y_borders_right[1])
            #     print("-------------right")
            #     print(f"{len(x_axis[1])} points for comparison, {points_aboveTop} points are above top, {points_belowBottom} points under bottom")
                
            if criterion:
                return pattern_name
        return None
                

       
            
            
            
            
        
    def plot_data_forPredict(self,point_index):
        print("*****************=============")
        data_plotPoint=self.data_forPredict.loc[self.data_forPredict['point_index'] == point_index]
        axs = (plt.figure(constrained_layout=True).subplots(1, 2, sharex=True))
        
        pattern_names=["buildUp","drawDown"]
        x_axis=[data_plotPoint["pressure_time_left"].values[0],
                data_plotPoint["pressure_time_right"].values[0]]
        y_axis=[data_plotPoint["pressure_measure_left"].values[0],
                data_plotPoint["pressure_measure_right"].values[0]]

        for ax, pattern_name in zip(axs, pattern_names):
            ax.set(title=f"{pattern_name}, point_index:{point_index}")
            
            #left
            y_borders_left=[data_plotPoint[pattern_name].values[0]['left_top'],
                            data_plotPoint[pattern_name].values[0]['left_bottom']]
            ax.plot(x_axis[0], y_borders_left[0],"k")
            ax.plot(x_axis[0], y_borders_left[1], "k")
            ax.scatter(x_axis[0],y_axis[0],s=2**2)
            print(f"in {pattern_name} pattern")
            # print(len(x_axis[0]),len(y_axis[0]),len(y_borders_left[0]),len(y_borders_left[1]))
            points_aboveTop=sum(y_axis[0]>y_borders_left[0])
            points_belowBottom=sum(y_axis[0]<y_borders_left[1])
            print("-------------left")
            print(f"{len(x_axis[0])} points for comparison, {points_aboveTop} points are above top, {points_belowBottom} points under bottom")
                
            #right
            y_borders_right=[data_plotPoint[pattern_name].values[0]['right_top'],
                            data_plotPoint[pattern_name].values[0]['right_bottom']]
            ax.plot(x_axis[1], y_borders_right[0], "k")
            ax.plot(x_axis[1], y_borders_right[1], "k")
            ax.scatter(x_axis[1],y_axis[1],s=2**2)
            
            points_aboveTop=sum(y_axis[1]>y_borders_right[0])
            points_belowBottom=sum(y_axis[1]<y_borders_right[1])
            print("-------------right")
            print(f"{len(x_axis[1])} points for comparison, {points_aboveTop} points are above top, {points_belowBottom} points under bottom")
            
            criterion=(sum(np.array(y_axis[0][0:-1])>=np.array(y_borders_left[1][0:-1]))>0.8*len(y_axis[0][0:-1]) and 
                sum(np.array(y_axis[0][0:-1])<=np.array(y_borders_left[0][0:-1]))>0.8*len(y_axis[0][0:-1]) and
                sum(np.array(y_axis[1][0:-1])>=np.array(y_borders_right[1][0:-1]))>0.8*len(y_axis[1][0:-1]) and 
                sum(np.array(y_axis[1][0:-1])<=np.array(y_borders_right[0][0:-1]))>0.8*len(y_axis[1][0:-1]))
            print("-----------------criterion:",criterion)
            print("sum(np.array(y_axis[0][0:-1])>=np.array(y_borders_left[1][0:-1])),len(y_axis[0][0:-1]",sum(np.array(y_axis[0][0:-1])>=np.array(y_borders_left[1][0:-1])),len(y_axis[0][0:-1]))
            print("sum(np.array(y_axis[0][0:-1])<=np.array(y_borders_left[0][0:-1])),len(y_axis[0][0:-1])",sum(np.array(y_axis[0][0:-1])<=np.array(y_borders_left[0][0:-1])),len(y_axis[0][0:-1]))
            print("sum(np.array(y_axis[1][0:-1])>=np.array(y_borders_right[1][0:-1])),len(y_axis[1][0:-1])",sum(np.array(y_axis[1][0:-1])>=np.array(y_borders_right[1][0:-1])),len(y_axis[1][0:-1]))
            print("sum(np.array(y_axis[1][0:-1])<=np.array(y_borders_right[0][0:-1])),len(y_axis[1][0:-1])",sum(np.array(y_axis[1][0:-1])<=np.array(y_borders_right[0][0:-1])),len(y_axis[1][0:-1]))
                  
            

        plt.show()
    
    def calculate_tuningParameters(self,point_index,pattern_name):
        data_plotPoint=self.data_forPredict.loc[self.data_forPredict['point_index'] == point_index]
           
        x_axis=[data_plotPoint["pressure_time_left"].values[0],
                data_plotPoint["pressure_time_right"].values[0]]
        y_axis=[data_plotPoint["pressure_measure_left"].values[0],
                data_plotPoint["pressure_measure_right"].values[0]]


            
        #left
        y_borders_left=[data_plotPoint[pattern_name].values[0]['left_top'],
                        data_plotPoint[pattern_name].values[0]['left_bottom']]
    
        # print(pattern_name)
        # print(len(x_axis[0]),len(y_axis[0]),len(y_borders_left[0]),len(y_borders_left[1]))
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

    
    
    def tuning(self):
        groundTruth_buildUp,groundTruth_drawDown=self.breakpoints_forLearn.values()
        detected_buildUp,detected_drawDown=self.detectedpoints.values()
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
        
        print("----self.fine_tuning",self.fine_tuning)
        
        
        
    def detect_breakpoint_type(self,pressure_measure:List[float],
                               pressure_time:List[float],
                               points:List[int]
                              )->(List[int],List[int]):
        """
        detect the break points are buildup or drawdown
        do the linear fitting,
        if slope_left>slope_right: drawdown
        else: buildup
        """
        print("detect_breakpoint.....")
        self.buildUp_or_drawDown=""
        data_inWindow=self.extract_points_inTimeWindow(pressure_measure,pressure_time,points,self.time_halfWindow_forPredict)
        parameters_allCurves=self.calculate_Parameters_allCurve(data_inWindow,fitting_type="linear")
        # self.parameters_allCurves
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

        
    def choose_fittingFunction(self,fitting_type):
        if fitting_type == "linear":
            fitting_func=linear_func_wrapper
        elif fitting_type == "polynomial":
            fitting_func=polyval_func_wrapper
        elif fitting_type == "log":
            fitting_func=log_func_wrapper
        else:
            print('fitting type must be "linear", "polynomial" or "log"')
        return fitting_func
    
    def produce_parameters_givenPoints(self,pressure_measure,pressure_time,points,time_halfWindow,loadedParameters_pattern,fitting_type):
        """
        extract the data of the given points in the timewindow
        and
        calculate the parameter for all curves fitting these points
        """
 
#         if filePath_loadPattern!=None:
#             self.load_pattern(filePath_loadPattern)
#             self.PatternLoaded=True
        # print("-----produce_parameters_givenPoints",fitting_type)
        if self.buildUp_or_drawDown!="":
            print(f"start to learn '{self.buildUp_or_drawDown}' pattern...")
        data_inWindow=self.extract_points_inTimeWindow(pressure_measure,pressure_time,points,time_halfWindow)
        parameters_allCurves=self.calculate_Parameters_allCurve(data_inWindow,fitting_type=fitting_type)
        return parameters_allCurves
        # parameters_pattern=self.calculate_parameters_pattern(fitting_type,loadedParameters_pattern)
        # return parameters_pattern
        
    def get_pattern(self,fitting_type,loadedParameters_pattern=None):
        for pattern_name in self.pattern_names:  
            self.buildUp_or_drawDown=pattern_name     
            parameters_pattern=self.calculate_parameters_pattern(fitting_type,loadedParameters_pattern)
            self.parameters_twoPatterns[pattern_name]=parameters_pattern
    
    def learn(self,pressure_measure,pressure_time,ground_truth,fitting_type,filePath_loadPattern=None):
        # print("-----------------learn",fitting_type)
        self.ground_truth = ground_truth
        ground_truth_buildUp,ground_truth_drawDown=self.detect_breakpoint_type(pressure_measure,
                                                                           pressure_time,
                                                                           ground_truth)
        print(f"------In the input ground truth: {len(ground_truth_buildUp)} points are detected as buildup, {len(ground_truth_drawDown)} points are detected as drawDown")
        self.breakpoints_forLearn={"buildUp":ground_truth_buildUp,
                                   "drawDown":ground_truth_drawDown}
        # self.breakpoints_forLearn_multipleLearn.append(self.breakpoints_forLearn)
        # print("----ground_truth_buildUp,ground_truth_drawDown",ground_truth_buildUp,ground_truth_drawDown)
#         if filePath_loadPattern!=None:
        loadedParameters_twoPattern=self.load_pattern(filePath_loadPattern)
#             self.PatternLoaded=True
        allPoints=[ground_truth_buildUp,ground_truth_drawDown]
#         buildUp_or_drawDown=["buildUp","drawDown"]
        for points,buildUp_or_drawDown,loadedParameters_pattern in zip(allPoints,
                                                          loadedParameters_twoPattern.keys(),
                                                          loadedParameters_twoPattern.values()):
            self.buildUp_or_drawDown=buildUp_or_drawDown
            self.parameters_allCurves_groundTruth[buildUp_or_drawDown]= self.produce_parameters_givenPoints(pressure_measure,
                                                                pressure_time,
                                                                points,
                                                                self.time_halfWindow_forLearn,
                                                                loadedParameters_pattern,
                                                                fitting_type)
        

    def sav_pattern(self):

        # data=identfication_UsePattern.parameters_pattern
        # data={key:list(value) for key,value in data.items()}
        data=self.parameters_twoPatterns
        with open(self.filePath_learnedPattern, 'w') as fp:
            json.dump(data, fp)

    def calculate_y_onBorders(self,x_left,x_right,fitting_type)->Dict[str,Dict[str,List[float]]]:
        """
        calculate y coordinates on borders coresponds to the input x coordinates
                self.parameters_twoPatterns={"buildUp":{"left_top":[],
                                               "left_bottom":[],
                                               "right_top":[],
                                               "right_bottom":[]},
                                    "drawDown":{"left_top":[],
                                               "left_bottom":[],
                                               "right_top":[],
                                               "right_bottom":[]}}
        """
        
     
                   
        y_borders_twoPattern=defaultdict(dict)
        x_left=np.array(x_left)
        x_right=np.array(x_right)
        xzip=[x_left,
            x_left,
            x_right,
            x_right]
        
        
        #todo      self.parameters_pattern 
        fitting_func=self.choose_fittingFunction(fitting_type)
        for pattern_name,parameters_pattern in self.parameters_twoPatterns.items():     
            for border_name, x, parameters in zip(self.border_names,xzip,parameters_pattern.values()):
                y_borders_twoPattern[pattern_name][border_name]=fitting_func(x, *parameters)
        
        return y_borders_twoPattern
    
    # def check_inPattern(self,
    #                     y_left:List[float],
    #                     y_right:List[float],
    #                     y_borders_twoPattern:Dict[str,Dict[str,List[float]]],
    #                     second_order_derivative:float
    #                    )->str:
    #     """
    #     check if all points besides the are fall into the field defined by the borders
    #     do not include the point for identification 
    #     """
    #     for pattern_name,y_borders in y_borders_twoPattern.items():
    #         # if (all(np.array(y_left[0:-1])>=np.array(y_borders["left_bottom"][0:-1])) and 
    #         #     all(np.array(y_left[0:-1])<=np.array(y_borders["left_top"][0:-1])) and
    #         #     all(np.array(y_right[0:-1])>=np.array(y_borders["right_bottom"][0:-1])) and 
    #         #     all(np.array(y_right[0:-1])<=np.array(y_borders["right_top"][0:-1]))):
    #         # if (sum(np.array(y_left[0:-1])>=np.array(y_borders["left_bottom"][0:-1]))>0.8*len(y_left[0:-1]) and 
    #         #     sum(np.array(y_left[0:-1])<=np.array(y_borders["left_top"][0:-1]))>0.8*len(y_left[0:-1]) and
    #         #     sum(np.array(y_right[0:-1])>=np.array(y_borders["right_bottom"][0:-1]))>0.8*len(y_right[0:-1]) and 
    #         #     sum(np.array(y_right[0:-1])<=np.array(y_borders["right_top"][0:-1]))>0.8*len(y_right[0:-1]) and 
    #         #     ((pattern_name==self.pattern_names[0] and second_order_derivative>0.05*self.std_2) 
    #         #      or 
    #         #      (pattern_name==self.pattern_names[1] and second_order_derivative<0.05*self.std_2)) ):
    #         if (sum(np.array(y_left[0:-1])>=np.array(y_borders["left_bottom"][0:-1]))>0.8*len(y_left[0:-1]) and 
    #             sum(np.array(y_left[0:-1])<=np.array(y_borders["left_top"][0:-1]))>0.8*len(y_left[0:-1]) and
    #             sum(np.array(y_right[0:-1])>=np.array(y_borders["right_bottom"][0:-1]))>0.8*len(y_right[0:-1]) and 
    #             sum(np.array(y_right[0:-1])<=np.array(y_borders["right_top"][0:-1]))>0.8*len(y_right[0:-1]) ):
                
            
    #         ##conbine with tangent prediction               
    #         # if (all(np.array(y_left[0:-1])>=np.array(y_borders["left_bottom"][0:-1])) and 
    #         #     all(np.array(y_left[0:-1])<=np.array(y_borders["left_top"][0:-1])) and
    #         #     all(np.array(y_right[0:-1])>=np.array(y_borders["right_bottom"][0:-1])) and 
    #         #     all(np.array(y_right[0:-1])<=np.array(y_borders["right_top"][0:-1])) and 
    #         #     ((pattern_name==self.pattern_names[0] and second_order_derivative>0.05*self.std_2) 
    #         #      or 
    #         #      (pattern_name==self.pattern_names[1] and second_order_derivative<0.05*self.std_2))):

    #             return pattern_name
            
    #         return "Not breakpoint"
        
    def predict(self,pressure_measure,pressure_time,second_order_derivative,mode="whole_dataset",fitting_type="polynomial"):
        self.std_2=statistics.stdev(second_order_derivative)
        self.data_forPredict=pd.DataFrame(columns=["pressure_time_left",
                                            "pressure_measure_left",
                                            "pressure_time_right",
                                            "pressure_measure_right",
                                            "left_top",
                                            "left_bottom",
                                            "right_top",
                                            "right_bottom"])
        
        # #point window
        # points=[point_index for point_index in range(self.point_halfWindow,len(pressure_measure)-self.point_halfWindow)]
        # self.extract_points_inTimeWindow(pressure_measure,pressure_time,points)    
        #time window
        if mode=="whole_dataset":   
            points=[point_index for point_index in range(len(pressure_measure))]
        elif mode=="refine_detection":
            points=[]
            for pattern_name, detected_points in self.detectedpoints.items():
                points.extend(detected_points)
                print("pattern name",pattern_name, len(detected_points))
                self.points_Detectedin1=[point for point in self.breakpoints_forLearn[pattern_name] if point in detected_points]
                print(f"---------there are {len(self.points_Detectedin1)} points_Detectedin1")
            print(f"---------there are {len(points)} inputted to be second prediction")
        else:
            print("check the mode, it must be 'whole_dataset' or 'refine_detection'...")
            
        borderData=pd.DataFrame()
        data_inWindow=self.extract_points_inTimeWindow(pressure_measure,pressure_time,points,self.time_halfWindow_forPredict)
        for index,curveData_singlePoint in data_inWindow.iterrows():  
            
            
            curveData_singlePoint["pressure_time_right"].reverse()
            curveData_singlePoint["pressure_measure_right"].reverse()
        
            y_borders_twoPattern=self.calculate_y_onBorders(curveData_singlePoint["pressure_time_left"],curveData_singlePoint["pressure_time_right"],fitting_type)
            borderData=borderData.append(y_borders_twoPattern,ignore_index=True)
#             display("-----y_borders-----",y_borders)
#             display("-----data_inWindow----:",data_inWindow)
         
        self.data_forPredict=pd.concat([data_inWindow, borderData], axis=1)  
   
        detectedpoints={"buildUp":[],
                        "drawDown":[]}
        for point_index in points:
            pattern_name=self.check_in_pattern(self.data_forPredict,point_index)  
            if pattern_name==None:
                continue
            else:
                detectedpoints[pattern_name].append(point_index)
                
        self.detectedpoints=detectedpoints
        
        builup, drawdown=self.detectedpoints.values()
        print(f"----after second prediction, before refine, there are {len(builup)} detected buildup points, {len(drawdown)} drawdown detected")
  
        
        detectedpoints_buildUp,detectedpoints_drawDown=self.refine_detectedPoints(pressure_measure,pressure_time)
        self.detectedpoints={"buildUp":detectedpoints_buildUp,
                        "drawDown":detectedpoints_drawDown}
        return detectedpoints_buildUp,detectedpoints_drawDown
            
            
        
        
#     def predict(self,pressure_measure,pressure_time,second_order_derivative,mode="whole_dataset",fitting_type="polynomial"):
#         #store border for every point
# #         self.borderData=pd.DataFrame(columns=self.border_names)
#         # self.get_pattern(fitting_type)
#         self.std_2=statistics.stdev(second_order_derivative)
#         self.data_forPredict=pd.DataFrame(columns=["pressure_time_left",
#                                             "pressure_measure_left",
#                                             "pressure_time_right",
#                                             "pressure_measure_right",
#                                             "left_top",
#                                             "left_bottom",
#                                             "right_top",
#                                             "right_bottom"])
#         borderData=pd.DataFrame()
#         detectedpoints_buildUp=[]
#         detectedpoints_drawDown=[]
#         # #point window
#         # points=[point_index for point_index in range(self.point_halfWindow,len(pressure_measure)-self.point_halfWindow)]
#         # self.extract_points_inTimeWindow(pressure_measure,pressure_time,points)
        
#         #time window
#         if mode=="whole_dataset":   
#             points=[point_index for point_index in range(len(pressure_measure))]
#         elif mode=="refine_detection":
#             points=[]
#             for detected_points in self.detectedpoints.values():
#                 points.extend(detected_points)
#             print(f"---------there are {len(points)} inputted to be second prediction")
#         else:
#             print("check the mode, it must be 'whole_dataset' or 'refine_detection'...")
            
       
#         data_inWindow=self.extract_points_inTimeWindow(pressure_measure,pressure_time,points,self.time_halfWindow_forPredict)

#         for index,curveData_singlePoint in data_inWindow.iterrows():  
            
            
#             curveData_singlePoint["pressure_time_right"].reverse()
#             curveData_singlePoint["pressure_measure_right"].reverse()
        
#             y_borders_twoPattern=self.calculate_y_onBorders(curveData_singlePoint["pressure_time_left"],curveData_singlePoint["pressure_time_right"],fitting_type)
#             borderData=borderData.append(y_borders_twoPattern,ignore_index=True)
# #             display("-----y_borders-----",y_borders)
# #             display("-----data_inWindow----:",data_inWindow)
#             point_type=self.check_inPattern(curveData_singlePoint["pressure_measure_left"],
#                               curveData_singlePoint["pressure_measure_right"],
#                               y_borders_twoPattern,
#                               second_order_derivative[curveData_singlePoint["point_index"]])
                
#             if(point_type=="buildUp"):
#                 detectedpoints_buildUp.append(curveData_singlePoint["point_index"])
#             elif(point_type=="drawDown"):
#                 detectedpoints_drawDown.append(curveData_singlePoint["point_index"])

        
#         self.detectedpoints={"buildUp":detectedpoints_buildUp,
#                              "drawDown":detectedpoints_drawDown}
#         self.data_forPredict=pd.concat([data_inWindow, borderData], axis=1)
#         print("----after second prediction, before refine",self.detectedpoints)
            
#         detectedpoints_buildUp,detectedpoints_drawDown=self.refine_detectedPoints(pressure_measure,pressure_time)
#         self.detectedpoints={"buildUp":detectedpoints_buildUp,
#                         "drawDown":detectedpoints_drawDown}
#         return detectedpoints_buildUp,detectedpoints_drawDown
    
    def get_tangent(self,parameters_allCurves:pd.DataFrame,fitting_type:str)->pd.DataFrame:
        #for polynomial fit the third parameter is the tangent
        if fitting_type=="polynomial":
            n=2
        elif fitting_type=="linear":
            n=0 
        else:  
            print("please check fitting type, need to be 'polynomial' or 'linear'")
        left_tangent=[parameters[n] for parameters in parameters_allCurves["left_curves_parameters"]] 
        right_tangent=[parameters[n] for parameters in parameters_allCurves["right_curves_parameters"]] 
        tangent_df=pd.DataFrame({"point_index":parameters_allCurves["point_index"],"left_tangent":left_tangent,"right_tangent":right_tangent})
        return tangent_df
    
    def truncateList_byPercentile(self,
                                  tempList:List[float],
                                  percentile_upperBound:float,
                                  percentile_lowerBound:float)->(float,float):
            upper_bound=np.percentile(tempList, percentile_upperBound, axis=0,method="normal_unbiased")
            lower_bound=np.percentile(tempList, percentile_lowerBound, axis=0,method="normal_unbiased")
            list_truncated=[item for item in tempList if item>=lower_bound and item<=upper_bound]
            list_max=max(list_truncated)
            list_min=min(list_truncated)
            return list_max,list_min
    
    def get_tangents_twoPatterns(self,fitting_type):
    
        for pattern_name in self.pattern_names:
            tangent_groundTruth_buildUpOrDrawDown=self.get_tangent(self.parameters_allCurves_groundTruth[pattern_name],fitting_type)
            #left side
            self.tangents_twoPatterns[pattern_name]["left_top"]=max(tangent_groundTruth_buildUpOrDrawDown["left_tangent"])
            self.tangents_twoPatterns[pattern_name]["left_bottom"]=min(tangent_groundTruth_buildUpOrDrawDown["left_tangent"])
            
            # percentile_upperBound,percentile_lowerBound=self.percentile_tuning[pattern_name]["left"]
            # self.tangents_twoPatterns[pattern_name]["left_top"],self.tangents_twoPatterns[pattern_name]["left_bottom"]=self.truncateList_byPercentile(
            #                                                         tangent_groundTruth_buildUpOrDrawDown["left_tangent"],
            #                                                         percentile_upperBound,
            #                                                         percentile_lowerBound)
            #right side
            self.tangents_twoPatterns[pattern_name]["right_top"]=max(tangent_groundTruth_buildUpOrDrawDown["right_tangent"])
            self.tangents_twoPatterns[pattern_name]["right_bottom"]=min(tangent_groundTruth_buildUpOrDrawDown["right_tangent"])
            # percentile_upperBound,percentile_lowerBound=self.percentile_tuning[pattern_name]["right"]
            # self.tangents_twoPatterns[pattern_name]["right_top"],self.tangents_twoPatterns[pattern_name]["right_bottom"]=self.truncateList_byPercentile(
            #                                                         tangent_groundTruth_buildUpOrDrawDown["right_tangent"],
            #                                                         percentile_upperBound,
            #                                                         percentile_lowerBound)

    
    def predict_usingTangent(self,pressure_measure,pressure_time,mode="whole_dataset",fitting_type="polynomial"):
        # self.get_tangents_twoPatterns(fitting_type)
        self.buildUp_or_drawDown=""
        if mode=="whole_dataset":   
            points=[point_index for point_index in range(len(pressure_measure))]
        elif mode=="refine_detection":
            points=[]
            for detected_points in self.detectedpoints.values():
                points.extend(detected_points)
        else:
            print("check the mode, it must be 'whole_dataset' or 'refine_detection'...")
        #calculate the parameters for points to be predicted
        parameters_allCurves=self.produce_parameters_givenPoints(pressure_measure,
                                                                 pressure_time,
                                                                 points,
                                                                 self.time_halfWindow_forPredict,
                                                                 None,
                                                                 fitting_type)
          
        tangent_forPredict=self.get_tangent(parameters_allCurves,fitting_type)
        
        self.tangent_forPredict=tangent_forPredict
        # print("---------==========",self.pattern_names)
        for pattern_name in self.pattern_names:

            sub_df=tangent_forPredict.loc[(tangent_forPredict["left_tangent"]>=self.tangents_twoPatterns[pattern_name]["left_bottom"]) &
                                    (tangent_forPredict["left_tangent"]<=self.tangents_twoPatterns[pattern_name]["left_top"]) &
                                    (tangent_forPredict["right_tangent"]>=self.tangents_twoPatterns[pattern_name]["right_bottom"]) &
                                    (tangent_forPredict["right_tangent"]<=self.tangents_twoPatterns[pattern_name]["right_top"])]

            # print("----pattern_name,sub_df",pattern_name,sub_df)
            self.detectedpoints[pattern_name]=list(sub_df["point_index"])
            # print("self.detectedpoints",self.detectedpoints)
        buildUp, drawDown=self.refine_detectedPoints(pressure_measure,pressure_time)
        self.detectedpoints={"buildUp":buildUp,
                    "drawDown":drawDown}
        return buildUp, drawDown
    
    def refine_detectedPoints(self,pressure_measure,pressure_time):
        points=[]
        for detected_points in self.detectedpoints.values():
            points.extend(detected_points)
        buildUp, drawDown=self.detect_breakpoint_type(pressure_measure,pressure_time,points)
        return buildUp, drawDown
            
        
            
        
      