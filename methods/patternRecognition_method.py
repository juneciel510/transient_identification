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

# from plot import plot_histogram
from base_classes import CurveParametersCalc,SaveNLoad




class PatternRecognitionMethod(CurveParametersCalc,SaveNLoad):
    """
    A class to learn pattern from ground_truth of PDG pressure data.
    And predict the buildUp and drawDown points for a given dataset.
    Args:
    
    """
    
    def __init__(self, 
                 min_pointsNumber:int=8,
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
        
        CurveParametersCalc.__init__(self)
    
        # self.breakpoints_forLearn_multipleLearn=[]
        self.min_pointsNumber=min_pointsNumber
        self.percentile_tuning=percentile_tuning
        self.fine_tuning=fine_tuning
        self.filePath_learnedPattern=filePath_learnedPattern

        
        self.border_names=["left_top","left_bottom","right_top","right_bottom"]

        self.data_forPredict=pd.DataFrame(columns=["point_index",
                                                   "pressure_time_left",
                                                    "pressure_measure_left",
                                                    "pressure_time_right",
                                                    "pressure_measure_right",
                                                    "buildUp",
                                                    "drawDown"])


        #to store patterns for buildup or drawdown
        self.parameters_twoPatterns={"buildUp":{"left_top":[],
                                               "left_bottom":[],
                                               "right_top":[],
                                               "right_bottom":[]},
                                    "drawDown":{"left_top":[],
                                               "left_bottom":[],
                                               "right_top":[],
                                               "right_bottom":[]}}
    
            
        self.for_tuning={"buildUp":{"left_top":[],
                                    "left_bottom":[],
                                    "right_top":[],
                                    "right_bottom":[]},
                        "drawDown":{"left_top":[],
                                    "left_bottom":[],
                                    "right_top":[],
                                    "right_bottom":[]}}
    
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
        parameters_half_pattern["top"]=self.fit_curve(x,y_allCurve_max,fitting_type,polynomial_order,plot_color)
        parameters_half_pattern["bottom"]=self.fit_curve(x,y_allCurve_min,fitting_type,polynomial_order,plot_color)
        
        return parameters_half_pattern
            
    def get_borders_onePattern(self,
                                     time_halfWindow_forLearn:float,
                                     fitting_type:str,
                                    parameters_allCurves_groundTruth:Dict[str,pd.DataFrame],
                                    buildUp_or_drawDown:str
                                    )->Dict[str,List[float]]:
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
        parameters_allCurves=parameters_allCurves_groundTruth[buildUp_or_drawDown]
            
        number=8
        y_left_allCurve = np.empty((0, number), float)
        y_right_allCurve = np.empty((0, number), float)
        x_left=np.linspace(start = -time_halfWindow_forLearn, stop = 0, num = number)
        x_right=np.linspace(start = 0, stop = time_halfWindow_forLearn, num = number)
    
        fig=plt.figure(figsize = (20, 10))
        # plt.xlim([-self.time_halfWindow_forPredict,self.time_halfWindow_forPredict])
        if buildUp_or_drawDown=="buildUp":
            plt.ylim([-50, 280])
        else:
            plt.ylim([-400, 100])
            
        curve_number=len(parameters_allCurves)
        
        print(f"-----start to calculate'{buildUp_or_drawDown}' pattern parameter, there are {curve_number} for calculation" )
        for i in range(curve_number):
            y_left=fitting_func(x_left, *parameters_allCurves["left_curves_parameters"][i])
            y_right=fitting_func(x_right, *parameters_allCurves["right_curves_parameters"][i])
            
            y_left_allCurve=np.append(y_left_allCurve,np.array([y_left]), axis=0)
            y_right_allCurve=np.append(y_right_allCurve,np.array([y_right]), axis=0)
                  
           
        for left_or_right, x,y_allCurve in zip(["left","right"],
                                                [x_left,x_right],
                                                [y_left_allCurve,y_right_allCurve]):
            percentile_upperBound,percentile_lowerBound=self.percentile_tuning[buildUp_or_drawDown][left_or_right]
            fine_tuning_max=self.fine_tuning[buildUp_or_drawDown][left_or_right+"_top"]
            fine_tuning_min=self.fine_tuning[buildUp_or_drawDown][left_or_right+"_bottom"]
            left_parameters_pattern=self.find_border(x,y_allCurve,fitting_type,percentile_upperBound,percentile_lowerBound,fine_tuning_max,fine_tuning_min)
        
            parameters_pattern[left_or_right+"_top"]=left_parameters_pattern["top"]
            parameters_pattern[left_or_right+"_bottom"]=left_parameters_pattern["bottom"]
            
        
        self.legend_unique_labels(fig)
        return parameters_pattern

    def legend_unique_labels(self,figure):
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        figure.legend(by_label.values(), by_label.keys(),shadow=True, fontsize='large')


    def get_borders_twoPattern(self,
                    parameters_allCurves_groundTruth:Dict[str,pd.DataFrame],
                    time_halfWindow_forLearn:float,
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
        
        parameters_twoPatterns={}
        
        for pattern_name in self.pattern_names:    
            parameters_pattern=self.get_borders_onePattern(time_halfWindow_forLearn,
                                                                 fitting_type,
                                                                parameters_allCurves_groundTruth,
                                                                pattern_name)
    
            parameters_twoPatterns[pattern_name]=parameters_pattern
            
        self.parameters_twoPatterns=parameters_twoPatterns
        print("pattern learned:",parameters_twoPatterns)
    
         
    
    
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
                  
     
    def plot_in_pattern(self, ax, x_axis, y_axis,y_borders,pattern_name,point_index):
        ax.set(title=f"{pattern_name}, point_index:{point_index}")
        ax.plot(x_axis, y_borders[0],"k")
        ax.plot(x_axis, y_borders[1], "k")
        ax.scatter(x_axis,y_axis,s=2**2)
        print(f"--in {pattern_name} pattern")
        points_aboveTop=sum(y_axis>y_borders[0])
        points_belowBottom=sum(y_axis<y_borders[1])     
        print(f"--{len(x_axis)} points for comparison, {points_aboveTop} points are above top, {points_belowBottom} points under bottom")
        
        
    
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
    
    def learn(self,
            pressure_measure:List[float],
            pressure_time:List[float],
            ground_truth:List[int],
            time_halfWindow_forLearn,
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
        parameters_allCurves_groundTruth=self.calculate_Parameters_allCurve_groundTruth(
                                                  pressure_measure,
                                                  pressure_time,
                                                  ground_truth,
                                                  time_halfWindow_forLearn,
                                                  self.min_pointsNumber,
                                                  fitting_type
                                                  )
        
        self.get_borders_twoPattern(parameters_allCurves_groundTruth,
                    time_halfWindow_forLearn,
                    fitting_type,
                    loadedParameters_pattern=None)
    
    
    
    def predict(self,
                pressure_measure:List[float],
                pressure_time:List[float],
                points:List[int],
                time_halfWindow:float=None,
                point_halfWindow:int=None,
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

        self.data_forPredict=pd.DataFrame(columns=["point_index",
                                                   "pressure_time_left",
                                                    "pressure_measure_left",
                                                    "pressure_time_right",
                                                    "pressure_measure_right",
                                                    "buildUp",
                                                    "drawDown"])
            
        borderData=pd.DataFrame()
        data_inWindow=self.extract_points_inWindow(pressure_measure,
                                                            pressure_time,
                                                            points,
                                                            time_halfWindow=time_halfWindow,
                                                            point_halfWindow=point_halfWindow,
                                                            min_pointsNumber=self.min_pointsNumber
                                                            )
        # display(data_inWindow)
        for index,curveData_singlePoint in data_inWindow.iterrows():     
            curveData_singlePoint["pressure_time_right"].reverse()
            curveData_singlePoint["pressure_measure_right"].reverse()
        
            y_borders_twoPattern=self.calculate_y_onBorders(curveData_singlePoint["pressure_time_left"],
                                                            curveData_singlePoint["pressure_time_right"],
                                                            fitting_type)
            borderData=borderData.append(y_borders_twoPattern,
                                         ignore_index=True)
         
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
  
        print(f"{len(points_buildUp)} buildup points detected, {len(points_drawDown)} drawdown points detected")
        return points_buildUp,points_drawDown
            
    
        
      