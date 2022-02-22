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
def test_func(x, a,b,c,d):
    y = a+b*x-c*np.exp(-d*x)
    return y
def test_func(x, a,b,c,d):
    y = a+b*x-c*np.exp(d*x)
    return y
def test_func(x, a,b,c,d):
    y = a+c*np.exp(d*x)
    return y

# def polyval_func_wrapper(x, parameters):
#     y = np.polyval(parameters,x)
#     return y
def polyval_func_wrapper(x, *parameters):
    y = np.polyval(parameters,x)
    return y
def linear_func_wrapper(x, a,b):
    # print("---linear-----")
    y = a*x+b
    return y
def log_func_wrapper(x,a,b,c,d):
    print("---log-----")
    y=a*np.log(x)+d+b*x+c*x*x
    return y

# def fitting_func_wapper(x, parameters):
#     a,b,c=parameters
#     y = a+b*x+c*x*x
#     return y

class PatternRecognition:
    def __init__(self, 
                 point_halfWindow:int=3,
                 time_halfWindow_forPredict:float=0.5,
                 time_halfWindow_forLearn:float=1,
#                  fitting_func=None,
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
                
#                  filePath_learnedPattern="../data_output/Learned_Pattern.csv",
                 filePath_learnedPattern="../data_output/Learned_Pattern.jason",
                ):
        
        #to store the input groundtruth, including buildup & drawdown together
        self.ground_truth =[]
        #to store the points for learn, classified as buildup & drawdown
        self.breakpoints_forLearn=defaultdict(list)
        self.breakpoints_forLearn_multipleLearn=[]
        #to store the points predicted 
        self.detectedpoints=defaultdict(list)
        self.point_halfWindow=point_halfWindow
        #time window for learn
        self.time_halfWindow_forLearn=time_halfWindow_forLearn
        #time window for predict
        self.time_halfWindow_forPredict=time_halfWindow_forPredict
        self.fitting_func=None
        self.percentile_tuning=percentile_tuning
        self.fine_tuning=fine_tuning
        self.filePath_learnedPattern=filePath_learnedPattern
#         self.filepath_curveDataLeft=filepath_curveDataLeft
#         self.filepath_curveDataRight=filepath_curveDataRight
        
        self.std_2=None
        self.buildUp_or_drawDown=""
        self.pattern_names=["buildUp","drawDown"]
        self.border_names=["left_top","left_bottom","right_top","right_bottom"]
        self.curveData=pd.DataFrame(columns=['point_index',
                                             'pressure_time_left', 
                                             'pressure_measure_left',
                                             'pressure_time_right', 
                                             'pressure_measure_right'])
        self.borderData=pd.DataFrame(columns=self.border_names)
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
#         self.parameters_pattern={}
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
        

#         self.detectedpoints_buildUp=[]
#         self.detectedpoints_drawDown=[]

#         self.x_leftPlot=[]
#         self.x_rightPlot=[]
#         self.patternLoaded=False
    
    
    def load_pattern(self,filePath_loadPattern):
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
    
        
    def extract_points_inPointWindow(self,pressure_measure,pressure_time,points): 
        """
        extract pressure measure & time data for all points 
        in window  [point_index-point_halfWindow,point_index+point_halfWindow]
        """
        self.curveData=pd.DataFrame(columns=['pressure_time_left', 
                                             'pressure_measure_left',
                                             'pressure_time_right', 
                                             'pressure_measure_right'])
        
        for point_index in points:
            #left   
            sub_measure=pressure_measure[point_index+1-self.point_halfWindow:point_index+1]
            sub_time=pressure_time[point_index+1-self.point_halfWindow:point_index+1]
            curve_pressure=[round(measure-sub_measure[-1],6) for measure in sub_measure]
            curve_time=[round(time-sub_time[-1],6) for time in sub_time]
            data={"pressure_time_left":curve_time,"pressure_measure_left":curve_pressure}
            # data={"pressure_time_left":np.asarray(curve_time),"pressure_measure_left":curve_pressure}
        
            
            #right
            sub_measure=pressure_measure[point_index:point_index+self.point_halfWindow]
            sub_time=pressure_time[point_index:point_index+self.point_halfWindow]
            curve_pressure=[round(measure-sub_measure[0],6) for measure in sub_measure]
            curve_time=[round(time-sub_time[0],6) for time in sub_time]
            data.update({"pressure_time_right":curve_time,
                         "pressure_measure_right":curve_pressure})
            # data.update({"pressure_time_right":np.asarray(curve_time),
            #              "pressure_measure_right":curve_pressure})
            
            
            self.curveData=self.curveData.append(data,ignore_index=True)
       
#         self.curveDataLeft.to_csv(self.filepath_curveDataLeft,index=False,float_format='%.4f',sep='\t')
#         self.curveDataRight.to_csv(self.filepath_curveDataRight,index=False,float_format='%.4f',sep='\t') 

    def extract_points_inTimeWindow(self,pressure_measure,pressure_time,points,time_halfWindow): 
            """
            extract pressure measure & time data for all points 
            in timewindow 
            """
            print("-------start to extract_points_inTimeWindow, empty 'self.curveData'")
            self.curveData=pd.DataFrame(columns=['point_index',
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
                
                    self.curveData=self.curveData.append(data,ignore_index=True)
                
    def extract_singlePoint_inPointWindow(self,pressure_measure,pressure_time,point_index,halfWinow_left,halfWinow_right): 
            """
            extract pressure measure & time data for a single point
            in window  [point_index-point_halfWindow,point_index+point_halfWindow]
            """
            #store point_index
            data={"point_index":int(point_index)}
            
            #left
            sub_measure=pressure_measure[point_index+1-halfWinow_left:point_index+1]
            sub_time=pressure_time[point_index+1-halfWinow_left:point_index+1]
            curve_pressure=[round(measure-sub_measure[-1],6) for measure in sub_measure]
            curve_time=[round(time-sub_time[-1],6) for time in sub_time]
            data.update({"pressure_time_left":curve_time,"pressure_measure_left":curve_pressure})
            
            #right
            sub_measure=pressure_measure[point_index:point_index+halfWinow_right]
            sub_time=pressure_time[point_index:point_index+halfWinow_right]
            curve_pressure=[round(measure-sub_measure[0],6) for measure in sub_measure]
            curve_time=[round(time-sub_time[0],6) for time in sub_time]
            data.update({"pressure_time_right":curve_time,
                        "pressure_measure_right":curve_pressure})
            return data
             
                
                
             

    def fit_curve(self,xdata,ydata,color,fitting_type,plot_title=""):
        # print("---------------fit_curve",fitting_type)
        x = np.asarray(xdata)
        y = np.asarray(ydata)
        
#         parameters, covariance = curve_fit(self.fitting_func, x, y)
#         y_fit = self.fitting_func(x, *parameters)
        
        if fitting_type=="polynomial":
            parameters=np.polyfit(x,y,3)
        if fitting_type=="linear" or fitting_type=="log":
            parameters, covariance = curve_fit(self.fitting_func, x, y)

        y_fit=self.fitting_func(x, *parameters)
        # if fitting_type=="linear":
        #     plt.figure(figsize = (10, 10))
        #     plt.axis([-1, 1, -10, 10])
        # plt.plot(x, y,color=color,  marker='o')
        if self.buildUp_or_drawDown!="":
            plt.scatter(x=x,y=y,color=color)
            plt.plot(x, y_fit, color=color,linestyle='-')
            plt.title(plot_title)
        # if fitting_type=="log" or fitting_type=="polynomial":
        #     plt.show()
        return parameters
    
    def calculate_Parameters_allCurve(self,fitting_type,plot_title=""):
        print("---------------calculate_Parameters_allCurve",fitting_type)
        curveLeft_parameters=[]
        curveRight_parameters=[]
        # self.parameters_allCurves=pd.DataFrame()
        parameters_allCurves=pd.DataFrame()
        
        plt.figure(figsize = (20, 10))

        for i in range(len(self.curveData)):
            #left side
                        
#             print("-----------i---------",i)
            xdata=self.curveData["pressure_time_left"][i]
            ydata=self.curveData["pressure_measure_left"][i]
#             print(xdata,ydata)
            color="yellow"  
            #when detect_breakpoint_type invoke this funtion,
            # set the title to be the breakpoint index
            if self.buildUp_or_drawDown!="":
                if fitting_type=="linear":
                    plot_title=f"{self.ground_truth[i]}"      
                if fitting_type=="polynomial" or fitting_type=="log":
                    plot_title=f"{self.buildUp_or_drawDown}---{self.ground_truth[i]}"   
            curveLeft_parameters.append(self.fit_curve(xdata,ydata,color,fitting_type,plot_title))

            
            #right side
            xdata=self.curveData["pressure_time_right"][i]
            ydata=self.curveData["pressure_measure_right"][i]
            if i==0:
                xRight_min=abs(xdata[-1])
                self.x_rightPlot=xdata
            if i>0 and abs(xdata[-1])<xRight_min:
                xRight_min=abs(xdata[-1])
                self.x_rightPlot=xdata
                
            color="blue" 
            curveRight_parameters.append(self.fit_curve(xdata,ydata,color,fitting_type,plot_title))
            
        parameters_allCurves=pd.DataFrame({"point_index":self.curveData["point_index"].values,
                                           "left_curves_parameters":curveLeft_parameters,
                                           "right_curves_parameters":curveRight_parameters})
        print("--------------self.buildUp_or_drawDown", self.buildUp_or_drawDown)
        # print("parameters_allCurves",parameters_allCurves)
        # if self.buildUp_or_drawDown!="":
        #     self.parameters_allCurves_groundTruth[self.buildUp_or_drawDown]=self.parameters_allCurves_groundTruth[self.buildUp_or_drawDown].append(parameters_allCurves, ignore_index = True)
        # print("-----------self.parameters_allCurves_groundTruth",self.parameters_allCurves_groundTruth)
        return parameters_allCurves
        
            
    def calculate_parameters_pattern(self,fitting_type,loadedParameters_pattern):
        """
        calculate the parameters of four border curves
        -------------
        "left_top"
        "left_bottom"
        "right_top"
        "right_bottom"
        -------------
        of buildup/drawdown pattern
        """
#         y_left_allCurve = np.empty((0, len(self.x_leftPlot)), float)
#         y_right_allCurve = np.empty((0, len(self.x_rightPlot)), float)
  
#         #left
#         x_left=np.asarray(self.x_leftPlot)
#         x_right=np.asarray(self.x_rightPlot)
        parameters_pattern={}
        
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
            y_left=self.fitting_func(x_left, *parameters_allCurves["left_curves_parameters"][i])
            y_right=self.fitting_func(x_right, *parameters_allCurves["right_curves_parameters"][i])
            
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


    # def find_border(self,x,y_allCurve,fitting_type,plot_title=""):
    #     """
    #     calculate parameters of top curve and buttom curve 
    #     for left or right side of buildup or drawndown pattern
    #     """
    #     parameters_half_pattern={}
        
    #     y_allCurve_max=y_allCurve.max(axis=0)
    #     color="green"
    #     parameters_half_pattern["top"]=self.fit_curve(x,y_allCurve_max,color,fitting_type,plot_title)
     
    #     y_allCurve_min=y_allCurve.min(axis=0)
    #     parameters_half_pattern["bottom"]=self.fit_curve(x,y_allCurve_min,color,fitting_type,plot_title)
        
    #     return parameters_half_pattern
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

        color="green"
        
        # # y_allCurve_max=y_allCurve.max(axis=0)
        # y_allCurve_max=np.percentile(y_allCurve, 90, axis=0,method="normal_unbiased")
        # # y_allCurve_min=y_allCurve.min(axis=0)
        # y_allCurve_min=np.percentile(y_allCurve, 10, axis=0,method="normal_unbiased")
        
        y_allCurve_min,y_allCurve_max=self.truncate_byPercentile(y_allCurve,percentile_upperBound,percentile_lowerBound)
        y_allCurve_min=y_allCurve_min*fine_tuning_min
        y_allCurve_max=y_allCurve_max*fine_tuning_max
        # plt.figure(figsize = (20, 10))
        parameters_half_pattern["top"]=self.fit_curve(x,y_allCurve_max,color,fitting_type,plot_title)
     

        parameters_half_pattern["bottom"]=self.fit_curve(x,y_allCurve_min,color,fitting_type,plot_title)
        
        return parameters_half_pattern
    
        
    def plot_data_forPredict(self,point_index):
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
            # print(pattern_name)
            # print(len(x_axis[0]),len(y_axis[0]),len(y_borders_left[0]),len(y_borders_left[1]))
                
        
            #right
            y_borders_right=[data_plotPoint[pattern_name].values[0]['right_top'],
                            data_plotPoint[pattern_name].values[0]['right_bottom']]
            ax.plot(x_axis[1], y_borders_right[0], "k")
            ax.plot(x_axis[1], y_borders_right[1], "k")
            ax.scatter(x_axis[1],y_axis[1],s=2**2)
            

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
        self.fitting_func=linear_func_wrapper
        self.extract_points_inTimeWindow(pressure_measure,pressure_time,points,self.time_halfWindow_forPredict)
        parameters_allCurves=self.calculate_Parameters_allCurve(fitting_type="linear")
        # self.parameters_allCurves
        breakpoint_buildUp=[]
        breakpoint_drawDown=[]
        for index,parameter in parameters_allCurves.iterrows():
            #compare slope and convert index in the parameters_allCurves to in pressure_df
            if parameter["left_curves_parameters"][0]<parameter["right_curves_parameters"][0]:
                if parameter["left_curves_parameters"][0]<0 and parameter["right_curves_parameters"][0]<0:
                    print(f"===============\n"
                          f"point '{points[index]}' is going to be classified as buildup,\n"
                          f"however the ground truth should be drawdown,\n"
                          f"there must be a bump in window thus should not be included into breakpoints for learning\n"
                          f"===============")
                else:
                    breakpoint_buildUp.append(points[index])
                
                
            if parameter["left_curves_parameters"][0]>parameter["right_curves_parameters"][0]:
                if parameter["left_curves_parameters"][0]>0 and parameter["right_curves_parameters"][0]>0:
                    print(f"===============\n"
                          f"point '{points[index]}' is going to be classified as drawdown,\n"
                          f"however the ground truth should be biuldup,\n"
                          f"there must be a cavity in window thus should not be included into breakpoints for learning\n"
                          f"===============")
                else:
                    breakpoint_drawDown.append(points[index])
                    
        self.breakpoints_forLearn={"buildUp":breakpoint_buildUp,
                                   "drawDown":breakpoint_drawDown}
        return breakpoint_buildUp,breakpoint_drawDown

        
    
    def produce_parameters_givenPoints(self,pressure_measure,pressure_time,points,time_halfWindow,loadedParameters_pattern,fitting_type):
        """
        extract the data of the given points in the timewindow
        and
        calculate the parameter for all curves fitting these points
        """
 
        # self.fitting_func=polyval_func_wrapper
#         if filePath_loadPattern!=None:
#             self.load_pattern(filePath_loadPattern)
#             self.PatternLoaded=True
        # print("-----produce_parameters_givenPoints",fitting_type)
        if fitting_type == "linear":
            self.fitting_func=linear_func_wrapper
        if fitting_type == "polynomial":
            self.fitting_func=polyval_func_wrapper
        if fitting_type == "log":
            self.fitting_func=log_func_wrapper
        if self.buildUp_or_drawDown!="":
            print(f"start to learn '{self.buildUp_or_drawDown}' pattern...")
        self.extract_points_inTimeWindow(pressure_measure,pressure_time,points,time_halfWindow)
        parameters_allCurves=self.calculate_Parameters_allCurve(fitting_type=fitting_type)
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
        self.breakpoints_forLearn_multipleLearn.append(self.breakpoints_forLearn)
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

    def calculate_y_onBorders(self,x_left,x_right)->Dict[str,Dict[str,List[float]]]:
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
        for pattern_name,parameters_pattern in self.parameters_twoPatterns.items():     
            for border_name, x, parameters in zip(self.border_names,xzip,parameters_pattern.values()):
                y_borders_twoPattern[pattern_name][border_name]=self.fitting_func(x, *parameters)
        
        return y_borders_twoPattern
    
    def check_inPattern(self,
                        y_left:List[float],
                        y_right:List[float],
                        y_borders_twoPattern:Dict[str,Dict[str,List[float]]],
                        second_order_derivative:float
                       )->str:
        """
        check if all points besides the are fall into the field defined by the borders
        do not include the point for identification 
        """
        for pattern_name,y_borders in y_borders_twoPattern.items():
            # if (all(np.array(y_left[0:-1])>=np.array(y_borders["left_bottom"][0:-1])) and 
            #     all(np.array(y_left[0:-1])<=np.array(y_borders["left_top"][0:-1])) and
            #     all(np.array(y_right[0:-1])>=np.array(y_borders["right_bottom"][0:-1])) and 
            #     all(np.array(y_right[0:-1])<=np.array(y_borders["right_top"][0:-1]))):
            if (sum(np.array(y_left[0:-1])>=np.array(y_borders["left_bottom"][0:-1]))>0.8*len(y_left[0:-1]) and 
                sum(np.array(y_left[0:-1])<=np.array(y_borders["left_top"][0:-1]))>0.8*len(y_left[0:-1]) and
                sum(np.array(y_right[0:-1])>=np.array(y_borders["right_bottom"][0:-1]))>0.8*len(y_right[0:-1]) and 
                sum(np.array(y_right[0:-1])<=np.array(y_borders["right_top"][0:-1]))>0.8*len(y_right[0:-1]) and 
                abs(second_order_derivative)>0.05*self.std_2):

                return pattern_name
             
        return "Not breakpoint"
        
    def predict(self,pressure_measure,pressure_time,second_order_derivative):
        #store border for every point
#         self.borderData=pd.DataFrame(columns=self.border_names)
        self.std_2=statistics.stdev(second_order_derivative)
        self.borderData=pd.DataFrame()
        detectedpoints_buildUp=[]
        detectedpoints_drawDown=[]
        # #point window
        # points=[point_index for point_index in range(self.point_halfWindow,len(pressure_measure)-self.point_halfWindow)]
        # self.extract_points_inTimeWindow(pressure_measure,pressure_time,points)
        
        #time window
        points=[point_index for point_index in range(len(pressure_measure))]
        self.extract_points_inTimeWindow(pressure_measure,pressure_time,points,self.time_halfWindow_forPredict)

        for index,curveData in self.curveData.iterrows():  
            
            
            curveData["pressure_time_right"].reverse()
            curveData["pressure_measure_right"].reverse()
        
            y_borders=self.calculate_y_onBorders(curveData["pressure_time_left"],curveData["pressure_time_right"])
            self.borderData=self.borderData.append(y_borders,ignore_index=True)
#             display("-----y_borders-----",y_borders)
#             display("-----curveData----:",curveData)
            point_type=self.check_inPattern(curveData["pressure_measure_left"],
                              curveData["pressure_measure_right"],
                              y_borders,
                              second_order_derivative[curveData["point_index"]])
                
            if(point_type=="buildUp"):
                detectedpoints_buildUp.append(curveData["point_index"])
            elif(point_type=="drawDown"):
                detectedpoints_drawDown.append(curveData["point_index"])
        
        self.detectedpoints={"buildUp":detectedpoints_buildUp,
                             "drawDown":detectedpoints_drawDown}
            
        self.data_forPredict=pd.concat([self.curveData, self.borderData], axis=1)
        return detectedpoints_buildUp,detectedpoints_drawDown
    
    def get_tangent(self,parameters_allCurves:pd.DataFrame)->pd.DataFrame:
        #for polynomial fit the third parameter is the tangent
        left_tangent=[parameters[2] for parameters in parameters_allCurves["left_curves_parameters"]] 
        right_tangent=[parameters[2] for parameters in parameters_allCurves["right_curves_parameters"]] 
        print(len(left_tangent),len(parameters_allCurves))
        print(len(right_tangent),len(parameters_allCurves))
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
    
    def get_tangents_twoPatterns(self):
        for pattern_name in self.pattern_names:
            tangent_groundTruth_buildUpOrDrawDown=self.get_tangent(self.parameters_allCurves_groundTruth[pattern_name])
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

    
    def predict_usingTangent(self,pressure_measure,pressure_time,fitting_type="polynomial"):
        self.buildUp_or_drawDown=""
        points=[point_index for point_index in range(len(pressure_measure))]
        parameters_allCurves=self.produce_parameters_givenPoints(pressure_measure,
                                                                 pressure_time,
                                                                 points,
                                                                 self.time_halfWindow_forPredict,
                                                                 None,
                                                                 fitting_type)
        
        self.get_tangents_twoPatterns()
        tangent_forPredict=self.get_tangent(parameters_allCurves)
        
        self.tangent_forPredict=tangent_forPredict
        # print("---------==========",self.pattern_names)
        for pattern_name in self.pattern_names:
            print("---------==========")

            sub_df=tangent_forPredict.loc[(tangent_forPredict["left_tangent"]>=self.tangents_twoPatterns[pattern_name]["left_bottom"]) &
                                    (tangent_forPredict["left_tangent"]<=self.tangents_twoPatterns[pattern_name]["left_top"]) &
                                    (tangent_forPredict["right_tangent"]>=self.tangents_twoPatterns[pattern_name]["right_bottom"]) &
                                    (tangent_forPredict["right_tangent"]<=self.tangents_twoPatterns[pattern_name]["right_top"])]

            # print("----pattern_name,sub_df",pattern_name,sub_df)
            self.detectedpoints[pattern_name]=list(sub_df["point_index"])
            # print("self.detectedpoints",self.detectedpoints)
        return self.detectedpoints
            
        
            
        
      