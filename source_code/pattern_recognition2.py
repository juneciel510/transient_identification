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
                 time_halfWindow:float=0.5,
                 time_halfWindow_forLearn:float=1,
#                  fitting_func=None,
#                  filePath_learnedPattern="../data_output/Learned_Pattern.csv",
                 filePath_learnedPattern="../data_output/Learned_Pattern.jason",
#                 filepath_curveDataLeft= "../data_output/curveDataLeft_forLearning.csv",
#                 filepath_curveDataRight= "../data_output/curveDataRight_forLearning.csv"
                ):
        
        #to store the input groundtruth, including buildup & drawdown together
        self.ground_truth =[]
        #to store the points for learn, classified as buildup & drawdown
        self.breakpoints_forLearn=defaultdict(list)
        #to store the points predicted 
        self.detectedpoints=defaultdict(list)
        self.point_halfWindow=point_halfWindow
        #time window for learn
        self.time_halfWindow_forLearn=time_halfWindow_forLearn
        #time window for predict
        self.time_halfWindow=time_halfWindow
        self.fitting_func=None
        self.filePath_learnedPattern=filePath_learnedPattern
#         self.filepath_curveDataLeft=filepath_curveDataLeft
#         self.filepath_curveDataRight=filepath_curveDataRight
        
        self.border_names=["left_top","left_bottom","right_top","right_bottom"]
        self.curveData=pd.DataFrame(columns=['pressure_time_left', 
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
        self.parameters_allCurves=pd.DataFrame(columns=["left_curves_parameters","right_curves_parameters"])
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

    def extract_points_inWindow(self,pressure_measure,pressure_time,points): 
            """
            extract pressure measure & time data for all points 
            in window  [point_index-point_halfWindow,point_index+point_halfWindow]
            """
            self.curveData=pd.DataFrame(columns=['pressure_time_left', 
                                                'pressure_measure_left',
                                                'pressure_time_right', 
                                                'pressure_measure_right'])
            
            for point_index in points:
                #convert timewindow to point window 
                time_leftStart=pressure_time[point_index]-self.time_halfWindow_forLearn
                halfWinow_left=point_index-bisect.bisect_left(pressure_time, time_leftStart) 
                if halfWinow_left<self.point_halfWindow:
                    halfWinow_left=self.point_halfWindow
                 
                time_rightEnd=pressure_time[point_index]+self.time_halfWindow_forLearn
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

            
            #left
            sub_measure=pressure_measure[point_index+1-halfWinow_left:point_index+1]
            sub_time=pressure_time[point_index+1-halfWinow_left:point_index+1]
            curve_pressure=[round(measure-sub_measure[-1],6) for measure in sub_measure]
            curve_time=[round(time-sub_time[-1],6) for time in sub_time]
            data={"pressure_time_left":curve_time,"pressure_measure_left":curve_pressure}
            
        
            
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
            parameters=np.polyfit(x,y,5)
        if fitting_type=="linear" or fitting_type=="log":
            parameters, covariance = curve_fit(self.fitting_func, x, y)

        y_fit=self.fitting_func(x, *parameters)
        # if fitting_type=="linear":
        #     plt.figure(figsize = (10, 10))
        #     plt.axis([-1, 1, -10, 10])
        # plt.plot(x, y,color=color,  marker='o')
        plt.scatter(x=x,y=y,color=color)
        plt.plot(x, y_fit, color=color,linestyle='-')
        plt.title(plot_title)
        if fitting_type=="log" or fitting_type=="polynomial":
            plt.show()
        return parameters
    
    def calculate_Parameters_allCurve(self,fitting_type,name_pattern="",plot_title=""):
        print("---------------calculate_Parameters_allCurve",fitting_type)
        curveLeft_parameters=[]
        curveRight_parameters=[]
        self.parameters_allCurves=pd.DataFrame()
        
        plt.figure(figsize = (20, 10))

        for i in range(len(self.curveData)):
            #left side
                        
#             print("-----------i---------",i)
            xdata=self.curveData["pressure_time_left"][i]
            ydata=self.curveData["pressure_measure_left"][i]
#             print(xdata,ydata)
            #get the smallest x range of all points
#             if i==0:
#                 xLeft_min=abs(xdata[0])
#                 self.x_leftPlot=xdata
#             if i>0 and abs(xdata[0])<xLeft_min:
#                 xLeft_min=abs(xdata[0])
#                 self.x_leftPlot=xdata
            color="yellow"  
            #when detect_breakpoint_type invoke this funtion,
            # set the title to be the breakpoint index
            if fitting_type=="linear":
                plot_title=f"{self.ground_truth[i]}"      
            if fitting_type=="polynomial" or fitting_type=="log":
                plot_title=f"{name_pattern}---{self.ground_truth[i]}"   
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
            
        self.parameters_allCurves=pd.DataFrame({"left_curves_parameters":curveLeft_parameters,
                         "right_curves_parameters":curveRight_parameters})
        
            
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

        #if has old patterns, load old patterns and compare with new fitting curves
        if len(loadedParameters_pattern)>0:
            old_pattern={"left_curves_parameters":[loadedParameters_pattern["left_bottom"],
                                                   loadedParameters_pattern["left_top"]],
                        "right_curves_parameters":[loadedParameters_pattern["right_bottom"],
                                                   loadedParameters_pattern["right_top"]]}
            self.parameters_allCurves.append(pd.DataFrame(old_pattern), ignore_index = True)
            
        number=30
        y_left_allCurve = np.empty((0, number), float)
        y_right_allCurve = np.empty((0, number), float)
        x_left=np.linspace(start = -self.time_halfWindow_forLearn, stop = 0, num = number)
        x_right=np.linspace(start = 0, stop = self.time_halfWindow_forLearn, num = number)
    
        fig=plt.figure(figsize = (20, 10))
        curve_number=len(self.parameters_allCurves)
        for i in range(curve_number):
            y_left=self.fitting_func(x_left, *self.parameters_allCurves["left_curves_parameters"][i])
            y_right=self.fitting_func(x_right, *self.parameters_allCurves["right_curves_parameters"][i])
            
            y_left_allCurve=np.append(y_left_allCurve,np.array([y_left]), axis=0)
            y_right_allCurve=np.append(y_right_allCurve,np.array([y_right]), axis=0)
            

            if len(loadedParameters_pattern)>0 and (i==0 or i==1):
                plt.plot(x_right, y_right, 'r+', 
                         label='Old_pattern',
                         color="red",
                         linewidth=4)
                plt.plot(x_left, y_left, 'r+', 
                         label='Old_patern',
                         color="red",
                         linewidth=4)
            else:
                plt.plot(x_right, y_right, '-', label='New_fit',
                     color="orange",linewidth=1)
                plt.plot(x_left, y_left, '-', label='New_fit',
                         color="orange",linewidth=1)        
            
        left_parameters_pattern=self.find_border(x_left,y_left_allCurve,fitting_type)
        right_parameters_pattern=self.find_border(x_right,y_right_allCurve,fitting_type)
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
    
    def find_border(self,x,y_allCurve,fitting_type,plot_title=""):
        """
        calculate parameters of top curve and buttom curve 
        for left or right side of buildup or drawndown pattern
        """
        parameters_half_pattern={}
        
        y_allCurve_max=y_allCurve.max(axis=0)
        color="green"
        parameters_half_pattern["top"]=self.fit_curve(x,y_allCurve_max,color,fitting_type,plot_title)
     
        y_allCurve_min=y_allCurve.min(axis=0)
        parameters_half_pattern["bottom"]=self.fit_curve(x,y_allCurve_min,color,fitting_type,plot_title)
        
        return parameters_half_pattern
    
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
        self.fitting_func=linear_func_wrapper
        self.extract_points_inWindow(pressure_measure,pressure_time,points)
        self.calculate_Parameters_allCurve(fitting_type="linear")
        # self.parameters_allCurves
        breakpoint_buildUp=[]
        breakpoint_drawDown=[]
        for index,parameter in self.parameters_allCurves.iterrows():
            #compare slope and convert index in the parameters_allCurves to in pressure_df
            if parameter["left_curves_parameters"][0]<parameter["right_curves_parameters"][0]:
                if parameter["left_curves_parameters"][0]<parameter["right_curves_parameters"][0]<0:
                    print(f"===============\n"
                          f"point '{points[index]}' is going to be classified as buildup,\n"
                          f"however the ground truth should be drawdown,\n"
                          f"there must be a bump in window thus should not be included into breakpoints for learning\n"
                          f"===============")
                else:
                    breakpoint_buildUp.append(points[index])
                
                
            if parameter["left_curves_parameters"][0]>parameter["right_curves_parameters"][0]:
                if parameter["left_curves_parameters"][0]>parameter["right_curves_parameters"][0]>0:
                    print(f"===============\n"
                          f"point '{points[index]}' is going to be classified as drawdown,\n"
                          f"however the ground truth should be biuldup,\n"
                          f"there must be a cavity in window thus should not be included into breakpoints for learning\n"
                          f"===============")
                else:
                    breakpoint_drawDown.append(points[index])
        return breakpoint_buildUp,breakpoint_drawDown

        
    
    def learn_buildUp_or_drawDown(self,pressure_measure,pressure_time,ground_truth,loadedParameters_pattern,name_pattern,fitting_type):
        # self.fitting_func=polyval_func_wrapper
#         if filePath_loadPattern!=None:
#             self.load_pattern(filePath_loadPattern)
#             self.PatternLoaded=True
        # print("-----learn_buildUp_or_drawDown",fitting_type)
        if fitting_type == "linear":
            self.fitting_func=linear_func_wrapper
        if fitting_type == "polynomial":
            self.fitting_func=polyval_func_wrapper
        if fitting_type == "log":
            self.fitting_func=log_func_wrapper
        print(f"start to learn '{name_pattern}' pattern...")
        self.extract_points_inWindow(pressure_measure,pressure_time,ground_truth)
        self.calculate_Parameters_allCurve(name_pattern=name_pattern,fitting_type=fitting_type)
        parameters_pattern=self.calculate_parameters_pattern(fitting_type,loadedParameters_pattern)
        return parameters_pattern
    
    def learn(self,pressure_measure,pressure_time,ground_truth,fitting_type,filePath_loadPattern=None):
        # print("-----------------learn",fitting_type)
        self.ground_truth = ground_truth
        ground_truth_buildUp,ground_truth_drawDown=self.detect_breakpoint_type(pressure_measure,
                                                                           pressure_time,
                                                                           ground_truth)
        print(f"------In the input ground truth: {len(ground_truth_buildUp)} points are detected as buildup, {len(ground_truth_drawDown)} points are detected as drawDown")
        self.breakpoints_forLearn={"buildUp":ground_truth_buildUp,
                                   "drawDown":ground_truth_drawDown}
        # print("----ground_truth_buildUp,ground_truth_drawDown",ground_truth_buildUp,ground_truth_drawDown)
#         if filePath_loadPattern!=None:
        loadedParameters_twoPattern=self.load_pattern(filePath_loadPattern)
#             self.PatternLoaded=True
        allPoints=[ground_truth_buildUp,ground_truth_drawDown]
#         name_pattern=["buildUp","drawDown"]
        for points,name_pattern,loadedParameters_pattern in zip(allPoints,
                                                          loadedParameters_twoPattern.keys(),
                                                          loadedParameters_twoPattern.values()):
            if len(points) ==0:
                self.parameters_twoPatterns[name_pattern]={}
            else:
                self.parameters_twoPatterns[name_pattern]=self.learn_buildUp_or_drawDown(pressure_measure,
                                                                pressure_time,
                                                                points,
                                                                loadedParameters_pattern,
                                                                name_pattern,
                                                                fitting_type)
        
        
        
#     def sav_pattern(self):
#         print("self.parameters_pattern",self.parameters_pattern)
#         pattern_df=pd.DataFrame({key:[value] for key,value in self.parameters_pattern.items()})
#         with open(self.filePath_learnedPattern, 'a') as f:
#             pattern_df.to_csv(f, mode='a', header=f.tell()==0)
# #         pattern_df.to_csv(self.filePath_learnedPattern,index=False,float_format='%.4f',sep=' ',mode="a")

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
                        y_borders_twoPattern:Dict[str,Dict[str,List[float]]]
                       )->str:
        """
        check if all points besides the are fall into the field defined by the borders
        do not include the point for identification 
        """
        for pattern_name,y_borders in y_borders_twoPattern.items():
            if (all(np.array(y_left[0:-1])>=np.array(y_borders["left_bottom"][0:-1])) and 
                all(np.array(y_left[0:-1])<=np.array(y_borders["left_top"][0:-1])) and
                all(np.array(y_right[0:-1])>=np.array(y_borders["right_bottom"][0:-1])) and 
                all(np.array(y_right[0:-1])<=np.array(y_borders["right_top"][0:-1]))):
#                 print("---------check_inPattern",pattern_name,y_borders)
                return pattern_name
             
        return "Not breakpoint"
        
    def predict(self,pressure_measure,pressure_time):
        #store border for every point
#         self.borderData=pd.DataFrame(columns=self.border_names)
        self.borderData=pd.DataFrame()
        detectedpoints_buildUp=[]
        detectedpoints_drawDown=[]
        points=[point_index for point_index in range(self.point_halfWindow,len(pressure_measure)-self.point_halfWindow)]
        
        self.extract_points_inWindow(pressure_measure,pressure_time,points)

        for index,curveData in self.curveData.iterrows():  
            
            curveData["pressure_time_right"].reverse()
            curveData["pressure_measure_right"].reverse()
        
            y_borders=self.calculate_y_onBorders(curveData["pressure_time_left"],curveData["pressure_time_right"])
            self.borderData=self.borderData.append(y_borders,ignore_index=True)
#             display("-----y_borders-----",y_borders)
#             display("-----curveData----:",curveData)
            point_type=self.check_inPattern(curveData["pressure_measure_left"],
                              curveData["pressure_measure_right"],
                              y_borders)
                
            if(point_type=="buildUp"):
                detectedpoints_buildUp.append(index+self.point_halfWindow)
            elif(point_type=="drawDown"):
                detectedpoints_drawDown.append(index+self.point_halfWindow)
            
        self.data_forPredict=pd.concat([self.curveData, self.borderData], axis=1)
        return detectedpoints_buildUp,detectedpoints_drawDown
            
        
      