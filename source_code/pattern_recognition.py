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

def test_func(x, a,b,c,d):
    y = a+b*x-c*np.exp(-d*x)
    return y
def test_func(x, a,b,c,d):
    y = a+b*x-c*np.exp(d*x)
    return y
def test_func(x, a,b,c,d):
    y = a+c*np.exp(d*x)
    return y

def polyval_func_wrapper(x, parameters):
    y = np.polyval(parameters,x)
    return y
def linear_func_wrapper(x, parameters):
    a,b=parameters
    y = a*x+b
    return y

# def fitting_func_wapper(x, parameters):
#     a,b,c=parameters
#     y = a+b*x+c*x*x
#     return y

class PatternRecognition:
    def __init__(self, 
                 delta_pointNumber:int=5,
                 deltaTime_learn:float=0.1,
                 fitting_func=polyval_func_wrapper,
#                  filePath_learnedPattern="../data_output/Learned_Pattern.csv",
                 filePath_learnedPattern="../data_output/Learned_Pattern.jason",
#                 filepath_curveDataLeft= "../data_output/curveDataLeft_forLearning.csv",
#                 filepath_curveDataRight= "../data_output/curveDataRight_forLearning.csv"
                ):
        

        self.delta_pointNumber=delta_pointNumber
        self.deltaTime_learn=deltaTime_learn
        self.fitting_func=fitting_func
        self.filePath_learnedPattern=filePath_learnedPattern
#         self.filepath_curveDataLeft=filepath_curveDataLeft
#         self.filepath_curveDataRight=filepath_curveDataRight
        
        self.border_names=["left_top","left_bottom","right_top","right_bottom"]
        self.curveData=pd.DataFrame(columns=['pressure_time_left', 
                                             'pressure_measure_left',
                                             'pressure_time_right', 
                                             'pressure_measure_right'])
        self.borderData=pd.DataFrame(columns=self.border_names)
        self.data_forPredict=pd.DataFrame()
        self.parameters_allCurves=pd.DataFrame(columns=["left_curves_parameters","right_curves_parameters"])
        self.parameters_pattern={}
        self.x_leftPlot=[]
        self.x_rightPlot=[]
        self.patternLoaded=False
    
    
    def load_pattern(self,filePath_savedPattern):
        """
        load the saved pattern
        """
        with open(filePath_savedPattern) as infile:
            self.parameters_pattern = json.load(infile)
        print(f"The pattern parameters {self.parameters_pattern} are loaded")
    
        
    def extract_points_inWindow(self,pressure_measure,pressure_time,points): 
        """
        extract pressure measure & time data for all points 
        in window  [point_index-delta_pointNumber,point_index+delta_pointNumber]
        """
        self.curveData=pd.DataFrame(columns=['pressure_time_left', 
                                             'pressure_measure_left',
                                             'pressure_time_right', 
                                             'pressure_measure_right'])
        
        for point_index in points:
            #left   
            sub_measure=pressure_measure[point_index+1-self.delta_pointNumber:point_index+1]
            sub_time=pressure_time[point_index+1-self.delta_pointNumber:point_index+1]
            curve_pressure=[round(measure-sub_measure[-1],6) for measure in sub_measure]
            curve_time=[round(time-sub_time[-1],6) for time in sub_time]
            data={"pressure_time_left":curve_time,"pressure_measure_left":curve_pressure}
        
            
            #right
            sub_measure=pressure_measure[point_index:point_index+self.delta_pointNumber]
            sub_time=pressure_time[point_index:point_index+self.delta_pointNumber]
            curve_pressure=[round(measure-sub_measure[0],6) for measure in sub_measure]
            curve_time=[round(time-sub_time[0],6) for time in sub_time]
            data.update({"pressure_time_right":curve_time,
                         "pressure_measure_right":curve_pressure})
            
            
            self.curveData=self.curveData.append(data,ignore_index=True)
       
#         self.curveDataLeft.to_csv(self.filepath_curveDataLeft,index=False,float_format='%.4f',sep='\t')
#         self.curveDataRight.to_csv(self.filepath_curveDataRight,index=False,float_format='%.4f',sep='\t') 


    def fit_curve(self,xdata,ydata,fitting_type:str="polynomial"):
        x = np.asarray(xdata)
        y = np.asarray(ydata)
        
#         parameters, covariance = curve_fit(self.fitting_func, x, y)
#         y_fit = self.fitting_func(x, *parameters)
        
        if fitting_type=="polynomial":
            parameters=np.polyfit(x,y,3)
        if fitting_type=="linear":
            parameters, covariance = curve_fit(self.fitting_func, x, y)
            
#         print(parameters)
#         y_fit=np.polyval(parameters,x)
        y_fit=self.fitting_func(x, parameters)

        plt.plot(x, y, 'o')
        plt.plot(x, y_fit, '-')
        plt.show()
        return parameters
    
    def calculate_Parameters_allCurve(self):
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
            if i==0:
                xLeft_min=abs(xdata[0])
                self.x_leftPlot=xdata
            if i>0 and abs(xdata[0])<xLeft_min:
                xLeft_min=abs(xdata[0])
                self.x_leftPlot=xdata
                
         
            curveLeft_parameters.append(self.fit_curve(xdata,ydata,"polynomial"))

            
            #right side
            xdata=self.curveData["pressure_time_right"][i]
            ydata=self.curveData["pressure_measure_right"][i]
            if i==0:
                xRight_min=abs(xdata[-1])
                self.x_rightPlot=xdata
            if i>0 and abs(xdata[-1])<xRight_min:
                xRight_min=abs(xdata[-1])
                self.x_rightPlot=xdata
          
            curveRight_parameters.append(self.fit_curve(xdata,ydata,"polynomial"))
            
        self.parameters_allCurves=pd.DataFrame({"left_curves_parameters":curveLeft_parameters,
                         "right_curves_parameters":curveRight_parameters})
        
            
    def calculate_parameters_pattern(self):
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
        #if has old patterns, load old patterns and compare with new fitting curves
        if len(self.parameters_pattern)>0:
            old_pattern={"left_curves_parameters":[self.parameters_pattern["left_bottom"],
                                                   self.parameters_pattern["left_top"]],
                        "right_curves_parameters":[self.parameters_pattern["right_bottom"],
                                                   self.parameters_pattern["right_top"]]}
            self.parameters_allCurves.append(pd.DataFrame(old_pattern), ignore_index = True)
            
        number=30
        timeInterval=0.05
        y_left_allCurve = np.empty((0, number), float)
        y_right_allCurve = np.empty((0, number), float)
        x_left=np.linspace(start = -timeInterval, stop = 0, num = number)
        x_right=np.linspace(start = 0, stop = timeInterval, num = number)
    
        fig=plt.figure(figsize = (20, 10))
        curve_number=len(self.parameters_allCurves)
        for i in range(curve_number):
            y_left=self.fitting_func(x_left, self.parameters_allCurves["left_curves_parameters"][i])
            y_right=self.fitting_func(x_right, self.parameters_allCurves["right_curves_parameters"][i])
            
            y_left_allCurve=np.append(y_left_allCurve,np.array([y_left]), axis=0)
            y_right_allCurve=np.append(y_right_allCurve,np.array([y_right]), axis=0)
            

            if self.patternLoaded and (i==0 or i==1):
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
                     color="yellow",linewidth=1)
                plt.plot(x_left, y_left, '-', label='New_fit',
                         color="yellow",linewidth=1)        
            
        left_parameters_pattern=self.find_border(x_left,y_left_allCurve)
        right_parameters_pattern=self.find_border(x_right,y_right_allCurve)
        self.parameters_pattern["left_top"]=left_parameters_pattern["top"]
        self.parameters_pattern["left_bottom"]=left_parameters_pattern["bottom"]
        self.parameters_pattern["right_top"]=right_parameters_pattern["top"]
        self.parameters_pattern["right_bottom"]=right_parameters_pattern["bottom"]
        
        self.legend_without_duplicate_labels(fig)
        
    def legend_without_duplicate_labels(self,figure):
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        figure.legend(by_label.values(), by_label.keys(),shadow=True, fontsize='large')


    def find_border(self,x,y_allCurve):
        """
        calculate parameters of top curve and buttom curve 
        for left or right side of buildup or drawndown pattern
        """
        parameters_half_pattern={}
        
        y_allCurve_max=y_allCurve.max(axis=0)
        parameters_half_pattern["top"]=self.fit_curve(x,y_allCurve_max,"polynomial")
     
        y_allCurve_min=y_allCurve.min(axis=0)
        parameters_half_pattern["bottom"]=self.fit_curve(x,y_allCurve_min,"polynomial")
        
        return parameters_half_pattern
    
    def detect_breakpoint_type(self,pressure_measure:List[float],
                               pressure_time:List[float],
                               points:List[int]
                              )->Dict[str,List[int]]:
        """
        detect the break points are buildup or drawdown
        do the linear fitting,
        if slope_left>slope_right: drawdown
        else: buildup
        """
        self.extract_points_inWindow(pressure_measure,pressure_time,points)
        
        
    
    def learn(self,pressure_measure,pressure_time,ground_truth,filePath_savedPattern=None):
        if filePath_savedPattern!=None:
            self.load_pattern(filePath_savedPattern)
            self.PatternLoaded=True
        self.extract_points_inWindow(pressure_measure,pressure_time,ground_truth)
        self.calculate_Parameters_allCurve()
        self.calculate_parameters_pattern()
        
#     def sav_pattern(self):
#         print("self.parameters_pattern",self.parameters_pattern)
#         pattern_df=pd.DataFrame({key:[value] for key,value in self.parameters_pattern.items()})
#         with open(self.filePath_learnedPattern, 'a') as f:
#             pattern_df.to_csv(f, mode='a', header=f.tell()==0)
# #         pattern_df.to_csv(self.filePath_learnedPattern,index=False,float_format='%.4f',sep=' ',mode="a")

    def sav_pattern(self):

        data=identfication_UsePattern.parameters_pattern
        data={key:list(value) for key,value in data.items()}
        with open(self.filePath_learnedPattern, 'w') as fp:
            json.dump(data, fp)

    def calculate_y_onBorders(self,x_left,x_right):
        """
        calculate y coordinates on borders coresponds to the input x coordinates
        """
        y_borders={}
        xzip=[x_left,
            x_left,
            x_right,
            x_right]
                     
        for border_name, x, parameters in zip(self.border_names,xzip,self.parameters_pattern.values()):
            y_borders[border_name]=self.fitting_func(x, parameters)
        
        return y_borders
    
    def check_inPattern(self,y_left:List[float],y_right:List[float],y_borders:Dict[str,List[float]])->bool:
        """
        check if all points besides the are fall into the field defined by the borders
        do not include the point for identification 
        """

        if (all(np.array(y_left[0:-1])>=np.array(y_borders["left_bottom"][0:-1])) and 
            all(np.array(y_left[0:-1])<=np.array(y_borders["left_top"][0:-1])) and
            all(np.array(y_right[0:-1])>=np.array(y_borders["right_bottom"][0:-1])) and 
            all(np.array(y_right[0:-1])<=np.array(y_borders["right_top"][0:-1]))):
            return True
        
        return False
        
    def predict(self,pressure_measure,pressure_time):
        #store border for every point
        self.borderData=pd.DataFrame(columns=self.border_names)
        breakpoints=[]
        points=[point_index for point_index in range(self.delta_pointNumber,len(pressure_measure)-self.delta_pointNumber)]
        
        self.extract_points_inWindow(pressure_measure,pressure_time,points)

        for index,curveData in self.curveData.iterrows():  
            
            curveData["pressure_time_right"].reverse()
            curveData["pressure_measure_right"].reverse()
        
            y_borders=self.calculate_y_onBorders(curveData["pressure_time_left"],curveData["pressure_time_right"])
            self.borderData=self.borderData.append(y_borders,ignore_index=True)
#             display("-----y_borders-----",y_borders)
#             display("-----curveData----:",curveData)
                
            if(
                self.check_inPattern(curveData["pressure_measure_left"],
                              curveData["pressure_measure_right"],
                              y_borders)
            ):
                breakpoints.append(index+self.delta_pointNumber)
            
        self.data_forPredict=pd.concat([self.curveData, self.borderData], axis=1)
        return breakpoints