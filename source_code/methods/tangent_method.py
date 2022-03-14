import numpy as np
import pandas as pd
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

from base_classes import CurveParametersCalc,SaveNLoad

class TangentMethod(CurveParametersCalc,SaveNLoad):
    def __init__(self, 
                time_halfWindow:float=None,
                point_halfWindow:int=None,
                min_pointsNumber:int=8,
                time_halfWindow_forDetectType:float=0.5,
                tangent_type:str="single_point",
                polynomial_order:int =3,
                filePath_learnedPattern="../data_output/Learned_Pattern.jason",
            )->None:
    
        CurveParametersCalc.__init__(self)
        self.min_pointsNumber=min_pointsNumber
        self.tangent_type=tangent_type
        self.polynomial_order=polynomial_order
        self.time_halfWindow_forDetectType=time_halfWindow_forDetectType
        self.time_halfWindow=time_halfWindow
        self.point_halfWindow=point_halfWindow
        
        self.tangents_twoPatterns={"buildUp":{"left_top":float('NaN'),
                                    "left_bottom":float('NaN'),
                                    "right_top":float('NaN'),
                                    "right_bottom":float('NaN')},
                        "drawDown":{"left_top":float('NaN'),
                                    "left_bottom":float('NaN'),
                                    "right_top":float('NaN'),
                                    "right_bottom":float('NaN')}}
        
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
                                    data_type:str="single_point",
                                    time_halfWindow:float=None,
                                    point_halfWindow:int=None,
                                    polynomial_order:int=3,
                                    point_halfWindow_tagentPlot:int=5
                                    )->pd.DataFrame:
        """
        for the given points, get the left & right tangents for each point in pointWindow or timeWindow
        """
        fitting_type="polynomial"     
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
        parameters_allCurves=self.calculate_Parameters_allCurve(data_inWindow,fitting_type,polynomial_order,show_plot=False)
        # display(parameters_allCurves)
        
        tangent_inWindow=pd.DataFrame()
        
        for index,parameters in parameters_allCurves.iterrows():
            
            
            
            if data_type=="for_plot" or data_type=="average":
                pressure_time_left=data_inWindow.iloc[index]["pressure_time_left"]
                pressure_time_right=data_inWindow.iloc[index]["pressure_time_right"]       
                tangent_left=self.calculte_tangent_nDegreePolynomial(np.asarray(pressure_time_left),parameters["left_curves_parameters"])
                tangent_right=self.calculte_tangent_nDegreePolynomial(np.asarray(pressure_time_right),parameters["right_curves_parameters"])
                if data_type=="for_plot":
                    pressure_measure_left=data_inWindow.iloc[index]["pressure_measure_left"]
                    pressure_measure_right=data_inWindow.iloc[index]["pressure_measure_right"]
                    
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
                if data_type=="average":      
                    tangent_left_average=sum(tangent_left)/len(tangent_left)
                    tangent_right_average=sum(tangent_right)/len(tangent_right)
                    data={"point_index":parameters["point_index"],
                        "tangent_left":tangent_left_average,
                        "tangent_right":tangent_right_average}
                tangent_inWindow=tangent_inWindow.append(data,ignore_index=True)        
            elif data_type=="single_point":   
                n=len(parameters_allCurves["left_curves_parameters"][0])-2           
                data={"point_index":parameters["point_index"],
                    "tangent_left":parameters["left_curves_parameters"][n],
                    "tangent_right":parameters["right_curves_parameters"][n]}   
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
                      
    
    # def get_tangent(self,
    #                 parameters_allCurves:pd.DataFrame,
    #                 fitting_type:str)->pd.DataFrame:
    #     """ 
    #     from the parameters of all fitted curves,
    #     get the tangent values of all curves
    #     Args:
    #         parameters_allCurves: pd.DataFrame in which each row 
    #         represents the parameters of left and right curves of a point
    #         fitting_type: the type for the fitting function            
    #     Returns:
    #         pd.DataFrame in which each row represents the left and right tangent of a point
    #     """  
    #     #for polynomial fit the third parameter is the tangent
    #     if fitting_type=="polynomial":
    #         n=len(parameters_allCurves["left_curves_parameters"][0])-2
    #     elif fitting_type=="linear":
    #         n=0 
    #     else:  
    #         print("please check fitting type, need to be 'polynomial' or 'linear'")
    #     tangent_left=[parameters[n] for parameters in parameters_allCurves["left_curves_parameters"]] 
    #     tangent_right=[parameters[n] for parameters in parameters_allCurves["right_curves_parameters"]] 
    #     tangent_df=pd.DataFrame({"point_index":parameters_allCurves["point_index"],"tangent_left":tangent_left,"tangent_right":tangent_right})
    #     return tangent_df
    
    
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
 

    
    # def get_tangents_twoPatterns(self,fitting_type:str):
    #     """ 
    #     from the groundtruth, get upperBound lowerBound of the 'buildUp' and 'drawDown'
    #     Args:
    #         fitting_type: the type for the fitting function        
    #     Returns:
    #         dictionary.
    #         ----------
    #         self.tangents_twoPatterns={"buildUp":{"left_top":float,
    #                                            "left_bottom":float,
    #                                            "right_top":float,
    #                                            "right_bottom":float},
    #                                 "drawDown":{"left_top":float,
    #                                            "left_bottom":float,
    #                                            "right_top":float,
    #                                            "right_bottom":float}}
    #         ----------
    #     """  
    #     print("==================")
    #     print(f"start to get tangent pattern..., using '{fitting_type}' fitting")
    #     for pattern_name in self.pattern_names:
    #         tangent_groundTruth_buildUpOrDrawDown=self.get_tangent(self.parameters_allCurves_groundTruth[pattern_name],fitting_type)
            
    #         #left side
    #         self.tangents_twoPatterns[pattern_name]["left_top"]=max(tangent_groundTruth_buildUpOrDrawDown["tangent_left"])
    #         self.tangents_twoPatterns[pattern_name]["left_bottom"]=min(tangent_groundTruth_buildUpOrDrawDown["tangent_left"])
                    
    #         #right side
    #         self.tangents_twoPatterns[pattern_name]["right_top"]=max(tangent_groundTruth_buildUpOrDrawDown["tangent_right"])
    #         self.tangents_twoPatterns[pattern_name]["right_bottom"]=min(tangent_groundTruth_buildUpOrDrawDown["tangent_right"])

    def check_in_deltaTangent(self,deltaTangent_criterion:float,tangent_forPredict:pd.DataFrame):
        buildup=[]
        drawdown=[]
        for index,row in tangent_forPredict.iterrows():
            if row["tangent_right"]>0 and (row["tangent_right"]-row["tangent_left"])>deltaTangent_criterion:
                buildup.append(int(row["point_index"]))
            if row["tangent_right"]<0 and (row["tangent_right"]-row["tangent_left"])<-deltaTangent_criterion:
                drawdown.append(int(row["point_index"]))
                    
        return buildup,drawdown
      
    def learn_TangentRange(self,
            pressure_measure:List[float],
            pressure_time:List[float],
            ground_truth:List[int],
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
        print(f"start to learn..., using polynomial curve fitting")
        ground_truth_buildUp,ground_truth_drawDown=self.detect_breakpoint_type(pressure_measure,
                                                                           pressure_time,
                                                                           ground_truth, 
                                                                           self.time_halfWindow_forDetectType,
                                                                           self.min_pointsNumber)
        for pattern_name,points in zip(self.pattern_names,[ground_truth_buildUp,ground_truth_drawDown]):
            tangent_groundTruth_buildUpOrDrawDown=self.produce_tangent_inWindow(pressure_measure,
                                                        pressure_time,
                                                        points,
                                                        self.tangent_type,
                                                        self.time_halfWindow,
                                                        self.point_halfWindow,
                                                        self.polynomial_order)
            
            #left side
            self.tangents_twoPatterns[pattern_name]["left_top"]=max(tangent_groundTruth_buildUpOrDrawDown["tangent_left"])
            self.tangents_twoPatterns[pattern_name]["left_bottom"]=min(tangent_groundTruth_buildUpOrDrawDown["tangent_left"])
                    
            #right side
            self.tangents_twoPatterns[pattern_name]["right_top"]=max(tangent_groundTruth_buildUpOrDrawDown["tangent_right"])
            self.tangents_twoPatterns[pattern_name]["right_bottom"]=min(tangent_groundTruth_buildUpOrDrawDown["tangent_right"])


    
    def predict_useDeltaTangent(self,
                             pressure_measure:List[float],
                             pressure_time:List[float],
                             points:List[int],
                             deltaTangent_criterion:float=40,
                             )->(List[int],List[int]):
        """
        predict the breakpoint use the difference between left tangent and right tangent
        """
        print("==================")
        print("start to predict using tangent...")
#         self.buildUp_or_drawDown=""
        
        # deltaTangent_criterion=self.get_deltadeltaTangent_criterion(fitting_type)
        # data_inWindow=self.extract_points_inPointWindow(pressure_measure,pressure_time,points,point_halfWindow)
        # parameters_allCurves=self.calculate_Parameters_allCurve(data_inWindow,fitting_type,polynomial_order)     
        # tangent_forPredict_get=self.get_tangent(parameters_allCurves,fitting_type)
        # display(tangent_forPredict_get)
        tangent_forPredict=self.produce_tangent_inWindow(pressure_measure,
                                                        pressure_time,
                                                        points,
                                                        self.tangent_type,
                                                        self.time_halfWindow,
                                                        self.point_halfWindow,
                                                        self.polynomial_order)
        self.tangent_forPredict=tangent_forPredict
        # display(tangent_forPredict)
        points_buildUp,points_drawDown=self.check_in_deltaTangent(deltaTangent_criterion,
                                               tangent_forPredict)
        print(f"detect {len(points_buildUp)} buildups and {len(points_drawDown)} drawdowns " )
        return points_buildUp,points_drawDown
        
    
    def predict_usingTangentRange(self,
                             pressure_measure:List[float],
                             pressure_time:List[float],
                             points:List[int]):
        """ 
        identify the breakpoints if the left tangent and right tangent are in some coresponding range.
        # if the mode is "whole_dataset", check every point in the dataset
        # if the mode is "refine_detection", check the points which already detected by the previous detection.
        Args:
            pressure_measure: pressure measure for the whole dataset
            pressure_time: pressure time for the whole dataset
            points: indices of points to be identify
            
        Returns:
            two lists for buildUp and drawDown break points indices
        """  
        print("==================")
        print("start to predict using tangent...")

        
        # data_inWindow=self.extract_points_inTimeWindow(pressure_measure,pressure_time,points,self.time_halfWindow_forPredict)
        # parameters_allCurves=self.calculate_Parameters_allCurve(data_inWindow,fitting_type,polynomial_order =3,show_plot=True)        
        # tangent_forPredict=self.get_tangent(parameters_allCurves,fitting_type)
        tangent_forPredict=self.produce_tangent_inWindow(pressure_measure,
                                                        pressure_time,
                                                        points,
                                                        self.tangent_type,
                                                        self.time_halfWindow,
                                                        self.point_halfWindow,
                                                        self.polynomial_order) 
        self.tangent_forPredict=tangent_forPredict
        
        points_buildUp,points_drawDown=self.check_in_tangentRange(tangent_forPredict)     
            
        print(f"before filter, the length of buildup {len(points_buildUp)}, the length of drawdown {len(points_drawDown)}")
        points=points_buildUp+points_drawDown
        points_buildUp,points_drawDown=self.detect_breakpoint_type(pressure_measure,
                                                                   pressure_time,
                                                                   points,
                                                                   self.time_halfWindow_forDetectType,
                                                                   self.min_pointsNumber)
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
    

        
            
        
            