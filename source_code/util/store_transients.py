from distutils.log import error
from typing import Callable, Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import math
import statistics
from tangent_method import TangentMethod

class StoreTransients(TangentMethod):
    def __init__(self, 
                 pressure_df:pd.DataFrame,
                 points_buildUp:List[int]=None,
                 points_drawDown:List[int]=None,
                 colum_names:Dict[str,Dict[str,str]]
                   ={"pressure":{"time":"Elapsed time",
                                 "measure":"Data",
                                 "first_order_derivative":"first_order_derivative",
                                 "second_order_derivative":"second_order_derivative"},
                    "rate":{"time":"Elapsed time","measure":"Liquid rate"}},
                   mode:str="Derivative")->None:
        self.pressure_df=pressure_df 
        self.colum_names=colum_names
        self.pressure_measure=list(pressure_df[self.colum_names["pressure"]["measure"]])
        self.pressure_time=list(pressure_df[self.colum_names["pressure"]["time"]])
        self.points_buildUp=points_buildUp
        self.points_drawDown=points_drawDown
        self.mode=mode
        self.std_pressure=statistics.stdev(self.pressure_measure)
        print("self.std_pressure",self.std_pressure)
        self.major_drawDown=None
        self.major_buildUp=None
     
        if points_buildUp!=None and points_drawDown!=None:
            self.shutInperiods=self.find_shutInPeriods()
            # self.flowingPeriods=self.find_flowingPeriods()
        
    def find_shutInPeriods(self)->List[Tuple[int,int]]:
        #copy self.points_drawDown list
        points_drawDown=[item for item in self.points_drawDown]
        
        shutInperiod=[]
        while len(points_drawDown)>0:
            drawdown_cluster=[]
            buildup_larger=list(filter(lambda i: i > points_drawDown[0], self.points_buildUp))
            if len(buildup_larger)==0:
                break
            point_nextbuildup=buildup_larger[0]
            # point_nextbuildup = list(filter(lambda i: i > points_drawDown[0], self.points_buildUp))[0]

            while len(points_drawDown)>0:
                if points_drawDown[0]>point_nextbuildup:
                    break
                drawdown_cluster.append(points_drawDown[0])
                points_drawDown.remove(points_drawDown[0])

            if self.mode=="Tangent":
                # point_drawDown_major=self.get_point_minDeltaTangent(drawdown_cluster)
                point_drawDown_major=self.get_point_minLeftTangent(drawdown_cluster)
            elif self.mode=="Derivative":
                point_drawDown_major=self.get_point_minDerivative(drawdown_cluster)
            else:
                raise Exception("The mode must be a string 'Tangent' or 'Derivative'")
            shutInperiod.append((point_drawDown_major,point_nextbuildup))
            
        return shutInperiod
            
                
    def get_point_minDerivative(self,drawdown_cluster:List[int])->int:
        FODs=[self.pressure_df["first_order_derivative"][index] for index in drawdown_cluster]
        index_minFOD=FODs.index(min(FODs))
        return drawdown_cluster[index_minFOD]
    
    def get_point_minLeftTangent(self,drawdown_cluster:List[int])->int:
        tangent_df=self.produce_tangent_inWindow(self.pressure_measure,
                                                 self.pressure_time,
                                                 drawdown_cluster,
                                                 polynomial_order=1,
                                                 point_halfWindow=10)
        # print(tangent_df)
        left_tangents=list(abs(tangent_df["tangent_left"]))
        index_minLeftTangents=left_tangents.index(min(left_tangents))
        
        return drawdown_cluster[index_minLeftTangents]
    
    def get_point_minDeltaTangent(self,drawdown_cluster:List[int])->int:
        tangent_df=self.produce_tangent_inWindow(self.pressure_measure,
                                                 self.pressure_time,
                                                 drawdown_cluster,
                                                 polynomial_order=1,
                                                 point_halfWindow=10)
        # print(tangent_df)
        delta_tangents=list(tangent_df["tangent_right"]-tangent_df["tangent_left"])
        index_minDeltaTangent=delta_tangents.index(min(delta_tangents))
   
        return drawdown_cluster[index_minDeltaTangent]
    
    # def get_point_maxStd(self,
    #                      points:List[int],
    #                         time_halfWindow:float=None,
    #                         point_halfWindow:int=None,)->int:
    #     pressure_inWindow=self.get_pressure_inWindow(self.pressure_measure,
    #                                                 self.pressure_time,
    #                                                 points:List[int],
    #                                                 time_halfWindow,
    #                                                 point_halfWindow)
    #     std_transients=statistics.stdev(transients_pressure)
        
        
        
    def get_pressure_inWindow(self,
                              pressure_measure:List[float],
                            pressure_time:List[float],
                            points:List[int],
                            time_halfWindow:float=None,
                            point_halfWindow:int=None):
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
        
        # pressure_inWindow=data_inWindow['pressure_measure_left']+data_inWindow['pressure_measure_right']
        return data_inWindow
    
    def find_flowingPeriods(self,shutInPeriods)->List[Tuple[int,int]]:
        major_buildUp,major_drawDown=self.convert_to_twoLsts(shutInPeriods)
        flowingPeriods=[]
        for buildup in major_buildUp:
            drawdown_larger=list(filter(lambda i: i > buildup, major_drawDown))
            if len(drawdown_larger)==0:
                break
            drawdown =drawdown_larger[0] 
            flowingPeriods.append((buildup,drawdown))
        return flowingPeriods
    
    def remove_minorTransients(self,shutInperiods,minor_threshold:float):
        filtered_shutIns=[]
        std_transients_lst=[]
        for drawDown, buildUp in shutInperiods:
            transients_pressure=self.pressure_df[self.colum_names["pressure"]["measure"]].iloc[drawDown:buildUp]
            std_transients=statistics.stdev(transients_pressure)
            std_transients_lst.append(std_transients)    
            if abs(std_transients)>minor_threshold*abs(self.std_pressure):
                filtered_shutIns.append((drawDown, buildUp))
        print("std_transients_lst",std_transients_lst)
        return filtered_shutIns
    
    def convert_to_twoLsts(self,shutInPeriods)->(List[int],List[int]):
        major_drawDown=[]
        major_buildUp=[]
        #if the first buildup point is smaller than first drawdown, 
        #set it as first major buildup by default
        if self.points_buildUp[0]<shutInPeriods[0][0]:
            major_buildUp.append(self.points_buildUp[0])
        for drawDown, buildUp in shutInPeriods:
            major_drawDown.append(drawDown)
            major_buildUp.append(buildUp)
        return major_buildUp,major_drawDown
            
            