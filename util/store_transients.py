from distutils.log import error
from typing import Callable, Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import math
import statistics
import sys, os.path
methods_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
+ '/methods/')
sys.path.append(methods_dir)
from tangent_method import TangentMethod
from base_classes import CurveParametersCalc
from extract_points import ExtractPoints_inWindow

class StoreTransients(TangentMethod):
    def __init__(self, 
                 pressure_df:pd.DataFrame,
                 minor_threshold_shutIn:float,
                 minor_threshold_Flowing:float,
                 points_buildUp:List[int]=None,
                 points_drawDown:List[int]=None,
                 colum_names:Dict[str,Dict[str,str]]
                   ={"pressure":{"time":"Elapsed time",
                                 "measure":"Data",
                                 "first_order_derivative":"first_order_derivative",
                                 "second_order_derivative":"second_order_derivative"},
                    "rate":{"time":"Elapsed time","measure":"Liquid rate"}},
                   mode:str="Derivative",
                   time_halfWindow:float=None,
                    point_halfWindow:int=10)->None:
        TangentMethod.__init__(self)
        self.pressure_df=pressure_df 
        self.colum_names=colum_names
        self.pressure_measure=list(pressure_df[self.colum_names["pressure"]["measure"]])
        self.pressure_time=list(pressure_df[self.colum_names["pressure"]["time"]])
        self.points_buildUp=points_buildUp
        self.points_drawDown=points_drawDown
        self.mode=mode
        self.std_pressure=statistics.stdev(self.pressure_measure)
        # print("self.std_pressure",self.std_pressure)
        self.pressure_max=max(pressure_df[colum_names["pressure"]["measure"]])
        self.pressure_min=min(pressure_df[colum_names["pressure"]["measure"]])
        self.major_drawDown=None
        self.major_buildUp=None
        self.shutInperiods=None
        self.flowingPeriods=None
        self.flowingTransient_objects=None
        self.allPointsStored=None
     
        if points_buildUp!=None and points_drawDown!=None:
            self.twoSequentIdentification(minor_threshold_shutIn,
                                          minor_threshold_Flowing)
            self.allPointsStored=self.get_allPointsStored()
            
        else:
            print("No points_buildUp or points_drawDown detected" )
            
    # def find_majorDrawDown(self)->List[int]:
    #     points_drawDown=self.points_drawDown.copy()
        
    #     major_drawDown=[]
    #     while len(points_drawDown)>0:
    #         # print("========================")
    #         drawdown_cluster=[]
    #         buildup_larger=list(filter(lambda i: i > points_drawDown[0], self.points_buildUp))
            
    #         if len(buildup_larger)==0:
    #             break
    #         point_nextbuildup=buildup_larger[0]
        
    #         while len(points_drawDown)>0:
    #             if points_drawDown[0]>point_nextbuildup:
    #                 break
    #             drawdown_cluster.append(points_drawDown[0])
    #             points_drawDown.remove(points_drawDown[0])
                
    #         # print("drawdown_cluster",drawdown_cluster)

    #         if self.mode=="Tangent":
    #             point_drawDown_major=self.get_point_minDeltaTangent(drawdown_cluster)
    #             # point_drawDown_major=self.get_point_minLeftTangent(drawdown_cluster)
    #             # print("point_drawDown_major",point_drawDown_major)
    #         elif self.mode=="Derivative":
    #             point_drawDown_major=self.get_point_minDerivative(drawdown_cluster)
    #         elif self.mode=="Std":
    #             point_drawDown_major=self.get_point_maxStd(drawdown_cluster)
    #         else:
    #             raise Exception("The mode must be a string 'Tangent' ,'Derivative' or 'Std'")
    #         major_drawDown.append(point_drawDown_major)
            
    #     return major_drawDown
    
    # def find_majorBuildUp(self,major_drawDown)->List[int]:
    #     major_buildUp=[]
    #     for i in range(len(major_drawDown)-1):
    #         start_point=major_drawDown[i]
    #         end_point=major_drawDown[i+1]
    #         sub_df_pressure=self.pressure_df.iloc[start_point:end_point]
    #         index=sub_df_pressure.loc[sub_df_pressure[self.colum_names["pressure"]["measure"]] == min(sub_df_pressure[self.colum_names["pressure"]["measure"]])].index.tolist()[0]
    #         major_buildUp.append(index)
    #     return major_buildUp
    
    # def find_majorBuildUp(self,major_drawDown)->List[int]:
    #     major_buildUp=[]
    #     for i in range(len(major_drawDown)-1):
    #         start_point=major_drawDown[i]
    #         end_point=major_drawDown[i+1]  
    #         points_inTwoDrawdown=[point for point in self.points_buildUp if point>start_point and point<end_point]
                
    #         std_points=[statistics.stdev(self.pressure_df.iloc[start_point:point][self.colum_names["pressure"]["measure"]]) for point in points_inTwoDrawdown] 
    #         index_maxStd=std_points.index(max(std_points))
    #         major_buildUp.append(points_inTwoDrawdown[index_maxStd])
            
    #     return major_buildUp
    
    
         
        
    def find_shutInPeriods(self)->List[Tuple[int,int]]:
        #copy self.points_drawDown list
        points_drawDown=[item for item in self.points_drawDown]
        
        shutInperiod=[]
        while len(points_drawDown)>0:
            # print("========================")
            drawdown_cluster=[]
            buildup_larger=list(filter(lambda i: i > points_drawDown[0], self.points_buildUp))
            
            if len(buildup_larger)==0:
                break
            point_nextbuildup=buildup_larger[0]
            # point_nextbuildup = list(filter(lambda i: i > points_drawDown[0], self.points_buildUp))[0]
            # print("points_drawDown[0],point_nextbuildup",points_drawDown[0],point_nextbuildup)
            while len(points_drawDown)>0:
                if points_drawDown[0]>point_nextbuildup:
                    break
                drawdown_cluster.append(points_drawDown[0])
                points_drawDown.remove(points_drawDown[0])
                
            # print("drawdown_cluster",drawdown_cluster)

            if self.mode=="Tangent":
                point_drawDown_major=self.get_point_minDeltaTangent(drawdown_cluster)
                # point_drawDown_major=self.get_point_minLeftTangent(drawdown_cluster)
                # print("point_drawDown_major",point_drawDown_major)
            elif self.mode=="Derivative":
                point_drawDown_major=self.get_point_minDerivative(drawdown_cluster)
            elif self.mode=="Std":
                point_drawDown_major=self.get_point_maxStd(drawdown_cluster)
            else:
                raise Exception("The mode must be a string 'Tangent' ,'Derivative' or 'Std'")
            shutInperiod.append((point_drawDown_major,point_nextbuildup))
            
        return shutInperiod
            
                
    def get_point_minDerivative(self,drawdown_cluster:List[int])->int:
        # print("=============minDerivative===========drawdown_cluster",drawdown_cluster)
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
        print("===================",drawdown_cluster)
        print(tangent_df)
        delta_tangents=list(tangent_df["tangent_right"]-tangent_df["tangent_left"])
        index_minDeltaTangent=delta_tangents.index(min(delta_tangents))
   
        return drawdown_cluster[index_minDeltaTangent]
    
    def get_point_maxStd(self,
                         points:List[int],
                            time_halfWindow:float=None,
                            point_halfWindow:int=15)->int:
        pressure_inWindow=self.get_pressure_inWindow(self.pressure_measure,
                                                    self.pressure_time,
                                                    points,
                                                    time_halfWindow,
                                                    point_halfWindow)
        std_points=[statistics.stdev(pressure) for pressure in pressure_inWindow]
        index_maxStd=std_points.index(max(std_points))
        return points[index_maxStd]
    
    def get_points_signSwitch(self,
                         pressure_measure:List[float],
                        pressure_time:List[float],
                         points:List[int],
                            time_halfWindow:float=None,
                            point_halfWindow:int=15)->List[int]:
        """
        search a point where the derivative changed its sign and 
        then remains the same sign for following points (period) 
        - in this case, first point of the derivative change is the break-point
        """
        points_buildUp=[]
        points_drawDown=[]
        data_inWindow=self.extract_points_inWindow(pressure_measure,
                                            pressure_time,
                                            points,
                                            time_halfWindow,
                                            point_halfWindow)
        for i in range(len(data_inWindow)):
            pressure_measure_rightWindow=data_inWindow["pressure_measure_right"][i]
            point_type=self.data_rightWindow(pressure_measure_rightWindow)
            if point_type=="buildUp":
                points_buildUp.append(data_inWindow["point_index"][i])
            elif point_type=="drawDown":
                points_drawDown.append(data_inWindow["point_index"][i])
            else:
                pass
                     
    
    def checkPoint_bysign(self,data_rightWindow):
    
        sign_remainings=[data_rightWindow[i]>0 for i in range(1,len(data_rightWindow)) if data_rightWindow[i]!=0]

        if all(sign_remainings) and data_rightWindow[0]<=0:
            return "buildUp"
        elif (not any(sign_remainings)) and data_rightWindow[0]>=0:
            return "drawDown"
        else:
            return "Not break point"    
        
    def get_pressure_inWindow(self,
                              pressure_measure:List[float],
                            pressure_time:List[float],
                            points:List[int],
                            time_halfWindow:float=None,
                            point_halfWindow:int=None):

        
        data_inWindow=self.extract_points_inWindow(pressure_measure,
                                                    pressure_time,
                                                    points,
                                                    time_halfWindow,
                                                    point_halfWindow)
        pressure_inWindow=data_inWindow['pressure_measure_left']+data_inWindow['pressure_measure_right']
        return pressure_inWindow
    
    def find_flowingPeriods(self,shutInPeriods)->List[Tuple[int,int]]:
        major_buildUp,major_drawDown=self.convert_to_twoLists(shutInPeriods)
        
        #set the first point of dataset as first major buildup by default
        # if self.points_buildUp[0]<self.shutInPeriods[0][0]:
        major_buildUp=[0]+major_buildUp
        flowingPeriods=[]
        for buildup in major_buildUp:
            drawdown_larger=list(filter(lambda i: i > buildup, major_drawDown))
            if len(drawdown_larger)==0:
                break
            drawdown =drawdown_larger[0] 
            flowingPeriods.append((buildup,drawdown))
            
        #process buildup points after last major drawdown    
        if self.major_buildUp[-1]>self.major_drawDown[-1]:
            flowingPeriods.append((major_buildUp[-1],self.points_buildUp[-1]))
        return flowingPeriods
    
    def remove_minorTransients_shutIn(self,shutInperiods,minor_threshold:float):
        filtered_shutIns=[]
        std_transients_lst=[]
        for drawDown, buildUp in shutInperiods:
            transients_pressure=self.pressure_df[self.colum_names["pressure"]["measure"]].iloc[drawDown:buildUp]
            if len(transients_pressure)<2:
                continue
            std_transients=statistics.stdev(transients_pressure)
            std_transients_lst.append(std_transients)    
            if abs(std_transients)>minor_threshold*abs(self.std_pressure):
                filtered_shutIns.append((drawDown, buildUp))
        # print("std_transients_lst",std_transients_lst)
        return filtered_shutIns
    
    # def remove_minorTransients_shutIn(self,shutInperiods,minor_threshold:float):
    #     filtered_shutIns=[]
    #     for drawDown, buildUp in shutInperiods:
    #         delta_pressure=self.pressure_df[self.colum_names["pressure"]["measure"]].iloc[drawDown]-self.pressure_df[self.colum_names["pressure"]["measure"]].iloc[buildUp]
    
    #         if abs(delta_pressure)>minor_threshold*(self.pressure_max-self.pressure_min):
    #             filtered_shutIns.append((drawDown, buildUp))
    #     # print("std_transients_lst",std_transients_lst)
    #     return filtered_shutIns
    
  
    
    def convert_to_twoLists(self,shutInPeriods)->(List[int],List[int]):
        major_drawDown=[]
        major_buildUp=[]
        # #if the first buildup point is smaller than first drawdown, 
        # #set it as first major buildup by default
        # if self.points_buildUp[0]<shutInPeriods[0][0]:
        #     major_buildUp.append(self.points_buildUp[0])
        for drawDown, buildUp in shutInPeriods:
            major_drawDown.append(drawDown)
            major_buildUp.append(buildUp)
        return major_buildUp,major_drawDown
    
    def find_breakPoints_inFlowingPeriods(self,
                                          flowingPeriods:List[Tuple[int,int]],
                                          minor_threshold:float):
        flowingTransient_lst=[]
        for flowingPeriod in flowingPeriods:
            flowingTransient_lst.append(FlowingTransient(self.pressure_df, 
                                                         flowingPeriod,
                                                         minor_threshold,
                                                         self.points_buildUp,
                                                         self.colum_names))
        return flowingTransient_lst
    
    def twoSequentIdentification(self,minor_threshold_shutIn,minor_threshold_Flowing):
        shutInperiods=self.find_shutInPeriods()
        print("====detected_shutIns",len(shutInperiods))
        filtered_shutIns=self.remove_minorTransients_shutIn(shutInperiods,minor_threshold_shutIn)
        print("====filtered_shutIns",len(filtered_shutIns))
        if len(filtered_shutIns)==0:
            raise Exception("No shut-in detected.")
        self.shutInperiods=filtered_shutIns
        self.major_buildUp,self.major_drawDown=self.convert_to_twoLists(filtered_shutIns)
        flowingPeriods=self.find_flowingPeriods(filtered_shutIns) 
        print("====len(flowingPeriods)",len(flowingPeriods))
        self.flowingPeriods=flowingPeriods
        
        self.flowingTransient_objects=self.find_breakPoints_inFlowingPeriods(flowingPeriods,minor_threshold_Flowing)
        
    def get_allPointsStored(self)->Dict[str,List[int]]:
        #copy list
        points_buildUp=self.major_buildUp.copy()
        for flowingTransient_object  in self.flowingTransient_objects:
            points_buildUp+=flowingTransient_object.points_inFlowTransient
            points_buildUp.sort()
            
        print(f"====finally detect buildUp:{len(points_buildUp)},drawDown:{len(self.major_drawDown)}")
            
        return {"buildUp":points_buildUp,
                "drawDown":self.major_drawDown}

        
        
            
            
    
    
    
 
        
        
class FlowingTransient:
    def __init__(self, 
                 pressure_df:pd.DataFrame,
                 flowing_period:Tuple[int,int],
                 minor_threshold:float,
                 points_buildUp:List[int]=None,
                #  points_drawDown:List[int]=None,
                 colum_names:Dict[str,Dict[str,str]]
                   ={"pressure":{"time":"Elapsed time",
                                 "measure":"Data",
                                 "first_order_derivative":"first_order_derivative",
                                 "second_order_derivative":"second_order_derivative"},
                    "rate":{"time":"Elapsed time","measure":"Liquid rate"}}
                   )->None:
        self.pressure_df=pressure_df 
        self.flowing_period=flowing_period
        self.minor_threshold=minor_threshold
        self.colum_names=colum_names
        self.pressure_measure=list(pressure_df[self.colum_names["pressure"]["measure"]])
        self.pressure_time=list(pressure_df[self.colum_names["pressure"]["time"]])
        self.std_FOD=statistics.stdev(list(pressure_df[self.colum_names["pressure"]["first_order_derivative"]]))
        self.points_buildUp=points_buildUp
        # self.points_drawDown=points_drawDown
    
        self.std_pressure=statistics.stdev(self.pressure_measure)
        self.points_inFlowTransient=self.remove_minorTransients_flowing(self.flowing_period,self.minor_threshold)
        
    def remove_minorTransients_flowing(self,flowing_period,minor_threshold):
        # print("================remove minor in flowing period",flowing_period)
        filtered_breakPoints=[]
        points_FlowingTransients=[point for point in self.points_buildUp if point>flowing_period[0] and point<flowing_period[1]]
        # print("+++++++++++++before remove, the number of points:",len(points_FlowingTransients))
       

        for point in points_FlowingTransients:
            if len(filtered_breakPoints)==0:
                start_point=flowing_period[0]
            else:
                start_point=filtered_breakPoints[-1]
            
            if not self.is_minorTransient(start_point,point,minor_threshold):
                filtered_breakPoints.append(point)  
                # print("filtered_breakPoints",filtered_breakPoints)
      
        # print("++++++++++++++filtered points ",len(filtered_breakPoints),filtered_breakPoints)
        return filtered_breakPoints
        
    # def remove_minorTransients_flowing(self,flowing_period,minor_threshold):
    #     # print("================remove minor in flowing period",flowing_period)
    #     filtered_breakPoints=[]
    #     points_FlowingTransients=[point for point in self.points_buildUp if point>flowing_period[0] and point<flowing_period[1]]
    #     # print("+++++++++++++before remove, the number of points:",len(points_FlowingTransients))
    #     exist_minor=True
    #     count=0
    #     while exist_minor:
    #         count+=1
    #         # print("***********while loop")
    #         exist_minor=False
    #         # print("--------exist minor",exist_minor)
    #         for i in range(len(points_FlowingTransients)-1):
    #             start_point=points_FlowingTransients[i]
    #             end_point=points_FlowingTransients[i+1]
    #             if self.is_minorTransient(start_point,end_point,minor_threshold):
    #                 exist_minor=True
    #                 # print("--------exist minor",exist_minor)
    #             else:
    #                 filtered_breakPoints+=[start_point,end_point]
    #             # print("filtered_breakPoints",filtered_breakPoints)
    #         filtered_breakPoints=list(set(filtered_breakPoints))
    #         filtered_breakPoints.sort()
    #         # print("++++++++++++++filtered points ",len(filtered_breakPoints),filtered_breakPoints)
    #         if len(filtered_breakPoints)==len(points_FlowingTransients):
    #             break
    #         points_FlowingTransients=filtered_breakPoints
    #         filtered_breakPoints=[]
            
    #     # print("while loop times",count)
    #     # print("final----points_FlowingTransients",len(points_FlowingTransients),points_FlowingTransients)
    #     return points_FlowingTransients
                
    
    def is_minorTransient(self,start_point,end_point,minor_threshold)->bool:
        transients_pressure=self.pressure_df[self.colum_names["pressure"]["measure"]].iloc[start_point:end_point]
        # print("transients_pressure",type(transients_pressure), len(transients_pressure))
        if len(transients_pressure)<2:
            return True
        std_transients=statistics.stdev(transients_pressure)
        # print("start_point,end_point,std_transients,minor_threshold:",start_point,end_point,std_transients,minor_threshold*abs(self.std_pressure))
        if abs(std_transients)<=minor_threshold*abs(self.std_pressure):
            return True
    
    # def is_minorTransient(self,start_point,end_point,minor_threshold)->bool:
    #     start_data=self.pressure_df.iloc[start_point]
    #     end_data=self.pressure_df.iloc[end_point]
    #     transients_pressure=self.pressure_df[self.colum_names["pressure"]["measure"]].iloc[start_point:end_point]
    #     # print("transients_pressure",type(transients_pressure), len(transients_pressure))
    #     if len(transients_pressure)<2:
    #         return True
    #     std_transients=statistics.stdev(transients_pressure)
    #     transient_slope=(start_data[self.colum_names["pressure"]["measure"]]-end_data[self.colum_names["pressure"]["measure"]])/start_data[self.colum_names["pressure"]["time"]]-end_data[self.colum_names["pressure"]["time"]]
    #     # print("start_point,end_point,std_transients,minor_threshold:",start_point,end_point,std_transients,minor_threshold*abs(self.std_pressure))
    #     if not (abs(std_transients)>=minor_threshold*abs(self.std_pressure) and abs(transient_slope)>=0.9*abs(self.std_FOD)):
    #         return True
        
    # def is_minorTransient(self,start_point,end_point,minor_threshold)->bool:
    #     start_data=self.pressure_df.iloc[start_point]
    #     end_data=self.pressure_df.iloc[end_point]
    #     transient_slope=(start_data[self.colum_names["pressure"]["measure"]]-end_data[self.colum_names["pressure"]["measure"]])/start_data[self.colum_names["pressure"]["time"]]-end_data[self.colum_names["pressure"]["time"]]
        
    #     # print("start_point,end_point,std_transients,minor_threshold:",start_point,end_point,std_transients,minor_threshold*abs(self.std_pressure))
    #     if abs(transient_slope)<=minor_threshold*abs(self.std_FOD):
    #         return True
        
            
            