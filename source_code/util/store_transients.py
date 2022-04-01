from typing import Callable, Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import math
import statistics

class StoreTransients:
    def __init__(self, 
                 pressure_df:pd.DataFrame,
                 points_buildUp:List[int],
                 points_drawDown:List[int],
                 colum_names:Dict[str,Dict[str,str]]
                   ={"pressure":{"time":"Elapsed time",
                                 "measure":"Data",
                                 "first_order_derivative":"first_order_derivative",
                                 "second_order_derivative":"second_order_derivative"},
                    "rate":{"time":"Elapsed time","measure":"Liquid rate"}})->None:
        self.pressure_df=pressure_df 
        self.points_buildUp=points_buildUp
        self.points_drawDown=points_drawDown
        self.colum_names=colum_names
        self.major_drawDown=None
        self.major_buildUp=None
     
        self.shutInperiods=self.find_shutInPeriods()
        self.flowingPeriods=self.find_flowingPeriods()
        
    def find_shutInPeriods(self)->List[Tuple[int,int]]:
        #copy self.points_drawDown list
        points_drawDown=[item for item in self.points_drawDown]
        
        major_drawDown=[]
        major_buildUp=[]
        #if the first buildup point is smaller than first drawdown, 
        #set it as first major buildup by default
        if self.points_buildUp[0]<points_drawDown[0]:
            major_buildUp.append(self.points_buildUp[0])
        shutInperiod=[]
        while len(points_drawDown)>0:
            drawdown_cluster=[]
            point_nextbuildup = list(filter(lambda i: i > points_drawDown[0], self.points_buildUp))[0]
            major_buildUp.append(point_nextbuildup)

            while len(points_drawDown)>0:
                if points_drawDown[0]>point_nextbuildup:
                    break
                drawdown_cluster.append(points_drawDown[0])
                points_drawDown.remove(points_drawDown[0])

            point_drawDown_minFOD=self.find_point_minDerivative(drawdown_cluster)
            shutInperiod.append((point_drawDown_minFOD,point_nextbuildup))
            major_drawDown.append(point_drawDown_minFOD)
            
        self.major_drawDown=major_drawDown
        self.major_buildUp=major_buildUp
        return shutInperiod
            
                
    def find_point_minDerivative(self,drawdown_cluster:List[int])->int:
        FODs=[self.pressure_df["first_order_derivative"][index] for index in drawdown_cluster]
        index_minFOD=FODs.index(min(FODs))
        return drawdown_cluster[index_minFOD]
    
    def find_flowingPeriods(self)->List[Tuple[int,int]]:
        flowingPeriods=[]
        for buildup in self.major_buildUp:
            drawdown_larger=list(filter(lambda i: i > buildup, self.major_drawDown))
            if len(drawdown_larger)==0:
                break
            drawdown =drawdown_larger[0] 
            flowingPeriods.append((buildup,drawdown))
        return flowingPeriods
            