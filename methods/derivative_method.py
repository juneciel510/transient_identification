from typing import Callable, Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import math
import statistics
import bisect

class DerivativeMethod:
    def __init__(self,
                 pressure_df:pd.DataFrame,
                colum_names:Dict[str,Dict[str,str]]
                ={"pressure":{"time":"Elapsed time",
                            "measure":"Data",
                            "first_order_derivative":"first_order_derivative",
                            "second_order_derivative":"second_order_derivative"},
                "rate":{"time":"Elapsed time",
                        "measure":"Liquid rate"}}
                ):
        self.pressure_df=pressure_df
        self.pressure_measure=pressure_df[colum_names["pressure"]["measure"]]
        self.pressure_time=pressure_df[colum_names["pressure"]["time"]]
        self.first_order_derivative=pressure_df[colum_names["pressure"]["first_order_derivative"]]
        self.second_order_derivative=pressure_df[colum_names["pressure"]["second_order_derivative"]]
        self.std_1=statistics.stdev(self.first_order_derivative)
        
    def percentile_FOD(self, 
                        percentile_upperBound:float,
                        percentile_lowerBound:float)->List[int]:
        data=self.first_order_derivative
        upper_bound=np.percentile(data, percentile_upperBound, axis=0,method="normal_unbiased")
        lower_bound=np.percentile(data, percentile_lowerBound, axis=0,method="normal_unbiased")
        print(f"fliter derivatives which are larger than {upper_bound}, smaller than {lower_bound}")
        filtered_df=self.pressure_df.loc[(self.first_order_derivative <= lower_bound) | (self.first_order_derivative >= upper_bound)]
        return list(filtered_df.index)
    
    def detect_max_FOD(self,
                   time_step:float=2
                    )->List[int]:
        """
        get the indices of the points with maximum first_order_derivative in every time step
        """
        
        self.pressure_df["abs(first_order_derivative)"]=self.first_order_derivative.abs()
        max_time=list(self.pressure_time)[-1]
        group_number=math.ceil(max_time/time_step)
        #devide the pressure_df into multiple sub df according to time step
        sub_pressure_dfs=[self.pressure_df.loc[(self.pressure_time >= i*time_step) & (self.pressure_time  <= (i+1)*time_step)] for i in range(group_number)]
        #get the index of max absolute value of first order derivative
        index_max_FOD=[sub_pressure_df["abs(first_order_derivative)"].idxmax() for sub_pressure_df in sub_pressure_dfs if len(sub_pressure_df)>0]
        
        # filtered_points=[point_index for point_index in index_max_FOD if self.first_order_derivative[point_index]>0.02*self.std_1 ]
        return index_max_FOD
    
    def detect_first_FOD(self,
                   time_step:float=2
                    )->List[int]:
        """
        get the indices of the points with maximum first_order_derivative in every time step
        """
        
        self.pressure_df["abs(first_order_derivative)"]=self.first_order_derivative.abs()
        max_time=list(self.pressure_time)[-1]
        group_number=math.ceil(max_time/time_step)
        #devide the pressure_df into multiple sub df according to time step
        sub_pressure_dfs=[self.pressure_df.loc[(self.pressure_time >= i*time_step) & (self.pressure_time  <= (i+1)*time_step)] for i in range(group_number)]
        #get the index of max absolute value of first order derivative
        # index_max_FOD=[sub_pressure_df["abs(first_order_derivative)"].idxmax() for sub_pressure_df in sub_pressure_dfs if len(sub_pressure_df)>0]
        
        index_first_FOD=[sub_pressure_df.index[0] for sub_pressure_df in sub_pressure_dfs if len(sub_pressure_df)>0]
        # filtered_points=[point_index for point_index in index_max_FOD if self.first_order_derivative[point_index]>0.02*self.std_1 ]
        return index_first_FOD

    def FOD_above_threshold(self, points,noise_threshold):
            filtered_points=[]
            for point_index in points:
                    if abs(self.first_order_derivative[point_index])>noise_threshold*abs(self.std_1):
                            filtered_points.append(point_index)
            return filtered_points   
         
    def detect_breakpoints_singlePoint(self,
                        #    derivative_lst:List[float],
                           points:List[int],
                           close_zero_threshold:float,
                           tuning_parameters:float
                           )->(List[int],List[int]):
        buildUp=[]
        drawDown=[]
        derivative_lst=list(self.first_order_derivative)
        for point_index in points:
            if point_index+1>len(derivative_lst)-1:
                continue
            if abs(derivative_lst[point_index])<close_zero_threshold and derivative_lst[point_index+1]>tuning_parameters*abs(self.std_1):
                    buildUp.append(point_index+1)
            elif abs(derivative_lst[point_index])<close_zero_threshold and derivative_lst[point_index+1]<-tuning_parameters*abs(self.std_1):
                        drawDown.append(point_index+1)
        return buildUp, drawDown
        
    
            
    def avg_derivative_inTimeWindow(self,
                                        derivative_lst:List[float],
                                        pressure_time:List[float],
                                        points:List[int],
                                        time_halfWindow:float,
                                        min_pointsNumber:int=8)->pd.DataFrame: 
        """
        extract pressure derivative & time data for 'points' 
        in 'timewindow' 
        if the number of points in the half timewindow is less than 'min_pointsNumber'
        then we extract 'min_pointsNumber' points
        
        Args:
            derivative_lst: pressure derivative for the whole dataset
            pressure_time: pressure time for the whole dataset
            points: a list contains index of points 
            time_halfWindow: half timewindow
            
        Returns:
            a dataframe containing five columns, each row for a point
            -------------
            columns=['point_index',
                    'avg_derivative_left', 
                    'avg_derivative_right']
            -------------
        """
        # print("-------start to extract points data inTimeWindow")
        pressure_time=list(pressure_time)
        avg_derivative_inWindow=pd.DataFrame(columns=['point_index',
                                'avg_derivative_left',
                                'avg_derivative_right'])
        
        
        for point_index in points:
            if point_index>(len(derivative_lst)-min_pointsNumber) or point_index<min_pointsNumber:
                continue
            #convert timewindow to point window 
            time_leftStart=pressure_time[point_index]-time_halfWindow
            time_rightEnd=pressure_time[point_index]+time_halfWindow
            if time_leftStart>=0 and time_rightEnd<=pressure_time[-1]:
                halfWinow_left=point_index-bisect.bisect_left(pressure_time, time_leftStart) 
                
                if halfWinow_left<min_pointsNumber:
                    halfWinow_left=min_pointsNumber
                
                halfWinow_right=bisect.bisect_left(pressure_time, time_rightEnd)-1-point_index
                if halfWinow_right<min_pointsNumber:
                    halfWinow_right=min_pointsNumber
            
                data=self.extract_singlePoint_inPointWindow(derivative_lst,point_index,halfWinow_left,halfWinow_right)
            
                # data={"point_index":int(point_index)}
        
                # #left side
                # sub_derivative=derivative_lst[point_index+1-halfWinow_left:point_index+1]
                # avg_derivative=round(sum(sub_derivative)/len(sub_derivative),6)
                # data.update({"avg_derivative_left":avg_derivative})
                
                # #right side
                # sub_derivative=derivative_lst[point_index:point_index+halfWinow_right]
                # avg_derivative=round(sum(sub_derivative)/len(sub_derivative),6)
                # data.update({"avg_derivative_right":avg_derivative})
                avg_derivative_inWindow=avg_derivative_inWindow.append(data,ignore_index=True)
        return avg_derivative_inWindow

    def avg_derivative_inPointWindow(self,
                                    derivative_lst:List[float],
                                    points:List[int],
                                    point_halfWindow:int)->pd.DataFrame: 
        """
        extract pressure measure & time data for 'points' 
        in 'pointwindow' 
        e.g. 8 points at the left & 8 points at the right
        
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
        avg_derivative_inWindow=pd.DataFrame(columns=['point_index',
                                'avg_derivative_left',
                                'avg_derivative_right'])
        
        points_valid=[point for point in points if point-point_halfWindow>=0 and point+point_halfWindow<len(derivative_lst) ]
        for point_index in points_valid: 
                # data={"point_index":int(point_index)}
                # #left side
                # sub_derivative=derivative_lst[point_index+1-point_halfWindow:point_index+1]
                # avg_derivative=round(sum(sub_derivative)/len(sub_derivative),6)
                # data.update({"avg_derivative_left":avg_derivative})
                
                # #right side
                # sub_derivative=derivative_lst[point_index:point_index+point_halfWindow]
                # avg_derivative=round(sum(sub_derivative)/len(sub_derivative),6)
                # data.update({"avg_derivative_right":avg_derivative})
                data=self.extract_singlePoint_inPointWindow(derivative_lst,point_index,point_halfWindow,point_halfWindow)
            
                avg_derivative_inWindow=avg_derivative_inWindow.append(data,ignore_index=True)
        return avg_derivative_inWindow

                             
    def extract_singlePoint_inPointWindow(self,
                                          derivative_lst:List[float],
                                          point_index:int,
                                          halfWinow_left:int,
                                          halfWinow_right:int
                                          )->Dict[str,List[float]]: 

        """
        extract pressure derivative & time data for a single point
        in point window  [point_index-halfWinow_left,point_index+halfWinow_right]
        
        Args:
            derivative_lst: pressure derivative for the whole dataset
            pressure_time: pressure time for the whole dataset
            point_index: index of points 
            halfWinow_left: the number of points to be extracted on the left side of the point_index
            halfWinow_right: the number of points to be extracted on the left side of the point_index
            
        Returns:
            a dictionary, the keys() are as follows:
            -------------
            ['point_index',
            'avg_derivative_left',
            'avg_derivative_right']
            -------------
        """
        data={"point_index":int(point_index)}
        
        #left side
        sub_derivative=derivative_lst[point_index+1-halfWinow_left:point_index+1]
        # sub_time=pressure_time[point_index+1-halfWinow_left:point_index+1]
        # curve_pressure=[round(derivative-sub_derivative[-1],6) for derivative in sub_derivative]
        # curve_time=[round(time-sub_time[-1],6) for time in sub_time]
        avg_derivative=round(sum(sub_derivative)/len(sub_derivative),6)
        data.update({"avg_derivative_left":avg_derivative})
        
        #right side
        sub_derivative=derivative_lst[point_index:point_index+halfWinow_right]
        # sub_time=pressure_time[point_index:point_index+halfWinow_right]
        # curve_pressure=[round(derivative-sub_derivative[0],6) for derivative in sub_derivative]
        # curve_time=[round(time-sub_time[0],6) for time in sub_time]
        avg_derivative=round(sum(sub_derivative)/len(sub_derivative),6)
        data.update({"avg_derivative_right":avg_derivative})
        return data

    def detect_breakpoints(self,
                           avg_derivative:pd.DataFrame,
                           close_zero_threshold:float,
                           tuning_parameters:float
                           )->(List[int],List[int]):
        buildUp=[]
        drawDown=[]
        for index,row in avg_derivative.iterrows():
            if abs(row["avg_derivative_left"])<close_zero_threshold and row["avg_derivative_right"]>tuning_parameters*abs(self.std_1):
                    buildUp.append(row["point_index"])
            elif abs(row["avg_derivative_left"])<close_zero_threshold and row["avg_derivative_right"]<-tuning_parameters*abs(self.std_1):
                        drawDown.append(row["point_index"])
        return buildUp, drawDown
        
    def detect_breakpoints_2(self,
                             avg_derivative:pd.DataFrame,
                             deltaDerivative_tuning:float
                             )->(List[int],List[int]):
        buildup=[]
        drawdown=[]
        for index,row in avg_derivative.iterrows():
            if row["avg_derivative_right"]>0 and (row["avg_derivative_right"]-row["avg_derivative_left"])>deltaDerivative_tuning*abs(self.std_1):
                    buildup.append(int(row["point_index"]))
            if row["avg_derivative_right"]<0 and (row["avg_derivative_right"]-row["avg_derivative_left"])<-deltaDerivative_tuning*abs(self.std_1):
                    drawdown.append(int(row["point_index"]))
                
        return buildup,drawdown
        
    def detect_breakpoints_3(self,
                             points:pd.DataFrame,
                             close_zero_threshold:float,
                             tuning_parameters:float,
                             point_halfWindow:int=None,
                             time_halfWindow:float=None,
                             )->(List[int],List[int]):
          
        if point_halfWindow!=None and time_halfWindow!=None:
            print("point_halfWindow and time_halfWindow, one of them should be none")
            
        if point_halfWindow!=None:            
            avg_derivative_inWindow=self.avg_derivative_inPointWindow(self.first_order_derivative,
                                                                    points,
                                                                    point_halfWindow)
        if time_halfWindow!=None:            
            avg_derivative_inWindow=self.avg_derivative_inTimeWindow(self.first_order_derivative,
                                                                     self.pressure_time,
                                                                    points,
                                                                    time_halfWindow)
        
        buildup,drawdown=self.detect_breakpoints(avg_derivative_inWindow,
                                                close_zero_threshold,
                                                tuning_parameters)
        
        return buildup,drawdown
    
    def detect_breakpoints_deltaAvgFOD(self,
                             points:pd.DataFrame,
                             deltaDerivative_tuning:float,
                             point_halfWindow:int=None,
                             time_halfWindow:float=None,
                             )->(List[int],List[int]):
          
        if point_halfWindow!=None and time_halfWindow!=None:
            print("point_halfWindow and time_halfWindow, one of them should be none")
            
        if point_halfWindow!=None:            
            avg_derivative_inWindow=self.avg_derivative_inPointWindow(self.first_order_derivative,
                                                                    points,
                                                                    point_halfWindow)
        if time_halfWindow!=None:            
            avg_derivative_inWindow=self.avg_derivative_inTimeWindow(self.first_order_derivative,
                                                                     self.pressure_time,
                                                                    points,
                                                                    time_halfWindow)
        
        buildup=[]
        drawdown=[]
        for index,row in avg_derivative_inWindow.iterrows():
            if row["avg_derivative_right"]>0 and (row["avg_derivative_right"]-row["avg_derivative_left"])>deltaDerivative_tuning*abs(self.std_1):
                    buildup.append(int(row["point_index"]))
            if row["avg_derivative_right"]<0 and (row["avg_derivative_right"]-row["avg_derivative_left"])<-deltaDerivative_tuning*abs(self.std_1):
                    drawdown.append(int(row["point_index"]))
        
        return buildup,drawdown
        
                