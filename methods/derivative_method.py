from typing import Callable, Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import math
import statistics
import bisect
from extract_points import ExtractPoints_inWindow

class DerivativeMethod(ExtractPoints_inWindow):
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
        ExtractPoints_inWindow.__init__(self,
                                coordinate_names=["pressure_time","first_order_derivative"],
                                mode="extractOriginData")
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
        index_first_FOD=[sub_pressure_df.index[0] for sub_pressure_df in sub_pressure_dfs if len(sub_pressure_df)>0]
        return index_first_FOD

    def FOD_above_threshold(self, points,noise_threshold):
            filtered_points=[]
            for point_index in points:
                    if abs(self.first_order_derivative[point_index])>noise_threshold*abs(self.std_1):
                            filtered_points.append(point_index)
            return filtered_points   
         
    def detect_breakpoints_rangeSingleFOD(self,
                           derivative_lst:List[float],
                           points:List[int],
                           close_zero_threshold:float,
                           tuning_parameters:float
                           )->(List[int],List[int]):
        buildUp=[]
        drawDown=[]
        for point_index in points:
            if point_index+1>len(derivative_lst)-1:
                continue
            if abs(derivative_lst[point_index])<close_zero_threshold and derivative_lst[point_index+1]>tuning_parameters*abs(self.std_1):
                    buildUp.append(point_index+1)
            elif abs(derivative_lst[point_index])<close_zero_threshold and derivative_lst[point_index+1]<-tuning_parameters*abs(self.std_1):
                        drawDown.append(point_index+1)
        return buildUp, drawDown
        

    
    def avg_derivative_inWindow(self,
                                derivative_lst:List[float],       
                                pressure_time:List[float],
                                points:List[int],
                                time_halfWindow:float=None,
                                point_halfWindow:int=None,
                                min_pointsNumber:int=8)->pd.DataFrame: 
        """
        extract derivative & time data for 'points' 
        in 'window' 
        
        Args:
            derivative_lst: derivative for the whole dataset
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
        avg_derivative_inWindow=pd.DataFrame(columns=['point_index',
                                'avg_derivative_left',
                                'avg_derivative_right'])
        
        for point_index in points: 
            data=self.extract_singlePoint_inWindow(derivative_lst,
                                                        pressure_time,
                                                        point_index,
                                                        time_halfWindow,
                                                        point_halfWindow,
                                                        min_pointsNumber)
            if data is None:
                continue
            #left side
            sub_derivative=data[f"{self.coordinate_names[1]}_left"]
            avg_derivative=round(sum(sub_derivative)/len(sub_derivative),6)
            data.update({"avg_derivative_left":avg_derivative})
            
            #right side
            sub_derivative=data[f"{self.coordinate_names[1]}_right"]
            avg_derivative=round(sum(sub_derivative)/len(sub_derivative),6)
            data.update({"avg_derivative_right":avg_derivative})
            
        
            avg_derivative_inWindow=avg_derivative_inWindow.append(data,ignore_index=True)
        return avg_derivative_inWindow
    
    
    def use_rangeFOD(self,
                           avg_derivative:pd.DataFrame,
                           close_zero_threshold:float,
                           tuning_parameters:float
                           )->(List[int],List[int]):
        """
        Use derivative range to detect
        """
        buildUp=[]
        drawDown=[]
        for index,row in avg_derivative.iterrows():
            if abs(row["avg_derivative_left"])<close_zero_threshold and row["avg_derivative_right"]>tuning_parameters*abs(self.std_1):
                    buildUp.append(row["point_index"])
            elif abs(row["avg_derivative_left"])<close_zero_threshold and row["avg_derivative_right"]<-tuning_parameters*abs(self.std_1):
                        drawDown.append(row["point_index"])
        return buildUp, drawDown
        
    def use_deltaFOD(self,
                             avg_derivative:pd.DataFrame,
                             deltaDerivative_tuning:float
                             )->(List[int],List[int]):
        """
        Use derivative delta to detect
        """
        buildup=[]
        drawdown=[]
        for index,row in avg_derivative.iterrows():
            if row["avg_derivative_right"]>0 and (row["avg_derivative_right"]-row["avg_derivative_left"])>deltaDerivative_tuning*abs(self.std_1):
                    buildup.append(int(row["point_index"]))
            if row["avg_derivative_right"]<0 and (row["avg_derivative_right"]-row["avg_derivative_left"])<-deltaDerivative_tuning*abs(self.std_1):
                    drawdown.append(int(row["point_index"]))
                
        return buildup,drawdown
        
    def detect_breakpoints_rangeAvgFOD(self,
                             points:pd.DataFrame,
                             close_zero_threshold:float,
                             tuning_parameters:float,
                             point_halfWindow:int=None,
                             time_halfWindow:float=None,
                             min_pointsNumber:int=8
                             )->(List[int],List[int]):
          
        avg_derivative_inWindow=self.avg_derivative_inTimeWindow(self.first_order_derivative,
                                                                self.pressure_time,
                                                                points,
                                                                time_halfWindow,
                                                                point_halfWindow,
                                                                min_pointsNumber)
        
        buildup,drawdown=self.use_rangeFOD(avg_derivative_inWindow,
                                                close_zero_threshold,
                                                tuning_parameters)
        
        return buildup,drawdown
    
    def detect_breakpoints_deltaAvgFOD(self,
                             points:pd.DataFrame,
                             deltaDerivative_tuning:float,
                             point_halfWindow:int=None,
                             time_halfWindow:float=None,
                             min_pointsNumber:int=8
                             )->(List[int],List[int]):
            
        avg_derivative_inWindow=self.avg_derivative_inWindow(self.first_order_derivative,
                                                                self.pressure_time,
                                                                points,
                                                                time_halfWindow,
                                                                point_halfWindow,
                                                                min_pointsNumber)
        
        buildup,drawdown=self.use_deltaFOD(avg_derivative_inWindow,deltaDerivative_tuning)
        
        return buildup,drawdown
    
    def detect_breakpoints_signSwitch(self,
                         derivative_lst:List[float],
                        pressure_time:List[float],
                         points:List[int],
                            time_halfWindow:float=None,
                            point_halfWindow:int=15,
                            min_pointsNumber:int=8)->List[int]:
        """
        search a point where the derivative changed its sign and 
        then remains the same sign for following points (period) 
        - in this case, first point of the derivative change is the break-point
        """
        buildup=[]
        drawdown=[]
        data_inWindow=self.extract_points_inWindow(derivative_lst,
                                            pressure_time,
                                            points,
                                            time_halfWindow,
                                            point_halfWindow,
                                            min_pointsNumber)
        for i in range(len(data_inWindow)):
            data_rightWindow=data_inWindow[f"{self.coordinate_names[1]}_right"][i]
            point_type=self.checkPoint_bysign(data_rightWindow)
            if point_type=="buildUp":
                buildup.append(data_inWindow["point_index"][i])
            elif point_type=="drawDown":
                drawdown.append(data_inWindow["point_index"][i])
            else:
                pass
        return buildup,drawdown
                     
    
    def checkPoint_bysign(self,data_rightWindow):
      
        sign_remainings=[data_rightWindow[i]>0 for i in range(1,len(data_rightWindow)) if data_rightWindow[i]!=0]
 
        if all(sign_remainings) and data_rightWindow[0]<=0:
            return "buildUp"
        elif (not any(sign_remainings)) and data_rightWindow[0]>=0:
            return "drawDown"
        else:
            return "Not break point"    
        
        
                