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


class ExtractPoints_inWindow:
    """
    extract points in time window or point window
    Args:
        mode: set to be "extractOriginData" when want to extract orginal points data 
              set to be "forPatternRecognition" when use it in PatternRecognition class
    """
    def __init__(self,
                 coordinate_names:List[str]
                 =["pressure_time","pressure_measure"],
                 mode:str="forPatternRecognition"
                 ):
        self.coordinate_names=coordinate_names
        self.mode=mode

    def extract_singlePoint_inPointWindow(self,
                                          yCoordinate:List[float],
                                          xCoordinate:List[float],
                                          point_index:int,
                                          halfWinow_left:int,
                                          halfWinow_right:int
                                          )->Dict[str,List[float]]: 

        """
        extract data for a single point
        in point window  [point_index-halfWinow_left,point_index+halfWinow_right]
        
        Args:
            yCoordinate: pressure measure for the whole dataset
            xCoordinate: pressure time for the whole dataset
            point_index: index of points 
            halfWinow_left: the number of points to be extracted on the left side of the point_index
            halfWinow_right: the number of points to be extracted on the left side of the point_index
            
        Returns:
            a dictionary, a example of keys() is as follows:
            -------------
            ['point_index',
            'pressure_time_left', 
            'pressure_measure_left',
            'pressure_time_right', 
            'pressure_measure_right']
            -------------
        """
        yCoordinate=list(yCoordinate)
        xCoordinate=list(xCoordinate)
        if point_index-halfWinow_left<0 or point_index+halfWinow_right>=len(yCoordinate):
            return None
        
        data={"point_index":int(point_index)}
        
        #left side
        sub_measure_left=yCoordinate[point_index+1-halfWinow_left:point_index+1]
        sub_time_left=xCoordinate[point_index+1-halfWinow_left:point_index+1]
        
        
        #right side
        sub_measure_right=yCoordinate[point_index:point_index+halfWinow_right]
        sub_time_right=xCoordinate[point_index:point_index+halfWinow_right]
        
        
        
        names=["left","right"]
        sub_measures=[sub_measure_left,sub_measure_right]
        sub_times=[sub_time_left,sub_time_right]
       
        
        for name,sub_measure, sub_time in zip(names,sub_measures, sub_times):
            sub_measure = list(np.around(np.array(sub_measure),6))
            sub_time = list(np.around(np.array(sub_time),6))
            if self.mode=="forPatternRecognition":
                if name=="left":
                    curve_measure=[round(measure-sub_measure_left[len(sub_measure_left)-1],6) for measure in sub_measure_left]
                    curve_time=[round(time-sub_time_left[len(sub_time_left)-1],6) for time in sub_time_left]
                if name=="right":
                    curve_measure=[round(measure-sub_measure_right[0],6) for measure in sub_measure_right]
                    curve_time=[round(time-sub_time_right[0],6) for time in sub_time_right]
                
                data.update({f"{self.coordinate_names[0]}_{name}":curve_time,
                                f"{self.coordinate_names[1]}_{name}":curve_measure})
            elif self.mode=="extractOriginData":
                data.update({f"{self.coordinate_names[0]}_{name}":sub_time,
                                f"{self.coordinate_names[1]}_{name}":sub_measure})
            else:
                raise Exception("'mode' must be 'forPatternRecognition' or 'extractOriginData'")
                
        return data
    
    
    
    def extract_singlePoint_inTimeWindow(self,
                                    yCoordinate:List[float],
                                    xCoordinate:List[float],
                                    point_index:int,
                                    time_halfWindow:float,
                                    min_pointsNumber:int=8)->Dict[str,List[float]]: 
        """
        extract pressure measure & time data for 'points' 
        in 'timewindow' 
        if the number of points in the half timewindow is less than 'min_pointsNumber'
        then we extract 'min_pointsNumber' points
        
        Args:
            yCoordinate: pressure measure for the whole dataset
            xCoordinate: pressure time for the whole dataset
            points: a list contains index of points 
            time_halfWindow: half timewindow
            min_pointsNumber: minimum number of points if the time window  contains less points than 'min_pointsNumber'  
            
        Returns:
            a dictionary, a example of keys() is as follows:
            -------------
            ['point_index',
            'pressure_time_left', 
            'pressure_measure_left',
            'pressure_time_right', 
            'pressure_measure_right']
            -------------
        """     
        #convert timewindow to point window 
        time_leftStart=xCoordinate[point_index]-time_halfWindow
        time_rightEnd=xCoordinate[point_index]+time_halfWindow
        if time_leftStart>=0 and time_rightEnd<=xCoordinate[len(xCoordinate)-1]:
            halfWinow_left=point_index-bisect.bisect_left(xCoordinate, time_leftStart) 
            
            if halfWinow_left<min_pointsNumber:
                halfWinow_left=min_pointsNumber
            
            halfWinow_right=bisect.bisect_left(xCoordinate, time_rightEnd)-1-point_index
            if halfWinow_right<min_pointsNumber:
                halfWinow_right=min_pointsNumber
        
            data=self.extract_singlePoint_inPointWindow(yCoordinate,
                                                        xCoordinate,
                                                        point_index,
                                                        halfWinow_left,
                                                        halfWinow_right)
            
                
            return data
    
    def extract_singlePoint_inWindow(self,
                                yCoordinate:List[float],
                                xCoordinate:List[float],
                                point_index:int,
                                time_halfWindow:float=None,
                                point_halfWindow:int=None,
                                min_pointsNumber:int=8)->Dict[str,List[float]]:
        """
        extract data for a single point in a time window or point
        Args:
            yCoordinate: measurements
            xCoordinate: time
            time_halfWindow: set to be None, when use point window
            point_halfWindow: set to be None, when use time window
            min_pointsNumber: the minimum number of points that should be included for a time window
        Returns:
            A dictionary, a example of keys is as follows:
            ------------------
            ['point_index',
            'pressure_time_left', 
            'pressure_measure_left',
            'pressure_time_right', 
            'pressure_measure_right']
            --------------------- 
        """ 
      
        if time_halfWindow!=None and point_halfWindow!=None:
            raise Exception("either 'time window' or 'point_halfWindow' should set to be None")
        if time_halfWindow!=None:
            data=self.extract_singlePoint_inTimeWindow(yCoordinate,
                                                                xCoordinate,
                                                                point_index,
                                                                time_halfWindow,
                                                                min_pointsNumber)
            return data
        if point_halfWindow!=None:
            data=self.extract_singlePoint_inPointWindow(yCoordinate,
                                                            xCoordinate,
                                                            point_index,
                                                            point_halfWindow,
                                                            point_halfWindow)
            return data
        
        raise Exception(f"check time window:{time_halfWindow} and point window:{point_halfWindow}")
        
        
        
        
        
    
    def extract_points_inWindow(self,
                                yCoordinate:List[float],
                                xCoordinate:List[float],
                                points:List[int],
                                time_halfWindow:float=None,
                                point_halfWindow:int=None,
                                min_pointsNumber:int=8)->pd.DataFrame:
        """
         Args: 
            yCoordinate: values of y coordinates
            xCoordinate: values of x coordinates
            points: list of point indices for extraction
            time_halfWindow: set to be None if use point window
            point_halfWindow: set to be None if use time window
            min_pointsNumber: minimum number of points if the time window  contains less points than 'min_pointsNumber'  
        Returns:
            data frame, a example of columns name
            ------------------
            ['point_index',
            'pressure_time_left', 
            'pressure_measure_left',
            'pressure_time_right', 
            'pressure_measure_right']
            --------------------- 
        """
        data_inWindow=pd.DataFrame(columns=['point_index',
                                f"{self.coordinate_names[0]}_left", 
                                f"{self.coordinate_names[1]}_left",
                                f"{self.coordinate_names[0]}_right", 
                                f"{self.coordinate_names[1]}_right"])
        for point_index in points:
            data=self.extract_singlePoint_inWindow(yCoordinate,
                                                   xCoordinate,
                                                   point_index,
                                                   time_halfWindow,
                                                    point_halfWindow,
                                                    min_pointsNumber)
            
            if data is not None:
                data_inWindow=data_inWindow.append(data,ignore_index=True)
        # print("--------data_inWindow:",data_inWindow)
        return data_inWindow
            

