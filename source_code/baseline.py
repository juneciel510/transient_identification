import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from matplotlib import rcParams
from operator import itemgetter
from typing import Callable, Dict, List, Set, Tuple


def detect_breakpoints(first_order_derivative):
    breakpoints=[]
    std=statistics.stdev(first_order_derivative)
    print(std)
    for i in range(len(first_order_derivative)-1):
        if (first_order_derivative[i]>0)^(first_order_derivative[i+1]>0) and abs(first_order_derivative[i+1])>2*std :
            breakpoints.append(i+1)

    return breakpoints




def detect_breakpoints_2(first_order_derivative,second_order_derivative):
    breakpoints=[]
    std_1=statistics.stdev(first_order_derivative)
    std_2=statistics.stdev(second_order_derivative)
    print(std_1,std_2)
    compare_point=first_order_derivative[0]
    for i in range(len(first_order_derivative)-1):
        if (compare_point>0)^(first_order_derivative[i+1]>0):
            if (abs(second_order_derivative[i+1])>2.4*std_2) and (abs(first_order_derivative[i+1])>2.4*std_1) :
                breakpoints.append(i+1)
                compare_point=first_order_derivative[i+1]
        else:
            compare_point=first_order_derivative[i+1]
            
 
    return breakpoints


def detect_breakpoints_3(first_order_derivative,second_order_derivative):
    breakpoints=[]
#     denoised_first_order_derivative=[]
    std_1=statistics.stdev(first_order_derivative)
    std_2=statistics.stdev(second_order_derivative)
    print(std_1,std_2)
    compare_point=first_order_derivative[0]
    for i in range(len(first_order_derivative)-1):
#         denoised_first_order_derivative.append(compare_point)
        if (first_order_derivative[i]>0)^(first_order_derivative[i+1]>0) and abs(first_order_derivative[i+1])>2.4*std_1:
            breakpoints.append(i+1)  
        elif (compare_point>0)^(first_order_derivative[i+1]>0):
            if (abs(second_order_derivative[i+1])>2.4*std_2) and (abs(first_order_derivative[i+1])>2.4*std_1) :
                breakpoints.append(i+1)
                compare_point=first_order_derivative[i+1]
        else:
            compare_point=first_order_derivative[i+1]
  
 
    return breakpoints


def detect_breakpoints_all(first_order_derivative):
    breakpoints=[]
    std=statistics.stdev(first_order_derivative)
    for i in range(len(first_order_derivative)):
        if first_order_derivative[i]>2.5*std:
            breakpoints.append(i+1)
    return breakpoints