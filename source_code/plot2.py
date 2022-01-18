import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from matplotlib import rcParams
from operator import itemgetter
from typing import Callable, Dict, List, Set, Tuple

def group_index(data, bin_start, bin_end, bin_step):
    x = np.array(data)
    bin_edges = np.arange(bin_start, bin_end + bin_step, bin_step)
    bin_number = bin_edges.size - 1
    cond = np.zeros((x.size, bin_number), dtype=bool)
    for i in range(bin_number):
        cond[:, i] = np.logical_and(bin_edges[i] < x,
                                    x < bin_edges[i+1])
    return [list(x[cond[:, i]]) for i in range(bin_number)]

# data=breakpoints_detected 
# bin_start=0
# bin_end=len(pressure_df[pressure_time])
# bin_step=100
# group_index(breakpoints_detected, 0, len(pressure_df[pressure_time]), N)


def plot_breakpoints(ax,ax_index,breakpoints_detected,ground_truth,pressure_df,pressure_time)->None:
    
    # vline_color="cyan"
    vline_color="dodgerblue"
    color_breakpoints_missed="gold"
    color_breakpoints_faultyDetected="fuchsia"
      
    #if ground truth not given, just plot detected points       
    if len(ground_truth)==0:
        for breakpoint in breakpoints_detected: 
            #specify the index of points in the second plot
            if ax_index==1:
                # ax.text(pressure_df[pressure_time][breakpoint], .5, f'{breakpoint}', rotation=70)
                ax.axvline(x=pressure_df[pressure_time][breakpoint],color=vline_color)
            else:
                ax.axvline(x=pressure_df[pressure_time][breakpoint],color=vline_color)
    
    #if ground truth is given, plot missed points, faulty detected points as well           
    if len(ground_truth)>0:  
        breakpoints_faultyDetected=[point for point in breakpoints_detected if point not in ground_truth]
        breakpoints_missed=[point for point in ground_truth if point not in breakpoints_detected]
        all_breakpoints=set(breakpoints_detected+ground_truth)
        
        for point in all_breakpoints:     
            if point in breakpoints_faultyDetected:
                ax.axvline(x=pressure_df[pressure_time][point],color=color_breakpoints_faultyDetected)
            elif point in breakpoints_missed:
                ax.axvline(x=pressure_df[pressure_time][point],color=color_breakpoints_missed)     
            else:
                ax.axvline(x=pressure_df[pressure_time][point],color=vline_color)
                
    return None
 
   
def plot_4_metrics(pressure_df:pd.DataFrame,
                   rate_df:pd.DataFrame,
                   breakpoints_detected:List[int],
                   ground_truth:List[int],
                   colum_names:Dict[str,List[str]]
                   ={"pressure":["Elapsed time","Data","first_order_derivative","second_order_derivative"],
                    "rate":["Elapsed time","Liquid rate"]})->None:
    
    pressure_time, pressure_measure,first_order_derivative,second_order_derivative=colum_names["pressure"]
    rate_time, rate_measure = colum_names["rate"]
   
    
    plt.close('all')
    rcParams.update({'figure.autolayout': True})
    fig, axs = plt.subplots(nrows=4, sharex=True, dpi=100,figsize=(20,15), gridspec_kw={'height_ratios': [5, 3,3,3]})
    fig.suptitle('pressure ~ rate ~ first derivative ~ second derivative', 
              **{'family': 'Arial Black', 'size': 22, 'weight': 'bold'},x=0.5, y=1.005)
    
    x_coordinates=[pressure_df[pressure_time],
                   rate_df[rate_time],
                   pressure_df[pressure_time],
                   pressure_df[pressure_time]]
    y_coordinates=[pressure_df[pressure_measure],
                   rate_df[rate_measure],
                   pressure_df[first_order_derivative],
                   pressure_df[second_order_derivative]]
    # scatter_colors=['blue','green','black','limegreen']
    scatter_colors=['red','green','orangered','limegreen']
    scatter_sizes=[4**2,6**2,5**2,5**2]   
    y_labels=[pressure_measure,rate_measure,first_order_derivative,second_order_derivative]
    
    for i,(ax, x,y,color,size,y_label) in enumerate(zip(axs, x_coordinates,y_coordinates,scatter_colors,scatter_sizes,y_labels)):
        ax.scatter(x=x,y=y,color=color,s=size) 
        ax.set_ylabel(y_label,fontsize=16) 
        plot_breakpoints(ax,i,breakpoints_detected,ground_truth,pressure_df,pressure_time)
    

    fig.subplots_adjust(bottom=0.1, top=0.9)
    plt.show()       
    return None
  
  
def plot_4_metrics_details(data_inOneRow:int,
                           pressure_df:pd.DataFrame,
                           rate_df:pd.DataFrame,
                           breakpoints_detected:List[int],
                           ground_truth:List[int],
                           colum_names:Dict[str,List[str]]
                               ={"pressure":["Elapsed time","Data","first_order_derivative","second_order_derivative"],
                                "rate":["Elapsed time","Liquid rate"]})->None:
    
    pressure_time, pressure_measure,first_order_derivative,second_order_derivative=colum_names["pressure"]
    rate_time, rate_measure = colum_names["rate"]
    
    size=len(pressure_df)
    grouped_pressure_df = [pressure_df.iloc[x:x+data_inOneRow,:] for x in range(0, len(pressure_df), data_inOneRow)]
    grouped_breakpoints=group_index(breakpoints_detected, 0, size, data_inOneRow)
    print(f"The plot is devided into {len(grouped_breakpoints)} rows")
    grouped_gound_truth=group_index(ground_truth, 0, size, data_inOneRow)
    # count_breakpoints=0

    for i,(sub_pressure_df,sub_breakpoints, sub_ground_truth) in enumerate(zip(grouped_pressure_df,grouped_breakpoints,grouped_gound_truth)):
        #print
        if len(grouped_gound_truth)==0:
            print(f"------row {i+1}-----detected points:{sub_breakpoints}")
        if len(grouped_gound_truth)!=0:
            print(f"------row {i+1}-----faulty points:{[point for point in sub_breakpoints if point not in sub_ground_truth]}")
            print(f"------row {i+1}-----missed points:{[point for point in sub_ground_truth if point not in sub_breakpoints]}")  

        # count_breakpoints+=len(sub_breakpoints)   
        #extract rate data for subplot     
        start_time=sub_pressure_df.iloc[0][pressure_time]
        end_time=sub_pressure_df.iloc[-1][pressure_time]
        sub_rate_df=rate_df.loc[(rate_df[rate_time] >= start_time) & (rate_df[rate_time] <= end_time)]
        
        plot_4_metrics(sub_pressure_df,
                       sub_rate_df,
                       sub_breakpoints,
                       sub_ground_truth,
                       colum_names)
    # print("count_breakpoints",count_breakpoints)
    return None  

def plot_detection_statistics(breakpoints_detected:List[int],ground_truth:List[int])->None:
    if len(ground_truth)==0:
        return None
    breakpoints_faultyDetected=[point for point in breakpoints_detected if point not in ground_truth]
    breakpoints_correctDetected=[point for point in breakpoints_detected if point in ground_truth]
    print("breakpoints_faultyDetected",breakpoints_faultyDetected)
    breakpoints_missed=[point for point in ground_truth if point not in breakpoints_detected]
    print("breakpoints_missed",breakpoints_missed)

    # creating the dataset for bar plot
    data = {'points correct':round(len(breakpoints_correctDetected)/len(ground_truth) ,3), 
            'points faulty':round(len(breakpoints_faultyDetected)/len(ground_truth) ,3),  
            'points missed':round(len(breakpoints_missed)/len(ground_truth) ,3), }
    bars = list(data.keys())
    values = list(data.values())
    
    # creating the bar plot
    fig = plt.figure(figsize = (7, 4))
    plt.bar(bars, values, color=['dodgerblue','fuchsia',  'gold'])

    for index, value in enumerate(values):
        plt.text(index-0.06, value+0.01,f"{round(value*100,3)}%")
        
    plt.xlabel("")
    plt.ylabel("Percentage")
    plt.title("")
    plt.show()
    return None