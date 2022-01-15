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

# data=breakpoints 
# bin_start=0
# bin_end=len(pressure_df[pressure_time])
# bin_step=100
# group_index(breakpoints, 0, len(pressure_df[pressure_time]), N)


# def plot_4_metrics(pressure_df:pd.DataFrame,
#                    rate_df:pd.DataFrame,
#                    breakpoints:List[int],
#                    colum_names:Dict[str,List[str]]
#                    ={"pressure":["Elapsed time","Data","first_order_derivative","second_order_derivative"],
#                     "rate":["Elapsed time","Liquid rate"]})->None:
    
#     pressure_time, pressure_measure,first_order_derivative,second_order_derivative=colum_names["pressure"]
#     rate_time, rate_measure = colum_names["rate"]
#     vline_color="cyan"
    
#     plt.close('all')
#     # plt.figure()
#     rcParams.update({'figure.autolayout': True})
#     fig, axs = plt.subplots(nrows=4, sharex=True, dpi=100,figsize=(20,14), gridspec_kw={'height_ratios': [5, 3,3,3]})
#     fig.suptitle('pressure ~ rate ~ first derivative ~ second derivative', 
#               **{'family': 'Arial Black', 'size': 22, 'weight': 'bold'})
 
#     #plot pressure and breakpoints
#     axs[0].scatter(x=pressure_df[pressure_time],y=pressure_df[pressure_measure],color='red',s=3**2) 
#     axs[0].set_ylabel(pressure_measure,fontsize=16)
 
#     for breakpoint_index in breakpoints:
# #         axs[0].text(pressure_df[pressure_time][breakpoint_index], .5, f'{breakpoint_index}', rotation=70)
#         axs[0].axvline(x=pressure_df[pressure_time][breakpoint_index],color=vline_color)
                
                

#     #plot rate and breakpoints       
#     start_time=pressure_df.iloc[0][pressure_time]
#     end_time=pressure_df.iloc[-1][pressure_time]
#     rate_df=rate_df.loc[(rate_df[rate_time] >= start_time) & (rate_df[rate_time] <= end_time)]

#     axs[1].scatter(x=rate_df[rate_time],y=rate_df[rate_measure],color='orangered',s=4**2) 
#     axs[1].set_ylabel(rate_measure,fontsize=16)
#     for breakpoint_index in breakpoints:
#         axs[1].text(pressure_df[pressure_time][breakpoint_index], .5, f'{breakpoint_index}', rotation=70)
#         axs[1].axvline(x=pressure_df[pressure_time][breakpoint_index],color=vline_color)
            

#     #plot first derivative and breakpoints
#     axs[2].scatter(x=pressure_df[pressure_time],y=pressure_df[first_order_derivative],color='blue',s=3**2) 
#     axs[2].set_ylabel(first_order_derivative,fontsize=16)

#     for breakpoint_index in breakpoints:
#             axs[2].axvline(x=pressure_df[pressure_time][breakpoint_index],color=vline_color)
   

#     #plot second derivative and breakpoints
#     axs[3].scatter(x=pressure_df[pressure_time],y=pressure_df[second_order_derivative],color='green',s=3**2) 
#     axs[3].set_ylabel(second_order_derivative,fontsize=16)
#     for breakpoint_index in breakpoints:
#             axs[3].axvline(x=pressure_df[pressure_time][breakpoint_index],color=vline_color)
    
#     plt.show()       
#     return None
    
    
    
# def plot_4_metrics_details(data_inOneRow:int,
#                            pressure_df:pd.DataFrame,
#                            rate_df:pd.DataFrame,
#                            breakpoints:List[int],
#                            colum_names:Dict[str,List[str]]
#                                ={"pressure":["Elapsed time","Data","first_order_derivative","second_order_derivative"],
#                                 "rate":["Elapsed time","Liquid rate"]})->None:
    
#     pressure_time, pressure_measure,first_order_derivative,second_order_derivative=colum_names["pressure"]
#     rate_time, rate_measure = colum_names["rate"]
    
#     size=len(pressure_df)
#     sub_pressure_df = [pressure_df.iloc[x:x+data_inOneRow,:] for x in range(0, len(pressure_df), data_inOneRow)]
#     grouped_breakpoints=group_index(breakpoints, 0, size, data_inOneRow)
# #     print(grouped_breakpoints)
#     print(len(grouped_breakpoints))
#     count_breakpoints=0

#     for sub_pressure_df,sub_breakpoints in zip(sub_pressure_df,grouped_breakpoints):
#         count_breakpoints+=len(sub_breakpoints)
            
#         #extract rate data for subplot     
#         start_time=sub_pressure_df.iloc[0][pressure_time]
#         end_time=sub_pressure_df.iloc[-1][pressure_time]
#         sub_rate_df=rate_df.loc[(rate_df[rate_time] >= start_time) & (rate_df[rate_time] <= end_time)]
        
#         plot_4_metrics(sub_pressure_df,
#                sub_rate_df,
#                sub_breakpoints,colum_names)
#     print("count_breakpoints",count_breakpoints)
#     return None


def plot_breakpoints(ax,ax_index,breakpoints,faulty_detectedBreakpoints,pressure_df,pressure_time)->None:
    
    vline_color="cyan"
    transient_color="fuchsia"
    
    for breakpoint in breakpoints:          
        if len(faulty_detectedBreakpoints)==0:
            if ax_index==1:
                ax.text(pressure_df[pressure_time][breakpoint], .5, f'{breakpoint}', rotation=70)
                ax.axvline(x=pressure_df[pressure_time][breakpoint],color=vline_color)
            else:
                ax.axvline(x=pressure_df[pressure_time][breakpoint],color=vline_color)
        else:       
            if breakpoint in faulty_detectedBreakpoints:
                #plot faulty detected transients
                temp_index=faulty_detectedBreakpoints.index(breakpoint)
                if temp_index%2==0:
                    faulty_detected_transient=[faulty_detectedBreakpoints[temp_index],faulty_detectedBreakpoints[temp_index+1]] 
                else:
                    faulty_detected_transient=[faulty_detectedBreakpoints[temp_index-1],faulty_detectedBreakpoints[temp_index]] 
                
                #if a transient is divided into two subplot
                if faulty_detected_transient[0]<pressure_df.index[0]:
                    faulty_detected_transient[0]=pressure_df.index[0]
                if faulty_detected_transient[1]>pressure_df.index[-1]:
                    faulty_detected_transient[1]=pressure_df.index[-1]

                ax.axvspan(pressure_df[pressure_time][faulty_detected_transient[0]], pressure_df[pressure_time][faulty_detected_transient[1]], alpha=0.5, color=transient_color)
            else:
                ax.axvline(x=pressure_df[pressure_time][breakpoint],color=vline_color)
                
    return None
    
def plot_4_metrics(pressure_df:pd.DataFrame,
                   rate_df:pd.DataFrame,
                   breakpoints:List[int],
                   faulty_detectedBreakpoints:List[int],
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
                   pressure_df[second_order_derivative],
                   pressure_df[second_order_derivative]]
    scatter_colors=['blue','green','black','limegreen']
    scatter_sizes=[4**2,6**2,5**2,5**2]
    
    for i,(ax, x,y,color,size) in enumerate(zip(axs, x_coordinates,y_coordinates,scatter_colors,scatter_sizes)):
        ax.scatter(x=x,y=y,color=color,s=size) 
        ax.set_ylabel(pressure_measure,fontsize=16) 
        plot_breakpoints(ax,i,breakpoints,faulty_detectedBreakpoints,pressure_df,pressure_time)
    

    fig.subplots_adjust(bottom=0.1, top=0.9)
    plt.show()       
    return None
  
  
def plot_4_metrics_details(data_inOneRow:int,
                           pressure_df:pd.DataFrame,
                           rate_df:pd.DataFrame,
                           breakpoints:List[int],
                           faulty_detectedBreakpoints:List[int],
                           colum_names:Dict[str,List[str]]
                               ={"pressure":["Elapsed time","Data","first_order_derivative","second_order_derivative"],
                                "rate":["Elapsed time","Liquid rate"]})->None:
    
    pressure_time, pressure_measure,first_order_derivative,second_order_derivative=colum_names["pressure"]
    rate_time, rate_measure = colum_names["rate"]
    
    size=len(pressure_df)
    sub_pressure_df = [pressure_df.iloc[x:x+data_inOneRow,:] for x in range(0, len(pressure_df), data_inOneRow)]
    grouped_breakpoints=group_index(breakpoints, 0, size, data_inOneRow)
    print(len(grouped_breakpoints))
    count_breakpoints=0

    for sub_pressure_df,sub_breakpoints in zip(sub_pressure_df,grouped_breakpoints):
        count_breakpoints+=len(sub_breakpoints)
            
        #extract rate data for subplot     
        start_time=sub_pressure_df.iloc[0][pressure_time]
        end_time=sub_pressure_df.iloc[-1][pressure_time]
        sub_rate_df=rate_df.loc[(rate_df[rate_time] >= start_time) & (rate_df[rate_time] <= end_time)]
        
        plot_4_metrics(sub_pressure_df,
                       sub_rate_df,
                       sub_breakpoints,
                       faulty_detectedBreakpoints,
                       colum_names)
    print("count_breakpoints",count_breakpoints)
    return None  