import streamlit as st
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from typing import Callable, Dict, List, Set, Tuple
import tempfile
from fpdf import FPDF 
import base64
import uuid,re
from scipy.signal import savgol_filter

class LoadNPreprocessData:
    def __init__(self,
                 pressure_df:pd.DataFrame, 
                 rate_df:pd.DataFrame, 
                 colum_names:Dict[str,Dict[str,str]], 
                 skip_rows:int=2,
                 sheet_name:Dict[str,str]={"pressure":"Pressure",
                                           "rate":"Rate"},
                 use_SG_smoothing:bool=True,
                 window_length:int=199,
                 polyorder:int=3)->None:
        # self.pressure_filePath=pressure_filePath
        # self.rate_filePath=rate_filePath
        self.colum_names=colum_names
        self.skip_rows=skip_rows
        self.sheet_name=sheet_name
        
        self.use_SG_smoothing=use_SG_smoothing
        self.window_length=window_length
        self.polyorder=polyorder
        
        self.pressure_df=pressure_df
        self.rate_df=rate_df
        self.pressureNrate_df=pd.DataFrame()
        
        print("---load data from 'txt' or 'xlsx' files...")
        self.load_data()
        
        if self.use_SG_smoothing:
            print("---denoising data using S-G smoothing...")
            self.SG_smoothing()
        self.produce_pressure_4metrics()
        print("---The first & second order derivative has been calculated and appended to pressure dataframe")
        self.concatenate_pressureNRate()
    
    def load_data(self)->None:
        '''
        load data from 'txt' or 'xlsx' files
        convert time to hours in float if it is "timestamp" format
        '''
        #check if file is txt or xlsx
        # if self.pressure_filePath.split(".")[-1]=="txt" and self.rate_filePath.split(".")[-1]=="txt":
        #     self.pressure_df = pd.read_csv(self.pressure_filePath, 
        #                           delimiter=" ",
        #                           skiprows=self.skip_rows, 
        #                           names=[self.colum_names["pressure"]["time"], 
        #                                  self.colum_names["pressure"]["measure"]],
        #                           skipinitialspace = True) 
        #     self.rate_df = pd.read_csv(self.rate_filePath, 
        #                         delimiter=" ",
        #                         skiprows=self.skip_rows, 
        #                         names=[self.colum_names["rate"]["time"], 
        #                                 self.colum_names["rate"]["measure"]], 
        #                         skipinitialspace = True) 
        # elif self.pressure_filePath.split(".")[-1]=="xlsx" and self.rate_filePath.split(".")[-1]=="xlsx":
        #     self.pressure_df = pd.DataFrame(pd.read_excel(self.pressure_filePath, sheet_name=self.sheet_name["pressure"]))
        #     self.rate_df = pd.DataFrame(pd.read_excel(self.rate_filePath, sheet_name=self.sheet_name["rate"]))
        # else:
        #     print("only can load data from 'txt' or 'xlsx' files")
        
        #if time is timestamp, convert to float representing hours
        pressure_time_type= type(self.pressure_df[self.colum_names["pressure"]["time"]][0])  
        rate_time_type= type(self.rate_df[self.colum_names["rate"]["time"]][0])  
        
        if  (pressure_time_type is float) and  (rate_time_type is float):
            return None
        
        if (pressure_time_type is pd.Timestamp) and (rate_time_type is pd.Timestamp):
            timestamps=self.pressure_df[self.colum_names["pressure"]["time"]]
            start_timestamp=timestamps[0]
            self.pressure_df[self.colum_names["pressure"]["time"]]=self.convert_timestamp2hour(timestamps,start_timestamp)
            
            timestamps=self.rate_df[self.colum_names["rate"]["time"]]
            self.rate_df[self.colum_names["rate"]["time"]]=self.convert_timestamp2hour(timestamps,start_timestamp)
        else:
            print("check the time type")
        return None

    
    def concatenate_pressureNRate(self):
        self.pressureNrate_df = pd.concat([self.pressure_df, self.rate_df]).sort_values(by=self.colum_names["pressure"]["time"]) 
    
    def convert_timestamp2hour(self,timestamps,start_timestamp)->List[float]:
        hours=[0.0]*len(timestamps)
        for i in range(len(timestamps)-1):
            hours[i+1]=(timestamps[i+1]-start_timestamp).total_seconds()/3600        
        return hours
    
    def calculate_derivative(self,x_coordinate:List[float],y_coordinate:List[float])->List[float]:
        """
        calculate forward derivative, the last point use the backforward derivative
        Args:
                x_coordinate: the value of x coordinate
                y_coordinate: the value of y coordinate

            Returns:
                derivative.
        """
        if len(x_coordinate)!=len(y_coordinate):
            print(f"the length of x_coordinate '{len(x_coordinate)}' is not equal to the length of y_coordinate '{len(y_coordinate)}'")
            return None
        
        length=len(y_coordinate)
        
        derivative=[0.0]*length
        for i in range(length-1):
            derivative[i]=(y_coordinate[i+1]-y_coordinate[i])/(x_coordinate[i+1]-x_coordinate[i])

        #calculate for the last point
        derivative[-1]=(y_coordinate[length-1]-y_coordinate[length-2])/(x_coordinate[length-1]-x_coordinate[length-2])
        return derivative
    
    def produce_pressure_4metrics(self):
        x_coordinate=self.pressure_df[self.colum_names["pressure"]["time"]]
        first_order_derivative=self.calculate_derivative(x_coordinate,self.pressure_df[self.colum_names["pressure"]["measure"]])
        second_order_derivative=self.calculate_derivative(x_coordinate,first_order_derivative)

        #add first and second derivative to pressure_df dataframe
        self.pressure_df["first_order_derivative"]=first_order_derivative
        self.pressure_df["second_order_derivative"]=second_order_derivative
        # pd.set_option('display.max_rows', pressure_df.shape[0]+1)
        return None
        
    def SG_smoothing(self):
        denoised_pressure_measures=savgol_filter(self.pressure_df[self.colum_names["pressure"]["measure"]],
                                             self.window_length,
                                             self.polyorder)
        self.pressure_df[self.colum_names["pressure"]["measure"]]=denoised_pressure_measures


class PlotNSave:
    """
    plot the given dataset and the detected points
    if the ground truth points is given,
    plot the false detection in purple color, missed detection in golden color.
    Args:
        pressure_df: contains 4 columns: 
                        pressure measurements,
                        pressure time, 
                        first order derivative, 
                        second order derivative
        rate_df: contains 2 columns:
                        flow rate measurements,
                        flow rate time
        points_detected_dict: indices of detected points
                        e.g. {"buildUp":[1,6,10],"drawDown":[2,7,12]}
        ground_truth: indices of ground truth points
        data_inOneRow: the number of data points in one row of the plot
        plot_name: the name of the plot
        filename_toSave_whole: the file name to save the figure in which all dataset are plotted in one row
                        if do not need to save the figure, let it to be ""
        filename_toSave_details: the file name to save the figure in which dataset are plotted in multiple rows,
                        the number of points in each row is detemined by arg 'data_inOneRow'
                        if do not need to save the figure, let it to be ""
        plot_statistics:set it True if the statistics of detected points needed,otherwise False
        plot_whole:set it True if want to plot the whole dataset in one row
        plot_details:set it True if want to plot the whole dataset into multiple rows,
                        the number of points in each row is detemined by arg 'data_inOneRow'
        colum_names: the column names of the pressure dataframe and rate dataframe
               
    Returns:
        None
    """
    def __init__(self, 
                   pressure_df:pd.DataFrame,
                   rate_df:pd.DataFrame,
                   points_detected_dict:Dict[str,List[int]],
                   ground_truth:List[int],
                   data_inOneRow:int,
                   plot_name:str,
                   txt:str="",
                   filename_toSave_whole:str="",
                   filename_toSave_details:str="",
                   plot_statistics:bool=True,
                   plot_whole:bool=True,
                   plot_details:bool=True,
                   colum_names:Dict[str,Dict[str,str]]
                   ={"pressure":{"time":"Elapsed time",
                                 "measure":"Data",
                                 "first_order_derivative":"first_order_derivative",
                                 "second_order_derivative":"second_order_derivative"},
                    "rate":{"time":"Elapsed time","measure":"Liquid rate"}}):
        print("---initializing...")
        self.pressure_df=pressure_df
        self.rate_df=rate_df
        self.points_detected_dict=points_detected_dict
        self.points_detected=self.produce_detectedPoints_lst()
        self.ground_truth=ground_truth
        self.data_inOneRow=data_inOneRow
        self.plot_name=plot_name
        self.txt=txt
        self.filename_toSave_whole=filename_toSave_whole
        self.filename_toSave_details=filename_toSave_details
        self.colum_names=colum_names
        self.plot_statistics=plot_statistics
        self.plot_whole=plot_whole
        self.plot_details=plot_details
        self.plots_toSave=[]
        
        print("---plotting...")
        if self.plot_statistics:
            self.plot_detection_statistics()
        st.markdown("- ######  Overview Plot")
        self.plot_4_metrics(self.pressure_df,
                                self.rate_df,
                                self.points_detected,
                                self.ground_truth, 
                                self.plot_whole,
                                filename_toSave_whole!="")
        st.markdown("- ###### Zoom-in Plot")
        self.plot_4_metrics_details(self.plot_details)
    
    def produce_detectedPoints_lst(self):
        points_lst=list(self.points_detected_dict.values())
        points_lst=points_lst[0]+points_lst[1]
        points_lst.sort()
        return points_lst
    def get_metrics(self, 
                    pressure_df:pd.DataFrame,
                    rate_df:pd.DataFrame)->(List[float],List[float],List[float],List[float],List[float],List[float]):
        pressure_time=pressure_df[self.colum_names["pressure"]["time"]]
        pressure_measure=pressure_df[self.colum_names["pressure"]["measure"]]
        pressure_first_order_derivative=pressure_df[self.colum_names["pressure"]["first_order_derivative"]]
        pressure_second_order_derivative=pressure_df[self.colum_names["pressure"]["second_order_derivative"]]
        rate_time=rate_df[self.colum_names["rate"]["time"]]
        rate_measure=rate_df[self.colum_names["rate"]["measure"]]
        return (pressure_time, pressure_measure, pressure_first_order_derivative,pressure_second_order_derivative,rate_time,rate_measure)
    
    def group_index(self,
                    data:List[int], 
                    bin_start:int, 
                    bin_end:int, 
                    bin_step:int
                    )->List[List[int]]:
        """
        for numbers in a list, devide these numbers into multiple sub lists,
        according to the specified step
        Args: 
            data: a list of sorted numbers, 
            bin_start: the start of number, 
            bin_end: the end of number, 
            bin_step: step
        Returns:
            a list contains sublists   
        """
        x = np.array(data)
        bin_edges = np.arange(bin_start, bin_end + bin_step, bin_step)
        bin_number = bin_edges.size - 1
        cond = np.zeros((x.size, bin_number), dtype=bool)
        for i in range(bin_number):
            cond[:, i] = np.logical_and(bin_edges[i] <= x,
                                        x < bin_edges[i+1])
        return [list(x[cond[:, i]]) for i in range(bin_number)]

    def detected_points_categories_2(self,points_detected,ground_truth):
 
        '''
        Classify the detected points into 3 categories: points correct, points faulty, points missed.
        
        if the detected point is 10 points ahead or behind the true breakpoint,
        we think that point is correctly detected.
        Since sometimes smoothing may cause deviation of breakpoints.
        Args:
            breakpoints_detected: detected points,
            ground_truth: the ground truth of breakpoints
        Returns:
            3 lists: points_correct,points_faulty, points_missed
        ''' 
        points_faulty=points_detected.copy()
        points_missed=ground_truth.copy()
        points_correct=[]
        for point_detected in points_detected:
            for point_true in ground_truth:
                if abs(point_true-point_detected)<=10:
                    points_correct.append(point_detected)
                    points_faulty.remove(point_detected)
                    points_missed.remove(point_true)              
                    break
        
        points_correct.sort()
        points_faulty.sort()
        points_missed.sort()
        return points_correct,points_faulty, points_missed
        
    def detected_points_categories(self,points_detected,ground_truth):
        '''
        Classify the detected points into 3 categories: points correct, points faulty, points missed.
        Args:
            breakpoints_detected: detected points,
            ground_truth: the ground truth of breakpoints
        Returns:
            3 lists: points_correct,points_faulty, points_missed
        '''                   
        points_correct=[point for point in points_detected if point in ground_truth]
        points_correct.sort()
        points_faulty=[point for point in points_detected if point not in ground_truth]
        points_faulty.sort()
        points_missed=[point for point in ground_truth if point not in points_detected]
        points_missed.sort()
    
        return points_correct,points_faulty, points_missed

    def plot_breakpoints(self,
                         ax,
                         ax_index:int,
                         pressure_df:pd.DataFrame,
                         points_detected:List[int],
                         ground_truth:List[int],
                         )->None:
        '''
        plot vertical lines for breakpoint.
        if ground_truth=[], all points_detected are plot in same color.
        otherwise, also plot points faulty and points missed in other colors.
        Args:
            ax: subplot
            ax_index: index of subplot,
            pressure_df: pd.DataFrame contains 
                pressure measurements, pressure time, first order derivative, second order derivative
            points_detected:a list of indices of detected points
            ground_truth: a list of indices of ground truth points
        Returns:
            None
        '''  
        
        # vline_color="cyan"
        vline_color_buildUp="dodgerblue"
        vline_color_drawDown="gold"
        color_points_missed="gold"
        color_points_faulty="fuchsia"
        
        pressure_time=pressure_df[self.colum_names["pressure"]["time"]]
        
        #if ground truth not given, just plot detected points       
        if len(ground_truth)==0:
            for point in points_detected: 
                if point in self.points_detected_dict["buildUp"]:
                    ax.axvline(x=pressure_time[point],color=vline_color_buildUp,label='buildUp')
                else:
                    ax.axvline(x=pressure_time[point],color=vline_color_drawDown,label='drawDown')
        
        #if ground truth is given, also plot missed points, faulty detected points using different color           
        if len(ground_truth)>0:  
            points_correct,points_faulty,points_missed=self.detected_points_categories_2(points_detected,ground_truth)
            all_points=set(points_detected+ground_truth)
            
            for point in all_points:     
                if point in points_faulty:
                    ax.axvline(x=pressure_time[point],color=color_points_faulty,label='Points faulty')
                elif point in points_missed:
                    ax.axvline(x=pressure_time[point],color=color_points_missed,label='Points missed')     
                else:
                    ax.axvline(x=pressure_time[point],color=vline_color,label='Points correct')
            
        #many labels are produced, just lengend unique ones        
        handles, labels = ax.get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        ax.legend(handles, labels, loc='best',shadow=True, fontsize='large')
                     
        return None

    def plot_4_metrics(self, 
                       pressure_df:pd.DataFrame,
                       rate_df:pd.DataFrame,
                       points_detected:List[int],
                       ground_truth:List[int],
                       plot_show:bool=True,
                       plot_save:bool=False)->None: 
        """
        for the given dataset, plot 4 subplots: 
            pressure measurements
            rate measurements
            first order derivative
            second order derivative
        Args: 
            pressure_df: pd.DataFrame contains 
                pressure measurements, pressure time, first order derivative, second order derivative
            rate_df:pd.DataFrame contains 
                rate measurements, rate time
            points_detected: a list for the indices of detected points
            ground_truth: a list for the indices of ground truth points
        Returns:
            None  
        """
        rcParams.update({'figure.autolayout': True})
        fig, axs = plt.subplots(nrows=4, sharex=True, dpi=100,figsize=(20,13), gridspec_kw={'height_ratios': [3, 3,3,3]})
        (pressure_time, 
        pressure_measure, 
        pressure_first_order_derivative,
        pressure_second_order_derivative,
        rate_time,
        rate_measure)=self.get_metrics(pressure_df,rate_df)
        x_coordinates=[pressure_time,
                    rate_time,
                    pressure_time,
                    pressure_time]
        y_coordinates=[pressure_measure,
                    rate_measure,
                    pressure_first_order_derivative,
                    pressure_second_order_derivative]
        # scatter_colors=['blue','green','black','limegreen']
        scatter_colors=['red','green','orangered','limegreen']
        scatter_sizes=[4**2,6**2,5**2,5**2]   
        # y_labels=[self.colum_names["pressure"]["measure"],
        #           self.colum_names["rate"]["measure"],
        #           self.colum_names["pressure"]["first_order_derivative"],
        #           self.colum_names["pressure"]["second_order_derivative"]]
        y_labels=["Pressure (psia)",
                  "Liquid rate (STB/D)",
                  "First_order_derivative",
                  "Second_order_derivative"]
        hline_color="purple"
        
        for i,(ax, x,y,color,size,y_label) in enumerate(zip(axs, x_coordinates,y_coordinates,scatter_colors,scatter_sizes,y_labels)):
            ax.scatter(x=x,y=y,color=color,s=size) 
            ax.set_ylabel(y_label,fontsize=16) 
            ax.set_xlabel("Time (hr)",fontsize=16) 
            ax.tick_params(labelbottom=True)
            # ax.xaxis.tick_top()
            if len(points_detected)>0:
                self.plot_breakpoints(ax,
                                    i,
                                    pressure_df,
                                    points_detected,
                                    ground_truth)
            #only plot horizontal lines in subplot of rate measures
            if i==1:
                ax.axhline(y=0,color=hline_color)
                
        fig.subplots_adjust(bottom=0.1, top=0.9)
        fig.align_ylabels()
        self.plots_toSave.append(fig)
        fig.suptitle(f'{self.plot_name}--Row {len(self.plots_toSave)}', 
            **{'family': 'Arial Black', 'size': 22, 'weight': 'bold'},x=0.5, y=0.98)
 
            # plt.show()       
        # print("plot_save",plot_save)
        if plot_save:
            print("save whole....")
            pdf = FPDF('P', 'in', (30, 19))
            pdf.add_page()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                print("tmpfile.name",tmpfile.name)
                fig.savefig(tmpfile.name,bbox_inches="tight")
                pdf.image(tmpfile.name)
            print("pdf",pdf)
    
            # pp = PdfPages(self.filename_toSave_whole)
            # # for fig in self.plots_toSave:
            # fig.savefig(pp, format='pdf')
            html = self.create_download_link(pdf.output(dest="S").encode("latin-1"), self.filename_toSave_whole)
            # # print("html",html)
            st.markdown(html, unsafe_allow_html=True)
            # st.header("")
            
            # with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            #     print("tmpfile.name",tmpfile.name)
            #     with open(tmpfile.name, "rb") as file:
            #         st.download_button(
            #                 label="Download image",
            #                 data=file,
            #                 file_name="flower.png",
            #                 mime="image/png"
                        # )
        # if plot_show:
            # st.pyplot(fig)
            
            # pp = PdfPages(self.filename_toSave_whole)
            # fig.savefig(pp, format='pdf')
            # pp.close()
            # with open(self.filename_toSave_whole, "rb") as pdf_file:
            #     PDFbyte = pdf_file.read()

            # st.download_button(label="Export_Report",
            #                     data=PDFbyte,
            #                     file_name=self.filename_toSave_whole,
            #                     mime='application/octet-stream')
            # with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            #     print("tmpfile.name",tmpfile.name)
            # st.image(tmpfile.name)
    
            # result = Image.fromarray(cropped_image.astype('uint8'), 'RGB')
                
            # img = Image.open(result)            

            # btn = st.download_button(
            #     label="Download image",
            #     data=img,
            #     file_name="imagename.png",
            #     mime="image/png")
            
        plt.close(fig)
        return None
  
        
#     def add_page(self,txt):
#         last_page = plt.figure(figsize=(20,13))
# #     first_page.clf()
#         last_page.text(0.2,0.5,txt, transform=last_page.transFigure, size=24, ha="left" )
#         self.plots_toSave.append(last_page)
#         plt.close()
        
    def create_download_link(self,val, filename):
        b64 = base64.b64encode(val)  # val looks like b'...'
        pdf_display = f'<iframe src="data:application/pdf;base64,{b64.decode()}" width="800" height="800" type="application/pdf"></iframe>'
        return pdf_display
        
        # return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">Download file</a>'
        
    def save_multi_plots(self,filename_toSave):
        """
        save plots to pdf file
        """
        pdf = FPDF('P', 'in', (30, 19))
        for fig in self.plots_toSave:
            pdf.add_page()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    fig.savefig(tmpfile.name,bbox_inches="tight")
                    pdf.image(tmpfile.name)
        html = self.create_download_link(pdf.output(dest="S").encode("latin-1"), filename_toSave)
        st.markdown(html, unsafe_allow_html=True)
        # st.st.markdown("### button")
        # self.add_page(self.txt)
        # pp = PdfPages(filename_toSave)
        # for fig in self.plots_toSave:
        #     fig.savefig(pp, format='pdf')
        # with open(filename_toSave, "rb") as pdf_file:
        #         PDFbyte = pdf_file.read()

        # st.download_button(label="Export_Report",
        #                     data=PDFbyte,
        #                     file_name=filename_toSave,
        #                     mime='application/octet-stream')
        # pp.close()

    def plot_4_metrics_details(self,
                               plot_show:bool=True)->None:
        """
        devide the dataset into multiple plots
        the number of plots depends on the self.data_inOneRow
        """
        print("=============start to plot details")

        print(f"detected {len(self.points_detected)} points as breakpoints")
        
        #seperate data into different rows 
        size=len(self.pressure_df)
        grouped_pressure_df = [self.pressure_df.iloc[x:x+self.data_inOneRow,:] for x in range(0, len(self.pressure_df), self.data_inOneRow)]
        
        grouped_breakpoints=self.group_index(self.points_detected, 0, size, self.data_inOneRow)
        print(f"The plot is devided into {len(grouped_breakpoints)} rows")
        grouped_gound_truth=self.group_index(self.ground_truth, 0, size, self.data_inOneRow)

        
        #to store the plots for all rows
        self.plots_toSave=[]

        for i,(sub_pressure_df,sub_breakpoints, sub_ground_truth) in enumerate(zip(grouped_pressure_df,grouped_breakpoints,grouped_gound_truth)):
            #print
            if len(self.ground_truth)==0:
                sub_breakpoints.sort()
                print(f"------row {i+1}-----detected points:{sub_breakpoints}")
            if len(self.ground_truth)!=0:
                points_correct,points_faulty,points_missed=self.detected_points_categories_2(sub_breakpoints,sub_ground_truth)
                print(f"------row {i+1}-----correctly detected points:{points_correct}")
                print(f"------row {i+1}-----faulty detected points:{points_faulty}")
                print(f"------row {i+1}-----missed breakpoints:{points_missed}")  

            #extract rate data for a certain row     
            start_time=sub_pressure_df.iloc[0][self.colum_names["pressure"]["time"]]
            end_time=sub_pressure_df.iloc[-1][self.colum_names["pressure"]["time"]]
            rate_time=self.rate_df[self.colum_names["rate"]["time"]]
            sub_rate_df=self.rate_df.loc[(rate_time >= start_time) & (rate_time <= end_time)]
            
            self.plot_4_metrics(sub_pressure_df,
                        sub_rate_df,
                        sub_breakpoints,
                        sub_ground_truth,
                        plot_show)
        
        #save multifigs
        # if self.filename_toSave_details!="":
        print("------------------save button")
        self.save_multi_plots(self.filename_toSave_details)
        return None  



    def plot_detection_statistics(self)->None:
        """
        statistics bar plot for the the number of 
        ground_truth, points_correct, points_faulty, points_missed
        """
        if len(self.ground_truth)==0:
            print("No ground truth defined")
            return None
        
        points_correct,points_faulty,points_missed=self.detected_points_categories_2(self.points_detected,self.ground_truth)
        print("the number of ground_truth",len(self.ground_truth))
        print("the number of points_correct",len(points_correct))
        print("the number of points_faulty",len(points_faulty))
        print("the number of points_missed",len(points_missed))

        # creating the dataset for bar plot
        data = {'points correct':round(len(points_correct)/len(self.ground_truth) ,3), 
                'points faulty':round(len(points_faulty)/len(self.ground_truth) ,3),  
                'points missed':round(len(points_missed)/len(self.ground_truth) ,3), }
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

def plot_histogram(data, xlabel:str, ylabel:str,title:str)->None:
    number_bins=300
    # plt.style.use('ggplot')
    plt.hist(data, bins=number_bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.legend(legend)
    plt.title(title)
    plt.show
    
def plot_nonDenoised_VS_Denoised(pressure_df:pd.DataFrame,
                                 pressure_df_denoised:pd.DataFrame,
                                 colum_names:Dict[str,Dict[str,str]]
                                            ={"pressure":{"time":"Elapsed time",
                                                            "measure":"Data",
                                                            "first_order_derivative":"first_order_derivative",
                                                            "second_order_derivative":"second_order_derivative"},
                                                "rate":{"time":"Elapsed time","measure":"Liquid rate"}})->None:
    """
    line plot for both pressure measurements and denoised pressure measurements
    """
    fig=plt.figure(figsize=(9,4))
    ax=fig.subplots()
    p=ax.plot(pressure_df[colum_names["pressure"]["time"]],pressure_df[colum_names["pressure"]["measure"]],label='raw data')
    p=ax.plot(pressure_df[colum_names["pressure"]["time"]],pressure_df_denoised[colum_names["pressure"]["measure"]],label='S-G filtered')
    legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
    # legend.get_frame().set_facecolor('C0')
    plt.show()
    
    
def download_button(object_to_download, download_filename, button_text):
    """
    Generates a link to download the given object_to_download.

    From: https://discuss.streamlit.io/t/a-download-button-with-custom-css/4220

    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.

    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.

    Returns:
    -------
    (str): the anchor tag to download object_to_download

    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')

    """
    # if pickle_it:
    #    try:
    #        object_to_download = pickle.dumps(object_to_download)
    #    except pickle.PicklingError as e:
    #        st.write(e)
    #        return None

    # if:
    if isinstance(object_to_download, bytes):
        pass

    elif isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    # Try JSON encode for everything else
    else:
        object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()
    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub("\d+", "", button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = (
        custom_css
        + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br><br>'
    )
    # dl_link = f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}"><input type="button" kind="primary" value="{button_text}"></a><br></br>'

    st.markdown(dl_link, unsafe_allow_html=True)
    