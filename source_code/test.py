    def produce_parameters_givenPoints(self,pressure_measure,pressure_time,points,time_halfWindow,loadedParameters_pattern,fitting_type):
        """
        extract the data of the given points in the timewindow
        and
        calculate the parameter for all curves fitting these points
        """
 
        # self.fitting_func=polyval_func_wrapper
#         if filePath_loadPattern!=None:
#             self.load_pattern(filePath_loadPattern)
#             self.PatternLoaded=True
        # print("-----produce_parameters_givenPoints",fitting_type)
        if fitting_type == "linear":
            self.fitting_func=linear_func_wrapper
        if fitting_type == "polynomial":
            self.fitting_func=polyval_func_wrapper
        if fitting_type == "log":
            self.fitting_func=log_func_wrapper
        if self.buildUp_or_drawDown!="":
            print(f"start to learn '{self.buildUp_or_drawDown}' pattern...")
        self.extract_points_inTimeWindow(pressure_measure,pressure_time,points,time_halfWindow)
        parameters_allCurves=self.calculate_Parameters_allCurve(fitting_type=fitting_type)
        return parameters_allCurves
        # parameters_pattern=self.calculate_parameters_pattern(fitting_type,loadedParameters_pattern)
        # return parameters_pattern