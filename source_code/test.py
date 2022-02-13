def plot_detection_statistics(self)->None:
    if len(self.ground_truth)==0:
        print("No ground truth defined")
        return None
    
    points_correct,points_faulty,points_missed=self.detected_points_categories_2(self.points_detected,self.ground_truth)
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