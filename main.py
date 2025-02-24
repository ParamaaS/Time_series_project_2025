
import cv2
import numpy as np
import pandas as pd
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

def fast_dtw(time_series_datas):
    # Example: Fast Dynamic Time Warping (FastDTW) Algorithm
    distance_matrix = np.zeros((len(time_series_datas), len(time_series_datas)))
    for i in range(len(time_series_datas)):
        for j in range(len(time_series_datas)):
            distance, _ = fastdtw(time_series_datas[i], time_series_datas[j], dist=euclidean)
            print(distance)
            distance_matrix[i, j] = distance    
    return distance_matrix

def plot_chart(time_series_datas):
    # Example: Plotting Time Series Data
    for i in range(len(time_series_datas)):
        plt.plot(time_series_datas[i], label=f"Time Series {i}")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title(f"Time Series Data {i}")       
        plt.legend()
        plt.show()

def picture_gen():
    # Example: Extract Average Pixel Intensity from a Sequence of Images
    image_names = ["snoopy2a", "snoopy2b", "snoopy2c"] 
    time_series_datas = []

    for idx, img_name in enumerate(image_names):
        img_path="pictures"+"/"+img_name+".jpg"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        # avg_intensity = np.mean(img)  # Compute average pixel intensity        
        df = pd.DataFrame(img)
        idx+=1
        print(df)
        # df= df.stack().reset_index(drop=True)
        df= df.stack().reset_index(drop=True).to_frame(name="Values")
        # df.to_csv(f"pixel_{img_name}.csv",index=False)
        time_series_datas.append(df)
    return time_series_datas

def main():
    time_series_datas=picture_gen()
    # plot_chart(time_series_datas)
    print(fast_dtw(time_series_datas))

if __name__=="__main__":
    main()