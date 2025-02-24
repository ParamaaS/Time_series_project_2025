
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
        for j in range(i+1,len(time_series_datas)):
            distance, path  = fastdtw(time_series_datas[i], time_series_datas[j], dist=euclidean)
            print(distance,path,sep="\n")
            distance_matrix[i, j] = distance    
    return distance_matrix

def plot_chart(time_series_datas):
    n = len(time_series_datas)  # Number of time series
    colors = plt.cm.tab10(np.arange(n))  # Generate 'n' distinct colors

    fig, axes = plt.subplots(n, 1, figsize=(8, 2 * n), sharex=True)  # Create subplots

    for i, (series, color) in enumerate(zip(time_series_datas, colors)):
        axes[i].plot(series, label=f"Time Series {i}", color=color, marker='o', linestyle='',markersize=1)  # Plot each series
        axes[i].set_ylabel("Value")
        axes[i].set_title(f"Time Series of line {i+1}")
        # axes[i].legend()
        axes[i].grid(True)

    axes[-1].set_xlabel("Index")  # Set X label only for the last subplot
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

def picture_gen():
    # Example: Extract Average Pixel Intensity from a Sequence of Images
    image_names = ["line1", "line2", "line3","line4"] 
    time_series_datas = []

    for idx, img_name in enumerate(image_names):
        img_path="pictures"+"/"+img_name+".png"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        # avg_intensity = np.mean(img)  # Compute average pixel intensity        
        df = pd.DataFrame(img)
        idx+=1
        print(df)
        # df= df.stack().reset_index(drop=True)
        df= df.stack().reset_index(drop=True).to_frame(name="Values")
        df.to_csv(f"pixel_{img_name}.csv",index=False)
        time_series_datas.append(df)
    return time_series_datas

def main():
    time_series_datas=picture_gen()
    plot_chart(time_series_datas)
    # print(fast_dtw(time_series_datas))

if __name__=="__main__":
    main()