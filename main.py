
import cv2
import numpy as np
import pandas as pd

def cal():
    # Example: Extract Average Pixel Intensity from a Sequence of Images
    image_paths = ["pictures/snoopy1.png", "pictures/snoopy2.jpg", "pictures/snoopy4.png"] 
    time_series_data = []

    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        # avg_intensity = np.mean(img)  # Compute average pixel intensity        
        df = pd.DataFrame(img)
        idx+=1
        print(df)
        df_1d = df.stack().reset_index(drop=True).to_frame(name="Values")
        df_1d.to_csv(f"pixel_intensity{idx}.csv",index=False)


def main():
    cal()

if __name__=="__main__":
    main()