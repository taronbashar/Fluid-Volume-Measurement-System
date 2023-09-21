import cv2
from scipy import stats
import numpy as np

def find_height(column):
    threshold = 128
    for i, p in enumerate(column):
        if p > threshold:
            return i

def fix_outliers(idx, outliers):
    
    new = []
    
    for i in idx:
        if i in outliers:
            if i+1 in outliers:
                new.append(i-1)
            else:
                new.append(i + 1)
        else:
            new.append(i)
    
    return np.array(new)

def process_image(image, n=10):
    
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    y = 20 # set to height to crop from top
    x = 0  # we don't crop x
    
    img_crop = img_gray[y:, x:]
    
    img_blur = cv2.GaussianBlur(img_crop,(3,3), SigmaX=0, SigmaY=0) 
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    
    heights = []
    for col in edges.T:
        heights.append(col)
        
    heights = np.array(heights)
    z = np.abs(stats.zscore(heights))
    
    outliers = np.where(z > 3)
    
    # evenly spaced points
    idx = np.round(np.linspace(0, len(heights) - 1, n)).astype(int)

    idx = fix_outliers(idx, outliers)
    
    return heights[idx]