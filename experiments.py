import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.special import binom
import matplotlib.patches as patches

from algos import *

#####################################################
# Test 0: Grayscale
#####################################################
np.random.seed(2024)
I1 = np.vstack( (np.random.normal(loc=0,scale=1,size=(240,240)),np.random.normal(loc=0,scale=2,size=(240,240))) )
I2 = np.vstack( (np.random.normal(loc=0,scale=4,size=(240,240)),np.random.normal(loc=0,scale=1,size=(240,240))) )
I = np.hstack((I1,I2))
plt.imshow(I,cmap='gray')
plt.axis('off')
plt.show()

B = np.random.normal(loc=0,scale=1,size=(480,480))
plt.imshow(B,cmap='gray')
plt.axis('off')
plt.show()

Delta = I-B
plt.imshow(Delta**2,cmap='gray',)
plt.axis('off')
plt.show()

n = 4 # number of sizes to test
W = np.zeros((n,2))
W = np.array([[40, 40],
       [20, 20],
       [40, 20],
       [20,40]])
out = algo2(I,B,W,sigma2=1)

plt.imshow(np.array([I-B,I-B,I-B]).T)
plt.axis('off')
plt.show()

#####################################################
# Test 1: Intrusive object detection
#####################################################
# We know the background and want to detect an intrusive object
np.random.seed(2024)
I = np.zeros((640,480)) + np.random.normal(loc=0,scale=0.5,size=(640,480))
# Add noise for measurement error and the intruder
# Add intruders
# Intruder smaller than a window size
I[240:270,240:250] = 100*np.ones((30,10))
# Intruder larger than a window size
I[350:390,240:360] = 100*np.ones((40,120))
plt.imshow(I,cmap='gray')
plt.axis('off')
plt.show()

B = np.zeros((640,480))
plt.imshow(B,cmap='gray')
plt.axis('off')
plt.show()

Delta = I-B
plt.imshow(Delta**2,cmap='gray',)
plt.axis('off')
plt.show()

n = 4 # number of sizes to test
W = np.zeros((n,2))
W = np.array([[40, 40],
       [20, 20],
       [40, 20],
       [20,40]])

# Apply the algorithm
out = algo2(I,B,W,sigma2=1)

#####################################################
# Test 2: Moving object detection
#####################################################
np.random.seed(2024)
I = np.zeros((640,480)) + np.random.normal(loc=0,scale=0.5,size=(640,480))
I[50:80,70:100] = 100 * np.ones((30,30))
plt.imshow(I,cmap='gray')
plt.axis('off')
plt.show()

B = np.zeros((640,480))
B[60:90,90:120] = 100 * np.ones((30,30))
plt.imshow(B,cmap='gray')
plt.axis('off')
plt.show()

Delta = I-B
plt.imshow(Delta**2,cmap='gray',)
plt.axis('off')
plt.show()

n = 4 # number of sizes to test
W = np.zeros((n,2))
W = np.array([[40, 40],
       [20, 20],
       [40, 20],
       [20,40]])
# Apply the algorithm
out = algo2(I,B,W,sigma2=1)

#####################################################
# Test 3: light switching on hides movement
#####################################################

def add_circle(array, center, radius, intensity):
    # Create a grid of x and y coordinates for each point in the array
    x, y = np.meshgrid(np.arange(array.shape[1]), np.arange(array.shape[0]))
    
    # Calculate the distances of each grid point from the center of the circle
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Replace the values in the array by 1 if they are inside the circle
    array[distances <= radius] += intensity
    
    return array

np.random.seed(2024)
I = np.zeros((640,480)) + np.random.normal(loc=0,scale=0.5,size=(640,480))
# Add noise for measurement error and the intruder
# Add intruders
I += 60*np.ones((640,480))
I = add_circle(I, (240, 210), 40,80)
I = add_circle(I, (240, 640-210+20), 30,50)
plt.imshow(I,cmap='gray')
plt.axis('off')
plt.show()

B = np.zeros((640,480))
B[0,0] = 140
B = add_circle(B, (240, 210), 40,50)
B = add_circle(B, (240, 640-210), 30,50)
plt.imshow(B,cmap='gray')
plt.axis('off')
plt.show()

Delta = I-B
plt.imshow(Delta**2,cmap='gray',)
plt.axis('off')
plt.show()

n = 4 # number of sizes to test
W = np.zeros((n,2))
W = np.array([[40, 40],
       [20, 20],
       [40, 20],
       [20,40]])
# Apply the algorithm
out = algo2(I,B,W,sigma2=1)