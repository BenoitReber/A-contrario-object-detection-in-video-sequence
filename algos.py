import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.special import binom
import matplotlib.patches as patches

# in this framework, all pixels are known, which simplifies the algorithm
# This is a variant of algo1_bis at the end of the script
def algo1(I,B,sigma2=1):
    """
    Identifies a region of interest based on the squared difference between two images.
    
    Parameters
    ----------
    I : numpy.ndarray
        The input image.
    B : numpy.ndarray
        The background image.
    sigma2 : float, optional
        The variance of the noise. The default is 1.

    Returns
    -------
    Dhat : numpy.ndarray
        An array containing the indices of the detected pixels.
    """
    # Calculate the squared difference between the input image and the background.
    D2 = (I-B)**2
    # Flatten the array for easier processing.
    D2 = D2.flatten()
    # Get the indices that would sort the flattened array.
    eps = D2.argsort()
    # Initialize variables for the cumulative sum of squared differences and lists of indices.
    d2 = 0
    D = []
    Dhat = []
    # Initialize the minimum NFA (Number of False Alarms) to a very large value.
    NFA1m = 10**50
    # Total number of pixels.
    K = I.size
    
    # Iterate through the sorted squared differences.
    for i in range(K):
        # Add the current squared difference to the cumulative sum.
        d2 += D2[eps[i]]
        # Append the index of the current pixel to the list of detected pixels.
        D += [eps[i]]
        
        # Calculate NFA1 (Number of False Alarms for a single pixel)
        # We have i+1 degrees of freedom
        # We want inf*0 = 0
        tmp = chi2.cdf(d2/sigma2, df=i+1)
        if tmp > 0: 
            NFA1 = binom(K,i+1) * tmp
        else: 
            NFA1 = 0
        
        # If the current NFA is less than or equal to the minimum, update the minimum and the list of detected pixels.
        if NFA1 <= NFA1m:
            NFA1m = NFA1
            Dhat = np.array(D)
    
    return(Dhat)
    
# Calculate the significance.
def Sign(p,pW,nu,kappa,K):
    """
    Calculates the significance value for a given window.

    Parameters
    ----------
    p : float
        The proportion of object points in the entire image.
    pW : float
        The proportion of object points in the current window.
    nu : int
        The number of pixels in the current window.
    kappa : int
        The number of object points in the current window.
    K : int
        The total number of pixels in the image.

    Returns
    -------
    tmp : float
        The calculated significance value.
    """
    if pW<p: 
        tmp =  -10000000000#0
    else: 
        tmp = nu*(pW * np.log(pW/p) + (1-pW) * np.log((1-pW)/(1-p)) )
    return tmp + np.log(nu) - nu*np.log(2)

def algo2(I,B,W,sigma2=1):
    """
    Detects moving objects in an image by applying a sliding window approach.

    Parameters
    ----------
    I : numpy.ndarray
        The input image.
    B : numpy.ndarray
        The background image.
    W : numpy.ndarray
        A 2D array of window sizes (width, height).
    sigma2 : float, optional
        The variance of the noise. The default is 1.

    Returns
    -------
    final : numpy.ndarray
        A 2D array of the detected object windows.
    """
    (X,Y) = I.shape
    # Apply algorithm 1 to find the initial set of interesting pixels.
    tst = algo1(I,B,sigma2=1)
    # Create a mask of the interesting pixels.
    Im=np.ones((X*Y))
    Im[tst]=0
    Im = Im.reshape((X,Y))
    # Calculate the proportion of object points in the entire image.
    p = Im.sum()/Im.size
    K = I.size
    # Get the number of window sizes to test.
    n = W.shape[0]
    # Get the minimum stride for y and x
    sr = W[:,1].min()
    sd = W[:,0].min()
    
    # Display the initial mask of detected points.
    im = np.array([Im,Im,Im]).T
    plt.imshow(im)
    plt.axis('off')
    plt.show()

    # for each window, we store
    # top left corner
    # bottom right corner
    # significance
    S = np.zeros((n*int(I.shape[1]/sr)*int(I.shape[0]/sd),8))
    # Initialize significance to a very small number.
    S[:,4] = -100000000000 * np.ones((S.shape[0]))
    # We scan the image like a book (row by row)
    x,y=0,0
    for j in range(int(I.shape[1]/sr)):
        y = j * sr
        for i in range(int(I.shape[0]/sd)):
            x = i * sd
            # Calculate significances
            for k in range(n):
                corner = np.array([x+W[k,0],y+W[k,1]])
                if (x+W[k,0])<=I.shape[0] and (y+W[k,1])<=I.shape[1]:
                    # Sum the object points within the current window.
                    kappa = np.sum( Im[x:(x+W[k,0]),y:(y+W[k,1])] )
                    # Calculate the number of pixels in the window.
                    nu = W[k,0] * W[k,1]
                    # Calculate the proportion of object points in the window.
                    pW = kappa/nu
                    # Calculate the significance of the window.
                    Stmp = Sign(p,pW,nu,kappa,K)
                else : 
                    # If the window goes out of bounds, set significance to a very small number.
                    Stmp = -100000000000
                
                # Store the window's properties and significance.
                S[k*int(I.shape[1]/sr)*int(I.shape[0]/sd)+j*int(I.shape[0]/sd)+i,:] = np.array([x,y,corner[0],corner[1],Stmp,kappa,nu,pW]) 
    
    # We manage the exclusions
    # 1 - Sort by decreasing significance.
    order = (-S[:,4]).argsort()
    extract = S[order,:]
    cleared = [] # The windows we remove.
    
    # Iterate through the sorted windows to find overlapping windows.
    for i in range(S.shape[0]):
        if i not in cleared:
            (a,b) = extract[i, 0:2]
            (x,y) = extract[i, 2:4]
            for j in np.arange(i+1,S.shape[0]):
                # Check if the window is available.
                if not j in cleared:
                    # We compare
                    # we allow overlaps on the edges with this code
                    if extract[j,2] > a and extract[j,3] > b and extract[j,0] < x and extract[j,1] < y:
                        cleared += [j]
    
    # Keep the non-overlapping windows.
    keeped = np.array([ k for k in range(S.shape[0]) if k not in cleared])
    extracted = extract[keeped,:]
    
    # Display all disjoint windows.
    im = np.array([Im,Im,Im]).T # for grayscale
    plt.imshow(im)
    ax = plt.gca()
    for i in range(extracted.shape[0]):
        (a,b,c,d) = extracted[i,:4]
        rect = patches.Rectangle((a,b), c-a, d-b, linewidth=1, facecolor='none',edgecolor='r')
        ax.add_patch(rect)
    
    plt.axis('off')
    plt.show()
    
    # Selecting the final windows.
    order = (-extracted[:,-1]).argsort()
    psorted = extracted[order,:]
    
    N = 0
    Nu = 0
    Kappa = 0
    Smin = -10000000000
    for i in range(psorted.shape[0]):
        Nu += psorted[i,-2]
        Kappa += psorted[i,-3]
        pk = Kappa / Nu
        Sk = Sign(p,pk,Nu,Kappa,K)
        
        if Sk >= Smin:
            Smin = Sk
            N +=1
    
    final = psorted[np.arange(0,N),:]
    im = np.array([Im,Im,Im]).T # for grayscale
    plt.imshow(im)
    ax = plt.gca()
    for i in range(final.shape[0]):
        (a,b,c,d) = final[i,:4]
        rect = patches.Rectangle((a,b), c-a, d-b, linewidth=1, facecolor='none',edgecolor='r')
        ax.add_patch(rect)
    
    plt.axis('off')
    plt.show()
    
    return(final)

# A variant of algo1 to go faster
# It assumes there are few object points
# It starts in reverse
def algo1_bis(I,B,sigma2=1):
    """
    A faster variant of algo1, assuming a small proportion of object points.
    
    Parameters
    ----------
    I : numpy.ndarray
        The input image.
    B : numpy.ndarray
        The background image.
    sigma2 : float, optional
        The variance of the noise. The default is 1.

    Returns
    -------
    Dhat : numpy.ndarray
        An array containing the indices of the detected pixels.
    """
    D2 = (I-B)**2
    D2 = D2.flatten()
    # Get the indices that would sort the flattened array.
    eps = D2.argsort()
    # Initialize the cumulative sum with the total sum of squared differences.
    d2 = D2.sum()
    D = eps.copy()
    Dhat = []
    # Initialize the minimum NFA to a very large value.
    NFA1m = 10**50
    # Total number of pixels.
    K = I.size
    
    # We assume less than 20% of points are objects
    for i in range(int(K/5)):
        # Subtract the largest squared difference from the cumulative sum.
        d2 -= D2[eps[K-i-1]]
        D[K-1-i] = 0
        
        # Calculate NFA1
        # We have i+1 degrees of freedom
        # We want inf*0 = 0
        tmp = chi2.cdf(d2/sigma2, df=K-i-1)
        if tmp > 0: 
            NFA1 = binom(K,K-i-1) * tmp
        else: 
            NFA1 = 0
        
        # If the NFA is 0, we've found the optimal set of pixels.
        if NFA1 == 0:
            NFA1m = NFA1
            Dhat = D.copy()
            break
        elif NFA1 <= NFA1m:
            NFA1m = NFA1
            Dhat = D.copy()
    
    return(Dhat[Dhat!=0])