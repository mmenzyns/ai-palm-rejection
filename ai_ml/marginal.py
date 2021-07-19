# MIT License

# Copyright (c) 2020 yumi

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np

def mean(arr,dim="x", meanNN_TF=False):
    '''
    E(x) = int x f(x) dx
    
    arr      : np.array containing 2-d heatmap, shape is (height, width) 
    dim      : if "x", the marginal mean of X (width) is calcualted
                if "y", the marginal mean of Y (height)is calcualted 
    meanNN_TF: if False, the marginal mean is calculated using all the pixels.
                if True, then the marginal mean is calculated using the neighboor of the high marginal density region.
                The "neighboor" is calculated in two steps:
                (1) order the intensity of the marginal density 
                in the decreasing manner, 
                (2) include the top densities until the corresponding pixel coordinate hits the pixel border.
                
    '''
    if dim == "x":
        axis = 0
    elif dim == "y":
        axis = 1
    mardens = arr.sum(axis=axis)
    asort   = np.argsort(mardens)[::-1]
    
    if meanNN_TF:
        Npixel = np.min(np.where((asort == 0) | (asort == (len(asort)-1) ))) + 1
    else: 
        Npixel = -1 
    mardens = mardens[asort][:Npixel]
    mardens = mardens / np.sum(mardens) ## rescale so that marginal density adds up to 1
    
    xcoor   = asort[:Npixel]
    
    meanx   = np.sum((xcoor*mardens)[:Npixel])
    return(meanx)
    
def std(arr,dim="x",verbose=False,meanNN_TF=False):
    '''
        E( (x - barx)**2 ) = int (x-barx)**2 f(x) dx
        
    arr     : np.array containing 2-d heatmap, shape is (height, width) 
    dim     : if "x", the marginal mean of X (width) is calculated
                if "y", the marginal mean of Y (height)is calculated 
    meanNN_TF: if False, the marginal mean is calculated using all the pixels.
                if True, then the marginal mean is calculated using the neighboor of the high marginal density region.
                The "neighboor" is calculated in two steps:
                (1) order the intensity of the marginal density 
                in the decreasing manner, 
                (2) include the top densities until the corresponding pixel coordinate hits the pixel border.
    '''
    if dim == "x":
        axis = 0
    elif dim == "y":
        axis = 1
    mardens = arr.sum(axis=axis)
    xcoor   = np.arange(len(mardens))
    
    meanx   = mean(arr, dim=dim, meanNN_TF=meanNN_TF)
    varx    = np.sqrt(np.sum((xcoor - meanx)**2*mardens))
    if verbose:
        print("{} total dens = {:4.3f}, mean = {:5.2f}".format(
                dim,
                np.sum(mardens),
                meanx)
                )
    return(varx)