
import vmo 
import vmo.analysis as van
import vmo.analysis.segmentation as vse
import csv
import numpy as np
#np.set_printoptions(threshold='nan')

from collections import Counter

import math
import entropy
from pyentrp import entropy as ent

import matplotlib as mpl 
import matplotlib.pyplot as plt

import matplotlib.cm as cm
from matplotlib import patches
import plotly.plotly as py


import librosa

from nolds import lyap_e, lyap_r, corr_dim

import scipy as sc
import scipy.sparse as sp
from scipy.sparse import spdiags
import scipy.stats

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import mutual_info_score

import pywt
from pywt import wavedec
from pywt import swt

import pylab 
import scipy.io.wavfile as wav

import operator

# from pyunicorn.timeseries import RecurrencePlot, RecurrenceNetwork

#from python_speech_features import mfcc
#from python_speech_features import logfbank

from pylab import * # show, scatter, xlim, ylim
from random import randint
from random import randrange

