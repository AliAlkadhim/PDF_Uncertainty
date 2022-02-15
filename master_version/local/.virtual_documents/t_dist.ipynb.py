import math
import scipy.stats as st
from math import gamma
import numpy as np


t = []
i = -6
while i < 6:
    t.append(round(i,3))
    i+=0.01
#t is our sampling space, we are choosing t to go from -6 to 6 in 0.01 increments
t[:3]


def numerator(df):
    n = (dof+1)/2
    num = gamma(n)
    return num
# def denomenator(df):
#     n = df/2
    



