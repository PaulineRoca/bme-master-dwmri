# -*- coding: utf-8 -*-
"""
4. Diffusion MR signal
"""
import numpy as np

n = 3 #3D space
D = 2.4 *(10**-6)**2 # en m2/ms = 2.4 (micro m)**2/ms = 2.4 * (10**-6)**2 m2/ms
t = 50 # ms

mean_squared_displacement = 2*n*D*t
print mean_squared_displacement
std_displacement = np.sqrt(mean_squared_displacement)
print "std of displacements:", std_displacement