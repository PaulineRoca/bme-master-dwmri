# -*- coding: utf-8 -*-
"""
Local Modeling in diffusion MRI
===============================

Step-by-step example of local modeling diffusion tensor model 
to experimental data and visualizing the results.

1. A sequence of diffusion MRI volumes are loaded
2. Masking and cropping of the data
3. Diffusion tensor model is computed from the data
4. Visualization of the resulting maps (FA, MD, diffusoids)
"""

# for numerical computation:
import numpy as np
# for loading imaging datasets
import nibabel as nib

# dataset:
from dipy.data import fetch_stanford_hardi
fetch_stanford_hardi() #you only need to fetch once

from dipy.data import read_stanford_hardi

import matplotlib.pyplot as plt # To make 2D plots

#################################################
# Choice of the local modeling
#################################################
# for reconstruction with the diffusion tensor model
import dipy.reconst.dti as dti
# for reconstruction with constrained spherical deconvolution
import dipy.reconst.csdeconv as csd

#################################################
# Load Data 
#################################################

img, gtab = read_stanford_hardi()
data = img.get_data()
print('data.shape (%d, %d, %d, %d)' % data.shape)
# Selection of one slice of the dataset :
data_small = data[20:50, 55:85, 38:39]

#######################################
# Diffusion tensor model
#######################################

tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(data_small) #environ 10min : 2s for the cc slice

print('Computing anisotropy measures (FA, MD, RGB)')
from dipy.reconst.dti import fractional_anisotropy, color_fa, lower_triangular
FA = fractional_anisotropy(tenfit.evals)
FA[np.isnan(FA)] = 0
FA = np.clip(FA, 0, 1)
RGB = color_fa(FA, tenfit.evecs)
#################################################
# Constrained spherical deconvolution model
#################################################
# fiber response function estimation:
response, ratio = csd.auto_response(gtab, data, roi_radius=10, fa_thr=0.7)#1min
csd_model = csd.ConstrainedSphericalDeconvModel(gtab, response, sh_order=8)
csd_fit = csd_model.fit(data_small) # 1min.

from dipy.data import get_sphere
sphere = get_sphere('symmetric724')
csd_odf = csd_fit.odf(sphere)
csd_odf.shape

from dipy.viz import fvtk#ok mais pb de visu: on peut baisser encore la taille
ren = fvtk.ren()
fodf_spheres = fvtk.sphere_funcs(csd_odf, sphere, scale=1.3, norm=False)
fvtk.add(ren, fodf_spheres)#14h19-21 (2min)
fvtk.show(ren)
#print('Saving illustration as csd_odfs.png')
#fvtk.record(ren, out_path='csd_odfs.png', size=(600, 600))
#################################################
# Create diffusion maps: tensor model
#################################################
#GFA = csd_fit.gfa[:,:,0]
#GFA[np.isnan(GFA)] = 0
#GFA.shape

#######################################
# Create snapshots of diffusion maps
#######################################

first_b0 = data_small[:,:,0,0]

plt.figure()
plt.subplot(121)
plt.imshow(first_b0.T,cmap='gray',origin="lower")
plt.subplot(122)
plt.imshow(FA.squeeze().T,cmap='gray',origin="lower")
plt.show()


#######################################
# Create snapshots of 
# tensor ellipsoids
#######################################

#visualization of tensor ellipsoids:
print('Computing tensor ellipsoids in a part of the splenium of the CC')

from dipy.data import get_sphere
sphere = get_sphere('symmetric724')

from dipy.viz import fvtk
ren = fvtk.ren()
#evals = tenfit.evals[13:43, 44:74, 28:29]
#evecs = tenfit.evecs[13:43, 44:74, 28:29]
evals = tenfit.evals
evecs = tenfit.evecs

cfa = RGB
cfa /= RGB.max()
fvtk.add(ren, fvtk.tensor(evals, evecs, cfa, sphere))#2min
fvtk.record(ren, n_frames=1, out_path='tensor_ellipsoids.png', size=(600, 600))
#######################################
# Create snapshots of tensor 
# orientation distribution functions
#######################################
