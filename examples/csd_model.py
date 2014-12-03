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

#################################################
# Choice of the local modeling
#################################################
# for reconstruction with constrained spherical deconvolution
import dipy.reconst.csdeconv as csd

#################################################
# Load Data 
#################################################

img, gtab = read_stanford_hardi()
data = img.get_data()
print('data.shape (%d, %d, %d, %d)' % data.shape)

#################################################
# Constrained spherical deconvolution model
#################################################
# fiber response function estimation:
response, ratio = csd.auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
csd_model = csd.ConstrainedSphericalDeconvModel(gtab, response, sh_order=8)

# Selection of one slice of the dataset :
data_small = data[20:50, 55:85, 38:39]
csd_fit = csd_model.fit(data_small)

from dipy.data import get_sphere
sphere = get_sphere('symmetric724')
csd_odf = csd_fit.odf(sphere)
csd_odf.shape

from dipy.viz import fvtk
ren = fvtk.ren()
fodf_spheres = fvtk.sphere_funcs(csd_odf, sphere, scale=1.3, norm=False)
fvtk.add(ren, fodf_spheres)#14h19-21 (2min)
fvtk.show(ren)
print('Saving illustration as csd_odfs.png')
fvtk.record(ren, out_path='dipy_csd_odfs.png', size=(600, 600))


