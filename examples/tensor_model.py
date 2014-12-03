# -*- coding: utf-8 -*-
"""
Local Modeling in diffusion MRI
===============================

Step-by-step example of fitting a diffusion tensor model 
to experimental data and visualizing the results :

1. A sequence of diffusion MRI volumes are loaded
2. Masking and cropping of the data
3. Diffusion tensor model is computed from the data
4. Visualization of the resulting maps (FA, MD, diffusoids)
"""

# for numerical computation:
import numpy as np
# for loading imaging datasets
import nibabel as nib
# for reconstruction with the diffusion tensor model
import dipy.reconst.dti as dti

# dataset:
from dipy.data import fetch_stanford_hardi
fetch_stanford_hardi() #you only need to fetch once

from dipy.data import read_stanford_hardi

#######################################
# Load Data 
#######################################

img, gtab = read_stanford_hardi()
data = img.get_data()
print('data.shape (%d, %d, %d, %d)' % data.shape)

#######################################
# Segment and mask data 
#######################################

from dipy.segment.mask import median_otsu
maskdata, mask = median_otsu(data, 3, 1, True,
                             vol_idx=range(10, 50), dilate=2)
print('maskdata.shape (%d, %d, %d, %d)' % maskdata.shape)

# To select only on slice and bow around the corpus callosum:
maskdata = maskdata[13:43, 44:74, 28:29,:]

#######################################
# Diffusion tensor model
#######################################

tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(maskdata) #environ 10min : 2s for the cc slice

print('Computing anisotropy measures (FA, MD, RGB)')
from dipy.reconst.dti import fractional_anisotropy, color_fa, lower_triangular

#######################################
# Create diffusion maps
#######################################

FA = fractional_anisotropy(tenfit.evals)

FA[np.isnan(FA)] = 0

fa_img = nib.Nifti1Image(FA.astype(np.float32), img.get_affine())
nib.save(fa_img, 'tensor_fa.nii.gz')

evecs_img = nib.Nifti1Image(tenfit.evecs.astype(np.float32), img.get_affine())
nib.save(evecs_img, 'tensor_evecs.nii.gz')

MD1 = dti.mean_diffusivity(tenfit.evals)
nib.save(nib.Nifti1Image(MD1.astype(np.float32), img.get_affine()), 'tensors_md.nii.gz')

FA = np.clip(FA, 0, 1)
RGB = color_fa(FA, tenfit.evecs)
nib.save(nib.Nifti1Image(np.array(255 * RGB, 'uint8'), img.get_affine()), 'tensor_rgb.nii.gz')

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
fvtk.record(ren, n_frames=1, out_path='dipy_tensor_ellipsoids.png', size=(600, 600))

