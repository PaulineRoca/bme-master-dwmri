# -*- coding: utf-8 -*-
"""
Local Modeling in diffusion MRI
===============================

Step-by-step example of fitting a diffusion tensor model 
to experimental data and visualizing the results :

1. A sequence of diffusion MRI volumes is loaded
2. Diffusion tensor model is estimated from the data
4. Visualization of the resulting maps (FA, MD, diffusoids)
"""

# for numerical computation:
import numpy as np
# for loading imaging datasets
import nibabel as nib
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
# for reconstruction with the diffusion tensor model
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy, color_fa

#######################################
# Define data filenames
#######################################

from os.path import expanduser, join

home = expanduser('~')
standfordhardi_dirname = join(home,'tp_bme_dwi','data','stanford_hardi')
fdwi = join(standfordhardi_dirname,'HARDI150.nii.gz')
fbval = join(standfordhardi_dirname, 'HARDI150.bval')
fbvec = join(standfordhardi_dirname,'HARDI150.bvec')

#######################################
# Load Data 
#######################################

img = nib.load(fdwi)
data = img.get_data()

print(data.shape)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)

print('data.shape (%d, %d, %d, %d)' % data.shape)

data_small = data[20:50, 55:85, 38:39]

#######################################
# Diffusion tensor model
#######################################

tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(data_small)

#######################################
# Create diffusion maps
#######################################
print('Computing anisotropy measures (FA, MD, RGB)')
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
evals = tenfit.evals
evecs = tenfit.evecs

cfa = RGB
cfa /= RGB.max()
fvtk.add(ren, fvtk.tensor(evals, evecs, cfa, sphere))
fvtk.record(ren, n_frames=1, out_path='dipy_tensor_ellipsoids.png', size=(600, 600))

