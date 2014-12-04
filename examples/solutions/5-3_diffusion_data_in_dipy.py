# -*- coding: utf-8 -*-
"""
5.3 : Manipulation of diffusion data using dipy
"""
import numpy as np # Python package for handling arrays
import os # to have access to commands of the operating system
from os.path import expanduser, join
home = expanduser('~')
print home
data1_dirname = join(home,'tp_bme_dwi','data','fsl_fdt1','subj1')

dwi_fname = join(data1_dirname,'data.nii.gz')
print dwi_fname
bvecs_fname = join(data1_dirname ,'bvecs')
print bvecs_fname
bvals_fname = join(data1_dirname,'bvals')
print bvals_fname

import nibabel as nib
img = nib.load(dwi_fname)
data = img.get_data()

from dipy.io import read_bvals_bvecs
bvals, bvecs = read_bvals_bvecs(bvals_fname, bvecs_fname)
# Cell to look at bval and bvecs
cc_profile = data[47,35,31,:] #corpus callosum voxel profile
print 'corpus callosum signal at b=0: ', cc_profile[0]
print 'corpus callosum signal at dir 1: ', cc_profile[1]
print 'corpus callosum signal at dir 11: ', cc_profile[11]
print 'direction 1', bvecs[1]
print 'direction 11', bvecs[11]

gm_profile = data[63,36,49,:]
print 'gray matter voxel:', gm_profile[0]
print 'gray matter voxel, dir1:', gm_profile[1]
print 'gray matter voxel, dir 11:', gm_profile[11]