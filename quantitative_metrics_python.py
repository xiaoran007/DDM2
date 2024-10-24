import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import TensorModel

from matplotlib import pyplot as plt
import os

# root = '...' # **YOUR ROOT OF DENOISED DATA**
root = "experiments/hardi150_denoise_241023_193159/results/"
denoised_data = {}

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
data, affine = load_nifti(hardi_fname)
raw_max =  np.max(data, axis=(0,1,2), keepdims=True)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)

# non b0 mask
sel_b = bvals != 0
b0_size = 10

# create a new gradient table on non b0 directions
gtab = gradient_table(bvals[sel_b], bvecs[sel_b])
raw_gtab = gradient_table(bvals, bvecs)

# PLEASE DOUBLE CHECK THE PATH BELOW
# THE DENOISED DATA MUST CONTAIN B0 DIRETCIONS!!
# denoised_data['n2s'], _ = load_nifti(root + 'n2s_denoised.nii.gz')
# denoised_data['n2n'], _ = load_nifti(root + 'n2n_denoised.nii.gz')
# denoised_data['dip'], _ = load_nifti(root + 'dip_denoised.nii.gz')
# denoised_data['p2s'], _ = load_nifti(root + 'p2s_denoised.nii.gz')
# denoised_data['our'], _ = load_nifti(root + 'ours_denoised.nii.gz')
denoised_data['exp'], _ = load_nifti(root + 'hardi150_denoised.nii.gz')

# normalize denoised data
for k in denoised_data.keys():
    denoised_data[k] -= np.min(denoised_data[k], axis=(0,1), keepdims=True)
    denoised_data[k] /= np.max(denoised_data[k], axis=(0,1), keepdims=True)
    denoised_data[k] = np.concatenate((data[:,:,:,:b0_size], denoised_data[k]), axis=-1)

print('Data loading done!')


print('Computing brain mask...')
b0_mask, mask = median_otsu(data, vol_idx=[0])

print('Computing tensors...')
tenmodel = TensorModel(raw_gtab)
tensorfit = tenmodel.fit(data, mask=mask)

print('Computing worst-case/best-case SNR using the corpus callosum...')
from dipy.segment.mask import segment_from_cfa
from dipy.segment.mask import bounding_box

threshold = (0.6, 1, 0, 0.1, 0, 0.1)

CC_box = np.zeros_like(data[..., 0])

mins, maxs = bounding_box(mask)
mins = np.array(mins)
maxs = np.array(maxs)
diff = (maxs - mins) // 4
bounds_min = mins + diff
bounds_max = maxs - diff

CC_box[bounds_min[0]:bounds_max[0],
       bounds_min[1]:bounds_max[1],
       bounds_min[2]:bounds_max[2]] = 1

mask_cc_part, cfa = segment_from_cfa(tensorfit, CC_box, threshold,
                                     return_cfa=True)


import matplotlib.pyplot as plt

region = 40

# visualize the masks and ROI
fig = plt.figure('Corpus callosum segmentation')
plt.subplot(1, 5, 1)
plt.title("CC")
plt.axis('off')
red = cfa[..., 0]
plt.imshow(np.rot90(red[...,region]))

plt.subplot(1, 5, 2)
plt.title("CC mask")
plt.axis('off')
plt.imshow(np.rot90(mask_cc_part[...,region]), cmap='gray')

from scipy.ndimage.morphology import binary_dilation
mask_noise = binary_dilation(mask, iterations=10)

mask_noise = ~mask_noise
print(mask_noise.shape)

plt.subplot(1, 5, 3)
plt.title("noise mask")
plt.axis('off')
plt.imshow(mask_noise[:,:,40], cmap='gray')

plt.subplot(1, 5, 4)
plt.title("mask")
plt.axis('off')
plt.imshow(mask[:,:,40], cmap='gray')


plt.subplot(1, 5, 5)
plt.title("data")
plt.axis('off')
plt.imshow(data[:,:,40,40], cmap='gray')

# normalize raw data for metric computation
data_normalized = data - np.min(data, axis=(0,1), keepdims=True)
data_normalized = (data_normalized.astype(np.float32) / np.max(data_normalized, axis=(0,1), keepdims=True))

mean_signal = np.mean(data_normalized[mask_cc_part], axis=0)
noise_std = np.std(data_normalized[mask_noise, :], axis=0)
mean_bg = np.mean(data_normalized[mask_noise, :], axis=0)

mean_signal_denoised = {}
denoised_noise_std = {}
denoised_mean_bg = {}

# normalize denoised data for metric calculation
for k, v in denoised_data.items():
    mean_signal_denoised[k] = np.mean(v[mask_cc_part], axis=0)
    denoised_noise_std[k] = np.std(v[mask_noise, :], axis=0)
    denoised_mean_bg[k] = np.mean(v[mask_noise, :], axis=0)

# metric calculation starts here

SNRs = {}
CNRs = {}

SNR = mean_signal / noise_std
CNR = (mean_signal - mean_bg) / noise_std

SNR = SNR[sel_b]
CNR = CNR[sel_b]

SNRs['raw'] = SNR
CNRs['raw'] = CNR

print('raw', '[SNR] mean: %.4f std: %.4f' % (np.mean(SNR), np.std(SNR)))
print('raw', '[CNR] mean: %.4f std: %.4f' % (np.mean(CNR), np.std(CNR)))

for k in denoised_data.keys():
    SNR = mean_signal_denoised[k] / (denoised_noise_std[k]+1e-7)
    CNR = (mean_signal_denoised[k] - denoised_mean_bg[k]) / (denoised_noise_std[k]+1e-7)

    SNR = SNR[sel_b]
    CNR = CNR[sel_b]
    SNR -= SNRs['raw']
    CNR -= CNRs['raw']
    SNRs[k] = SNR
    CNRs[k] = CNR
    print("=================")
    print(k, '[SNR delta] mean: %.4f std: %.4f best: %.4f worst: %.4f' % (np.mean(SNR), np.std(SNR), np.max(SNR), np.min(SNR)))
    print(k, '[CNR delta] mean: %.4f std: %.4f best: %.4f worst: %.4f' % (np.mean(CNR), np.std(CNR), np.max(CNR), np.min(CNR)))

