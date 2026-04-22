import nibabel as nib
import numpy as np

img = nib.load("output/preprocessed.nii.gz")
data = img.get_fdata()

print("Min:", np.min(data))
print("Max:", np.max(data))