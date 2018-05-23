import tomopy
import dxchange
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage
import h5py
from skimage.feature import register_translation
print('\n*** Libraries imported')


##########################################################################################################
fdir = '/local/data/2018-04/Dubacq/'
file_name_Im1 = 'tomo_manip7G-sc_7124eV_1200prj_354.h5'
file_name_Im2 = 'tomo_manip7G-sc_7200eV_1200prj_352.h5'
prj = 0
binning = 1
medfilt_size = 3
crop = [200,300]
##########################################################################################################

# Reading Images to align:
Im1, flat1, dark1, theta = dxchange.read_aps_32id(fdir+file_name_Im1, proj=(prj, prj+1, 1))
Im2, flat2, dark2, theta = dxchange.read_aps_32id(fdir+file_name_Im2, proj=(prj, prj+1, 1))

Im1 = Im1/np.mean(flat1, axis = 0)
Im2 = Im2/np.mean(flat2, axis = 0)

if medfilt_size>0:
    Im1 = ndimage.median_filter(Im1,footprint=np.ones((1, medfilt_size, medfilt_size)))
    Im2 = ndimage.median_filter(Im2,footprint=np.ones((1, medfilt_size, medfilt_size)))

if binning>0:
    Im1 = tomopy.downsample(Im1, level=binning)
    Im1 = tomopy.downsample(Im1, level=binning, axis=1)

    Im2 = tomopy.downsample(Im2, level=binning)
    Im2 = tomopy.downsample(Im2, level=binning, axis=1)


if 1:
    plt.figure(),
    plt.subplot(1,2,1), plt.imshow(np.squeeze(Im1), cmap='gray', aspect="auto", interpolation='none'), plt.colorbar()
    plt.subplot(1,2,2), plt.imshow(np.squeeze(Im2), cmap='gray', aspect="auto", interpolation='none'), plt.colorbar()
    plt.show()


def transform_image(img, rotation=0, translation=(0, 0), crop=False):
    """Take a set of transformations and apply them to the image.
    
    Rotations occur around the center of the image, rather than the
    (0, 0).
    
    Parameters
    ----------
    translation : 2-tuple, optional
      Translation parameters in (vert, horiz) order.
    rotation : float, optional
      Rotation in degrees.
    scale : 2-tuple, optional
      Scaling parameters in (vert, horiz) order.
    crop : bool, optional
      If true (default), clip the dimensions of the image to avoid
      zero values.
    
    Returns
    -------
    out : np.ndarray
      Similar to input array but transformed. Dimensions will be
      different if ``crop=True``.
    crops : 4-tuple
      The dimensions used for cropping in order (v_min, v_max, h_min,
      h_max)
    
    """
    rot_center = (img.shape[1] / 2, img.shape[0] / 2)
    xy_trans = (translation[1], translation[0])
    M0 = _transformation_matrix(tx=-rot_center[0], ty=-rot_center[1])
    M1 = _transformation_matrix(r=np.radians(rotation), tx=xy_trans[0], ty=xy_trans[1])
    M2 = _transformation_matrix(tx=rot_center[0], ty=rot_center[1])
    # python 3.6
    # M = M2 @ M1 @ M0
    MT = np.dot(M1, M0)
    M = np.dot(M2, MT)
    tr = FundamentalMatrixTransform(M)
    out = warp(img, tr, preserve_range=True)
    # Calculate new boundaries if needed
    # Adjust for rotation
    h_min = 0.5 * img.shape[0] * np.tan(np.radians(rotation))
    v_min = 0.5 * img.shape[1] * np.tan(np.radians(rotation))
    # Adjust for translation
    v_max = min(img.shape[0], img.shape[0] - v_min - translation[0])
    h_max = min(img.shape[1], img.shape[1] - h_min - translation[1])
    v_min = max(0, v_min - translation[0])
    h_min = max(0, h_min - translation[1])
    crops = (int(v_min), int(v_max), int(h_min), int(h_max))
    # Apply the cropping
    if crop:
        out = out[crops[0]:crops[1],
                  crops[2]:crops[3]]
    return out, crops
