import os
import sys
import logging
import pkg_resources
import numpy as np
import pyqtgraph as pg
from argparse import ArgumentParser
from contextlib import contextmanager
from PyQt4 import QtGui, QtCore, uic

import tomopy
import dxchange
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage
import h5py
from skimage.feature import register_translation
print('\n*** Libraries imported')

LOG = logging.getLogger(__name__)


def file_io():
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

# ##########################################################################################
#                              added to link GUI with tasks
# ##########################################################################################

def remove_extrema(data):
    upper = np.percentile(data, 99)
    lower = np.percentile(data, 1)
    data[data > upper] = upper
    data[data < lower] = lower
    return data

class OverlapViewer(QtGui.QWidget):
    """
    Presents two images by subtracting the flipped second from the first.

    To get the current deviation connect to the *slider* attribute's
    valueChanged signal.
    """
    def __init__(self, parent=None):
        super(OverlapViewer, self).__init__()
        image_view = pg.ImageView()
        image_view.getView().setAspectLocked(True)
        self.image_item = image_view.getImageItem()

        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.update_image)

        self.main_layout = QtGui.QVBoxLayout()
        self.main_layout.addWidget(image_view)
        self.main_layout.addWidget(self.slider)
        self.setLayout(self.main_layout)
        self.first, self.second = (None, None)

    def set_images(self, first, second):
        """Set *first* and *second* image."""
        self.first = remove_extrema(first.T)
        self.second = remove_extrema(np.flipud(second.T))

        if self.first.shape != self.second.shape:
            LOG.warn("Shape {} of {} is different to {} of {}".
                     format(self.first.shape, self.first, self.second.shape, self.second))

        self.slider.setRange(0, self.first.shape[0])
        self.slider.setSliderPosition(self.first.shape[0] / 2)
        self.update_image()

    def set_position(self, position):
        self.slider.setValue(int(position))
        self.update_image()

    def update_image(self):
        """Update the current subtraction."""
        if self.first is None or self.second is None:
            LOG.warn("No images set yet")
        else:
            pos = self.slider.value()
            moved = np.roll(self.second, self.second.shape[0] / 2 - pos, axis=0)
            self.image_item.setImage(moved - self.first)

    
class ApplicationWindow(QtGui.QMainWindow):
    def __init__(self, app):
        QtGui.QMainWindow.__init__(self)
        self.app = app
        ui_file = pkg_resources.resource_filename(__name__, 'gui.ui')
        self.ui = uic.loadUi(ui_file, self)
        self.ui.show()
    
        # set up run-time widgets
        self.overlap_viewer = OverlapViewer()
        self.ui.overlap_layout.addWidget(self.overlap_viewer)

        # connect signals
        #self.overlap_viewer.slider.valueChanged.connect(self.center_slider_changed)
        self.ui.arrow_right.clicked.connect(self.arrow_right_clicked)
        self.ui.arrow_left.clicked.connect(self.arrow_left_clicked)
        self.ui.arrow_up.clicked.connect(self.arrow_up_clicked)
        self.ui.arrow_down.clicked.connect(self.arrow_down_clicked)
        self.ui.rotate_clock.clicked.connect(self.rotate_clock_clicked)
        self.ui.rotate_cclock.clicked.connect(self.rotate_cclock_clicked)

    def arrow_right_clicked(self, checked):
        print("clicking right")

    def arrow_left_clicked(self, checked):
        print("clicking left")

    def arrow_up_clicked(self, checked):
        print("clicking up")

    def arrow_down_clicked(self, checked):
        print("clicking down")

    def rotate_clock_clicked(self, checked):
        print("rotating clock")

    def rotate_cclock_clicked(self, checked):
        print("rotating counter clock")

def main():
    app = QtGui.QApplication(sys.argv)
    ApplicationWindow(app)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
