# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 15:04:31 2016

@author: kyle
"""

from __future__ import (absolute_import, division,print_function, unicode_literals)
from future.builtins import (bytes, dict, int, list, object, range, str, ascii, chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)

import numpy as np
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from time import time
import global_vars as g
from process.BaseProcess import BaseProcess
import pyqtgraph as pg
from window import Window

from spimagine import volshow




class Light_Sheet_Analyzer(BaseProcess):
    def __init__(self):
        super().__init__()

    def __call__(self, nSteps, shift_factor, keepSourceWindow=False):
        g.m.statusBar().showMessage("Generating 4D movie ...")
        t = time()
        self.start(keepSourceWindow)
        A=self.tif
        mt,mx,my=A.shape
        mv=int(np.floor(mt/nSteps)) #number of volumes
        A=A[:mv*nSteps]
        B=np.reshape(A,(mv,nSteps,mx,my))
        
        B=B.swapaxes(1,2)
        
        mv,mz,mx,my=B.shape
        newx=mx+shift_factor*mz
        C=np.zeros((mv,mz,newx,my))
        shifted=0
        for z in np.arange(mz):
            shifted=z*shift_factor
            C[:,z,shifted:shifted+mx,:]=B[:,z,:,:]
        
        
        
        
        g.m.statusBar().showMessage("Successfully generated movie ({} s)".format(time() - t))
        
        volshow(C)     
        return Window(np.squeeze(C[0,:,:,:]))


    def closeEvent(self, event):
        self.ui.close()
        event.accept()

    def gui(self):
        self.gui_reset()
        self.nSteps = pg.SpinBox(int=True, step=1)
        self.nSteps.setMinimum(1)
        
        self.shift_factor = pg.SpinBox(int=False, step=.1)
        
        self.items.append({'name': 'nSteps', 'string': 'Number of steps per volume', 'object': self.nSteps})
        self.items.append({'name': 'shift_factor', 'string': 'Shift Factor', 'object': self.shift_factor})
        super().gui()
        
light_sheet_analyzer = Light_Sheet_Analyzer()







