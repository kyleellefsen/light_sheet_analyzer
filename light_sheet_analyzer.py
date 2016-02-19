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
from process.BaseProcess import BaseProcess, SliderLabel
import pyqtgraph as pg
from window import Window

from skimage.transform import resize


#from spimagine import volshow




class Light_Sheet_Analyzer(BaseProcess):
    def __init__(self):
        super().__init__()

    def __call__(self, nSteps, shift_factor, keepSourceWindow=False):
        g.m.statusBar().showMessage("Generating 4D movie ...")
        t = time()
        self.start(keepSourceWindow)
        A=self.tif
        '''
        A=g.m.currentWindow.image
        nSteps=250
        shift_factor=1
        
        '''
        mt,mx,my=A.shape
        mv=int(np.floor(mt/nSteps)) #number of volumes
        A=A[:mv*nSteps]
        B=np.reshape(A,(mv,nSteps,mx,my))
        
        B=B.swapaxes(1,3)
        
        mv,mz,mx,my=B.shape
        newy=my+shift_factor*mz
        C=np.zeros((mv,mz,mx,newy),dtype=A.dtype)
        shifted=0
        for z in np.arange(mz):
            minus_z=mz-z
            shifted=minus_z*shift_factor
            C[:,z,:,shifted:shifted+my]=B[:,z,:,:]
            
        # shift aspect ratio
        #D=resize(C,(mt,mx,newy*shift_factor))

        g.m.statusBar().showMessage("Successfully generated movie ({} s)".format(time() - t))
        
        #volshow(C)     
        w = Window(np.squeeze(C[:,0,:,:]))
        w.volume=C
        Volume_Viewer(w)
        return 


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


class Volume_Viewer(QWidget):
    closeSignal=Signal()
    def __init__(self,window=None,parent=None):
        super(Volume_Viewer,self).__init__(parent) ## Create window with ImageView widget
        g.m.volume_viewer=self
        self.window=window
        self.setWindowTitle('Light Sheet Volume View Controller')
        self.setWindowIcon(QIcon('images/favicon.png'))
        self.setGeometry(QRect(422, 35, 222, 86))
        self.l = QVBoxLayout()
        
        mv,mz,mx,my=window.volume.shape   

        self.zSlider=SliderLabel(0)
        self.zSlider.setRange(0,mz)
        self.zSlider.label.valueChanged.connect(self.zSlider_updated)
        self.zSlider.slider.mouseReleaseEvent=self.zSlider_release_event
        self.l.addWidget(self.zSlider)
        self.setLayout(self.l)
        self.show()

    def closeEvent(self, event):
        event.accept() # let the window close
        
    def zSlider_updated(self,val):
        currentIndex=self.window.currentIndex
        vol=self.window.volume
        testimage=np.squeeze(vol[currentIndex,val,:,:])
        self.window.imageview.setImage(testimage,autoLevels=False)    
        
    def zSlider_release_event(self,ev):
        z=self.zSlider.value()
        vol=self.window.volume
        image=np.squeeze(vol[:,z,:,:])
        self.window.imageview.setImage(image)
        QSlider.mouseReleaseEvent(self.zSlider.slider, ev)

#v=Volume_Viewer(g.m.currentWindow)


    
