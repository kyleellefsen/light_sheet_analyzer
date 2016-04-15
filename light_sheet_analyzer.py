# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 15:04:31 2016

@author: kyle
"""

from __future__ import (absolute_import, division,print_function, unicode_literals)
from future.builtins import (bytes, dict, int, list, object, range, str, ascii, chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)

import os
import numpy as np
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from time import time
import global_vars as g
from process.BaseProcess import BaseProcess, SliderLabel, CheckBox
import pyqtgraph as pg
from window import Window
from skimage.transform import resize
import tifffile

#from spimagine import volshow




class Light_Sheet_Analyzer(BaseProcess):
    def __init__(self):
        if g.settings['light_sheet_analyzer'] is None:
            s = dict()
            s['nSteps']=1
            s['shift_factor']=1
            g.settings['light_sheet_analyzer']=s
        super().__init__()


    def __call__(self, nSteps, shift_factor, keepSourceWindow=False):
        g.settings['light_sheet_analyzer']['nSteps']=nSteps
        g.settings['light_sheet_analyzer']['shift_factor']=shift_factor
        g.m.statusBar().showMessage("Generating 4D movie ...")
        t = time()
        self.start(keepSourceWindow)
        A=self.tif
        A=A[1:] # Ian Parker said to hard code removal of the first frame.
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
        B=np.repeat(B,shift_factor,axis=3)
        
        mv,mz,mx,my=B.shape
        newy=my+mz
        C=np.zeros((mv,mz,mx,newy),dtype=A.dtype)
        shifted=0
        for z in np.arange(mz):
            minus_z=mz-z
            shifted=minus_z
            C[:,z,:,shifted:shifted+my]=B[:,z,:,:]
        C=C[:,::-1,:,:]
        # shift aspect ratio
        #C=resize(C,(mt,mx,newy*shift_factor))   #This doesn't look like it works, it runs into memory problems

        g.m.statusBar().showMessage("Successfully generated movie ({} s)".format(time() - t))
        w = Window(np.squeeze(C[:,0,:,:]))
        w.volume=C

        Volume_Viewer(w)
        return 


    def closeEvent(self, event):
        self.ui.close()
        event.accept()

    def gui(self):
        s=g.settings['light_sheet_analyzer']
        self.gui_reset()
        self.nSteps = pg.SpinBox(int=True, step=1)
        self.nSteps.setMinimum(1)
        self.nSteps.setValue(s['nSteps'])
        
        self.shift_factor = pg.SpinBox(int=False, step=.1)
        self.shift_factor.setValue(s['shift_factor'])
        self.shift_factor
        
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
        self.layout = QVBoxLayout()
        self.vol_shape=window.volume.shape
        mv,mz,mx,my=window.volume.shape
        self.currentAxisOrder=[0,1,2,3]
        self.current_v_Index=0
        self.current_z_Index=0
        self.current_x_Index=0
        self.current_y_Index=0
        
        self.formlayout=QFormLayout()
        self.formlayout.setLabelAlignment(Qt.AlignRight)
        
        self.xzy_position_label=QLabel('Z position')
        self.zSlider=SliderLabel(0)
        self.zSlider.setRange(0,mz-1)
        self.zSlider.label.valueChanged.connect(self.zSlider_updated)
        self.zSlider.slider.mouseReleaseEvent=self.zSlider_release_event
        
        self.sideViewOn=CheckBox()
        self.sideViewOn.setChecked(False)
        self.sideViewOn.stateChanged.connect(self.sideViewOnClicked)
        
        self.sideViewSide = QComboBox(self)
        self.sideViewSide.addItem("X")
        self.sideViewSide.addItem("Y")
        
        self.MaxProjButton = QPushButton('Max Intenstiy Projection')
        self.MaxProjButton.pressed.connect(self.make_maxintensity)
        
        self.exportVolButton = QPushButton('Export Volume')
        self.exportVolButton.pressed.connect(self.export_volume)
        
        self.formlayout.addRow(self.xzy_position_label,self.zSlider)
        self.formlayout.addRow('Side View On',self.sideViewOn)
        self.formlayout.addRow('Side View Side',self.sideViewSide)
        self.formlayout.addRow('', self.MaxProjButton)
        self.formlayout.addRow('', self.exportVolButton)
        
        self.layout.addWidget(self.zSlider)
        self.layout.addLayout(self.formlayout)
        self.setLayout(self.layout)
        self.setGeometry(QRect(381, 43, 416, 110))
        self.show()

    def closeEvent(self, event):
        event.accept() # let the window close
        
    def zSlider_updated(self,z_val):
        self.current_v_Index=self.window.currentIndex
        vol=self.window.volume
        testimage=np.squeeze(vol[self.current_v_Index,z_val,:,:])
        self.window.imageview.setImage(testimage,autoLevels=False)   
        
    def zSlider_release_event(self,ev):
        vol=self.window.volume
        if self.currentAxisOrder[1]==1: # 'z'
            self.current_z_Index=self.zSlider.value()
            image=np.squeeze(vol[:,self.current_z_Index,:,:])
        elif self.currentAxisOrder[1]==2: # 'x'
            self.current_x_Index=self.zSlider.value()
            image=np.squeeze(vol[:,self.current_x_Index,:,:])
        elif self.currentAxisOrder[1]==3: # 'y'
            self.current_y_Index=self.zSlider.value()
            image=np.squeeze(vol[:,self.current_y_Index,:,:])
            
        self.window.imageview.setImage(image,autoLevels=False)
        self.window.imageview.setCurrentIndex(self.current_v_Index)
        QSlider.mouseReleaseEvent(self.zSlider.slider, ev)
    
    def sideViewOnClicked(self, checked):
        self.current_v_Index=self.window.currentIndex
        vol=self.window.volume
        if checked==2: #checked=True
            assert self.currentAxisOrder==[0,1,2,3]
            side = self.sideViewSide.currentText()
            if side=='X':
                vol=vol.swapaxes(1,2)
                self.currentAxisOrder=[0,2,1,3]
                
                vol=vol.swapaxes(2,3)
                self.currentAxisOrder=[0,2,3,1]
                
                
            elif side=='Y':
                vol=vol.swapaxes(1,3)
                self.currentAxisOrder=[0,3,2,1]
        else: #checked=False
            if self.currentAxisOrder == [0,3,2,1]:
                vol=vol.swapaxes(1,3)
                self.currentAxisOrder=[0,1,2,3]
            elif self.currentAxisOrder == [0,2,3,1]:
                vol=vol.swapaxes(2,3)
                vol=vol.swapaxes(1,2)
                self.currentAxisOrder=[0,1,2,3]
                
                
        if self.currentAxisOrder[1]==1: # 'z'
            idx=self.current_z_Index
            self.xzy_position_label.setText('Z position')
            self.zSlider.setRange(0,self.vol_shape[1]-1)
        elif self.currentAxisOrder[1]==2: # 'x'
            idx=self.current_x_Index
            self.xzy_position_label.setText('X position')
            self.zSlider.setRange(0,self.vol_shape[2]-1)
        elif self.currentAxisOrder[1]==3: # 'y'
            idx=self.current_y_Index
            self.xzy_position_label.setText('Y position')
            self.zSlider.setRange(0,self.vol_shape[3]-1)
            
        image=np.squeeze(vol[:,idx,:,:])
        self.window.imageview.setImage(image,autoLevels=False)
        self.window.volume=vol
        self.window.imageview.setCurrentIndex(self.current_v_Index)
        self.zSlider.setValue(idx)
    def make_maxintensity(self):
        vol=self.window.volume
        new_vol=np.max(vol,1)
        if self.currentAxisOrder[1]==1: # 'z'
            name='Max Z projection'
        elif self.currentAxisOrder[1]==2: # 'x'
            name = 'Max X projection'
        elif self.currentAxisOrder[1]==3: # 'y'
            name = 'Max Y projection'
        Window(new_vol, name=name)
        
    def export_volume(self):
        vol=self.window.volume
        export_path='C:/Users/Admin/Desktop/light_sheet_vols'
        i=0
        while os.path.isdir(export_path+str(i)):
            i+=1
        export_path=export_path+str(i)
        os.mkdir(export_path) 
        for v in np.arange(len(vol)):
            A=vol[v]
            filename=os.path.join(export_path,str(v)+'.tiff')
            if len(A.shape)==3:
                A=np.transpose(A,(0,2,1)) # This keeps the x and the y the same as in FIJI
            elif len(A.shape)==2:
                A=np.transpose(A,(1,0))
            tifffile.imsave(filename, A)

#v=Volume_Viewer(g.m.currentWindow)


    
