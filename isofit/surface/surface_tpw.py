#! /usr/bin/env python3
#
#  Copyright 2020 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ISOFIT: Imaging Spectrometer Optimal FITting
# Author: David R Thompson, david.r.thompson@jpl.nasa.gov
#

import numpy as np
from scipy.linalg import block_diag, norm
from scipy.io import loadmat
from scipy.optimize import minimize

from ..core.common import svd_inv, VectorInterpolator
from .surface import Surface
from isofit.configs import Config


class TPWSurface(Surface):
    """Three Phases of Water Surface Model, as in Bohn et al., RSE 2019,
    Thompson et al., RSE 2015, Green et al., WRR 2006.
    """

    def __init__(self, full_config: Config):
        """."""

        super().__init__(full_config)

        config = full_config.forward_model.surface

        # Models are stored as dictionaries in .mat format
        model_dict = loadmat(config.surface_file)
        self.wl = model_dict['wl'][0]
        self.n_wl = len(self.wl)
        self.ice = model_dict['ice'][0]
        self.lqd = model_dict['lqd'][0]
        self.windows = model_dict['windows']
        self.window_idx = []
        self.lqd_abscf = []
        self.ice_abscf = []
        self.statevec_names = []
        self.bounds = []
        self.scale = []
        self.init = []
        self.idx_lamb = []
        rmin, rmax = 0, 100.0

        for wi,(lo,hi) in enumerate(self.windows):
            blo = np.argmin(abs(self.wl-lo))
            bhi = np.argmin(abs(self.wl-hi))
            self.window_idx.append((blo,bhi))

        for i in np.arange(self.n_wl):
            lamb = True
            for self.wi,(lo,hi) in enumerate(self.window_idx):
                if self.wl[i]>=lo and self.wl[i]<hi:
                    lamb = False
            if lamb:
                self.idx_lamb.append(int(i))
                self.statevec_names.append('RFL_%04i'%i)
                self.bounds.append([rmin,rmax])
                self.scale.append(1.0)
                self.init.append(0.15)
        self.idx_lamb = np.array(self.idx_lamb)

        # Variables retrieved: each channel maps to a reflectance model parameter
        for wi,(blo,bhi) in enumerate(self.window_idx):
            self.init = self.init + [1e-6,1e-6,1e-6,1e-6]
            self.scale = self.scale + [1,1,1,1]
            self.bounds = self.bounds + [[-100,100],[-100,100],[0,100],[0,100]]
            self.statevec_names = self.statevec_names + \
                ['WIN%i_OFFS'%wi,'WIN%i_SLOPE'%wi,
                 'WIN%i_LQD'%wi,'WIN%i_ICE'%wi]
        
        self.n_lamb = len(self.idx_lamb)
        self.n_state = len(self.statevec_names)


    def xa(self, x_surface, geom):
        """Mean of prior distribution."""
        
        mu = np.zeros(self.n_state)
        return mu

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x."""

        sigma = np.ones(self.n_state)*1000
        variance = pow(sigma,2)
        return np.diag(variance)

    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate, fit a state vector."""

        def err(x, transm, blo, bhi):
            lqd, ice = x
            transm_est = np.exp(-ice*self.ice[blo:bhi]) *\
                            np.exp(-lqd*self.lqd[blo:bhi])
            return np.sum(pow(transm-transm_est,2))

        x_surface = np.zeros(self.n_state)
        for i,li in enumerate(self.idx_lamb):
            x_surface[i] = rfl_meas[li]
        for wi, (blo, bhi) in enumerate(self.window_idx):
            offs = rfl_meas[blo]
            slp = (rfl_meas[bhi]-rfl_meas[blo]) \
                    / float(bhi-blo)
            x_surface[self.n_lamb+wi*4+1] = slp
            x_surface[self.n_lamb+wi*4] = offs
            continuum = offs + slp*np.arange(bhi-blo)
            ctmrm = rfl_meas[blo:bhi] / continuum
            x0 = [0,0]
            res = minimize(err, x0, args=(ctmrm,blo,bhi))
            lqd, ice = res.x
            x_surface[self.n_lamb+wi*4+2] = lqd
            x_surface[self.n_lamb+wi*4+3] = ice
        return x_surface

    def calc_rfl(self, x_surface, geom):
        """Non-Lambertian reflectance."""

        return self.calc_lamb(x_surface, geom)

    def calc_lamb(self, x_surface, geom):
        """Lambertian reflectance.  Be sure to incorporate BRDF-related
          LUT dimensions such as solar and view zenith."""

        lamb = np.zeros(self.n_wl)
        lamb[self.idx_lamb] = x_surface[:self.n_lamb]
        for wi, (blo, bhi) in enumerate(self.window_idx):
            offs,slp,lqd,ice = x_surface[(self.n_lamb+wi*4):(self.n_lamb+(wi+1)*4)]
            lqd_abs = np.exp(-lqd*self.lqd[blo:bhi])
            ice_abs = np.exp(-ice*self.ice[blo:bhi])
            lamb[blo:bhi] = (offs + slp*np.arange(bhi-blo)) * ice_abs * lqd_abs
        return lamb

    def drfl_dsurface(self, x_surface, geom):
        """Partial derivative of reflectance with respect to state vector, 
        calculated at x_surface."""

        return self.dlamb_dsurface(x_surface, geom)

    def dlamb_dsurface(self, x_surface, geom):
        """Partial derivative of Lambertian reflectance with respect to 
        state vector, calculated at x_surface."""

        dlamb = np.zeros((self.n_wl,self.n_state))
        for i, li in enumerate(self.idx_lamb):
            dlamb[li,i] = 1.0
        for wi, (blo, bhi) in enumerate(self.window_idx):
            offs,slp,lqd,ice = x_surface[(self.n_lamb+wi*4):(self.n_lamb+(wi+1)*4)]
            continuum = offs + slp*np.arange(bhi-blo)
            lqd_abs = np.exp(-lqd*self.lqd[blo:bhi])
            ice_abs = np.exp(-ice*self.ice[blo:bhi])
            dlqd_abs_dstate = -self.lqd[blo:bhi]*np.exp(-lqd*self.lqd[blo:bhi])
            dice_abs_dstate = -self.ice[blo:bhi]*np.exp(-ice*self.ice[blo:bhi])
            dlamb[blo:bhi,self.n_lamb+wi*4] = ice_abs * lqd_abs
            dlamb[blo:bhi,self.n_lamb+wi*4+1] = np.arange(bhi-blo) * ice_abs * lqd_abs
            dlamb[blo:bhi,self.n_lamb+wi*4+2] = \
                continuum * ice_abs * dlqd_abs_dstate
            dlamb[blo:bhi,self.n_lamb+wi*4+3] = \
                continuum * lqd_abs * dice_abs_dstate
        return dlamb
                                                             
    def calc_Ls(self, x_surface, geom):
        """Emission of surface, as a radiance."""

        return np.zeros(self.n_wl, dtype=float)

    def dLs_dsurface(self, x_surface, geom):
        """Partial derivative of surface emission with respect to state vector, 
        calculated at x_surface."""

        dLs = np.zeros((self.n_wl, self.n_state), dtype=float)
        return dLs

    def summarize(self, x_surface, geom):
        """Summary of state vector."""

        if len(x_surface) < 1:
            return ''
        return 'LQD: %4.2f, ICE: %4.2f' % (x_surface[self.n_lamb+2],
                                          x_surface[self.n_lamb+3])
