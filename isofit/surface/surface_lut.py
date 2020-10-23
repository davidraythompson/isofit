#! /usr/bin/env python3
#
#  Copyright 2018 California Institute of Technology
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

from ..core.common import svd_inv, VectorInterpolator
from .surface import Surface
from isofit.configs import Config


class LUTSurface(Surface):
    """A model of the surface based on a collection of multivariate 
    Gaussians, with one or more equiprobable components and full 
    covariance matrices. 

    To evaluate the probability of a new spectrum, we calculate the
    Mahalanobis distance to each component cluster, and use that as our
    Multivariate Gaussian surface model.
    """

    def __init__(self, full_config: Config):
        """."""

        super().__init__(full_config)

        config = full_config.forward_model.surface

        # Models are stored as dictionaries in .mat format
        model_dict = loadmat(config.surface_file)
        self.lut_grid = [grid[0] for grid in model_dict['grids'][0]]
        self.lut_names = [l.strip() for l in model_dict['lut_names']]
        self.statevec_names = [sv.strip() for sv in model_dict['statevec_names']]
        self.data = model_dict['data']
        interp_types = np.array(['n' for n in self.lut_grid])
        self.itp = VectorInterpolator(self.lut_grid, self.data, interp_types)
        self.wl = model_dict['wl'][0]
        self.n_wl = len(self.wl)
        self.bounds = model_dict['bounds']
        self.scale = model_dict['scale'][0]
        self.init = model_dict['init'][0]
        self.mean = model_dict['mean'][0]
        self.sigma = model_dict['sigma'][0]
        self.n_state = len(self.statevec_names)
        self.n_lut = len(self.lut_names)
        self.idx_lamb = np.arange(self.n_state)


    def xa(self, x_surface, geom):
        """Mean of prior distribution, calculated at state x. We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function. This always uses the
        Lambertian (non-specular) version of the surface reflectance."""
        
        mu = np.zeros(self.n_state)
        mu[self.idx_lamb] = self.mean.copy()
        return mu

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x. We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function."""

        variance = pow(self.sigma,2)
        Cov = np.diag(variance)

        # If there are no other state vector elements, we're done.
        if len(self.statevec_names) == len(self.idx_lamb):
            return Cov

        # Embed into a larger state vector covariance matrix
        nprefix = self.idx_lamb[0]
        nsuffix = len(self.statevec_names) - self.idx_lamb[-1] - 1
        Cov_prefix = np.zeros((nprefix, nprefix))
        Cov_suffix = np.zeros((nsuffix, nsuffix))
        return block_diag(Cov_prefix, Cov, Cov_suffix)

    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate, fit a state vector."""

        x_surface = self.mean.copy()
        return x_surface

    def calc_rfl(self, x_surface, geom):
        """Non-Lambertian reflectance."""

        return self.calc_lamb(x_surface, geom)

    def calc_lamb(self, x_surface, geom):
        """Lambertian reflectance."""

        point = np.zeros(self.n_lut)
        for v,name in zip(x_surface, self.statevec_names):
          point[self.lut_names.index(name)] = v
        if 'SOLZEN' in self.lut_names:
          solzen_ind = self.lut_names.index('SOLZEN')
          point[solzen_ind] = geom.solar_zenith
        if 'VIEWZEN' in self.lut_names:
          viewzen_ind = self.lut_names.index('VIEWZEN')
          point[viewzen_ind] = geom.observer_zenith
        lamb = self.itp(point)
        return lamb

    def drfl_dsurface(self, x_surface, geom):
        """Partial derivative of reflectance with respect to state vector, 
        calculated at x_surface."""

        return self.dlamb_dsurface(x_surface, geom)

    def dlamb_dsurface(self, x_surface, geom):
        """Partial derivative of Lambertian reflectance with respect to 
        state vector, calculated at x_surface."""

        eps = 1e-6
        base = self.calc_lamb(x_surface, geom)
        dlamb = []
        for xi in range(self.n_state):
           x_new = x_surface.copy()
           x_new[xi] = x_new[xi] + eps
           perturbed = self.calc_lamb(x_new, geom)
           dlamb.append((perturbed-base)/eps)
        dlamb = np.array(dlamb).T
        nprefix = self.idx_lamb[0]
        nsuffix = self.n_state - self.idx_lamb[-1] - 1
        prefix = np.zeros((self.n_wl, nprefix))
        suffix = np.zeros((self.n_wl, nsuffix))
        return np.concatenate((prefix, dlamb, suffix), axis=1)

    def calc_Ls(self, x_surface, geom):
        """Emission of surface, as a radiance."""

        return np.zeros(self.n_wl, dtype=float)

    def dLs_dsurface(self, x_surface, geom):
        """Partial derivative of surface emission with respect to state vector, 
        calculated at x_surface."""

        dLs = np.zeros((self.n_wl, self.n_state), dtype=float)
        nprefix = self.idx_lamb[0]
        nsuffix = len(self.statevec_names) - self.idx_lamb[-1] - 1
        prefix = np.zeros((self.n_wl, nprefix))
        suffix = np.zeros((self.n_wl, nsuffix))
        return np.concatenate((prefix, dLs, suffix), axis=1)

    def summarize(self, x_surface, geom):
        """Summary of state vector."""

        if len(x_surface) < 1:
            return ''
        return 'Surface: '+' '.join([('%5.4f' % x) for x in x_surface])
