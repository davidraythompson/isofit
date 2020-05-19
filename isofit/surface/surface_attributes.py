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

import scipy as s
from scipy.io import loadmat

from ..core.common import eps, conditional_gaussian
from .surface_multicomp import MultiComponentSurface
from .surface_glint import GlintSurface
from isofit.configs import Config


class AttributeSurface(GlintSurface):
    """A model of the surface based on a collection of multivariate 
       Gaussians, extended with a surface glint term."""

    def __init__(self, full_config: Config):

        super().__init__(full_config)

        config = full_config.forward_model.surface
        model_dict = loadmat(config.surface_file)
        self.component_attrs = list(zip(model_dict['attribute_means'], 
                    model_dict['attribute_covs']))
        self.attributes = model_dict['attributes']
        self.n_attrs = len(self.attributes)

        self.statevec_names.extend(self.attributes)
        self.bounds.extend([[-100, 100] for w in self.attributes])
        self.scale.extend([1.0 for w in self.attributes])
        self.init.extend([0 for w in self.attributes])

        # we maintain the index of attributes in the state vector array
        # and the surface model. 
        self.idx_attr_sv = self.n_state + s.arange(self.n_attrs)
        self.idx_attr_sm = len(self.idx_lamb) + s.arange(self.n_attrs)

        self.n_state = len(self.statevec_names)

    def xa(self, x_surface, geom):
        """Mean of prior distribution, calculated at state x."""

        mu = GlintSurface.xa(self, x_surface, geom)
        ci = self.component(x_surface, geom)
        mu[self.idx_attr_sv] = self.component_attrs[ci][0][self.idx_attr_sm]
        return mu

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x."""

        Cov = GlintSurface.Sa(self, x_surface, geom)
        ci = self.component(x_surface, geom)
        idx_all = s.r_[self.idx_lamb, self.idx_attr_sv]
        for i,j in enumerate(idx_all):
            Cov[j,idx_all] = self.component_attrs[ci][1][i,:]
        return Cov

    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate and one or more emissive parameters, 
          fit a state vector."""

        x_surface = GlintSurface.fit_params(self, rfl_meas, geom)
        ci = self.component(x_surface, geom)
       # x_surface[self.idx_attr_sv] = self.component_attrs[ci][0][self.idx_attr_sm]
        x_surface[self.idx_attr_sv], x_surface_cov = \
                conditional_gaussian(self.component_attrs[ci][0],
                self.component_attrs[ci][1], 
                self.idx_attr_sm,
                self.idx_lamb,
                self.calc_lamb(x_surface, geom))
        return x_surface

    def dLs_dsurface(self, x_surface, geom):
        """Partial derivative of surface emission with respect to state vector, 
        calculated at x_surface.  We append a column of zeros to handle 
        the extra glint parameter"""

        dLs_dsurface = super().dLs_dsurface(x_surface, geom)
        dLs_dattrs = s.zeros((dLs_dsurface.shape[0],self.n_attrs))
        dLs_dsurface = s.hstack([dLs_dsurface, dLs_dattrs]) 
        return dLs_dsurface

    def summarize(self, x_surface, geom):
        """Summary of state vector."""

        return GlintSurface.summarize(self, x_surface, geom) + \
            ' %s: %5.3f' % (self.attributes[0],x_surface[self.idx_attr_sv[0]])
