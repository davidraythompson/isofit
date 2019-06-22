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
from common import load_spectrum, load_wavelen, eps
from scipy.interpolate import interp1d
from numpy.polynomial.legendre import Legendre


class PolySurface:
    """A model of the surface.
      Surface models are stored as MATLAB '.mat' format files"""

    def __init__(self, config):

        self.degree = config['degree']
        self.statevec = ['COEFF%i' % i for i in range(self.degree+1)]
        self.bounds = s.array([[-5,5] for i in range(self.degree+1)])
        self.init = s.array([0 for i in range(self.degree+1)])
        self.scale = s.array([1.0 for i in range(self.degree+1)])
        if 'coefficient_priors_file' in config:
            fn = config['coefficient_priors_file']
            self.prior_mean, self.prior_sigma = s.loadtxt(fn).T
        else:
            self.prior_sigma = s.array([0.1 for i in range(self.degree+1)])
            self.prior_mean = s.array([0 for i in range(self.degree+1)])
        self.bvec = []
        self.bval = s.array([])
        self.emissive = False
        self.reconfigure(config)
        self.wl, self.fwhm = load_wavelen(config['wavelength_file'])
        if 'domain' in config:
          self.domain = config['domain']
        else:
          self.domain = [self.wl[0], self.wl[-1]]
        self.n_wl = len(self.wl)

    def reconfigure(self, config):
        """Adjust the surface reflectance (for predefined reflectances)"""

        if 'reflectance' in config and config['reflectance'] is not None:
            self.rfl = config['reflectance']

    def xa(self, x_surface, geom):
        '''Mean of prior state vector distribution calculated at state x'''

        return s.array(self.prior_mean)

    def Sa(self, x_surface, geom):
        '''Covariance of prior state vector distribution calculated at state x.'''

        return s.diag(self.prior_sigma)

    def fit_params(self, rfl, Ls, geom):
        '''Given a directional reflectance estimate and one or more emissive 
           parameters, fit a state vector.'''
        leg = Legendre(self.init, domain = self.domain)
        use = s.logical_and(self.wl<self.domain[1], self.wl>self.domain[0])
        return leg.fit(self.wl[use], rfl[use], len(self.init)-1).coef

    def calc_lamb(self, x_surface, geom):
        '''Calculate a Lambertian surface reflectance for this state vector.'''

        return Legendre(x_surface, domain=self.domain)(self.wl)

    def calc_rfl(self, x_surface, geom):
        '''Calculate the directed reflectance (specifically the HRDF) for this
           state vector.'''

        return Legendre(x_surface, domain=self.domain)(self.wl)

    def drfl_dsurface(self, x_surface, geom):
        '''Partial derivative of reflectance with respect to state vector, 
           calculated at x_surface.  In the case that there are no free 
           paramters our convention is to return the vector of zeros.'''


        unperturb = Legendre(x_surface, domain=self.domain)(self.wl)
        drfl_dsurf = []
        for i,x in enumerate(x_surface):
          x_perturb = x_surface.copy()
          x_perturb[i] = x_surface[i] + eps 
          drfl_dsurf.append(Legendre(x_perturb, domain=self.domain)(self.wl))
        return s.array(drfl_dsurf).T

    def drfl_dsurfaceb(self, x_surface, geom):
        '''Partial derivative of reflectance with respect to unmodeled 
           variables, calculated at x_surface.  In the case that there are no
           free paramters our convention is to return the vector of zeros.'''

        return s.zeros((self.n_wl, self.degree+1))

    def calc_Ls(self, x_surface, geom):
        '''Emission of surface, as a radiance'''

        return s.zeros((self.n_wl,))

    def dLs_dsurface(self, x_surface, geom):
        '''Partial derivative of surface emission with respect to state vector, 
           calculated at x_surface.  In the case that there are no
           free paramters our convention is to return the vector of zeros.'''

        return s.zeros((self.n_wl, self.degree+1))

    def summarize(self, x_surface, geom):
        '''Summary of state vector'''

        return ' '.join(['%4f' % x for x in x_surface])
