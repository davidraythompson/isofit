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

import pandas as pd
import re
import time
import argparse
import os
import sys
import subprocess
import json
from os.path import realpath, split, abspath, expandvars
import scipy as s
from scipy.io import loadmat, savemat
import os
import sys
import glob
import argparse
import scipy.io as io
import numpy.random as random
import scipy as s
import time
import matplotlib
matplotlib.use('agg')
import pylab as plt
from common import find_header, expand_path, json_load_ascii, spectrumResample


eps = 1e-5  # used for finite difference derivative calculations


class NeuralNetworkRT:
    """A model of photon transport including the atmosphere."""

    def __init__(self, config, instrument):
        self.weights, self.biases = [], []

        # Determine top level parameters
        for q in ['weights1_file', 'weights2_file', 'biases1_file',
                  'biases2_file', 'solar_file', 'wavelength_file']:
            if q not in config:
                raise ValueError('Missing parameter: %s' % q)
            else:
                setattr(self, q, config[q])#, expand_path(self.config_dir, config[q]))

        # load wavelengths file
        q = s.loadtxt(self.wavelength_file)
        if q.shape[1] > 2:
            q = q[:, 1:]
        if q[0, 0] < 100:
            q = q * 1000.0
        self.wl = q[:,0]
        self.fwhm = q[:,1]
        nchan = len(self.wl)

        self.weights1 = s.load(self.weights1_file)
        self.weights2 = s.load(self.weights2_file)
        self.biases1 = s.load(self.biases1_file)
        self.biases2 = s.load(self.biases2_file)
        solar =  s.loadtxt(self.solar_file)
        self.solar_irr = spectrumResample(solar[:,1], solar[:,0], self.wl, 
                self.fwhm)
        self.solar_irr = self.solar_irr / 10.0 # convert nW / m2 to uW / cm2  
        self.atmosphere_inputvec = s.array(config['atmosphere_inputvec'])

        self.statevec = list(config['statevector'].keys())
        self.bvec = list(config['unknowns'].keys())
        self.n_state = len(self.statevec)

        # Retrieved variables.  We establish scaling, bounds, and
        # initial guesses for each state vector element.
        self.bounds, self.scale, self.init_val = [], [], []
        for key in self.statevec:
            element = config['statevector'][key]
            self.bounds.append(element['bounds'])
            self.scale.append(element['scale'])
            self.init_val.append(element['init'])
        self.bounds = s.array(self.bounds)
        self.scale = s.array(self.scale)
        self.init_val = s.array(self.init_val)
        self.bval = s.array([config['unknowns'][k] for k in self.bvec])

        if 'prior_sigma' in config['statevector']:
            self.prior_sigma = s.zeros((len(self.lut_grid),))
            for name, val in config['prior_sigma'].items():
                self.prior_sigma[self.statevec.index(name)] = val
        else:
            std_factor = 10.0
            self.prior_sigma = (s.diff(self.bounds) * std_factor).flatten()

    def calc_rdn(self, x_RT, rfl, Ls, geom):
        irr_scaling = self.solar_irr * geom.sundist() * geom.coszen() / s.pi
        rdn = self.calc_rho(x_RT, rfl, Ls, geom) * irr_scaling
        return s.squeeze(rdn)

    def calc_rho(self, x_RT, rfl, Ls=None, geom=None):

        # Gather the atmosphere and geometry input values
        n_atm = len(self.atmosphere_inputvec)
        atm = s.zeros(n_atm)
        for i, name in enumerate(self.atmosphere_inputvec):
            if name in self.statevec:
                atm[i] = x_RT[self.statevec.index(name)]
            elif name == 'phi':
                relaz = geom.RELAZ
                if relaz>180.0:
                  relaz = 360.0-relaz
                atm[i] = relaz/360.0*2.0*s.pi  # relative az, radians
            elif name == 'umu':
                atm[i] = geom.umu
            else:
                raise ValueError(
                    'State vector does not match NN. Needs geomdata?')

        # Replicate and append surface reflectance
        atm_mat = s.tile(atm, [len(rfl), 1])
        inp     = s.hstack([atm_mat, rfl[:, s.newaxis]])
        inp     = inp.reshape([len(rfl), n_atm+1, 1])

        # Feedforward with rectifying hidden and linear output layers
        hid = s.sum(inp * self.weights1, axis=1) + self.biases1
        hid[hid < 0] = 0
        hid = hid.reshape([hid.shape[0], hid.shape[1], 1])
        rho = s.sum(hid * self.weights2, axis=1) + self.biases2
        rho = s.squeeze(s.array(rho))

        return rho

    def xa(self):
        '''Mean of prior distribution, calculated at state x. This is the mean of 
           our LUT grid (why not).'''
        return self.init_val.copy()

    def Sa(self):
        '''Covariance of prior distribution. Our state vector covariance is 
           diagonal with very loose constraints.'''
        std_factor = 10.0
        return s.diagflat(pow(s.diff(self.bounds) * std_factor, 2))

    def estimate_Ls(self, x_RT, rfl, rdn, geom=None):
        """Estimate the surface emission for a given state vector and 
           reflectance/radiance pair"""
        Ls = zeros(len(rfl))
        return Ls

    def heuristic_atmosphere(self, rdn, geom=None):
        '''From a given radiance, estimate atmospheric state using band ratio
        heuristics.  Used to initialize gradient descent inversions.'''
        x = self.init_val
        rfl_est = self.invert_algebraic(x, rdn, geom=geom)
        return x, rfl_est

    def invert_algebraic(self, x, rdn, Ls=None, geom=None):
        '''Inverts radiance algebraically to get a reflectance. Ls is the surface 
           emission, if present'''
        nwl = len(self.wl)
        rdn_0 = self.calc_rdn(x, s.zeros(nwl), Ls=Ls, geom=geom)
        rdn_1 = self.calc_rdn(x, s.ones(nwl), Ls=Ls, geom=geom)
        est_rfl = (rdn-rdn_0)/(rdn_1-rdn_0)
        return est_rfl

    def Kb_RT(self, x_RT, rfl, Ls, geom):
        return s.zeros((1, len(self.wl.shape)))

    def K_RT(self, x_RT, x_surface, rfl, drfl_dsurface, Ls, dLs_dsurface,
             geom=None):
        """Jacobian of radiance with respect to RT and surface state vectors"""

        # first the radiance at the current state vector
        rdn = self.calc_rdn(x_RT, rfl, Ls=Ls, geom=geom)

        # perturb each element of the RT state vector (finite difference)
        K_RT = []
        for i in range(len(x_RT)):
            x_RT_perturb = x_RT.copy()
            x_RT_perturb[i] = x_RT[i] + eps
            rdne = self.calc_rdn(x_RT_perturb, rfl, Ls=Ls, geom=geom)
            K_RT.append((rdne-rdn) / eps)
        K_RT = s.array(K_RT).T

        # analytical jacobians for surface model state vector, via chain rule
        # KLUDGE WARNING - should generalize/refine this!!!
        rdne = self.calc_rdn(x_RT, rfl+eps, Ls=Ls, geom=geom)
        drdn_drfl = (rdne-rdn)/eps
        K_surface = s.dot(s.diag(drdn_drfl), drfl_dsurface)

        return K_RT, K_surface

    def summarize(self, x_RT, geom):
        '''Summary of state vector'''
        return 'Atmosphere: '+' '.join(['%5.3f' % xi for xi in x_RT])
