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

import os
import numpy as np
import logging
import multiprocessing
from collections import OrderedDict
import subprocess

from isofit.core import common
from isofit.configs import Config
from isofit.configs.sections.radiative_transfer_config import RadiativeTransferEngineConfig
from isofit.configs.sections.statevector_config import StateVectorElementConfig
from isofit.configs.sections.implementation_config import ImplementationConfig


### Functions ###

def spawn_rt(cmd):
    """Run a CLI command."""

    print(cmd)
    subprocess.call(cmd, shell=True)


### Classes ###

class FileExistsError(Exception):
    """FileExistsError with a message."""

    def __init__(self, message):
        super(FileExistsError, self).__init__(message)


class TabularRT:
    """A model of photon transport including the atmosphere."""

    def __init__(self, engine_config: RadiativeTransferEngineConfig, full_config: Config):

        self.implementation_config: ImplementationConfig = full_config.implementation
        self.wl, self.fwhm = common.load_wavelen(full_config.forward_model.instrument.wavelength_file)
        if engine_config.wavelength_range is not None:
            valid_wl = np.logical_and(self.wl >= engine_config.wavelength_range[0],
                                      self.wl <= engine_config.wavelength_range[1])
            self.wl = self.wl[valid_wl]
            self.fwhm = self.fwhm[valid_wl]

        self.n_chan = len(self.wl)

        self.auto_rebuild = full_config.implementation.rte_auto_rebuild
        self.configure_and_exit = full_config.implementation.rte_configure_and_exit

        # We use a sorted dictionary here so that filenames for lookup
        # table (LUT) grid points are always constructed the same way, with
        # consistent dimesion ordering). Every state vector element has
        # a lookup table dimension, but some lookup table dimensions
        # (like geometry parameters) may not be in the state vector.
        # TODO: enforce a requirement that makes all SV elements be inside the LUT
        full_lut_grid = full_config.forward_model.radiative_transfer.lut_grid
        # selectively get lut components that are in this particular RTE
        self.lut_grid_config = OrderedDict()
        if engine_config.lut_names is not None:
            lut_names = engine_config.lut_names
        else:
            lut_names = full_config.forward_model.radiative_transfer.lut_grid.keys()

        for key, value in full_lut_grid.items():
            if key in lut_names:
                self.lut_grid_config[key] = value

        # selectively get statevector components that are in this particular RTE
        full_sv_names = full_config.forward_model.radiative_transfer.statevector.get_element_names()
        if engine_config.statevector_names is not None:
            self.statevector_names = [x for x in full_sv_names if x in engine_config.statevector_names]
        else:
            self.statevector_names = full_sv_names

        self.lut_dir = engine_config.lut_path
        self.n_point = len(self.lut_grid_config)
        self.n_state = len(self.statevector_names)

        self.luts = {}

        # Retrieved variables.  We establish scaling, bounds, and
        # initial guesses for each state vector element.  The state
        # vector elements are all free parameters in the RT lookup table,
        # and they all have associated dimensions in the LUT grid.
        self.bounds, self.scale, self.init = [], [], []
        self.prior_mean, self.prior_sigma = [], []
        for key in self.statevector_names:
            element: StateVectorElementConfig = full_config.forward_model.radiative_transfer.statevector.get_single_element_by_name(
                key)
            self.bounds.append(element.bounds)
            self.scale.append(element.scale)
            self.init.append(element.init)
            self.prior_sigma.append(element.prior_sigma)
            self.prior_mean.append(element.prior_mean)
        self.bounds = np.array(self.bounds)
        self.scale = np.array(self.scale)
        self.init = np.array(self.init)
        self.prior_mean = np.array(self.prior_mean)
        self.prior_sigma = np.array(self.prior_sigma)

        # This array is used to handle the potential indexing mismatch between
        # the 'global statevector' (which may couple multiple radiative transform
        # models) and this statevector. It should never be modified
        full_to_local_statevector_position_mapping = []
        self.complete_statevector_names = full_config.forward_model.radiative_transfer.statevector.get_element_names()
        for sn in self.statevector_names:
            ix = self.complete_statevector_names.index(sn)
            full_to_local_statevector_position_mapping.append(ix)
        self._full_to_local_statevector_position_mapping = \
            np.array(full_to_local_statevector_position_mapping)


        #
        self.lut_dims = []
        self.lut_grids = []
        self.lut_names = []
        self.lut_interp_types = []
        for key, grid_values in self.lut_grid_config.items():

            # do some quick checks on the values
            if len(grid_values) == 1:
                err = 'Only 1 value in LUT grid {}. ' +\
                    '1-d LUT grids cannot be interpreted.'.format(key)
                raise ValueError(err)
            if grid_values != sorted(grid_values):
                logging.error('Lookup table grid needs ascending order')
                raise ValueError('Lookup table grid needs ascending order')

            # Store the values
            self.lut_grids.append(grid_values)
            self.lut_dims.append(len(grid_values))
            self.lut_names.append(key)

            # Store in an indication of the type of value each key is
            # (normal - n, degree - d, radian - r)
            if key in self.angular_lut_keys_radians:
                self.lut_interp_types.append('r')
            elif key in self.angular_lut_keys_degrees:
                self.lut_interp_types.append('d')
            else:
                self.lut_interp_types.append('n')

        # Cast as array for faster reference later
        self.lut_interp_types = np.array(self.lut_interp_types)

        # "points" contains all combinations of grid points
        # We will have one filename prefix per point
        self.points = common.combos(self.lut_grids)
        self.files = self.get_lut_filenames()

    def build_lut(self, rebuild=False):
        """Each LUT is associated with a source directory.  We build a lookup 
            table by: 
              (1) defining the LUT dimensions, state vector names, and the 
                    grid of values; 
              (2) running the radiative transfer solver if needed, with each 
                    run defining a different point in the LUT; and 
              (3) loading the LUTs, one per key atmospheric coefficient vector,
                  into memory as VectorInterpolator objects."""

       # Build the list of radiative transfer run commands. This
        # rebuild_cmd() function will be overriden by the child class to
        # perform setup activities unique to each RTM.
        rebuild_cmds = []
        for point, fn in zip(self.points, self.files):
            try:
                cmd = self.rebuild_cmd(point, fn)
                rebuild_cmds.append(cmd)
            except FileExistsError:
                pass

        if self.configure_and_exit:
            raise SystemExit
            # sys.exit(0)

        elif len(rebuild_cmds) > 0 and self.auto_rebuild:
            logging.info("rebuilding")

            # check to make sure lut directory is there, create if not
            cwd = os.getcwd()
            if os.path.isdir(self.lut_dir) is False:
                os.mkdir(self.lut_dir)

            # migrate to the appropriate directory and spool up runs
            os.chdir(self.lut_dir)

            if self.implementation_config.n_cores is None:
                n_cores = multiprocessing.cpu_count()
            else:
                n_cores = self.implementation_config.n_cores

            if self.implementation_config.runtime_nice_level is None:
                pool = multiprocessing.Pool(processes=n_cores)
            else:
                pool = multiprocessing.Pool(processes=n_cores, initializer=
                                            common.nice_me(self.implementation_config.runtime_nice_level))

            r = pool.map_async(spawn_rt, rebuild_cmds)
            r.wait()
            os.chdir(cwd)

    def get_lut_filenames(self):
        files = []
        for point in self.points:
            outf = '_'.join(['%s-%6.4f' % (n, x)
                             for n, x in zip(self.lut_names, point)])
            files.append(outf)
        return files

    def summarize(self, x_RT, geom):
        """Summary of state vector."""

        if len(x_RT) < 1:
            return ''
        return 'Atmosphere: '+' '.join(['%s: %5.3f' % (si, xi) for si, xi in
                                        zip(self.statevector_names, x_RT[self._full_to_local_statevector_position_mapping])])
