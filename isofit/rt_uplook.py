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

from sys import platform
import json
import julian
import os, sys
import re
import scipy as s
import datetime
from common import json_load_ascii, combos, VectorInterpolator
from common import recursive_replace, load_spectrum, load_wavelen
from copy import deepcopy
from scipy.stats import norm as normal
from scipy.interpolate import interp1d
from rt_lut import TabularRT, FileExistsError
from rt_modtran import ModtranRT

eps = 1e-5  # used for finite difference derivative calculations

rfm_template = '''*HDR
09NOV18 RFM Example: zenith transmittance
*FLG
   OBS ZEN TRA
*SPC
   {wn_start} {wn_end} {wn_del} 
*GAS
   H2O O3 CH4 CO2 O2 N2O
*ATM
   {profile_path}
*SEC
   {airmass_factor}
*OBS
   {bottom_alt}
*HIT
   {hit_path}
*OUT
   TRAFIL = {transmittance_path}
*END'''

rfm_script_template = '''
export origdir=`pwd`
cd {rfm_rundir_path}
{rfm_dir}/rfm 2>&1 > {rfm_log_path}
cd $origdir
'''

atm_template = '''! FASCOD Model 2. Midlatitude Summer Atmosphere                                
! Transformed to RFM .atm file format by program USARFM v.23-AUG-96            
{NLEV} ! No.Levels in profiles                                                  
*HGT [km]                               
{HGT}
*PRE [mb]                               
{PRE}
*TEM [K]                                
{TEM}
*H2O [ppmv]                                
{H2O}
*CO2 [ppmv]                                
{CO2}
*O3  [ppmv]                                
{O3}                             
*N2O [ppmv]                                
{N2O}                             
*CO  [ppmv]                                
{CO}
*CH4 [ppmv]                                
{CH4}
*O2  [ppmv]                                
{O2}
*END'''


class Profile:

    def __init__(self, fname):
        with open(fname,'r') as fin:
            state = 'header'
            lines = fin.readlines()
            X = {}
            for line in lines:
                if line.startswith('!'):
                    continue
                elif state == 'header':
                    X['NLEV'] = int(line.split()[0])
                    state = 'label'
                elif state == 'label':
                    label = line.split()[0].replace('*','')
                    if label == 'END':
                       break
                    X[label] = []
                    state = label
                else:
                    clean = re.sub(r"[\s]", "", line, flags=re.UNICODE)
                    clean.replace('E','e').replace('+','')
                    X[state].extend([float(q) for q in clean.split(',') 
                        if len(q)>0])
                    if len(X[state]) == X['NLEV']:
                        state = 'label'
            for key, val in X.items():
                X[key] = s.array(X[key])
            self.atm = X

    def __str__(self):
        strings = {'NLEV':'%i'%self.atm['NLEV']}
        for label, v in self.atm.items():
            if label == 'NLEV': 
                 continue
            st = ''
            for i in s.arange(0,len(v),5):
                st = st+','.join(['%10e'% v[j] for j in s.arange(i,i+5)])
                if i<(self.atm['NLEV']-5):
                    st = st+'\n'
            strings[label] = st[:].replace('e','E') # copy
        return atm_template.format(**strings)

    def rescale(self, param, val):
        self.atm[param] = self.atm[param]*val


class UplookRT(TabularRT):
    """A model of photon transport including the atmosphere, for upward-
       looking spectra."""

    def __init__(self, config):

        TabularRT.__init__(self, config)
        self.rfm_dir = self.find_basedir(config)
        self.wl, self.fwhm = load_wavelen(config['wavelength_file'])
        domain        = config['domain'] 
        self.wn_start = 1e7 / domain['end']  # wavenumbers in reverse order :)
        self.wn_end   = 1e7 / domain['start'] 
        self.nsteps   = domain['end']-domain['start'] 
        self.n_chan   = int((domain['end']-domain['start'])/domain['step'])
        self.wn_del   = (self.wn_end-self.wn_start)/float(self.n_chan-1)
        self.wn_grid  = s.linspace(self.wn_start, self.wn_end, self.n_chan)
        self.wl_grid  = 1e7 / s.flip(self.wn_grid,0)
        print('wn step: ',self.wn_del)
        self.rfm_grid_wn = \
            s.arange(self.wn_start, self.wn_end+self.wn_del, self.wn_del)
        self.rfm_grid = 1e7 / self.rfm_grid_wn
        self.rfm_dir = self.find_basedir(config)
        self.atm_path = config['rfm_atm_path']
        self.params = {'wn_start': self.wn_start, 
                       'wn_end': self.wn_end, 
                       'wn_del': self.wn_del,
                       'hit_path': config['hit_path'],
                       'bottom_alt': config['bottom_alt'],
                       'H2OSCL': 1.0,
                       'O3SCL': 1.0}

        if 'obs_file' in config:
            # A special case where we load the observation geometry
            # from a custom-crafted text file
            g = Geometry(obs=config['obs_file'])
            self.params['solzen'] = g.solar_zenith
            self.params['solaz'] = g.solar_azimuth
            self.params['viewzen'] = g.observer_zenith
            self.params['viewaz'] = g.observer_azimuth
        else:
            # We have to get geometry from somewhere, so we presume it is
            # in the configuration file.
            for f in ['solzen', 'viewzen', 'solaz', 'viewaz']:
                self.params[f] = config[f]
        
        airmass =  1.0/s.cos(self.params['viewzen']/360.0*2*s.pi)
        self.params['airmass_factor'] = airmass
                       
        self.esd = s.loadtxt(config['earth_sun_distance_path'])
        dt = julian.from_jd(config['julian_date'], fmt='mjd')
        self.day_of_year = dt.timetuple().tm_yday
        self.irr_factor = self.esd[self.day_of_year-1, 1]

        irr = s.loadtxt(config['irradiance_file'], comments='#')
        self.iwl, self.irr = irr.T
        self.irr = self.irr / 10.0  # convert, uW/nm/cm2
        self.irr = self.irr / self.irr_factor**2  # consider solar distance

        self.build_lut()


    def build_lut(self, rebuild=False):
        """ Each LUT is associated with a source directory.  We build a 
            lookup table by: 
              (1) defining the LUT dimensions, state vector names, and the grid 
                  of values; 
              (2) running modtran if needed, with each MODTRAN run defining a 
                  different point in the LUT; and 
              (3) loading the LUTs, one per key atmospheric coefficient vector,
                  into memory as VectorInterpolator objects."""

        TabularRT.build_lut(self, rebuild)

    
    def find_basedir(self, config):
        '''Seek out a sixs base directory'''

        if 'rfm_installation' in config:
            return config['rfm_installation']
        if 'RFM_DIR' in os.environ:
            return os.getenv('RFM_DIR')
        return None


    def rebuild_cmd(self, point, fn):

        rfm_profile_fn = 'LUT_'+fn+'.atm'
        rfm_profile_path = os.path.join(self.lut_dir, 'LUT_'+fn+'.atm')
        rfm_rundir_dn  = 'LUT_'+fn+'_rundir'
        rfm_rundir_path = os.path.join(self.lut_dir, rfm_rundir_dn)
        rfm_config_path = os.path.join(rfm_rundir_path, 'rfm.drv')
        rfm_script_fn  = 'LUT_'+fn+'.sh'
        rfm_script_path = os.path.join(self.lut_dir, rfm_script_fn)
        rfm_log_fn  = 'LUT_'+fn+'.log'
        rfm_log_path = os.path.join(self.lut_dir, rfm_log_fn)
        rfm_output_fn  = fn
        rfm_output_path = os.path.join(self.lut_dir, rfm_output_fn)
        
        # update the rfm configuration file
        vals = self.params.copy()
        vals.update(dict([(n, v) for n, v in zip(self.lut_names, point)]))
        vals['profile_path'] = rfm_profile_path
        vals['transmittance_path'] = rfm_output_path
        profile = Profile(self.atm_path)
        for label in profile.atm:
           for p in self.statevec:
               if label == p+'SCL':
                  profile.rescale(param, point[self.statevec.index(p)]) 
        rfm_config_str = rfm_template.format(**vals)

        # Check rebuild conditions: LUT is missing or from a different config
        if not os.path.exists(rfm_config_path) or\
           not os.path.exists(rfm_script_path) or\
           not os.path.exists(rfm_profile_path):
            rebuild = True
        else:
            with open(rfm_config_path, 'r') as f:
                existing = f.read()
                rebuild = (rfm_config_str.strip() != existing.strip())

        if not rebuild:
            raise FileExistsError('File exists')

        # write profile to file
        with open(rfm_profile_path, 'w') as fout:
            fout.write(str(profile))

        # write rfm configuration file
        if not os.path.exists(rfm_rundir_path):
            os.mkdir(rfm_rundir_path)
        with open(rfm_config_path, 'w') as fout:
            fout.write(rfm_config_str)

        # write rfm script that will copy the driver and recover output
        script = rfm_script_template[:]

        with open(rfm_script_path, 'w') as fout:
            fout.write(script.format(rfm_dir = self.rfm_dir,
                        rfm_rundir_path = rfm_rundir_path,
                        rfm_log_path = rfm_log_path))

        # Specify the command to run the script
        cmd = 'bash ' + rfm_script_path
        return cmd

    def load_rt(self, point, fn):
        trafile = self.lut_dir+'/'+fn
        transm = s.loadtxt(trafile, skiprows=4)
        wl = self.wl_grid
        assert(len(wl)==len(transm))
        sol = s.ones(transm.shape) * s.pi  # to get a radiance of unity
        rhoatm = s.zeros(transm.shape)
        sphalb = s.zeros(transm.shape)
        transup = s.zeros(transm.shape)
        solzen = 0.0
        return wl, sol, solzen, rhoatm, transm, sphalb, transup

    def get(self, x_RT, geom):
        rhoatm, sphalb, transm, transup = TabularRT.get(self, x_RT, geom)
        if 'BIAS' in self.statevec:
            idx_bias = self.statevec.index('BIAS')
            rhoatm = s.ones(transm.shape) * x_RT[idx_bias]
        return rhoatm, sphalb, transm, transup

if __name__ == '__main__':
   atm = AtmFile(sys.argv[1])
   atm.write(sys.argv[2])

