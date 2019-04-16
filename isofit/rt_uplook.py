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
from common import resample_spectrum
from copy import deepcopy
from scipy.stats import norm as normal
from scipy.interpolate import interp1d
from rt_lut import TabularRT, FileExistsError
from rt_modtran import ModtranRT
import logging

eps = 1e-5  # used for finite difference derivative calculations

rfm_template = '''*HDR
09NOV18 RFM Example: zenith transmittance
*FLG
   OBS ZEN TRA CTM
*SPC
   {wn_start} {wn_end} {wn_del} 
*GAS
   H2O O3 CH4 CO2 O2 N2O
*ATM
   {rfm_profile_path}
*SEC
   {airmass_factor}
*OBS
   {observer_altitude_km}
*HIT
   {hit_path}
*OUT
   TRAFIL = {rfm_output_path}
*END'''

script_template_old = '''
{sixs_exe_path} < {sixs_config_path} > {sixs_output_path}
'''


script_template = '''
export origdir=`pwd`
cd {rfm_rundir_path}
{rfm_dir}/rfm 2>&1 > {rfm_log_path}
{sixs_exe_path} < {sixs_config_path} > {sixs_output_path}
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

sixs_uplook_template = '''0 (User defined)
{viewzen} 0 {viewzen} 0 {month} {day}
0  (No absorption)
1
0
0
-{surface_elevation_km} (target level)
-1000 (sensor level)
-2 
{wl_inf}
{wl_sup}
0 Homogeneous surface
0 (no directional effects)
0
0
0
-1 No atm. corrections selected
'''

sixs_airborne_template = '''0 (User defined)
{solzen} {solaz} {viewzen} {viewaz} {month} {day}
7  (User defined atmospheric profile)
{profile_6sv}
{aermodel}
0
{AOT550}
-{surface_elevation_km} (target level)
-{observer_altitude_km} (sensor level)
-{H2OSTR}, -{O3}
{AOT550}
-2 
{wl_inf}
{wl_sup}
0 Homogeneous surface
0 (no directional effects)
0
0
0
-1 No atm. corrections selected
'''



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

    def air_molecules_per_m3(self, P, T):
        P_Pa = P * 100 # From mb 
        air_kg_per_m3 = P_Pa / T / 287.058 #J kg-1 K-1 gas_constant
        air_kg_per_mol = 0.02897
        air_moles_per_m3 = air_kg_per_m3 / air_kg_per_mol
        return air_moles_per_m3 * 6.022e23 

    def h2o_g_per_m3(self, P, T, h2o_ppmv):
        h2o_molecules_per_m3 = h2o_ppmv * self.air_molecules_per_m3(P,T) / 1e6
        h2o_g_per_mol, molecules_per_mol = 18.02, 6.022e23
        return h2o_g_per_mol / molecules_per_mol * h2o_molecules_per_m3

    def o3_g_per_m3(self, P, T, o3_ppmv):
        o3_molecules_per_m3  = o3_ppmv  * self.air_molecules_per_m3(P,T) / 1e6
        o3_g_per_mol, molecules_per_mol = 48.0, 6.022e23 
        return o3_g_per_mol  / molecules_per_mol * o3_molecules_per_m3

    def format_6sv(self, nlev = 34):
        """Translate the profile to 6SV format""" 
        levels = [int(s.floor(q)) for q in s.linspace(0,self.atm['NLEV']-1,nlev)]
        profile = ''
        for i, lev in enumerate(levels):
            alt      = self.atm['HGT'][lev]
            P        = self.atm['PRE'][lev]
            T        = self.atm['TEM'][lev]
            h2o_ppmv = self.atm['H2O'][lev]
            o3_ppmv  = self.atm['O3'][lev]
            lev_str  = '%f,%f,%f,%f,%f' % (alt, P, T, 
                    self.h2o_g_per_m3(P, T, h2o_ppmv), 
                    self.o3_g_per_m3(P, T, o3_ppmv))
            profile = profile + lev_str 
            if lev != levels[-1]:
              profile = profile + '\n'
        return profile

    def rescale(self, param, val):
        self.atm[param] = self.atm[param] * val

    def calc_column_h2o(self):
        h2o, alt         = self.atm['H2O'], self.atm['HGT']
        T, P             = self.atm['TEM'], self.atm['PRE']
        heights          = s.diff(alt)
        Ts               = (T[1:] + T[:-1])/2.0
        Ps               = P[:-1] *  s.exp(-s.log(P[:-1]/P[1:])/2.0)
        ppmvs            = (h2o[1:] + h2o[:-1])/2.0
        total_column_pwv = 0
        for h2o, P, T, hgt_km in zip(ppmvs, Ps, Ts, heights):
            hgt_m = hgt_km * 1000.0
            m2_per_cm2 = 1.0 / 10000.0
            pwv = self.h2o_g_per_m3(P, T, h2o) * hgt_m * m2_per_cm2
            total_column_pwv = total_column_pwv + pwv
        return total_column_pwv

    def calc_column_o3(self):
        o3, alt         = self.atm['O3'], self.atm['HGT']
        T, P            = self.atm['TEM'], self.atm['PRE']
        heights         = s.diff(alt)
        Ts              = (T[1:] + T[:-1])/2.0
        Ps              = P[:-1] *  s.exp(-s.log(P[:-1]/P[1:])/2.0)
        ppmvs           = (o3[1:] + o3[:-1])/2.0
        total_column_o3 = 0
        for o3, P, T, hgt_km in zip(ppmvs, Ps, Ts, heights):
            #print(o3,P,T,hgt_km,total_column_o3)
            hgt_m = hgt_km * 1000.0
            g_per_mol = 48.0 
            moles_o3_per_m2 = self.o3_g_per_m3(P, T, o3) * hgt_m  / g_per_mol
            gas_constant = 8.3144598
            m_o3 = gas_constant * T * moles_o3_per_m2 / P 
            mm_o3 = m_o3 * 1000.0
            total_column_o3 = total_column_o3 + mm_o3
        return total_column_o3


class UplookRT(TabularRT):
    """A model of photon transport including the atmosphere, for upward-
       looking spectra."""

    def __init__(self, config):

        TabularRT.__init__(self, config)
        self.wl, self.fwhm = load_wavelen(config['wavelength_file'])
        domain          = config['domain'] 
        self.hit_path   = config['hit_path']
        self.atm_path   = config['rfm_atm_path']
        self.observer_altitude_km = config['observer_altitude_km']
        self.surface_elevation_km = config['surface_elevation_km']
        self.uplook_overrides = True

        # Wavenumber grid at superhigh resolution for RTM absorptions
        self.wn_start   = 1e7 / domain['end']  # wavenumbers in reverse order
        self.wn_end     = 1e7 / domain['start'] 
        self.nsteps     = domain['end']-domain['start'] 
        self.n_chan_rfm = int((domain['end']-domain['start'])/domain['step'])
        self.wn_del     = (self.wn_end-self.wn_start)/float(self.n_chan_rfm-1)
        self.wn_grid    = s.linspace(self.wn_start, self.wn_end, self.n_chan_rfm)
        self.rfm_grid_wn = \
            s.arange(self.wn_start, self.wn_end + self.wn_del, self.wn_del)
        self.rfm_grid   = 1e7 / self.rfm_grid_wn
        self.wl_grid    = 1e7 / s.flip(self.wn_grid,0) # wavelengths lo -> hi

        # Sixs grid, initial
        sixs_grid_init  = s.arange(self.wl_grid[0], self.wl_grid[-1]+2.5, 2.5)
        self.sixs_ngrid = len(sixs_grid_init)
        self.wl_inf     = sixs_grid_init[0] / 1000.0 # convert to microns
        self.wl_sup     = sixs_grid_init[-1]/ 1000.0 # convert to microns
        if not self.uplook_overrides:
            self.AOT550, self.aermodel = config['AOT550'], config['aermodel']
        else:
            self.AOT550, self.aermodel = 0, 0

        # Find the RFM installation directory
        if 'rfm_installation' in config:
            self.rfm_dir = config['rfm_installation']
        elif 'RFM_DIR' in os.environ:
            self.rfm_dir = os.environ['RFM_DIR']
        else:
            logging.error('Must specify an RFM installation')

        # Find the 6SV installation directory
        if 'sixs_installation' in config:
            self.sixs_dir = config['sixs_installation']
        elif 'SIXS_DIR' in os.environ:
            self.sixs_dir = os.environ['SIXS_DIR']
        else:
            logging.error('Must specify a 6SV installation')
        self.sixs_exe_path = self.sixs_dir+'/sixsV2.1'
        
        # A special case where we load the observation geometry
        # from a custom-crafted text file
        if 'obs_file' in config:
            g = Geometry(obs=config['obs_file'])
            self.solzen  = g.solar_zenith
            self.solaz   = g.solar_azimuth
            self.viewzen = g.observer_zenith
            self.viewaz  = g.observer_azimuth
        else:
            # We presume it is in the configuration file.
            for f in ['solzen', 'viewzen', 'solaz', 'viewaz']:
                setattr(self,f,config[f])
                       
        self.esd = s.loadtxt(config['earth_sun_distance_path'])
        dt = julian.from_jd(config['julian_date'], fmt='mjd')
        self.timetuple = t = dt.timetuple()
        self.day_of_year = self.timetuple.tm_yday
        self.month, self.day = self.timetuple[1:3]
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


    def rebuild_cmd(self, point, fn):
        """ Every point in the grid defines an RFM run (to calculate gas 
            absorptions) and a 6SV run (to calculate scattering)"""

        # create the atmospheric profile, apply any gridpoint-specific scaling
        # of the gas concentrations, temperatures, etc.
        profile = Profile(self.atm_path)

        # set up filenames and paths for RFM
        rfm_profile_fn = 'LUT_'+fn+'.atm'
        rfm_profile_path = os.path.join(self.lut_dir, 'LUT_'+fn+'.atm')
        rfm_rundir_dn  = 'LUT_'+fn+'_rundir'
        rfm_rundir_path = os.path.join(self.lut_dir, rfm_rundir_dn)
        rfm_config_path = os.path.join(rfm_rundir_path, 'rfm.drv')
        script_fn  = 'LUT_'+fn+'.sh'
        script_path = os.path.join(self.lut_dir, script_fn)
        rfm_log_fn  = 'LUT_'+fn+'.log'
        rfm_log_path = os.path.join(self.lut_dir, rfm_log_fn)
        rfm_output_fn  = fn+'.rfm'
        rfm_output_path = os.path.join(self.lut_dir, rfm_output_fn)

        # set up filenames and paths for 6SV
        sixs_config_fn = 'LUT_'+fn+'.6sv'
        sixs_config_path = os.path.join(self.lut_dir, sixs_config_fn)
        sixs_output_fn = fn+'.sca'
        sixs_output_path = os.path.join(self.lut_dir, sixs_output_fn)

        # build the template format strings for this gridpoint
        params = {'rfm_profile_path':   rfm_profile_path,
                  'rfm_output_path':    rfm_output_path,
                  'rfm_log_path':       rfm_log_path,
                  'sixs_config_path':   sixs_config_path,
                  'sixs_output_path':   sixs_output_path,
                  'rfm_rundir_path':    rfm_rundir_path}

        for f in ['wn_start', 'wn_end', 'wn_del', 'hit_path', 'day', 'month',
                  'solzen', 'solaz', 'viewzen', 'viewaz', 'rfm_dir',
                  'sixs_dir', 'sixs_exe_path', 'aermodel', 'wl_inf', 'wl_sup', 
                  'observer_altitude_km', 'surface_elevation_km']:
            params[f] = getattr(self, f)

        # update various geometry elements airmass factor 
        airmass =  1.0/s.cos(params['viewzen']/360.0*2*s.pi)
        params['airmass_factor'] = airmass
               
        # update the configuration parameters to match the LUT gridpoint
        params.update(dict([(n, v) for n, v in zip(self.lut_names, point)]))

        # update the atmospheric profile
        for p in self.statevec:
           if 'SCL' in p:
              label = p.replace('SCL','')
              profile.rescale(label, point[self.statevec.index(p)]) 
        params['H2OSTR'] = profile.calc_column_h2o()
        params['O3']     = profile.calc_column_o3()
        print('H2OSTR',params['H2OSTR'],'O3',params['O3'])
        params['profile_6sv'] = profile.format_6sv()

        # by convention, we always update the OBSZEN variable, but must
        # keep the others consistent
        if "OBSZEN" in self.lut_grid:
          obszen_ind = self.lut_names.index("OBSZEN")
          params['viewzen'] = point[obszen_ind] # MODTRAN downlooking convention 
        amf = 1.0/s.cos((params['viewzen'])/360.0*2*s.pi)
        params['airmass_factor'] =  amf

        # create the up-to-date configuration strings
        rfm_config_str  = rfm_template.format(**params)
        script_str      = script_template.format(**params)
        if self.uplook_overrides:
            sixs_config_str = sixs_uplook_template.format(**params)
        else:
            sixs_config_str = sixs_airborne_template.format(**params)

        # Rebuild if an LUT is missing 
        rebuild = False
        if not os.path.exists(rfm_config_path) or\
           not os.path.exists(script_path) or\
           not os.path.exists(rfm_profile_path) or\
           not os.path.exists(sixs_config_path) or\
           not os.path.exists(sixs_output_path): 
            rebuild = True
        else:
           # Rebuild if the new configuration differs 
           with open(rfm_config_path, 'r') as f_rfm:
               existing_rfm = f_rfm.read()
           with open(sixs_config_path, 'r') as f_sixs:
               existing_sixs = f_sixs.read()
           if rfm_config_str.strip()  != existing_rfm.strip() or \
              sixs_config_str.strip() != existing_sixs.strip():
                  rebuild = True
        if not rebuild:
            raise FileExistsError('File exists')

        # set up run subdirectory and write all files
        if not os.path.exists(rfm_rundir_path):
            os.mkdir(rfm_rundir_path)
        with open(rfm_profile_path, 'w') as fout:
            fout.write(str(profile))
        with open(rfm_config_path, 'w') as fout:
            fout.write(rfm_config_str)
        with open(sixs_config_path, 'w') as fout:
            fout.write(sixs_config_str)
        with open(script_path, 'w') as fout:
            fout.write(script_str)

        # Specify the command to run the script
        cmd = 'bash ' + script_path 
        return cmd

    def load_rt(self, point, fn):
        '''Load both 6SV and RTM runs.'''

        # set up filenames 
        logging.debug('Loading LUT grid point %s'%fn)
        script_fn  = 'LUT_'+fn+'.sh'
        script_path = os.path.join(self.lut_dir, script_fn)
        rfm_output_fn  = fn+'.rfm'
        rfm_output_path = os.path.join(self.lut_dir, rfm_output_fn)
        sixs_config_fn = 'LUT_'+fn+'.6sv'
        sixs_config_path = os.path.join(self.lut_dir, sixs_config_fn)
        sixs_output_fn = fn+'.sca'
        sixs_output_path = os.path.join(self.lut_dir, sixs_output_fn)

        # load RFM gas transmissions
        gas_xm = s.flip(s.loadtxt(rfm_output_path, skiprows=4),0)
        wl = self.wl_grid
        assert(len(wl)==len(gas_xm))
        gas_xm  = resample_spectrum(gas_xm, self.wl_grid, self.wl, self.fwhm)
        rhoatm  = s.zeros(gas_xm.shape)
        sphalb  = s.zeros(gas_xm.shape)
        transup = s.zeros(gas_xm.shape)

        # load 6SV solar zenith configuration 
        with open(sixs_config_path, 'r') as l:
            inlines = l.readlines()
            solzen = float(inlines[1].strip().split()[0])

        # load 6SV scattering and strip header
        with open(sixs_output_path, 'r') as l:
            lines = l.readlines()
        for i, ln in enumerate(lines):
            if ln.startswith('*        trans  down   up'):
                lines = lines[(i + 1):(i + 1 + self.sixs_ngrid)]
                break

        sphalbs = s.zeros(len(lines))
        transups = s.zeros(len(lines))
        transms = s.zeros(len(lines))
        rhoatms = s.zeros(len(lines))
        self.grid = s.zeros(len(lines))

        for i, ln in enumerate(lines):
            ln = ln.replace('*', ' ').strip()
            w, gt, scad, scau, salb, rhoa, swl, step, sbor, dsol, toar = \
                ln.split()

            self.grid[i] = float(w) * 1000.0  # convert to nm
            sphalbs[i]   = float(salb) 
            transms[i]   = float(scau) * float(scad) * float(gt)
            rhoatms[i]   = float(rhoa) 

            if self.uplook_overrides:
                transms[i] = float(scau) # one direction only

        logging.debug('Resampling LUT grid point %s'%fn)
        transm  = interp1d(self.grid, transms)(self.wl)
        transm  = transm * gas_xm
        transup = s.zeros(transm.shape)
    
        if self.uplook_overrides:
            sphalb = s.zeros(transm.shape)
            rhoatm = s.zeros(transm.shape)
            irr    = s.ones(transm.shape) * s.pi  
        else:
            sphalb  = interp1d(self.grid, sphalbs)(self.wl)
            rhoatm  = interp1d(self.grid, rhoatms)(self.wl)
            irr = resample_spectrum(self.irr, self.iwl,  self.wl, self.fwhm)

        return self.wl, irr, solzen, rhoatm, transm, sphalb, transup


    def get(self, x_RT, geom):
        rhoatm, sphalb, transm, transup = TabularRT.get(self, x_RT, geom)
        if 'BIAS' in self.statevec:
            idx_bias = self.statevec.index('BIAS')
            rhoatm = s.ones(transm.shape) * x_RT[idx_bias]
        return rhoatm, sphalb, transm, transup

if __name__ == '__main__':
   atm = AtmFile(sys.argv[1])
   atm.write(sys.argv[2])

