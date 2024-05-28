#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 13:49:27 2022

@author: hagen
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import atmPy.aerosols.physics.shapes as atmshape

def split_dist(dist):
    """
    Splits the size distribution into accumulation mode and coarse mode along 750 nm

    Parameters
    ----------
    dist : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # dist = dist.copy()

    dist_smps = dist.zoom_diameter(start = 50,end = 750)
    dist_aps = dist.zoom_diameter(start = 750, end = 1.53e4) # the end is because the last 
    
    # ensure there are no nans anywhere
    dist_smps.data[dist_smps.data.isna()] = 0
    dist_aps.data[dist_aps.data.isna()] = 0
    
    assert(dist_aps.data.isna().sum().sum() == 0), 'found nans, not allowed'
    assert(dist_smps.data.isna().sum().sum() == 0), 'found nans, not allowed'
    
    return dict(dist_smps = dist_smps, dist_aps = dist_aps)

def lut2scatt_coeff(lut, dist, shape_c, n_real, n_imag, smoothen = True):
    # size parameter to axis ratio
    pshape = abs(shape_c) + 1 #ratio between parrlell and rotational axis of spheroid
    if shape_c > 0: # Thesar are t prolates. Prolates have a longer rotatational vs parallel axis -> invert
        pshape = 1/pshape
    assert(lut.pshape.min() <= pshape <= lut.pshape.max()), f'pshape ({pshape}) not in valid interval ([{float(lut.pshape.min())},{float(lut.pshape.max())}])'
    assert(lut.n_real.min() <= n_real <= lut.n_real.max()), f'n_real ({n_real}) not in valid interval ([{float(lut.n_real.min())},{float(lut.n_real.max())}])'
    assert(lut.n_imag.min() <= n_imag <= lut.n_imag.max()), f'pshape ({n_imag}) not in valid interval ([{float(lut.n_imag.min())},{float(lut.n_imag.max())}])'
    
    disttm = dist.copy()
    
    # interpolate lut to get size dependent scattering crossection
    # scattcross_tm = lut.scatt_cross_scect.interp({'pshape': 5, 'n_real' : 1.5, 'n_imag': 0.02}).to_pandas()
    scattcross_tm = lut.scatt_cross_scect.interp({'pshape': pshape, 'n_real' : n_real, 'n_imag': n_imag}).to_pandas()
    
    # smoothen
    if smoothen:
        scattcross_tm = scattcross_tm.rolling(5, center=True, min_periods= 1, win_type='gaussian').mean(std = 1.5)
    
    # map scattering coeeficient diameters onto size distribution
    scattcross_tm = scattcross_tm.reindex(scattcross_tm.index | disttm.bincenters).interpolate().reindex(disttm.bincenters)

    # make format so it can be fed into the sizedist.optical.parameters
    tm_res = {'mie':pd.DataFrame(scattcross_tm, columns=['scattering_crossection']), 'angular_scatt_func': np.nan}
  
    # apply to particular sizedistribution
    disttm.optical_properties.parameters.mie_result = tm_res
    return float(disttm.optical_properties.scattering_coeff.iloc[0].values)

def create_parameter_samples(lut,
                             pshape_lims = (-2.7,2.7),
                             n_real_cs = (1.5, 0.028/2),
                             n_imag_cs = (0.002, 0.0015,'log'),
                             d_inst_cs = (1, 0.02),
                             roh_cs = None, #(2.5, 0.15),
                             sample_size = 300):
    
    
    # lut = lut_accu
    # sigmax = 1
    rng = np.random.default_rng()
    
    if n_imag_cs[2] == 'log':
        n_imag = rng.lognormal(mean=np.log(n_imag_cs[0]), sigma=n_imag_cs[1]/n_imag_cs[0] ,size=sample_size)
    elif n_imag_cs[2] == 'lin':
        n_imag = (rng.standard_normal(sample_size) * n_imag_cs[1]) + n_imag_cs[0]
        
    paramdict = {'pshapes':rng.uniform(pshape_lims[0], pshape_lims[1], size = sample_size),
                       'n_real': (rng.standard_normal(sample_size) * n_real_cs[1]) + n_real_cs[0],
                       'n_imag': n_imag,
                       'ce': rng.standard_normal(sample_size),
                       'd': rng.standard_normal(sample_size) * d_inst_cs[1] + d_inst_cs[0],
                      }
    
    if not isinstance(roh_cs, type(None)):
        paramdict['roh'] = rng.standard_normal(sample_size) * roh_cs[1] + roh_cs[0]
    
    #### remove extreme values that are outside the LUT. The LUT was designed to cover 3 sigma!!
    # Keep in mind, normal distributions have long tails. Those values need to be removed.
    
    df = pd.DataFrame(paramdict)
    
    lims = float(-lut.pshape.max()) + 1 ,1/(float(lut.pshape.min())) - 1
    df.loc[df[np.logical_or(lims[0] > df.pshapes, df.pshapes > lims[1])].index, 'pshapes'] = np.nan
    df.dropna(inplace=True)
    # print(df.shape)
    
    lims = float(lut.n_real.min()),float(lut.n_real.max())
    df.loc[df[np.logical_or(lims[0] > df.n_real, df.n_real > lims[1])].index, 'n_real'] = np.nan
    df.dropna(inplace=True)
    # print(df.shape)
    
    lims = float(lut.n_imag.min()),float(lut.n_imag.max())
    df.loc[df[np.logical_or(lims[0] > df.n_imag, df.n_imag > lims[1])].index, 'n_real'] = np.nan
    df.dropna(inplace=True)
    # print(df.shape)
    
    # df.dropna(inplace=True)
    return df
    
def size_distribution_correction(dist,
                                 instrument = 'aps',
                                dd = 0.9,  
                                cc = 2., 
                                ce_log = None,
                                shape_c = 2.5 , 
                                roh = 2.7,
                                plot = True,
                                    ):
    """
    This corrects the size distribution (diameters, and particle numbers) according to assumed parameters.

    Parameters
    ----------
    dist : TYPE
        DESCRIPTION.
    daps : TYPE, optional
        diameter uncertainty of APS, this is the intrinsic uncertainty not the chi related. The default is 0.9.
    cc : TYPE, optional
        this is diameter dependen counting effeicnecy uncertatinty in coarse mode. This is the sigma of a normal distribution                                     shape_c : TYPE, optional
        The default is 2.5.
    shape_c: ...
        shape parameter. This is the parameter that goes from -x to +y in our case from -5 to 5                                     
    roh : TYPE, optional
        DESCRIPTION. The default is 2.7.
    plot : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    dist : TYPE
        DESCRIPTION.

    """

    #### 1. dist preconditioning
    #######
    dist_in = dist
    dist = dist_in.copy()
    dist = dist.convert2numberconcentration()



    #### 2. diameter precision & accuracy, I actually do not know which of the two it is?
    ##########
    dist_tmp = dist.copy()
    dist = dist.grow_sizedistribution(dd, raise_particle_loss_error=False)
    # sometime very small particle losses can accure, so I allow it (above) and double-check below
    diff = abs(dist.particle_number_concentration/dist_in.particle_number_concentration - 1)
    assert(diff < 1e-8), f'diff: {diff}, dd: {dd}'

    if plot:
        a = plt.subplot()
        dist_tmp.convert2numberconcentration().plot(ax = a)
        dist.convert2numberconcentration().plot(ax = a)

    #### 3. counting efficiency
    #############
    dist_tmp = dist.copy()
    ce = ce_log(cc).interp(diameter = dist.bincenters)
    dist.data = dist.data * ce.to_pandas()
    dist.data[dist.data.isna()] = 0 #for some reason it created a nan in first place
    if plot:
        f,aa = plt.subplots(2, sharex=True, gridspec_kw={'hspace':0})

        a = aa[0]
        dist_tmp.convert2dNdlogDp().plot(ax = a)

        a = aa[1]
        ce.plot(ax = a)

        a = aa[0]
        dist.convert2dNdlogDp().plot(ax = a)

        ######
        a = aa[0]
        a.set_yscale('log')

    #### 4. convertion to volume equivalent diameter
    #############    
    
    #### convert shape parameter to dynamic shape factor
    shape = atmshape.AspectRatio2DynamicShapeFactor()
    shape_dyn = float(shape.dataset.dynamic_shape_factor.interp(shape_parameter = shape_c))
    shape_dyn
    
    dist_tmp = dist.copy()
    if instrument == 'aps':
        # get correction and apply to dist
        # correction_fct = d_aps2d_vol(roh_p=roh, xsi = shape_dyn)
        correction_fct = atmshape.EquivalentDiameters(diameter=1, 
                                                      diameter_equivalent='aerodynamic', 
                                                      density_measured=roh, 
                                                      density_calibrated=2, 
                                                      dynamic_shape_factor=shape_dyn).volume_diameter
        
    elif instrument == 'smps':
        correction_fct = atmshape.EquivalentDiameters(diameter=1, 
                                                      diameter_equivalent='mobility', 
                                                      dynamic_shape_factor=shape_dyn).volume_diameter
    
        # correction_fct
    dist = dist.grow_sizedistribution(correction_fct, raise_particle_loss_error=False)
    diff = abs(dist.particle_number_concentration/dist_tmp.particle_number_concentration - 1)
    assert(diff < 1e-8), f'diff: {diff}, roh: {roh}, xi: {shape_dyn}, corrfct: {correction_fct}'
    
    if plot:
        a = plt.subplot()
        dist_tmp.convert2numberconcentration().plot(ax = a)
        dist.convert2numberconcentration().plot(ax = a)
    
    return dist

def montecarlo(instrument, dist, lut, ce_log, row):
    """
    Main function. It takes everything it needs to return the scattering coefficient

    Parameters
    ----------
    instrument : TYPE
        DESCRIPTION.
    dist : TYPE
        DESCRIPTION.
    lut : TYPE
        DESCRIPTION.
    ce_log : TYPE
        DESCRIPTION.
    row : TYPE
        DESCRIPTION.

    Returns
    -------
    scattcoeff : TYPE
        DESCRIPTION.

    """
    if instrument == 'aps':
        roh=row.roh
    elif instrument == 'smps':
        roh = None
    dist = size_distribution_correction(dist, 
                                        instrument = instrument,
                                        dd=row.d,
                                        cc=row.ce,
                                        ce_log = ce_log,
                                        shape_c=row.pshapes,
                                        roh=roh,
                                        plot=False)

    scattcoeff = lut2scatt_coeff(lut, dist, row.pshapes, row.n_real, row.n_imag, smoothen=True)
    return scattcoeff

#### all APS related

# def d_aps2d_vol(roh_p = 2, roh_0 = 2, xsi = 1):
#     """
#     Provides a correction factor to convert the aerodynamic diameter to the volume equivalent diameter
    
#     Parameters
#     ----------
#     roh_p: actual particles density
#     roh_0: calibration density (either the acuatl calibration material, e.g. PSL, but more often then not an additional adjustem to a more realistic density, e.g. 2 in case of SGP
#     xsi: dynamic shape factor
    
#     Returns
#     -------
#     correction factor for volume equivalent diameter
#     """
#     c = np.sqrt((roh_0 * xsi) / roh_p)
#     return c



def get_counting_efficiency_fct_aps():
    """Returns the function that describes the size dependent counting efficiency for a given sigma
    see here for plots and development: http://localhost:8000/lab/workspaces/auto-8/tree/projects/16_closure_of_arm_data/uncertainties/montecarlo/counting_efficiency.ipynb
    """
    
    # fname = '/mnt/telg/projects/fundgrube/g3data_gesnatched/Pfeifer_2016/data_from_pfeifer_2016.nc'
    fname = '/export/htelg/fundgrube/g3data_gesnatched/Pfeifer_2016/data_from_pfeifer_2016.nc'
    aps_pfeifer = xr.open_dataset(fname)

    meanlog_aps = np.log10(aps_pfeifer.size_dist_ambient_all).mean(dim = 'instrument')
    meanlog_aps = xr.concat([meanlog_aps, xr.DataArray([meanlog_aps[-1].values], {'diameter': [20.]})], dim = 'diameter')
    meanlog_aps = meanlog_aps.assign_coords(diameter = meanlog_aps.diameter * 1e3) # convert diameter to nanometer
    stdlog_aps = np.log10(aps_pfeifer.size_dist_ambient_all).std(dim = 'instrument')
    stdlog_aps[-1] = stdlog_aps[-2] #the last value was zero, probably because there was only one value?
    stdlog_aps = xr.concat([stdlog_aps, xr.DataArray([stdlog_aps[-1].values], {'diameter': [20.]})], dim = 'diameter') #expend to 20um
    stdlog_aps = stdlog_aps.assign_coords(diameter = stdlog_aps.diameter * 1e3) # convert diameter to nanometer
    ce_log_aps = lambda i:(10**((meanlog_aps + i * stdlog_aps)) / 10**(meanlog_aps))
    return ce_log_aps



def deprecated_size_distribution_correction_aps(dist,
                                     daps = 0.9,  
                                     cc = 2., 
                                     ce_log_aps = None,
                                     shape_c = 2.5 , 
                                     roh = 2.7,
                                     plot = True,
                                    ):
    """
    This corrects the size distribution (diameters, and particle numbers) according to assumed parameters.

    Parameters
    ----------
    dist : TYPE
        DESCRIPTION.
    daps : TYPE, optional
        diameter uncertainty of APS, this is the intrinsic uncertainty not the chi related. The default is 0.9.
    cc : TYPE, optional
        this is diameter dependen counting effeicnecy uncertatinty in coarse mode. This is the sigma of a normal distribution                                     shape_c : TYPE, optional
        The default is 2.5.
    shape_c: ...
        shape parameter. This is the parameter that goes from -x to +y in our case from -5 to 5                                     
    roh : TYPE, optional
        DESCRIPTION. The default is 2.7.
    plot : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    dist : TYPE
        DESCRIPTION.

    """

    # 3. dist preconditioning
    #######
    dist_in = dist
    dist = dist_in.copy()
    dist = dist.convert2numberconcentration()



    # 4. diameter precision & accuracy, I actually do not know which of the two it is?
    ##########
    dist_tmp = dist.copy()
    dist = dist.grow_sizedistribution(daps, raise_particle_loss_error=False)
    # sometime very small particle losses can accure, so I allow it (above) and double-check below
    diff = abs(dist.particle_number_concentration/dist_in.particle_number_concentration - 1)
    assert(diff < 1e-10), f'{diff}'

    if plot:
        a = plt.subplot()
        dist_tmp.convert2numberconcentration().plot(ax = a)
        dist.convert2numberconcentration().plot(ax = a)

    # 5. counting efficiency
    #############
    dist_tmp = dist.copy()
    ce = ce_log_aps(cc).interp(diameter = dist.bincenters)
    dist.data = dist.data * ce.to_pandas()
    dist.data[dist.data.isna()] = 0 #for some reason it created a nan in first place
    if plot:
        f,aa = plt.subplots(2, sharex=True, gridspec_kw={'hspace':0})

        a = aa[0]
        dist_tmp.convert2dNdlogDp().plot(ax = a)

        a = aa[1]
        ce.plot(ax = a)

        a = aa[0]
        dist.convert2dNdlogDp().plot(ax = a)

        ######
        a = aa[0]
        a.set_yscale('log')

    # 6. convertion to volume equivalent diameter
    #############
    dist_tmp = dist.copy()

    # convert shape parameter to dynamic shape factor
    shape = atmshape.AspectRatio2DynamicShapeFactor()
    shape_dyn = float(shape.dataset.dynamic_shape_factor.interp(shape_parameter = shape_c))

    # get correction and apply to dist
    # correction_fct = d_aps2d_vol(roh_p=roh, xsi = shape_dyn)
    correction_fct = atmshape.EquivalentDiameters(diameter=1, 
                                                  diameter_equivalent='aerodynamic', 
                                                  density_measured=roh, 
                                                  density_calibrated=2, 
                                                  dynamic_shape_factor=shape_dyn).volume_diameter

    # correction_fct
    dist = dist.grow_sizedistribution(correction_fct, raise_particle_loss_error=False)
    particleloss = dist.particle_number_concentration/dist_tmp.particle_number_concentration
    assert(round(particleloss,5) == 1), f'particle_loss exceeds arbitrary threshold: {particleloss}'
    
    if plot:
        a = plt.subplot()
        dist_tmp.convert2numberconcentration().plot(ax = a)
        dist.convert2numberconcentration().plot(ax = a)
    
    return dist




def deprecated_montecarlo_aps(dist_aps, lut_coarse, ce_log_aps, row):
    dist = size_distribution_correction(dist_aps, 
                                        instrument = 'aps',
                                            dd=row.d,
                                            cc=row.ce,
                                            ce_log = ce_log_aps,
                                            shape_c=row.pshapes,
                                            roh=row.roh,
                                            plot=False)

    scattcoeff = lut2scatt_coeff(lut_coarse, dist, row.pshapes, row.n_real, row.n_imag, smoothen=True)
    return scattcoeff



#### all SMPS realted

def get_counting_efficiency_fct_spmps():
    """Returns the function that describes the size dependent counting efficiency for a given sigma
    see here for plots and development: http://localhost:8000/lab/workspaces/auto-8/tree/projects/16_closure_of_arm_data/uncertainties/montecarlo/counting_efficiency.ipynb
    """
    # smps
    # fname = '/mnt/telg/projects/fundgrube/g3data_gesnatched/Wiedensohler_2012/project.nc'
    fname = '/export/htelg/fundgrube/g3data_gesnatched/Wiedensohler_2012/project.nc'
    smps_wieden = xr.open_dataset(fname, autoclose=True)
    
    expand2 = 750.
    meanlog_smps = np.log10(smps_wieden.data_interpolated).mean(dim = 'instruments')
    meanlog_smps = xr.concat([meanlog_smps, xr.DataArray([meanlog_smps[-1].values], {'diameter': [expand2]})], dim = 'diameter')
    
    stdlog_smps = np.log10(smps_wieden.data_interpolated).std(dim = 'instruments')
    stdlog_smps = xr.concat([stdlog_smps, xr.DataArray([stdlog_smps[-1].values], {'diameter': [expand2]})], dim = 'diameter') #expend to 750nm
    
    ce_log_smps = lambda i:(10**((meanlog_smps + i * stdlog_smps)) / 10**(meanlog_smps))
    return ce_log_smps




