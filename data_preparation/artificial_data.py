import batman
import ellc
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from pytransit import OblateStarModel, QuadraticModel
from data_preparation.data_processing_utils import min_max_norm_vectorized, resize, standardize


R_SUN2JUPYTER = 1.0 / 0.10045
M_EARTH_KG = 5.9723e24
M_SUN_KG = 1.9884e30
M_JUPYTER = 1.89813e27
M_JUPYTER2Sun = M_JUPYTER / M_SUN_KG
M_EARTH2SUN = M_EARTH_KG / M_SUN_KG
AU2kKM = 149597870.7
R_SUN = 696340.0
R_EARTH2SUN = 6371.0 / R_SUN


def prob_decrease(max_spots, mass=0.5, decay=0.5):
    assert 0.0 < decay < 1.0, "Invalid decay value! Must be between 0 and 1."
    probs = []
    current = mass
    for i in range(max_spots):
        current -= current * decay
        probs.append(current)
    remaining_mass = mass-sum(probs)
    return [prob + (remaining_mass/len(probs)) for prob in probs]


def sample_spot_parameters_realistic(b=0., max_spots=4, max_size=20., spotless_prob=0.0, latitude_offset_prob=0.5,
                                     latitude_offset_std=0.1):
    p = prob_decrease(max_spots, mass=1 - spotless_prob, decay=0.5)
    p.insert(0, spotless_prob)
    num_spots = np.random.choice(range(max_spots + 1), p=p)
    if num_spots == 0:
        return None
    spot_params = np.empty((4, num_spots))
    longitude_range = np.linspace(-60., 60, 360)
    for s, spot in enumerate(range(num_spots)):
        if np.random.choice([True, False], p=[latitude_offset_prob, 1.-latitude_offset_prob]):
            offset = np.random.normal(0, latitude_offset_std)
        else:
            offset = 0.
        #latitude = (b+offset)*90.
        latitude = -b * 60. + np.random.uniform(-5., 5.)
        longitude = np.random.choice(longitude_range)
        size = np.random.uniform(2., max_size-(5.*s))
        used_longitude = np.logical_and(longitude-size >= longitude_range, longitude_range <= longitude+size)
        longitude_range = longitude_range[~used_longitude]
        brightness = np.random.uniform(0.7, 1.3)
        spot_params[0, s] = longitude
        spot_params[1, s] = latitude
        spot_params[2, s] = size
        spot_params[3, s] = brightness

    return spot_params


def extract_parameters(path="All_Exoplanets_Params.csv", transit__method_only=True,
                       params_essential=('pl_orbper', 'pl_rade', 'pl_orbsmax', 'pl_orbincl', 'st_rad'),
                       params_optional=('pl_trandur', 'pl_orbeccen', 'pl_orblper', 'st_teff', 'st_logg', 'st_met')):
    exos = pd.read_csv(path, comment='#', sep=',')
    if transit__method_only:
        exos = exos[exos['discoverymethod'] == 'Transit']
    if params_essential is None:
        params_essential = ['pl_orbper', 'pl_trandur', 'pl_rade', 'pl_orbsmax', 'pl_orbeccen',
                           'pl_orbincl', 'st_rad']
    if params_optional is None:
        params_optional = []
    param_names = list(params_essential) + list(params_optional)
    exos_selected = exos.loc[:, param_names]
    # convert unit of 'a' from AU to ratio of stellar radii
    exos_selected.loc[:, 'pl_orbsmax'] = (AU2kKM * exos_selected['pl_orbsmax']) / (R_SUN * exos_selected['st_rad'])

    valid_exos = exos_selected.dropna(subset=params_essential)

    return valid_exos.where(pd.notnull(valid_exos), None)


def bin_parameters_by_impact(param_df, uniform_impact_bins=10):
    bin_edges = np.linspace(0, 1, uniform_impact_bins)
    bins = {bin_edge: [] for bin_edge in bin_edges}
    for r_idx, row in param_df.iterrows():
        a = row.get('pl_orbsmax')
        i = row.get('pl_orbincl')
        if a is None:
            a = np.random.uniform(2.3, 30.)
        if i is None:
            b = np.random.uniform(0., 1.)
            i = np.arccos(b / a) * 180 / np.pi
        # calculate impact parameter
        b = a * np.cos(i * np.pi / 180)
        # determine closest bin edge
        bin_idx = np.abs(bin_edges - b).argmin()
        bins[bin_edges[bin_idx]].append(row)
    return bins


def get_valid_range(value, constraints=('t_eff',), quantity='logg', limb_model='quad'):
    assert all([constraint in ('t_eff', 'logg', 'met') for constraint in constraints]),\
        f"Unknown value in constraints for argument constraint! Must only contain ('t_eff', 'logg' 'met')."
    assert quantity in ('t_eff', 'logg', 'met'),\
        f"Unknown value {quantity} for argument quantity! Must be one of ('t_eff', 'logg' 'met')."
    assert all([constraint != quantity for constraint in constraints]), "Argument constraints must not contain quantity!"
    assert len(value) == len(constraints), f"Arguments value and constraints have different lengths {len(value)} and " \
                                           f"{len(constraints)}. You need to provide a value for each constraint!"

    if limb_model == 'claret':
        table = pd.read_csv("TESS_Nonlinear_Limb_Darkening_Atlas.csv", comment='#', sep=',')
    elif limb_model == 'quad':
        table = pd.read_csv("TESS_Quadratic_Limb_Darkening_Atlas.csv", comment='#', sep=',')
    else:
        raise RuntimeError("Unknown limb-darkening model use one of ['quad', 'claret']")

    translate_arguments = lambda x: 'Teff [K]' if x == 't_eff' else 'logg [cm/s2]' if x == 'logg' else 'Z [Sun]'
    constraints_translated = [translate_arguments(constraint) for constraint in constraints]
    quantity_translated = translate_arguments(quantity)

    constraint_results = table
    for c, constraint in enumerate(constraints_translated):
        match_idx = (constraint_results[constraint] - value[c]).abs().argmin()
        matched_value = constraint_results[constraint].iloc[match_idx]
        constraint_results = constraint_results[constraint_results[constraint] == matched_value]

    joined_constraints = set(constraint_results[quantity_translated])

    if len(joined_constraints) > 0:
        return list(joined_constraints)
    else:
        return [None]


def match_stellar_params_with_table(table, T_eff, logg, met):
    T_eff_match = (table['Teff [K]'] - T_eff).abs().argmin()
    T_eff = table['Teff [K]'].iloc[T_eff_match]
    logg_match = (table['logg [cm/s2]'] - logg).abs().argmin()
    logg = table['logg [cm/s2]'].iloc[logg_match]
    met_match = (table['Z [Sun]'] - met).abs().argmin()
    met = table['Z [Sun]'].iloc[met_match]
    candidates = table.loc[(table['Teff [K]'] == T_eff) & (table['logg [cm/s2]'] == logg) & (table['Z [Sun]'] == met)]
    if len(candidates) > 1:
        candidates = candidates.loc[candidates['xi [km/s]'] == 2.0]
    return candidates


def lookup_limb_darkening(T_eff, logg, met, limb_model='claret', model='Atlas'):
    if model == 'Atlas':
        if limb_model == 'claret':
            table = pd.read_csv("TESS_Nonlinear_Limb_Darkening_Atlas.csv", comment='#', sep=',')
            candidates = match_stellar_params_with_table(table, T_eff, logg, met)
            if candidates.empty:
                return None
            else:
                return candidates['a1LSM'].item(), candidates['a2LSM'].item(), candidates['a3LSM'].item(), candidates['a4LSM'].item()
        elif limb_model == 'quad':
            table = pd.read_csv("TESS_Quadratic_Limb_Darkening_Atlas.csv", comment='#', sep=',')
            candidates = match_stellar_params_with_table(table, T_eff, logg, met)
            if candidates.empty:
                return None
            else:
                return candidates['aLSM'].item(), candidates['bLSM'].item()
        else:
            raise RuntimeError("Unknown limb-darkening model use one of ['quad', 'claret']")
    else:
        raise RuntimeError("Currently Atlas is the only Model implemented! Please use model='Atlas'")


def lookup_gravity_darkening(T_eff, logg, met):
    table = pd.read_csv("TESS_Gravity_Darkening.csv", comment='#', sep=',')
    candidates = match_stellar_params_with_table(table, T_eff, logg, met)
    if candidates.empty:
        return None
    else:
        return candidates['y'].item()


def batman_model(R_ratio, a, i, ecc, t0=0.0, period=2.5, duration=2.5, w=0.0, size=256,
                 limb_dark="quadratic", limb_coeff=(0.1, 0.3), mult=3.):
    params = batman.TransitParams()
    params.t0 = t0  # time of inferior conjunction
    params.per = period  # orbital period
    params.rp = R_ratio  # planet radius (in units of stellar radii)
    params.a = a  # semi-major axis (in units of stellar radii)
    params.inc = i  # orbital inclination (in degrees)
    params.ecc = ecc  # eccentricity
    params.w = w  # longitude of periastron (in degrees)
    params.u = limb_coeff  # limb darkening coefficients [u1, u2]
    params.limb_dark = limb_dark

    time = (duration / 24.) * (mult / 2.)
    # print(R_ratio, a)
    t = np.linspace(-time, time, size, endpoint=True)

    m = batman.TransitModel(params, t)  # initializes model
    return m.light_curve(params)


def ellc_transit(r1, r2, m_star=1.0, m_planet=None, period=1.0, duration=None, incl=90.0, ecc=None, w=None, a_sun=None,
                 star_shape='sphere', rot_per=None, T_eff=None, logg=None, met=None, limb_model="quad",
                 lambda_1=None, spots_1 = None, include_gd=True,
                 sbratio=0.0, t_zero=0, mult=3, size=256, accuracy='default', plot=False):
    if period == 1:
        window = 0.5
        a_sun = None
    else:
        window = (duration / 24.) * (mult / 2.)
    t_obs = np.linspace(-window, window, size, endpoint=True)
    # eccentricity
    if ecc is None or w is None:
        f_c = None
        f_s = None
    else:
        f_c = np.sqrt(ecc) * np.cos(w * np.pi / 180)
        f_s = np.sqrt(ecc) * np.sin(w * np.pi / 180)
    # limb and gravity darkening
    if T_eff is None or logg is None or met is None:
        ldc_1 = None
        gdc_1 = None
    else:
        ldc_1 = lookup_limb_darkening(T_eff, logg, met, limb_model=limb_model, model='Atlas')
        if include_gd:
            gdc_1 = lookup_gravity_darkening(T_eff, logg, met)
        else:
            gdc_1 = None
    grav = True if gdc_1 is not None else False
    # mass relation
    if m_planet is None:
        q = (M_JUPYTER * (r2 * R_SUN2JUPYTER) ** 3) / m_star
    else:
        q = m_planet / m_star
    # stellar shape
    if rot_per is None:
        async_rot = 1
    else:
        async_rot = period / rot_per

    f1 = ellc.lc(t_obs, r1, r2, sbratio, incl,
                         t_zero=t_zero, period=period,
                         a=a_sun,
                         q=q,
                         f_c=f_c, f_s=f_s,
                         ldc_1=ldc_1, ldc_2=None,
                         gdc_1=gdc_1, gdc_2=None,
                         didt=None,
                         domdt=None,
                         rotfac_1=async_rot, rotfac_2=1,
                         hf_1=1.5, hf_2=1.5,
                         bfac_1=None, bfac_2=None,
                         heat_1=None, heat_2=None,
                         lambda_1=lambda_1, lambda_2=None,
                         vsini_1=None, vsini_2=None,
                         t_exp=None, n_int=None,
                         grid_1=accuracy, grid_2=accuracy,
                         ld_1=limb_model, ld_2=None,
                         shape_1=star_shape, shape_2='sphere',
                         spots_1=spots_1, spots_2=None,
                         exact_grav=grav, verbose=0)
    if plot:
        plt.plot(t_obs, f1)
        plt.show()
    return f1


def allign_model_with_residuals(model, res_ingress, res_egress, res_size):
    model_ingress = min(np.argwhere(model < 1.))[0]
    model_egress = max(np.argwhere(model < 1.))[0]
    scale_factor = (res_egress - res_ingress) / (model_egress - model_ingress)
    model_size = len(model)
    new_model_size = int(model_size*scale_factor)
    resized = resize(model, new_model_size)
    resized_ingress = min(np.argwhere(resized < 1))[0]

    start_diff = res_ingress-resized_ingress
    if start_diff > 0:
        resized = np.insert(resized, 0, np.ones(start_diff))
    elif start_diff < 0:
        resized = resized[abs(start_diff):]

    size_diff = res_size-len(resized)
    if size_diff > 0:
        resized = np.append(resized, np.ones(size_diff))
    elif size_diff < 0:
        resized = resized[:res_size]

    return resized, scale_factor, start_diff


def fill_missing_params(row, max_repetitions=10):
    period = row.get('pl_orbper') or np.random.uniform(0.2, 18.)
    r_ratio = np.inf
    radii_iter = 0
    while r_ratio >= 1.:
        r_st = row.get('st_rad') or np.random.uniform(0.3, 4.)
        if radii_iter > 3:
            r_pl = 0.08*r_st
        else:
            r_pl = row.get('pl_rade') or np.random.uniform(0.3, 7.)
            r_pl *= R_EARTH2SUN
        r_ratio = r_pl / r_st
        radii_iter += 1

    repeat = 0
    while repeat < max_repetitions:

        rot_per = row.get('st_rotp') or np.random.uniform(10., 40.)
        st_mass = row.get('st_mass') or np.random.uniform(0.5, 2.5)
        pl_mass = row.get('pl_massj') or np.random.uniform(0.005, 2.5)
        pl_mass *= M_JUPYTER2Sun

        # impact parameter
        a = row.get('pl_orbsmax')
        i = row.get('pl_orbincl')
        if a is None:
            a = np.random.uniform(2.3, 30.)
            b_valid = np.random.uniform(0., 1.)
            if i is None:
                i = np.arccos(b_valid/a) * 180/np.pi
        else:
            if i is None:
                b_valid = np.random.uniform(0., 1.)
                i = np.arccos(b_valid/a) * 180/np.pi
            else:
                b_valid = a * np.cos(i * np.pi/180)

        # eccentricity
        ecc = row.get('pl_orbeccen') or np.random.uniform(0., 1.) if repeat < max_repetitions - 1 else 0.
        w = row.get('pl_orblper') or np.random.uniform(0., 90.)

        # limb darkening
        t_eff = row.get('st_teff') or np.random.uniform(3500., 12500.)
        logg = row.get('st_logg') or np.random.choice(
            get_valid_range((t_eff,), constraints=('t_eff',), quantity='logg'))
        met = row.get('st_met') or np.random.choice(
            get_valid_range((t_eff, logg), constraints=('t_eff', 'logg'), quantity='met'))
        limb_coeff = lookup_limb_darkening(t_eff, logg, met, limb_model='quad', model='Atlas') or \
                     np.random.uniform(-1., 1., size=2)

        duration = row.get('pl_trandur') or period * 24. / np.pi * np.arcsin(
            np.sqrt((r_st + r_pl) ** 2 - (b_valid * r_st) ** 2) / a)

        flux_original = ellc_transit(1./a, (1./a)*r_ratio, m_star=st_mass, m_planet=pl_mass, period=1.0, duration=duration,
                                     incl=i, ecc=ecc, w=w, a_sun=a*r_st, star_shape='sphere', rot_per=rot_per,
                                     T_eff=t_eff, logg=logg, met=met, limb_model="quad", lambda_1=None, spots_1=None)

        if max(flux_original) > min(flux_original):
            return {'a':a, 'i':i, 'b':b_valid, 'e':ecc, 'w':w, 'period':period, 'duration':duration,
                    'r_star':r_st, 'r_planet':r_pl, 'm_star':st_mass, 'm_planet':pl_mass, 'rot_per':rot_per,
                    't_eff':t_eff, 'logg':logg, 'met':met, 'ldc':limb_coeff}
        else:
            repeat += 1
    return {'a':a, 'i':i, 'b':b_valid, 'e':ecc, 'w':w, 'period':period, 'duration':duration,
            'r_star':r_st, 'r_planet':r_pl, 'm_star':st_mass, 'm_planet':pl_mass, 'rot_per':rot_per,
            't_eff':t_eff, 'logg':logg, 'met':met, 'ldc':limb_coeff}


def sample_horizontal_scale(mean_gaussian=3., std_gaussian=1., uniform_range=(1., 8.), uniform_prob=0.0):
    gaussian = np.random.choice([True, False], p=(1-uniform_prob, uniform_prob))
    if gaussian:
        return max(1., np.random.normal(mean_gaussian, std_gaussian))
    else:
        return np.random.uniform(*uniform_range)


def py_transit_model(radii_ratio, a, i,  period, duration, ldc, e=0., w=0., t0=0., mult=3., size=256):
    tmc = QuadraticModel(interpolate=False)
    window = (duration / 24.) * (mult / 2.)
    times = np.linspace(-window, window, size, endpoint=True)
    tmc.set_data(times)
    k = np.array([radii_ratio])
    i *= np.pi / 180

    return tmc.evaluate_ps(k, ldc, t0, period, a, i, e, w)


def py_transit_gd(radii_ratio, a, i,  period, duration, ldc, gdc, e=0., w=0.,
                  r_star=1., density=1., phi=90., az=0., rot_per=12., t_pole=6500., t0=0., mult=3., size=256):
    degree2rad = np.pi / 180
    tmo = OblateStarModel(sres=100, pres=8, rstar=r_star)
    window = (duration / 24.) * (mult / 2.)
    times = np.linspace(-window, window, size, endpoint=True)
    tmo.set_data(times)

    k = np.array([radii_ratio])
    i *= degree2rad
    phi *= degree2rad
    az *= degree2rad
    return tmo.evaluate_ps(k, density, rot_per, t_pole, phi, gdc, ldc, t0, period, a, i, l=az, e=e, w=w)


def sample_gd_params():
    gd_params = {}
    gd_params['rot_per'] = np.random.uniform(0.5, 2.5)
    gd_params['st_density'] = np.random.uniform(0.1, 1.9)
    gd_params['st_obliquity'] = np.random.uniform(0., 90.)
    gd_params['spin_orbit_angle'] = np.random.uniform(0., 90.)
    gd_params['t_pole'] = np.random.uniform(5700., 12000.)
    return gd_params


def create_transit_model(params, mode, feature=False, t0=0., size=256, res_data=None):
    if mode == 'normal' or (mode == 'dre' and not feature):
        flux = batman_model(params['r_planet'] / params['r_star'], params['a'], params['i'],
                            ecc=params['e'], w=params['w'], period=params['period'], duration=params['duration'],
                            limb_dark="quadratic", limb_coeff=params['ldc'], t0=t0, size=size)
        if feature:
            return flux, {}
        else:
            return flux
    elif mode == 'spots':
        if feature:
            spots = sample_spot_parameters_realistic(params['b'])
            apply_gd = False  # True
        else:
            spots = None
            apply_gd = False

        r1 = 1./params['a']  # radius of star in units of semi-major-axis (a is in units of stellar radii)
        flux = ellc_transit(r1, r1*params['r_planet']/params['r_star'], m_star=params['m_star'],
                            m_planet=params['m_planet'], period=params['period'],
                            duration=params['duration'], incl=params['i'], ecc=params['e'], w=params['w'],
                            a_sun=params['a'] * params['r_star'], star_shape='sphere',
                            rot_per=max(params['period'],params['rot_per']), T_eff=params['t_eff'], logg=params['logg'],
                            met=params['met'], limb_model="quad", lambda_1=None, include_gd=apply_gd, spots_1=spots,
                            t_zero=t0)
        # fix numerical issue pre transit flux slightly smaller 1.0
        flux += 1e-8
        gdc = lookup_gravity_darkening(params['t_eff'], params['logg'], params['met'])
        if feature:
            if spots is None:
                spot_params = {'num_spots':0}
            else:
                spot_params = {'num_spots':spots.shape[-1], 'longitudes':spots[0,:], 'latitudes':spots[1,:],
                          'spot_sizes':spots[2,:], 'spot_brightnesses':spots[3,:], 'gdc':gdc}
            return flux, spot_params
        else:
            return flux
    elif mode == 'gd':
        flux_no_gd = py_transit_model(params['r_planet'] / params['r_star'], params['a'], params['i'], params['period'],
                                      params['duration'], params['ldc'], params['e'], params['w'], t0=t0, size=size)
        if feature:
            mean_effect = 0.
            max_effect = 0.
            gdc = lookup_gravity_darkening(params['t_eff'], params['logg'], params['met'])
            gd_iter = 0
            while not 5e-5 < mean_effect < 5e-4 and not 1e-4 < max_effect < 1e-3:
                if gd_iter > 5:
                    return None, gd_params
                gd_params = sample_gd_params()
                gd_params['gdc'] = gdc or np.random.uniform(0., 0.6)
                flux = py_transit_gd(params['r_planet'] / params['r_star'], params['a'], params['i'], params['period'],
                                     params['duration'], params['ldc'], gd_params['gdc'], params['e'], params['w'],
                                     params['r_star'], gd_params['st_density'], gd_params['st_obliquity'],
                                     gd_params['spin_orbit_angle'], gd_params['rot_per'], gd_params['t_pole'],
                                     t0=t0, size=size)
                in_transit_idxs = np.nonzero(flux < 1.)[0]
                if len(in_transit_idxs) == 0:
                    gd_iter += 1
                    continue
                oot_idxs = np.nonzero(flux == 1.)[0]
                if np.logical_and(oot_idxs > min(in_transit_idxs), oot_idxs < max(in_transit_idxs)).any():
                    gd_iter += 1
                    continue
                abs_diff = abs(flux - flux_no_gd)
                diff_idxs = abs_diff > 0
                mean_effect = sum(abs_diff[diff_idxs]) / (sum(diff_idxs) + 1)
                c_mean_effect = sum(abs_diff) / len(abs_diff)
                max_effect = max(abs_diff)
                gd_iter += 1
            gd_params['gd_effect'] = mean_effect
            return flux, gd_params
        else:
            return flux_no_gd
    elif mode == 'dre' and feature:
        flux = batman_model(params['r_planet'] / params['r_star'], params['a'], params['i'],
                            ecc=params['e'], w=params['w'], period=params['period'], duration=params['duration'],
                            limb_dark="quadratic", limb_coeff=params['ldc'], t0=t0, size=size)

        residual = res_data[0][np.random.choice(len(res_data))]
        #transit_model = min_max_norm(flux)
        transit_model = flux
        alligned_model, dre_scale, dre_shift = allign_model_with_residuals(transit_model, residual['ingress'], residual['egress'],
                                                     len(residual['residual']))

        # scale and shift
        res_to_depth = max(abs(residual['residual'])) / (1.-min(alligned_model))
        tran_dur = (residual['egress'] - residual['ingress'])
        if res_data[1] is None:
            in_transit_ratio = tran_dur * 1. / len(residual['residual'])
            sampled_mult = max(1.1, sample_horizontal_scale())
            dre_mult = min(sampled_mult, 1./in_transit_ratio)
        else:
            dre_mult = res_data[1]
        select_size = int(dre_mult * tran_dur)
        epoch_idx = residual['ingress'] + (tran_dur // 2)
        start_dre = epoch_idx-(select_size//2)
        end_dre = start_dre + select_size
        dre_shift = 0
        if res_data[2]:
            left_in_data_constraint = residual['ingress'] - start_dre
            right_in_data_constraint = end_dre - residual['egress']
            left_out_data_constraint = start_dre
            right_out_data_constraint = len(residual['residual']) - 1 - residual['egress']
            left_constraint = min(left_in_data_constraint, left_out_data_constraint)
            right_constraint = min(right_in_data_constraint, right_out_data_constraint)
            #shift_range = max(1, int((dre_mult-1.)/2. * tran_dur))
            #dre_shift = np.random.randint(-shift_range, shift_range)
            dre_shift = np.random.randint(-left_constraint, right_constraint)
            start_dre += dre_shift
            end_dre += dre_shift

        feature_model = alligned_model[start_dre:end_dre]
        res_scale_factor = np.random.uniform(0.001, 0.5) / res_to_depth
        dre = feature_model + residual['residual'][start_dre:end_dre] * res_scale_factor

        # resize models
        feature_model = resize(feature_model, size)
        dre = resize(dre, size)

        return resize(feature_model, size), resize(dre, size), dre_mult, np.argmin(feature_model)-feature_model.shape[-1]//2

    else:
        raise RuntimeError(f"Unknown mode {mode} for transit creation use one of ('normal', 'spots', 'gd', 'dre').")


def sample_artificial_transits(num_transits, mode='normal', size=256, snr_min=1., snr_max=30., max_repetitions=10,
                               multiplier=3., horizontal_scale=False, horizontal_shift=False, uniform_params=False,
                               num_bins=20):
    assert mode in ('normal', 'spots', 'gd', 'dre')
    function_start = time()
    if mode == 'dre':
        res_data = (pickle.load(open("data/Kepler1520/residuals_dv_model_plain.pkl", "rb")),
                    multiplier, horizontal_shift)
    else:
        res_data = None
    known_exos_params = extract_parameters()
    bins = bin_parameters_by_impact(known_exos_params, uniform_impact_bins=num_bins)
    results = {'artificial_data':np.empty((num_transits, size)), 'transit_model':np.empty((num_transits, size)),
               'feature_model':np.empty((num_transits, size))}
    mata_columns = ['a', 'i', 'b', 'e', 'w', 'period', 'duration', 'multiplier',
                    'r_star', 'r_planet', 'm_star', 'm_planet', 'rot_per', 'st_density', 't_pole',
                    't_eff', 'logg', 'met', 'limb_model', 'ldc', 'gdc', 'st_obliquity', 'spin_orbit_angle', 'gd_effect',
                    'num_spots', 'longitudes', 'latitudes', 'spot_sizes', 'spot_brightnesses', 'mode',
                    'uniform_params', 'transit_depth', 'shift', 'ingress_idx', 'egress_idx', 'snr']
    meta = pd.DataFrame(columns=mata_columns, index=range(num_transits))
    transits_generated = 0
    loop_start = time()
    while transits_generated < num_transits:
        params = dict.fromkeys(mata_columns, np.nan)
        uniform_param_iter = 0
        # get theoretical model
        if uniform_params:
            raise NotImplementedError("This functionality is deprecated and not implemented anymore!")
        else:
            bin_idx = np.random.choice(num_bins)
            selected_bin = bins[list(bins.keys())[bin_idx]]
            row = selected_bin[np.random.choice(len(selected_bin))]
            params.update(fill_missing_params(row))

        params['mode'] = mode
        params['uniform_params'] = uniform_params
        params['limb_model'] = "quad"
        params['num_spots'] = 0

        flux_original = create_transit_model(params, mode=mode, feature=False, size=size)

        # skip invalid configurations
        if max(flux_original) - min(flux_original) <= 1e-8:
            continue

        in_transit_ratio = sum(flux_original < 1.) / len(flux_original)

        # if zoomed in too close into transit zoom out until ingress/egress edges become visible
        while in_transit_ratio == 1.:
            params['duration'] *= 5
            flux_original = create_transit_model(params, mode=mode, feature=False, size=size)
            in_transit_ratio = sum(flux_original < 1.) / len(flux_original)
            if params['duration'] >= params['period']/2.:
                break
        # if correction fails skip transit
        if in_transit_ratio >= 1.:
            print(f"Inaccurate parameters detected! Transit model will be skipped.")
            continue

        if mode == 'dre':
            feature_params = {}
            flux_feature, dre, dre_mult, dre_shift = create_transit_model(params, mode=mode, feature=True,
                                                                          size=size, res_data=res_data)
            results['artificial_data'][transits_generated, :] = dre

        # fix horizontal scale to given number of transit durations (multiplier)
        duration_correction = 1.
        if horizontal_scale:
            # always scale original (label) to 3 transit durations
            duration_correction_original = in_transit_ratio / (1. / 3.)
            params['duration'] *= duration_correction_original
            flux_original = create_transit_model(params, mode=mode, feature=False, size=size)
            params['duration'] /= duration_correction_original
            excess_in_transit = round(sum(flux_original < 1.) - len(flux_original)*(1./3.))
            if excess_in_transit > 1:
                fill_num = round(excess_in_transit/2.)
                flux_original = np.insert(flux_original, 0, np.ones(fill_num))
                flux_original = np.append(flux_original, np.ones(fill_num))
                flux_original = resize(flux_original, size=size)
            elif excess_in_transit < -1:
                discard_num = abs(excess_in_transit) - 1
                flux_original = resize(flux_original[discard_num:-discard_num], size=size)
            # scale feature model to specified ratio
            fix_scale = multiplier or sample_horizontal_scale()
            duration_correction = in_transit_ratio / (1. / fix_scale)
            if mode != 'dre':
                params['duration'] *= duration_correction
                flux_feature, feature_params = create_transit_model(params, mode=mode, feature=True, size=size)
                params['duration'] /= duration_correction
                params['multiplier'] = fix_scale
            else:
                params['multiplier'] = dre_mult
        else:
            if not mode == 'dre':
                flux_feature, feature_params = create_transit_model(params, mode=mode, feature=True, size=size)

        if flux_feature is None:
            continue

        in_transit_indices = np.argwhere(flux_original < 1.).flatten()
        if len(in_transit_indices) == 0:
            print("No cadences below 1 in theoretical transit model! Skip example.")
            continue
        ingress_idx = in_transit_indices.min()
        eggress_idx = in_transit_indices.max()

        # add horizontal shift
        shift = 0
        if horizontal_shift:
            shift = np.random.randint(-ingress_idx, (size - 1) - eggress_idx)
            if mode == 'spots':
                params['duration'] *= duration_correction
                window_size = (params['duration'] / 24.) * (params['multiplier'] / 2.)
                time_crop = np.linspace(-window_size, window_size, size, endpoint=True)
                t0 = time_crop[size // 2 + shift]
                flux_feature, feature_params = create_transit_model(params, mode=mode, feature=True, t0=t0, size=size)
                params['duration'] /= duration_correction
            else:
                if mode != 'dre':
                    flux_feature = np.roll(flux_feature, shift=shift)

        if np.isnan(flux_original).any() or np.isnan(flux_feature).any() or min(flux_feature) == max(flux_feature):
            continue

        params.update(feature_params)
        params['shift'] = shift if mode != 'dre' else dre_shift
        params['ingress_idx'] = ingress_idx + shift
        params['egress_idx'] = eggress_idx + shift
        params['transit_depth'] = min(flux_original)

        # add noise to model to create artificial data
        if not mode == 'dre':
            if mode == 'gd':
                tran_sig = 1.-params['transit_depth']
                gd_snr = tran_sig/params['gd_effect']
                sampled_snr = np.random.uniform(gd_snr, gd_snr*2)
            else:
                sampled_snr = np.random.uniform(snr_min, snr_max)
            params['snr'] = sampled_snr
            results['artificial_data'][transits_generated, :] = add_random_noise(flux_feature, inplace=False,
                                                                                 snr=sampled_snr, size=size)
        else:
            noise_scale = (sum(abs(results['artificial_data'][transits_generated, :]-flux_feature))/len(flux_feature))/2.
            dre_snr = (max(flux_feature)-min(flux_feature)) / noise_scale
            params['snr'] = dre_snr
            add_random_noise(results['artificial_data'][transits_generated, :], inplace=True, snr=dre_snr, size=size)

        results['transit_model'][transits_generated, :] = flux_original
        results['feature_model'][transits_generated, :] = flux_feature

        meta.iloc[transits_generated] = pd.Series(params)
        transits_generated += 1
        if transits_generated % 100 == 0:
            now = time()
            print(f"Transits_generated: {transits_generated}, Time ellapsed {(now-loop_start)/60.} min (last 100)"
                  f" {(now-function_start)/60.} min (total)")
            loop_start = time()

    return results, meta


def add_random_noise(transit, snr=10, inplace=False, size=256):
    noise = np.random.normal(size=(size,))
    signal = max(transit) - min(transit)
    if inplace:
        transit += noise * signal / snr
    else:
        return transit + noise * signal / snr

def concat_artificial_data_batches(data_batches):
    data_sizes = [batch[0]['artificial_data'].shape[1] for batch in data_batches]
    meta_sizes = [len(batch[1].columns) for batch in data_batches]
    assert len(set(data_sizes)) <= 1, "Incompatible data sizes detected! Make sure all batches have equal transit size."
    assert len(set(meta_sizes)) <= 1, "Incompatible meta data detected! Make sure all batches have equal meta columns."
    num_rows = [batch[0]['artificial_data'].shape[0] for batch in data_batches]
    data = {'artificial_data': np.empty((sum(num_rows), data_sizes[0])),
               'transit_model': np.empty((sum(num_rows), data_sizes[0])),
               'feature_model': np.empty((sum(num_rows), data_sizes[0]))}
    data['artificial_data'][:num_rows[0]] = data_batches[0][0]['artificial_data']
    data['transit_model'][:num_rows[0]] = data_batches[0][0]['transit_model']
    data['feature_model'][:num_rows[0]] = data_batches[0][0]['feature_model']
    meta = data_batches[0][1]
    rows_filled = num_rows[0]
    for b in range(1, len(data_batches)):
        batch_data, batch_meta = data_batches[b]
        data['artificial_data'][rows_filled:rows_filled+num_rows[b]] = batch_data['artificial_data']
        data['transit_model'][rows_filled:rows_filled+num_rows[b]] = batch_data['transit_model']
        data['feature_model'][rows_filled:rows_filled+num_rows[b]] = batch_data['feature_model']
        rows_filled += num_rows[b]
        meta = meta.append(batch_meta, ignore_index=True)
    return data, meta


def tensorfy_artificial_data(data, meta, meta_cols=('shift', 'multiplier'), meta_defaults=(None, 3.0),
                             norm_range=(-1., 1.), whiten=False):
    assert len(meta_cols) == len(meta_defaults)

    transit_data = torch.DoubleTensor(data['artificial_data'])
    transit_model = torch.DoubleTensor(data['transit_model'])
    feature_model = torch.DoubleTensor(data['feature_model'])

    if whiten:
        transit_data = standardize(min_max_norm_vectorized(transit_data)).type(torch.FloatTensor)
        transit_model = standardize(min_max_norm_vectorized(transit_model)).type(torch.FloatTensor)
        feature_model = standardize(min_max_norm_vectorized(feature_model)).type(torch.FloatTensor)
    else:
        scale = abs(norm_range[0] - norm_range[1])
        shift = norm_range[0]
        transit_data = min_max_norm_vectorized(transit_data).type(torch.FloatTensor) * scale + shift
        transit_model = min_max_norm_vectorized(transit_model).type(torch.FloatTensor) * scale + shift
        feature_model = min_max_norm_vectorized(feature_model).type(torch.FloatTensor) * scale + shift

    meta_tensors = []
    for c, col in enumerate(meta_cols):
        meta_info = meta.loc[:, col]
        nan_free = pd.notnull(meta_info).all()
        if not nan_free:
            assert meta_defaults[c] is not None, "Meta info contains NaNs and no default value was given."
            meta_info.fillna(value=meta_defaults[c], inplace=True)
        meta_tensors.append(torch.Tensor(meta_info.to_list()))
    return torch.utils.data.TensorDataset(transit_data, transit_model, feature_model, *meta_tensors)


def add_linear_trend(transits, feature_models):
    size = transits.shape[-1]
    num = transits.shape[0]
    influence_trend1 = np.linspace(1., 0., size)
    influence_trend2 = np.ones(size) - influence_trend1
    mini = feature_models.min(axis=1)
    maxi = feature_models.max(axis=1)
    transit_depths = maxi-mini
    lows = np.random.uniform(low=-transit_depths, high=transit_depths, size=num)
    highs = np.random.uniform(low=-transit_depths, high=transit_depths, size=num)
    trends = lows[:,None] * influence_trend1[None,:] + highs[:,None] * influence_trend2[None,:]
    return transits+trends, feature_models+trends, lows, highs


def remove_trend(transits, feature_models, lows=None, highs=None):
    size = transits.shape[-1]
    if lows is None or highs is None:
        lows = feature_models[:, 0]
        highs = feature_models[:, -1]
    influence_trend1 = np.linspace(1., 0., size)
    influence_trend2 = np.ones(size) - influence_trend1
    trends = lows[:,None] * influence_trend1[None,:] + highs[:,None] * influence_trend2[None,:]
    return transits-trends, feature_models-trends


def calculate_h_scale(transit_model, feature_model):
    in_tran1 = (transit_model < 1.).sum(axis=1)
    in_tran2 = (feature_model < 1.).sum(axis=1)
    return in_tran2/in_tran1


def remove_transits_with_nan(data, meta):
    new_nan_mask = np.zeros(data['artificial_data'].shape[0])
    for key in data.keys():
        nan_mask = np.isnan(data[key]).any(axis=1)
        new_nan_mask = np.logical_or(nan_mask, new_nan_mask)
    for key in data.keys():
        data[key] = data[key][~new_nan_mask]
    return data, meta.loc[~new_nan_mask]


def remove_transits_without_signal(data, meta):
    new_mask = np.zeros(data['artificial_data'].shape[0])
    for key in data.keys():
        mask = (data[key] - data[key].min(axis=1, keepdims=True)).sum(axis=1) == 0
        new_mask = np.logical_or(mask, new_mask)
    for key in data.keys():
        data[key] = data[key][~new_mask]
    return data, meta.loc[~new_mask]


def remove_transits_off_scale(data, meta, scale=0.3333):
    oot_ideal = (scale * data['transit_model'].shape[1])
    num_oot = (data['transit_model'] < 1.0).sum(axis=1)
    mask_1 = np.floor(oot_ideal) <= num_oot
    mask_2 = num_oot <= np.ceil(oot_ideal)
    mask = np.logical_and(mask_1, mask_2)
    for key in data.keys():
        data[key] = data[key][mask]
    return data, meta.loc[mask], mask


def remove_transits_custom(data, meta, mask, cutoff=None):
    for key in data.keys():
        filtered_data = data[key][~mask]
        if cutoff is not None:
            filtered_data = filtered_data[:cutoff]
        data[key] = filtered_data
    filtered_meta = meta.loc[~mask]
    if cutoff is not None:
        filtered_meta = filtered_meta.head(cutoff)
    return data, filtered_meta


def plot_artificial_sample(dataset, idx):
    sample_size = range(len(dataset[0]['artificial_data'][idx]))
    plt.subplot(121)
    plt.scatter(sample_size, dataset[0]['artificial_data'][idx])
    shift = dataset[1]['shift'].iloc[idx]
    plt.vlines(128+shift, min(dataset[0]['feature_model'][idx]), 1, colors=['r'], linestyles='dashed')
    #plt.plot(sample_size, dataset['transit_model'][idx], c='r')
    plt.plot(sample_size, dataset[0]['feature_model'][idx], c='r')
    plt.subplot(122)
    plt.plot(sample_size, dataset[0]['feature_model'][idx], c='r')
    plt.plot(sample_size, dataset[0]['transit_model'][idx], c='g')
    plt.show()


def build_artificial_dataset(batch_names=('normal', 'spot_transits', 'gd_transits', 'dres'), save_path=None):
    # load artificial data batches
    data_batches = [pickle.load(open("data/{}.pkl".format(f_name), "rb")) for f_name in batch_names]
    data, meta = concat_artificial_data_batches(data_batches)
    orig_len = len(data['artificial_data'])
    data, meta = remove_transits_with_nan(data, meta)
    nan_len = len(data['artificial_data'])
    print(f"Found {orig_len - nan_len} NaN rows")
    data, meta = remove_transits_without_signal(data, meta)
    zerovar_len = len(data['artificial_data'])
    print(f"Found {nan_len - zerovar_len} zero variance rows")
    data, meta, mask = remove_transits_off_scale(data, meta)
    print(f"Found {zerovar_len - len(data['artificial_data'])} out of scale rows")
    #pickle.dump((data, meta), open("data/concat_artificial_data_meta_remove_off_scale.pkl", "wb"))
    tensor_dataset = tensorfy_artificial_data(data, meta, norm_range=(-1., 1.), whiten=False)
    if save_path is not None:
        pickle.dump(tensor_dataset, open(save_path, "wb"))


if __name__ == "__main__":
    # example how to create 10k regular transits with random horizontal shifts and scales
    results_n_meta = sample_artificial_transits(10000, mode='normal', horizontal_scale=True, horizontal_shift=True,
                                                multiplier=None)


