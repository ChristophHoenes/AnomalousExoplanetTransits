import glob
import os
import re
from copy import deepcopy
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightkurve as lk
from wotan import flatten
from tqdm import tqdm
from data_preparation.data_processing_utils import resize, min_max_norm, normalize, time_window_binning


def crop_transit(folded_light_curve, period, duration, multiplier=3., plot=False):
    """ Crops all transits out of folded light curve
        Input: folded light curve (TESSLightCurve)
        :returns: flux and time of transits ordered by time (NumPy array)
    """
    # set time format and extract flux
    folded_light_curve.time_format = 'jd'
    flux_folded = folded_light_curve.flux_quantity

    # convert flux, phases and original time to numpy
    flux_folded_np = np.array(flux_folded)
    phase_folded_np = np.array(folded_light_curve.phase)
    original_time_np = np.array(folded_light_curve.time_original)

    # calculate transit time
    # duration in hours -> /24; period is 100% -> /period; two sides around zero -> /2
    transit_time = (duration*multiplier) / (24 * period * 2)

    # determine transit indices of time series
    lower_bound = (phase_folded_np > -transit_time).astype(bool)
    upper_bound = (phase_folded_np < transit_time).astype(bool)
    in_bound = np.logical_and(lower_bound, upper_bound)

    if plot:
        out_bound = np.invert(in_bound)
        # folded
        fig = plt.figure(figsize=(10, 5))
        mean = flux_folded_np.mean()
        plt.scatter(phase_folded_np[out_bound], flux_folded_np[out_bound]/mean, s=2, c='k')
        plt.scatter(phase_folded_np[in_bound], flux_folded_np[in_bound]/mean, s=2, c='r')
        plt.ylabel("Normalized Flux", fontsize=14)
        plt.xlabel("Phase", fontsize=14)
        plt.show()
        # unfolded
        fig = plt.figure(figsize=(10, 5))
        plt.scatter(original_time_np[out_bound], flux_folded_np[out_bound]/mean, s=2, c='k')
        plt.scatter(original_time_np[in_bound], flux_folded_np[in_bound]/mean, s=2, c='r')
        plt.ylabel("Normalized Flux", fontsize=14)
        plt.xlabel("Time [BTJD days]", fontsize=14)
        plt.show()

    # select transit data (crop transit)
    #plt.scatter(phase_folded_np, flux_folded_np, c='b')
    flux_folded_np = flux_folded_np[in_bound]
    #plt.scatter(phase_folded_np[in_bound], flux_folded_np, c='r')
    #plt.show()
    original_time_np = original_time_np[in_bound]
    phase_folded_np = phase_folded_np[in_bound]

    # sort by time to gather data points belonging to different transits
    original_time_indices = np.argsort(original_time_np)

    flux_folded_np = flux_folded_np[original_time_indices]
    original_time_np = original_time_np[original_time_indices]
    phase_folded_np = phase_folded_np[original_time_indices]

    return flux_folded_np, original_time_np, phase_folded_np


def get_crop_indices(phase, period, duration, multiplier):
    """ Calculates indices belonging to the transit crop
        :param phase (1D np.array) phases of the transit light curve
        :param period (float) orbital period of object in days
        :param duration (float) transit duration in hours
        :param multiplier (float) number of transit durations that should be cropped out
        :returns: (1D bool np.array) indices that belong to the specified window
    """
    # calculate transit time
    # duration in hours -> /24; period is 100% -> /period to get to phase space; two sides around zero -> /2
    transit_time = (duration*multiplier) / (24 * period * 2)

    # determine transit indices of time series
    lower_bound = (phase > -transit_time).astype(bool)
    upper_bound = (phase < transit_time).astype(bool)
    return np.logical_and(lower_bound, upper_bound)


def separate_transits(flux, time, phase, crop_window, min_datapoints=10, plot=False):

    sequence_lengths = []
    transit_start_indices = [0]

    for i, t in enumerate(time):
        if t - time[transit_start_indices[-1]] > crop_window:
            sequence_lengths.append(i - transit_start_indices[-1])
            transit_start_indices.append(i)
    # last length is from last transit start til the end of the array
    sequence_lengths.append(len(time) - transit_start_indices[-1])

    transit_file = []

    for i, t in enumerate(transit_start_indices):
        flux_transit = flux[t:t + sequence_lengths[i]]
        time_transit = time[t:t + sequence_lengths[i]]
        phase_transit = phase[t:t + sequence_lengths[i]]

        # check for missing data (only add transit file if sufficient data is present)
        if len(flux_transit) >= min_datapoints and len(time_transit) >= min_datapoints:
            if plot:
                plot_transit(time_transit, flux_transit, bin_size=13)
            transit_file.append((flux_transit, time_transit, phase_transit))

    return transit_file


def plot_transit(time_transit, flux_transit, bin_size=13, c='b'):
    plt.scatter(time_transit, flux_transit / flux_transit.mean(), c=c)
    binned = bin_data(flux_transit, bin_size=bin_size)
    plt.plot(time_transit[::bin_size], binned / binned.mean(), 'r--',
             time_transit[::bin_size], binned / binned.mean(), 'ro', markersize=15)
    plt.show()


def cdpp_estimate(light_curve, duration, cadence, day_window=2.0):
    """ Estimates CDPP noise level of a light curve
    :param light_curve: (lightkurve.LightCurve) input data
    :param duration: (float) transit duration in hours
    :param cadence: (float) duration of one cadence in days
    :param day_window: (float) trend filter window length in days
    :return: (float) cdpp noise level estimate of light curve
    """
    num_transit_cadences = round(duration/(cadence*24))
    x_day_num_cadences = round(day_window/cadence)
    x_day_num_cadences = x_day_num_cadences + 1 if x_day_num_cadences%2 == 0 else x_day_num_cadences
    return light_curve.estimate_cdpp(transit_duration=num_transit_cadences, savgol_window=x_day_num_cadences, savgol_polyorder=2, sigma=5.0)


def bin_data(data, bin_size=2):
    binned = [data[0:bin_size].mean()]
    prev = bin_size
    for i in range(prev, len(data), bin_size):
        binned.append(data[prev:prev+bin_size].mean())
        prev = prev+bin_size
    return np.asarray(binned)


def map_TIC_to_quality_flag(fits_folder=r'../../Data/TESS/Found_TOIs/*.fits', toi_source='tois_old.csv', save_path=None):
    mapping = {}
    tois = pd.read_csv(toi_source, comment='#', sep=',')
    for f, filepath in enumerate(tqdm(glob.iglob(fits_folder))):
        # extract sector from filename
        sector_str = re.findall(r"-s\d+-", filepath)
        sector = int(sector_str[0][-3:-1])

        # load light curve (fits file)
        test = lk.lightcurvefile.TessLightCurveFile(filepath)
        ticid = test.get_keyword("TICID", hdu=0)

        # search corresponding lines to TICID in the TOI sheet
        selected = tois.loc[tois['TIC ID'] == ticid]

        epoch = list(selected['Epoch Value'])
        period = list(selected['Orbital Period Value'])
        duration = list(selected['Transit Duration Value'])
        depth = list(selected['Transit Depth Value'])

        for obj in range(max([len(period), len(epoch), len(duration)])):

            pdc = test.get_lightcurve('PDCSAP_FLUX').remove_nans().remove_outliers(sigma_lower=100, sigma_upper=5)

            # estimate light curve cadence by median of time difference between data points
            cadence_estimate = np.median(np.array(pdc.time_original)[1:] - np.array(pdc.time_original)[:-1])

            # determine signal quality
            signal_quality, quality_flag = signal_quality_flag(pdc, depth[obj], cadence_estimate)#, duration=duration[obj])
            mapping['{}_{}_{}'.format(ticid,sector,period[obj])] = signal_quality

    if save_path is not None:
        pickle.dump(mapping, open(save_path, "wb"))

    return mapping


def aggregate_low_quality_data(flux, phase, period, target_cadence=2.0, plot=False):
    phase_sort_idx = np.argsort(phase)
    flux_sorted = flux[phase_sort_idx]
    phase_sorted = phase[phase_sort_idx]
    aggregation_ranges = []
    aggregation_start = phase_sorted[0]
    for t, value in enumerate(phase_sorted):
        phase_diff = np.abs(aggregation_start - value)
        if phase_diff*period < target_cadence/(60*24):
            continue
        aggregation_start = value
        aggregation_ranges.append(t)
    aggregated_flux = []
    aggregation_start = 0
    for agg in aggregation_ranges:
        aggregated_flux.append(flux_sorted[aggregation_start:agg].mean())
        aggregation_start = agg
    aggregated_flux.append(flux_sorted[aggregation_start:].mean())
    aggregated_flux = np.asarray(aggregated_flux)
    if plot:
        plot_transit(np.arange(len(aggregated_flux)), aggregated_flux, c='k')
    return aggregated_flux


class FluxChannel:

    def __init__(self, flux, time, phase):
        if flux is not None:
            assert flux.shape == time.shape == phase.shape, "Flux, time and phase must be the same size!"
        self._flux = flux
        self._time = time
        self._phase = phase

    def is_complete(self):
        return self._flux is not None and self._time is not None and self._phase is not None

    def num_datapoints(self):
        return len(self._flux)

    def gap_analysis(self, crop_window_time, period, target_cadence=2., tolerance=0.):
        if period is None:
            period = 2 * (max(self._time) - min(self._time))
        phase_boundary = crop_window_time / (2. * period)
        target_cadence_days = target_cadence / (60. * 24.)

        time_diff_cadences = np.empty(self._time.size+1)
        # calculate gaps before and after last point
        time_diff_cadences[0] = abs(min(self._phase) + phase_boundary) * period
        time_diff_cadences[-1] = abs(max(self._phase) - phase_boundary) * period
        # calculate time difference between data points
        time_diff_cadences[1:-1] = self._time[1:] - self._time[:-1]

        gap_idxs = time_diff_cadences > target_cadence_days + (tolerance*target_cadence_days)
        num_gaps = sum(gap_idxs)
        total_gap_sizes = time_diff_cadences[gap_idxs] / target_cadence_days
        percentage_gap_sizes = total_gap_sizes/(crop_window_time/target_cadence_days)
        total_num_missing_cadences = sum(total_gap_sizes)
        percentage_missing_cadences = total_num_missing_cadences/(self._time.size+total_num_missing_cadences)
        percentage_gap_location = np.argwhere(gap_idxs).squeeze()/(crop_window_time//target_cadence_days)
        return {"num_gaps":num_gaps, "gap_sizes_percent":percentage_gap_sizes, "gap_sizes_total": total_gap_sizes,
                "total_cadences_missing": total_num_missing_cadences, "percent_cadences_missing":percentage_missing_cadences,
                "gap_locations":percentage_gap_location}

    def fill_gaps(self, crop_window_time, period, target_cadence=2.):
        """ Fills the gaps with linear interpolation (edge points are repeated by default)
            :param crop_window_time (float) time within crop window in days (multiplier*transit_duration/24)
            :param period (float) orbital period in days
            :param target_cadence (float) the cadence after filling the gaps in minutes
        """
        phase_boundary = crop_window_time/(2.*period)
        target_cadence_days = target_cadence/(60.*24.)
        num_data_points = int(crop_window_time // target_cadence_days)
        new_phase = np.linspace(-phase_boundary, phase_boundary, num_data_points, endpoint=True)
        self._flux = np.interp(new_phase, self._phase, self._flux)
        self._time = np.interp(new_phase, self._phase, self._time)
        self._phase = new_phase

    def normalize(self, typ='median_flux'):
        if typ == 'median_flux':
            self._flux = normalize(self._flux, 'median')
        elif typ == 'mean_flux':
            self._flux = normalize(self._flux, 'mean')
        elif typ == 'min_max':
            self._flux = min_max_norm(self._flux)
        else:
            self._flux = normalize(self._flux, typ)

    def resize(self, target_size=256):
        self._flux = resize(self._flux, target_size)
        self._time = resize(self._time, target_size)
        self._phase = resize(self._phase, target_size)

    @property
    def flux(self):
        return deepcopy(self._flux)

    @property
    def time(self):
        return deepcopy(self._time)

    @property
    def phase(self):
        return deepcopy(self._phase)


class TransitCutOut:

    def __init__(self, multiplier, raw=FluxChannel(None,None,None), pdc=FluxChannel(None,None,None),
                 flat=FluxChannel(None,None,None), detrended=FluxChannel(None,None,None), **kwargs):
        self._target = None
        self._multiplier = multiplier
        self._min_time = min(pdc._time) if pdc._time is not None else min(raw._time) if raw._time is not None else None
        self._valid_channels = ("raw", "pdc", "flat", "detrended", *list(kwargs.keys()))
        self._channels = {"raw":raw,
                          "pdc":pdc,
                          "flat":flat,
                          "detrended":detrended}
        self._channels.update(kwargs)

    def __getitem__(self, key):
        if not key in self._valid_channels:
            raise KeyError(f"Invalid channel {key}! Use one of {self._valid_channels}")
        return self._channels[key]

    def _set_target(self, target):
        assert isinstance(target, Target), "Can only set a Target object as target for a TransitCutOut!"
        self._target = target

    def is_complete(self):
        return all([channel.is_complete() for channel in self._channels.values()])

    def has_pdc(self):
        return self._channels["pdc"].is_complete()

    def select_channels(self, channels):
        assert all([channel in self._valid_channels for channel in channels]),\
            f"Invalid channel detected! Only pick channels from {self._valid_channels}."
        for channel in list(self._channels.keys()):
            if channel not in channels:
                del self._channels[channel]

    def valid_num_datapoints(self, min_datapoints=26, max_datapoints=np.inf, channel=None, **kwargs):
        if channel is None:
            for channel in self._channels.values():
                if not min_datapoints <= channel.num_datapoints() <= max_datapoints:
                    return False
            return True
        else:
            return min_datapoints <= self._channels[channel].num_datapoints() <= max_datapoints

    def valid_gaps(self, num_gaps=np.inf, total_cadences_missing=np.inf, percent_cadences_missing=1.,
                   gap_sizes_cadence=np.inf, gap_sizes_percent=1., gap_locations_lower=np.inf, gap_locations_upper=-np.inf,
                   size_location=(np.inf, np.inf, -np.inf), channel='pdc', **kwargs):

        gap_results = self._channels[channel].gap_analysis(self.multiplier*self.duration/24., self.period)

        at_most_x_gaps = gap_results["num_gaps"] <= num_gaps
        at_most_x_cadences_missing = gap_results["total_cadences_missing"] <= total_cadences_missing
        at_most_x_percent_missing = gap_results["percent_cadences_missing"] <= percent_cadences_missing
        at_most_x_cadences_large = (gap_results["gap_sizes_total"] <= gap_sizes_cadence).all()
        at_most_x_percent_large = (gap_results["gap_sizes_percent"] <= gap_sizes_percent).all()
        location = np.logical_and(gap_locations_lower < gap_results["gap_locations"],
                                  gap_results["gap_locations"] < gap_locations_upper)
        not_in_location = not location.any()
        location2 = np.logical_and(size_location[1] < gap_results["gap_locations"],
                                  gap_results["gap_locations"] < size_location[2])
        at_most_x_size_in_location = (gap_results["gap_sizes_percent"][location2] <= size_location[0]).all()


        return at_most_x_gaps and at_most_x_cadences_missing and at_most_x_percent_missing and \
               at_most_x_cadences_large and at_most_x_percent_large and not_in_location and at_most_x_size_in_location

    def valid_cdpp_snr(self, cdpp_snr_threshold=7.1, **kwargs):
        return self._target._depth/self._target._cdpp > cdpp_snr_threshold

    def fill_gaps(self, channel=None, target_cadence=2.):
        period = self.period or self._target.meta['folding_period']
        if channel is None:
            for channel in self._channels.values():
                channel.fill_gaps(self.multiplier*self.duration/24., period, target_cadence=target_cadence)
        else:
            self._channels[channel].fill_gaps(self.multiplier*self.duration/24., period, target_cadence=target_cadence)

    def normalize(self, typ='median_flux'):
        for channel in self._channels.values():
            channel.normalize(typ)

    def resize(self, target_size=256):
        for channel in self._channels.values():
            channel.resize(target_size)

    def to_tensor(self, quantities=('flux',)):
        assert all([quantity in ('flux', 'time', 'phase') for quantity in quantities]),\
            "Invalid quantity detected! Valid quantities are: 'flux', 'time' or 'phase'."
        tensor_channels = []
        for channel in self._channels.values():
            if 'flux' in quantities:
                tensor_channels.append(torch.Tensor(channel.flux))
            if 'time' in quantities:
                tensor_channels.append(torch.Tensor(channel.time))
            if 'phase' in quantities:
                tensor_channels.append(torch.Tensor(channel.phase))
        return torch.stack(tensor_channels)

    def plot(self, xtick="time", channels=None):
        if len(self._channels) <= 1:
            if len(self._channels) == 0:
                raise RuntimeError("No channels available, nothing to plot!")
            else:
                ax = plt.gca()
                channel = list(self._channels.keys())[0]
                ax.set_title(channel)
                data = self._channels[channel]
                x = data._time if xtick == 'time' else data._phase
                plt.scatter(x, data._flux)
                plt.show()
        else:
            if channels is None:
                plot_size = np.sqrt(len(self._channels))
                channel_iterator = self._channels.items()
            else:
                plot_size = np.sqrt(len(channels))
                channel_iterator = [self._channels[c] for c in channels]
            rows = int(np.floor(plot_size))
            cols = int(np.ceil(plot_size))
            fig, axs = plt.subplots(rows, cols)
            for i, (channel, data) in enumerate(channel_iterator):
                x = data._time if xtick == 'time' else data._phase
                axs.flatten()[i].set_title(channel)
                axs.flatten()[i].scatter(x, data._flux)
            plt.show()

    @property
    def multiplier(self):
        return self._multiplier

    @property
    def epoch(self):
        assert self._target is not None, "TransitCutOut object only has property epoch when assigned to a Target!"
        return self._target.epoch

    @property
    def period(self):
        assert self._target is not None, "TransitCutOut object only has property period when assigned to a Target!"
        return self._target.period

    @property
    def duration(self):
        assert self._target is not None, "TransitCutOut object only has property duration when assigned to a Target!"
        return self._target.duration

    @property
    def time(self):
        return self._min_time


class Target:

    def __init__(self, ticid, epoch, period, duration, depth, cdpp, meta):
        self._ticid = ticid
        self._epoch = epoch
        self._period = period
        self._duration = duration
        self._depth = depth
        self._cdpp = cdpp
        self.transtis = []
        self.aggregations = {"full":None, "sector":[], "random":[]}
        self._sectors = []
        self.meta = meta

    def __len__(self):
        return len(self.transtis)

    def add_transit(self, transit, sector):
        assert isinstance(transit, TransitCutOut), "Class Target can only hold transits of type 'TransitCutOut'!"
        added = False
        for t, existing_transit in reversed(list(enumerate(self.transtis))):
            if existing_transit.time > transit.time:
                continue
            else:
                self.transtis.insert(t+1, transit)
                self._sectors.insert(t+1, sector)
                transit._set_target(self)
                added = True
                break
        if not added:
            self.transtis.insert(0, transit)
            self._sectors.insert(0, sector)
            transit._set_target(self)

    def aggregate_transits(self, channels=("raw", "pdc", "flat", "detrended"), multiplier=3., target_cadence=2.,
                           target_size=256, median=True, mode="full", sectors=None, num_transits=5, is_sorted=False,
                           plot=False):
        if mode == "full":
            selected_transits = self.transtis
        elif mode == "sector":
            assert sectors is not None, "Need to specify argument 'sectors'" \
                                        " (tuple of ints representing sector ids) for mode 'sector'!"
            selected_transits = self.filter("sector", action="get", sectors=sectors)
        elif mode == 'random':
            selected_transits = np.random.choice(self.transtis, num_transits)
        else:
            raise RuntimeError(f"Unknown mode {mode} for aggregation! Please choose from ['full', 'sector', 'rendom'].")

        fluxes = {channel: [transit._channels[channel].flux for transit in selected_transits] for channel in channels}
        #times = {channel: [transit._channels[channel].time for transit in selected_transits] for channel in channels}
        phases = {channel: [transit._channels[channel].phase for transit in selected_transits] for channel in channels}

        results = {}
        for channel in channels:
            transit_fluxes = np.hstack(fluxes[channel])
            #transit_times = np.hstack(times[channel])
            transit_phases = np.hstack(phases[channel])
            if not is_sorted:
                sort_idxs = np.argsort(transit_phases)
                transit_fluxes = transit_fluxes[sort_idxs]
                #transit_times = transit_times[sort_idxs]
                transit_phases = transit_phases[sort_idxs]

            period = self._period or self.meta['folding_period']
            crop_window_time = multiplier*self._duration/24.
            phase_boundary = crop_window_time / (2. * period)
            if target_cadence is None:
                num_data_points = target_size
            else:
                target_cadence_days = target_cadence / (60. * 24.)
                num_data_points = int(crop_window_time // target_cadence_days)
            new_phase = np.linspace(-phase_boundary, phase_boundary, num_data_points, endpoint=True)
            #flux = np.interp(new_phase, transit_phases, transit_fluxes)
            flux = time_window_binning(transit_fluxes, transit_phases, new_phase, median=median)
            if plot:
                plt.scatter(transit_phases, transit_fluxes, c='k')
                plt.scatter(new_phase, flux, c='r')
                plt.show()
            time = new_phase*period  # np.interp(new_phase, transit_phases, transit_times)
            results[channel] = FluxChannel(flux, time, new_phase)

        aggregated_transit = TransitCutOut(multiplier, **results)
        aggregated_transit.select_channels(channels)
        aggregated_transit._target = self
        if mode == "full":
            self.aggregations["full"] = aggregated_transit
        else:
            self.aggregations[mode].append(aggregated_transit)

    def _clear_empty_transits(self):
        self.filter('completeness', action='drop_others')

    def select_channels(self, channels):
        """ Note that all transits that do not contain ALL of the selected channels will be dropped"""
        for transit in self.transtis:
            transit.select_channels(channels)
        self._clear_empty_transits()

    def count_complete_transits(self):
        return sum([transit.is_complete() for transit in self.transtis])

    def count_incomplete_transits(self):
        return sum([not transit.is_complete() for transit in self.transtis])

    @property
    def ticid(self):
        return self._ticid

    @property
    def epoch(self):
        return self._epoch

    @property
    def period(self):
        return self._period

    @property
    def duration(self):
        return self._duration

    @property
    def available_sectors(self):
        return set(self._sectors)

    def get_transits_by_sector(self, sectors):
        if isinstance(sectors, int):
            selector = np.array(self._sectors) == sectors
        else:
            sector_array = np.array(self._sectors)
            sector_indices = []
            for sector in sectors:
                sector_indices.append(sector_array == sector)
            selector = np.logical_or.reduce(sector_indices)
        transit_array = np.asarray(self.transtis, dtype=object)
        return [transit for transit, select in zip(transit_array, selector) if select]

    def filter(self, quantity, action='get', **kwargs):
        if quantity == 'completeness':
            selector = [transit.is_complete() for transit in self.transtis]
        elif quantity == 'incompleteness':
            selector = [not transit.is_complete() for transit in self.transtis]
        elif quantity == 'pdc_available':
            selector = [transit.has_pdc() for transit in self.transtis]
        elif quantity == 'data_points':
            selector = [transit.valid_num_datapoints(**kwargs) for transit in self.transtis]
        elif quantity == 'data_gaps':
            selector = [transit.valid_gaps(**kwargs) for transit in self.transtis]
        elif quantity == 'CDPP_SNR':
            selector = [transit.valid_cdpp_snr(**kwargs) for transit in self.transtis]
        elif quantity == 'sector':
            selector = [self.transtis[s] for s, sector in enumerate(self._sectors) if sector in kwargs['sectors']]
        elif quantity == 'period':
            valid_period = kwargs.get('valid_period', True)
            if valid_period:
                correct = kwargs.get('period_lower', -np.inf) < self._period < kwargs.get('period_upper', np.inf)
                if self._period is None or not correct:
                    selector =  [False]*len(self.transtis)
                else:
                    selector = [True]*len(self.transtis)
            else:
                selector = [True]*len(self.transtis) if self._period is None else [False]*len(self.transtis)
        else:
            raise RuntimeError(f"Unknown filter quantity {quantity}!")

        transit_array = np.asarray(self.transtis, dtype=object)
        sector_array = np.asarray(self._sectors, dtype=object)
        result = [transit for transit, select in zip(transit_array, selector) if select]
        sector_selection = [sector for sector, select in zip(sector_array, selector) if select]

        if action == 'count':
            return len(result)
        elif action == 'get':
            return result
        elif action == 'drop_others':
            self.transtis = result
            self._sectors = sector_selection
        else:
            raise RuntimeError(f"Unknown action type {action} for filter.")

    def normalize(self, typ='median_flux'):
        for transit in self.transtis:
            transit.normalize(typ)
        if self.aggregations["full"] is not None:
            self.aggregations["full"].normalize(typ)
        for transit in self.aggregations["sector"]:
            transit.normalize(typ)
        for transit in self.aggregations["random"]:
            transit.normalize(typ)

    def resize(self, target_size=256):
        for transit in self.transtis:
            transit.resize(target_size)
        if self.aggregations["full"] is not None:
            self.aggregations["full"].resize(target_size)
        for transit in self.aggregations["sector"]:
            transit.resize(target_size)
        for transit in self.aggregations["random"]:
            transit.resize(target_size)

    def fill_gaps(self):
        for transit in self.transtis:
            transit.fill_gaps()


class TargetCollection(dict):

    def __init__(self):
        dict.__init__(self)
        self._normalized = False
        self._resized = False
        self._gaps_filled = False
        self._filtered = False

    def extend(self, ticid, epoch, period, duration, depth, cdpp, sector, transit_list, meta=None):
        if len(transit_list) == 0:
            return
        id = (ticid, epoch, period)
        target = self.get(id, None)
        if target is None:
            target = Target(ticid, epoch, period, duration, depth, cdpp, meta)
            self[id] = target
        for transit in transit_list:
            target.add_transit(transit, sector)

    def get_tensor_representation(self, aggregations=True, single_transits=True):
        target_ids = []
        transit_data = []
        for key, target in self.items():
            tic, epoch, period = key
            safe_key = (tic, epoch, period or np.nan)
            if single_transits:
                for transit in target.transtis:
                    target_ids.append(torch.Tensor((*safe_key, 0.)))
                    transit_data.append(transit.to_tensor())
            if aggregations:
                target_ids.append(torch.Tensor((*safe_key, 1.)))
                transit_data.append(torch.Tensor(target.aggregations["full"].to_tensor()))
                for agg in target.aggregations["sector"]:
                    target_ids.append(torch.Tensor((*safe_key, 1.)))
                    transit_data.append(torch.Tensor(agg.to_tensor()))
                for agg in target.aggregations["random"]:
                    target_ids.append(torch.Tensor((*safe_key, 1.)))
                    transit_data.append(torch.Tensor(agg.to_tensor()))
        return torch.stack(target_ids), torch.stack(transit_data)

    def _clear_empty_targets(self):
        for key in list(self.keys()):
            if len(self[key].transtis) == 0:
                del self[key]

    def aggregate_transits(self, channels=("raw", "pdc", "flat", "detrended"), multiplier=3., target_cadence=2.,
                           target_size=256, median=True, mode="full", sectors=None, num_transits=5, is_sorted=False,
                           plot=False):
        for target in self.values():
            target.aggregate_transits(channels=channels, multiplier=multiplier, target_cadence=target_cadence,
                                      target_size=target_size, median=median, mode=mode, sectors=sectors,
                                      num_transits=num_transits, is_sorted=is_sorted, plot=plot)

    def select_channels(self, channels):
        for target in self.values():
            target.select_channels(channels)
        self._clear_empty_targets()

    def filter(self, quantity, action='get', **kwargs):
        if action == 'get':
            results = []
            return_results = True
        elif action == 'count':
            results = 0
            return_results = True
        else:
            return_results = False

        for key in list(self.keys()):
            if return_results:
                results += self[key].filter(quantity, action, **kwargs)
            else:
                self[key].filter(quantity, action, **kwargs)
                # remove empty targets
                self._clear_empty_targets()

        if return_results:
            return results
        else:
            # keep track of filter changes in a string of the filter arguments
            if self._filtered:
                self._filtered += str(quantity) + str(kwargs)
            else:
                self._filtered = str(quantity) + str(kwargs)

    def normalize(self, typ='median_flux'):
        for target in self.values():
            target.normalize(typ)
        self._normalized = typ

    def resize(self, target_size=256):
        for target in self.values():
            target.resize(target_size)
        self._resized = target_size

    def fill_gaps(self):
        for target in self.values():
            target.fill_gaps()
        self._gaps_filled = True

    def num_transits(self):
        return sum([len(target) for target in self.values()])



def apply_indices_to_light_curve(light_curve, indices):
    light_curve.flux = light_curve.flux[indices]
    light_curve.time_original = light_curve.time_original[indices]
    light_curve.time = light_curve.phase[indices]


def crop_transits(raw_folded, pdc_channels, in_bound_raw, in_bound_pdc):
    """:param pdc_channels (tuple of lightkurve.FoldedlightCurve)"""
    apply_indices_to_light_curve(raw_folded, in_bound_raw)
    for channel in pdc_channels:
        apply_indices_to_light_curve(channel, in_bound_pdc)


def sort_by_time(raw_folded, pdc_channels, return_indices=False):
    """:param pdc_channels (tuple of lightkurve.FoldedlightCurve)"""
    sort_idx_raw = np.argsort(raw_folded.time_original)
    sort_idx_pdc = np.argsort(pdc_channels[0].time_original)

    apply_indices_to_light_curve(raw_folded, sort_idx_raw)
    for channel in pdc_channels:
        apply_indices_to_light_curve(channel, sort_idx_pdc)

    if return_indices:
        return sort_idx_raw, sort_idx_pdc


def get_transit_boundary_indices(time, transit_size):
    """ Determines transit boundaries from sorted time of transit cut out
        :param time (1D np.array) sorted times of transit cut out
        :param transit_size (float) size of the transit crop window in days
        :returns tuple:
                        [0] list of transit start indices (int)
                        [1] list of sequence lengths (int) of each transit
    """
    sequence_lengths = []
    transit_start_indices = [0]

    for i, t in enumerate(time):
        if t - time[transit_start_indices[-1]] > transit_size:
            sequence_lengths.append(i - transit_start_indices[-1])
            transit_start_indices.append(i)
    # last length is from last transit start til the end of the array
    sequence_lengths.append(len(time) - transit_start_indices[-1])

    return transit_start_indices, sequence_lengths


def _get_separated_transits(light_curve, transit_start_indices, sequence_lengths):
    fluxes = []
    times = []
    phases = []
    start_times = []

    for i, t in enumerate(transit_start_indices):
        time_transit = light_curve.time_original[t:t + sequence_lengths[i]]
        if len(time_transit) == 0:
            continue
        times.append(time_transit)
        start_times.append(time_transit[0])
        fluxes.append(light_curve.flux[t:t + sequence_lengths[i]])
        phases.append(light_curve.phase[t:t + sequence_lengths[i]])
    return fluxes, times, phases, start_times


def match_start_times(start_time, start_times2, crop_window_days):
    window = start_time + crop_window_days
    for st, start_time2 in enumerate(start_times2):
        window2 = start_time2 + crop_window_days
        if start_time <= start_time2 < window or start_time2 <= start_time < window2:
            return st
    return None


def separate_transits_by_time(raw_folded, pdc_channels, multiplier, duration):
    """:param pdc_channels (tuple of lightkurve.FoldedlightCurve)"""
    sort_by_time(raw_folded, pdc_channels)
    crop_window_days = multiplier * (duration / 24)
    transit_start_indices_raw, sequence_lengths_raw = get_transit_boundary_indices(raw_folded.time_original, crop_window_days)
    transit_start_indices_pdc, sequence_lengths_pdc = get_transit_boundary_indices(pdc_channels[0].time_original, crop_window_days)

    raw_fluxes, raw_times, raw_phases, raw_start_times = _get_separated_transits(raw_folded, transit_start_indices_raw,
                                                                                 sequence_lengths_raw)
    pdc_fluxes, pdc_times, pdc_phases, pdc_start_times = _get_separated_transits(pdc_channels[0], transit_start_indices_pdc,
                                                                                 sequence_lengths_pdc)

    flat_fluxes = _get_separated_transits(pdc_channels[1], transit_start_indices_pdc, sequence_lengths_pdc)[0]
    detr_fluxes = _get_separated_transits(pdc_channels[2], transit_start_indices_pdc, sequence_lengths_pdc)[0]

    transit_list = []
    for start_time in raw_start_times:
        matched_idx = match_start_times(start_time, pdc_start_times, crop_window_days)
        raw_flux = raw_fluxes.pop(0)
        raw_time = raw_times.pop(0)
        raw_phase = raw_phases.pop(0)
        if matched_idx is None:
            pdc_flux = None
            pdc_time = None
            pdc_phase = None
            flat_flux = None
            detr_flux = None
        else:
            pdc_start_times.pop(matched_idx)
            pdc_flux = pdc_fluxes.pop(matched_idx)
            pdc_time = pdc_times.pop(matched_idx)
            pdc_phase = pdc_phases.pop(matched_idx)
            flat_flux = flat_fluxes.pop(matched_idx)
            detr_flux = detr_fluxes.pop(matched_idx)
        transit_list.append(TransitCutOut(multiplier,
                                          raw=FluxChannel(raw_flux, raw_time, raw_phase),
                                          pdc=FluxChannel(pdc_flux, pdc_time, pdc_phase),
                                          flat=FluxChannel(flat_flux, pdc_time, pdc_phase),
                                          detrended=FluxChannel(detr_flux, pdc_time, pdc_phase)))

    for i in range(len(pdc_start_times)):
        raw_flux = None
        raw_time = None
        raw_phase = None
        pdc_start_times.pop(0)
        pdc_flux = pdc_fluxes.pop(0)
        pdc_time = pdc_times.pop(0)
        pdc_phase = pdc_phases.pop(0)
        flat_flux = flat_fluxes.pop(0)
        detr_flux = detr_fluxes.pop(0)
        transit_list.append(TransitCutOut(multiplier,
                                          raw=FluxChannel(raw_flux, raw_time, raw_phase),
                                          pdc=FluxChannel(pdc_flux, pdc_time, pdc_phase),
                                          flat=FluxChannel(flat_flux, pdc_time, pdc_phase),
                                          detrended=FluxChannel(detr_flux, pdc_time, pdc_phase)))

    return transit_list


def get_transit_parameters(selected):
    epoch = list(selected['Epoch (BJD)'])  # ['Epoch Value'])
    period = list(selected['Period (days)'])  # ['Orbital Period Value'])
    # replace nan values of period (when only one transit exists)
    for i, p in enumerate(period):
        if np.isnan(p) or p == 0:
            period[i] = None
        if np.isnan(epoch[i]):
            epoch[i] = None
    duration = list(selected['Duration (hours)'])  # ['Transit Duration Value'])
    depth = list(selected['Depth (ppm)'])  # ['Transit Depth Value'])
    return epoch, period, duration, depth


def get_transit_parameters_dv(selected):
    epoch = selected.get('tce_time0')
    if epoch is None:
        epoch = selected.get('transitEpochBtjd')
    epoch = list(epoch)
    period = selected.get('tce_period')
    if period is None:
        period = selected.get('orbitalPeriodDays')
    period = list(period)
    # replace nan values of period (when only one transit exists)
    for i, p in enumerate(period):
        if np.isnan(p) or p == 0:
            period[i] = None
        if np.isnan(epoch[i]):
            epoch[i] = None
    duration = selected.get('tce_duration')
    if duration is None:
        duration = selected.get('transitDurationHours')
    duration = list(duration)
    depth = selected.get('tce_depth')
    if depth is None:
        depth = selected.get('transitDepthPpm')
    depth = list(depth)
    return epoch, period, duration, depth


def flatten_biweight(pdc, window_length=0.5):
    negative_flux_flag = True
    nans_introduced_flag = False
    # check for negative flux
    if (pdc.flux < 0).any():
        negative_flux_flag = True
        print("WARNING: Negative flux encountered during detrending!")
        if (pdc.flux > 0).any():
            pdc.flux += 1000.0 + abs(min(pdc.flux))
        else:
            pdc.flux = np.abs(pdc.flux)

    flux, trend = flatten(pdc.time, pdc.flux, window_length=window_length, method='biweight', return_trend=True)
    time = pdc.time

    # check for introduced NaNs
    nans = np.isnan(flux)
    if nans.any():
        nans_introduced_flag = True
        print("Alert! NaN encountered. Will be interpolated.")
        flux[nans] = np.interp(nans.nonzero()[0], (~nans).nonzero()[0], flux[~nans])

    return flux, trend, negative_flux_flag, nans_introduced_flag


def extract_data(fits_folder=r'../../Data/TESS/Found_TOIs/*.fits', save_path=None,
                 toi_path='tois_latest.csv', multiplier=3., verbose=False, multi_sector=False, plot=False):
    """ Extracts single transits and aggregated transits of targets that are listed in the TOI catalogue (flux, time, phase)
        :param fits_folder (str) Path to light curve fits files
        :param save_path (str) Path where to save extracted data
        :param toi_path (str) Path to toi table
        :param multiplier (float) Specifies how many transit durations the cut out window is
        :param target_cadence (float) Specifies the level of aggregation in minutes cadences within a this range are aggregated (mean)
        :return (TargetCollection) a structured data format containing targets with meta information together with the transit cut outs (flux, time, phase for different channels)
    """
    # load TOI and  meta data csv files
    tois = pd.read_csv(toi_path, comment='#', sep=',')
    unique_toi_tic = set(tois['TIC ID'].tolist())
    dv_meta_path = "./data/dv_meta_data/"
    if os.path.exists(dv_meta_path):
        dv_meta_data_file_names = [f_name for f_name in os.listdir(dv_meta_path) if os.path.isfile(dv_meta_path+f_name)]
    else:
        dv_meta_data_file_names = []

    print("Start processing light curve (.fits) files:")
    target_collection = TargetCollection()
    for f, filepath in enumerate(tqdm(glob.iglob(fits_folder))):
        sector_str = re.findall(r"-s\d+-", filepath)[0]
        sector = int(sector_str[-4:-1])

        # load light curve (fits file)
        test = lk.lightcurvefile.TessLightCurveFile(filepath)
        ticid = test.get_keyword("TICID", hdu=0)
        if ticid not in unique_toi_tic:
            print("Alert! Non-TOI found!")

        # load data validation meta data
        meta_dv_df = None
        for f_name in dv_meta_data_file_names:
            if sector_str + sector_str[1:-1] in f_name:
                meta_dv_df = pd.read_csv(dv_meta_path + f_name, comment='#', sep=',')
                break
        selected_dv = None
        if meta_dv_df is not None:
            selected_dv = meta_dv_df.loc[meta_dv_df['ticid'] == ticid]

        # search corresponding lines to TICID in the TOI sheet
        selected = tois.loc[tois['TIC ID'] == ticid]

        if selected_dv is None:
            epoch, period, duration, depth = get_transit_parameters(selected)
        else:
            epoch, period, duration, depth = get_transit_parameters_dv(selected_dv)

        for obj in range(len(period)):
            # correct BJD to BTJD
            if epoch is None:
                print(f"WARNING: Epoch for TIC {ticid} is None!")
                continue
            if epoch[obj] > 2457000.0:
                epoch[obj] -= 2457000.0

            if verbose:
                print('TIC ' + str(ticid))
                print('epoch:', epoch[obj])
                print('period:', period[obj])
                print('duration:', duration[obj])

            # extract metadata
            meta_data = {c: selected[c].to_list()[obj%len(selected)] for c in list(tois.columns)}
            if selected_dv is not None:
                meta_data_dv = {"dv_"+c: selected_dv[c].to_list()[obj%len(selected_dv)] for c in list(meta_dv_df.columns)}
                meta_data.update(meta_data_dv)

            raw = test.get_lightcurve('SAP_FLUX').remove_nans().remove_outliers(sigma_lower=100, sigma_upper=5)
            folding_period = period[obj] or 2 * (max(raw.time) - min(raw.time))
            meta_data['folding_period'] = folding_period
            raw_folded = raw.fold(folding_period, epoch[obj])

            pdc = test.get_lightcurve('PDCSAP_FLUX').remove_nans().remove_outliers(sigma_lower=100, sigma_upper=5)
            pdc_copy = deepcopy(pdc)
            pdc_folded = pdc.fold(folding_period, epoch[obj])

            # old flatten
            pdc_flat_old_folded = pdc.flatten().fold(folding_period, epoch[obj])

            # new flatten (sometimes negative flux -> nan trend Solution take absolute flux)
            pdc_copy.flux, trend, neg_flux_flag, nan_flag = flatten_biweight(pdc_copy)
            pdc_detr_folded = pdc_copy.fold(folding_period, epoch[obj])
            meta_data['negative_flux_flag'] = neg_flux_flag
            meta_data['nan_in_detrending_flag'] = nan_flag

            # estimate noise level
            cadence_estimate = np.median(pdc.time[1:] - pdc.time[:-1])
            noise_level = cdpp_estimate(pdc, duration[obj], cadence_estimate, day_window=2.0)
            snr = depth[obj]/noise_level
            if verbose:
                print(f"Depth/CDPP: {snr}, SNR: {meta_data['Planet SNR']}")

            pdc_channels = (pdc_folded, pdc_flat_old_folded, pdc_detr_folded)

            # crop out transit window
            in_bound_raw = get_crop_indices(raw_folded.phase, folding_period, duration[obj], multiplier=multiplier)
            in_bound_pdc = get_crop_indices(pdc_folded.phase, folding_period, duration[obj], multiplier=multiplier)
            crop_transits(raw_folded, pdc_channels, in_bound_raw, in_bound_pdc)

            # separate into single transits
            transit_list = separate_transits_by_time(raw_folded, pdc_channels, multiplier, duration[obj])

            # add extracted transits of this fits file to target transits
            target_collection.extend(ticid, epoch[obj], period[obj], duration[obj], depth[obj], noise_level, sector,
                                     transit_list, meta=meta_data)

    if save_path is not None:
        pickle.dump(target_collection, open(save_path, "wb"))
    return target_collection


if __name__ == "__main__":
    # fits folder is the directory where the retrieved lc.fits files from MAST lie
    target_colection = extract_data(fits_folder=r'../../Data/TESS/Found_TOIs/*.fits',
                                    save_path="data/extracted_data.pkl", verbose=False)
