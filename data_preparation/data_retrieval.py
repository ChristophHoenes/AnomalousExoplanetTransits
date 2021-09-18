import os
import requests
import glob
import re
import numpy as np
import pandas as pd
import lightkurve as lk
import pickle
import urllib.request
from tqdm import tqdm
from time import time
from bs4 import BeautifulSoup
from astroquery.mast import Catalogs, Observations
from astropy.table import vstack


def recent_TOI_table(path="./"):
    url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    r = requests.get(url)

    with open(os.path.join(path, "tois_latest.csv"), 'wb') as f:
        f.write(r.content)


def is_one_sector_dv_meta(url):
    sector_numper_split = url.split('-s')
    if len(sector_numper_split) != 3:
        return False
    return sector_numper_split[1][:4] == sector_numper_split[2][:4]


def download_dv_meta_data_per_sector(save_path="./data/dv_meta_data"):

    url = "https://archive.stsci.edu/missions/tess/catalogs/tce"

    r = requests.get(url)
    html = r.text
    soup = BeautifulSoup(html, "html5lib")

    for link in soup.find_all('a'):
        link = url + '/' + link.get('href')
        #content_type = requests.head(link, allow_redirects=True).headers.get('content-type').lower()
        #if 'text' not in content_type and 'html' not in content_type:
        if link.endswith(".csv") and is_one_sector_dv_meta(link):
            file_name = link.rsplit('/', 1)[1]
            with open(os.path.join(save_path, file_name), 'wb') as f:
                f.write(requests.get(link).content)


def min_file_sector_coverage(prods):
    """ Selects a number of files that cover all available sectors.
    Result might not be the minimal set of files that covers all available sectors in general
    (to avoid computational and implementation overhead).
    But given the process of data validation file creation it is highly likely that the minimal set is found."""
    file_names = list(prods['dataURI'])
    coverages = [None] * len(file_names)
    all_sectors = set()
    for f, file_name in enumerate(file_names):
        result = re.findall(r"-s\d+", file_name)
        if len(result) != 2:
            raise RuntimeError("Invalid file name encountered!\n" +
                               f"File name '{file_name}' is not valid according to convention fot dvt.fits files.")
        start_sector = int(result[0][2:])
        end_sector = int(result[1][2:])
        coverages[f] = set(range(start_sector, end_sector+1))
        all_sectors = all_sectors.union(coverages[f])
    # add files with largest coverages to download until all sectors are covered
    selected_file_mask = np.zeros(len(file_names), dtype=bool)
    cover_lens = [len(i) for i in coverages]
    for i in np.asarray(cover_lens).argsort()[::-1]:
        if len(all_sectors) == 0:
            return prods[selected_file_mask]
        set_diff = all_sectors.difference(coverages[i])
        if len(set_diff) == len(all_sectors):
            continue
        else:
            selected_file_mask[i] = True
        all_sectors = all_sectors.difference(coverages[i])
    return prods[selected_file_mask]


def query_TIC_IDs(source_file="tois.csv", tess_only=True, start_from=0, download=False, file_ending="lc.fits"):
    tois = pd.read_csv(source_file, comment='#', sep=',')
    ticids = tois['TIC ID'].unique().tolist()[start_from:]
    results = []
    failed_tics = {'not_found': [], 'exception': []}
    print("Start querying {} TIC IDs:". format(len(ticids)))
    for t in tqdm(range(len(ticids))):
        try:
            # query each TIC ID
            obs = Observations.query_criteria(obs_collection='TESS', target_name=str(ticids[t]),
                                              dataproduct_type='timeseries')
            # only exact target
            obs = obs[obs['distance'] == 0]
            # filter results to only contain time series data
            obs = obs[obs['dataproduct_type'] == 'timeseries']

            # select only data collected by TESS mission
            if tess_only:
                obs = obs[obs['obs_collection'] == 'TESS']
            prods = Observations.get_product_list(obs)

            # filter result to only contain light curve files (no target pixel files or data validation pdf)
            lc_mask = []
            for row in prods['dataURI']:
                lc_mask.append(file_ending in row)
            prods = prods[np.asarray(lc_mask)]

            # avoid duplicate data
            if 'dvt' in file_ending:
                prods = min_file_sector_coverage(prods)

            if len(prods) == 0:
                failed_tics['not_found'].append(ticids[t])
            elif download:
                Observations.download_products(prods)

            results.append(prods)

        except Exception as err:
            print("TIC failed with: {}".format(err))
            failed_tics['exception'].append(ticids[t])

    return vstack(results), failed_tics


def create_download_script(table, save_path="download_tois.sh"):
    manifest = Observations.download_products(table, curl_flag=True)


def check_files_for_TOI(folder=r'../../Data/TESS', toi_source="tois.csv"):
    tois = pd.read_csv(toi_source, comment='#', sep=',')
    ticids = set(tois['TIC ID'].unique().tolist())

    found_ticids = set()
    found_folder = folder+'/Found_TOIs'
    if not os.path.exists(found_folder):
        os.mkdir(found_folder)

    for f, filepath in enumerate(tqdm(glob.iglob(folder+'/tess*/*lc.fits', recursive=True))):
        # load light curve (fits file)
        try:
            file = lk.lightcurvefile.TessLightCurveFile(filepath)
        except OSError:
            continue
        ticid = file.get_keyword("TICID", hdu=0)
        file.hdu.close()
        file_name = filepath.split(sep='\\')[-1]
        if ticid in ticids:
            os.rename(filepath, os.path.join(found_folder, file_name))
            found_ticids.add(ticid)
    print(
        "Found {} out of {} TIC IDs!\n".format(len(found_ticids), len(ticids)) +
        "Check output for list of unfound TIC IDs and find the found fits files under {}".format(found_folder))

    unfound = list(ticids.difference(found_ticids))
    pickle.dump(unfound, open(folder+'/unfound_ticids.pkl', "wb"))

    return unfound


if __name__ == "__main__":
    recent_TOI_table()
    query_results, failed = query_TIC_IDs(source_file="tois_latest.csv", tess_only=True, download=True, start_from=347,
                                          file_ending="lc.fits")
    check_files_for_TOI(folder=r'../../Data/TESS_large', toi_source="tois_latest.csv")
