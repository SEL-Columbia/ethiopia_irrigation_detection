import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import rasterio

from pysptools.abundance_maps.amaps import UCLS, NNLS, FCLS


def normalize(M):
    """
    Normalizes M to be in range [0, 1].
    Parameters:
      M: `numpy array`
          1D, 2D or 3D data.
    Returns: `numpy array`
          Normalized data.
    """
    minVal = np.min(M)
    maxVal = np.max(M)

    Mn = M - minVal

    if maxVal == minVal:
        return np.zeros(M.shape)
    else:
        return Mn / (maxVal-minVal)


def return_flattened_modis_and_chirps():

    modis_file = '/Volumes/sel_external/ethiopia_irrigation/modis/ethiopia_modis_06012011-06012021_1000m.tif'
    chirps_file = '/Volumes/sel_external/ethiopia_irrigation/chirps/temporal_stacks/ethiopia_16day_05152011_06152021_1000m.tif'

    shift = 1

    # LOAD CHIRPS
    with rasterio.open(chirps_file, 'r') as src:
        chirps_img = src.read()
        print(chirps_img.shape)

        chirps_img_reshaped = np.zeros(chirps_img.shape)
        chirps_img_reshaped[0:shift] = chirps_img[-shift::]
        chirps_img_reshaped[shift::] = chirps_img[0:-shift]

        # LOAD MODIS
    with rasterio.open(modis_file, 'r') as src:
        modis_img = src.read()
        meta = src.meta
        valid_px = np.where(~np.all(modis_img == src.meta['nodata'], axis=0))

    # RESHAPE
    modis_img = np.transpose(modis_img, (1,2,0))
    chirps_img_reshaped = np.transpose(chirps_img_reshaped, (1,2,0))


    return modis_img[valid_px], chirps_img_reshaped[valid_px], valid_px, meta


def load_dark_endmember():
    fn = '/Volumes/sel_external/ethiopia_irrigation/modis/4_tEMs_single_evergreen_double_dark.txt'

    df = pd.read_fwf(fn, skiprows=6, names=['ix', 'single', 'evergreen', 'double', 'dark'], delimeter='\t')

    return np.array((df['dark']))


def flattened_image_unmixing():

    modis_img, chirps_img_reshaped, valid_px, meta = return_flattened_modis_and_chirps()

    dark_em = load_dark_endmember()


    print(f'MODIS Shape: {modis_img.shape}')
    print(f'CHIRPS Shape: {chirps_img_reshaped.shape}')

    # modis_img = np.clip(modis_img, 0, 10000)

    amap = UCLS

    mse_img = np.zeros((1, meta['height'], meta['width']))
    maxe_img = np.zeros((1, meta['height'], meta['width']))
    mse_norm_img = np.full((1, meta['height'], meta['width']), np.nan)
    maxe_img_norm_img = np.full((1, meta['height'], meta['width']), np.nan)

    mean_evi = np.full((meta['height'], meta['width']), np.nan)
    max_evi = np.full((meta['height'], meta['width']), np.nan)
    std_evi = np.full((meta['height'], meta['width']), np.nan)

    for ix in tqdm(range(modis_img.shape[0])):
        evi = modis_img[ix][None,...]

        tems = chirps_img_reshaped[ix][None, ...]
        # tems = np.stack((chirps_img_reshaped[ix], dark_em), axis=0)

        # NORMALIZE
        evi = normalize(evi)
        tems = normalize(tems)


        # abundance_map_slice = amap(evi, tems)
        abundance_map_slice = 0
        mse_array, max_error = calculate_error(evi, tems, abundance_map_slice)
    #
    #     mse_img[0, valid_px[0][ix], valid_px[1][ix]] = mse_array
    #     mse_norm_img[0, valid_px[0][ix], valid_px[1][ix]] = mse_array/(np.mean(evi) + np.finfo(float).eps)
    #
        maxe_img[0, valid_px[0][ix], valid_px[1][ix]] = max_error
        maxe_img_norm_img[0, valid_px[0][ix], valid_px[1][ix]] = max_error / (np.mean(evi) + np.finfo(float).eps)

    # mean_evi[valid_px] = np.mean(modis_img, axis=-1)
    # max_evi[valid_px] = np.percentile(modis_img, 90, axis=-1)# - np.percentile(modis_img, 10, axis=-1)
    # std_evi[valid_px] = np.std(modis_img, axis=-1)

    mean_evi = mean_evi[None,...]
    max_evi = max_evi[None,...]
    std_evi = std_evi[None,...]


    out_name_mse = '/Volumes/sel_external/ethiopia_irrigation/modis/unmixed/modis_unmixed_pixelwise_chirps_mse.tif'
    out_name_maxe = '/Volumes/sel_external/ethiopia_irrigation/modis/unmixed/norm_modis_nomixing_pixelwise_chirps_shift_1_maxe_90.tif'
    out_name_mse_norm = '/Volumes/sel_external/ethiopia_irrigation/modis/unmixed/modis_unmixed_pixelwise_chirps_mse_norm.tif'
    out_name_maxe_norm = '/Volumes/sel_external/ethiopia_irrigation/modis/unmixed/norm_modis_nomixing_pixelwise_chirps_shift_1_maxe_90_norm.tif'

    out_name_mean = '/Volumes/sel_external/ethiopia_irrigation/modis/evi_stats/modis_mean_evi_1000m.tif'
    out_name_max = '/Volumes/sel_external/ethiopia_irrigation/modis/evi_stats/modis_90perc_evi_1000m.tif'
    out_name_std = '/Volumes/sel_external/ethiopia_irrigation/modis/evi_stats/modis_std_evi_1000m.tif'


    meta['count'] = 1
    meta['dtype'] = 'float32'
    meta['nodata'] = 'NaN'

    # with rasterio.open(out_name_mse, 'w', **meta) as dest:
    #     dest.write(mse_img.astype(np.float32))
    # with rasterio.open(out_name_mse_norm, 'w', **meta) as dest:
    #     dest.write(mse_norm_img.astype(np.float32))
    #
    with rasterio.open(out_name_maxe, 'w', **meta) as dest:
        dest.write(maxe_img.astype(np.float32))
    with rasterio.open(out_name_maxe_norm, 'w', **meta) as dest:
        dest.write(maxe_img_norm_img.astype(np.float32))

    # with rasterio.open(out_name_mean, 'w', **meta) as dest:
    #     dest.write(mean_evi.astype(np.float32))
    # with rasterio.open(out_name_max, 'w', **meta) as dest:
    #     dest.write(max_evi.astype(np.float32))
    # with rasterio.open(out_name_std, 'w', **meta) as dest:
    #     dest.write(std_evi.astype(np.float32))



def calculate_error(image_stack, endmember_array, abundance_map):

    # Mean square error calculation
    # image_recreated = np.matmul(abundance_map, endmember_array)

    image_recreated = endmember_array


    mse_array = ((image_stack - image_recreated)**2).sum(axis = -1)

    max_error = np.percentile(image_stack - image_recreated, 90)

    return mse_array, max_error


if __name__  == '__main__':
    flattened_image_unmixing()