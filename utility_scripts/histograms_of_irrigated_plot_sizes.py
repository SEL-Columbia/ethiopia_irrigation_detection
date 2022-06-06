import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from rasterio.mask import mask
import os
import pandas as pd
import seaborn as sns

colors_xkcd = ['very dark purple', "windows blue", "amber", "greyish",
                   "faded green", 'pastel blue', 'grape', 'dark turquoise', 'terracotta',
                   'salmon pink', 'evergreen'
                   ]
cmap = sns.color_palette(sns.xkcd_palette(colors_xkcd), desat=1)
sns.set_palette(sns.xkcd_palette(colors_xkcd))
# sns.set_style("whitegrid")


def compress_envi_segmented_output(region, year):
    min_px = 10

    tif_dir = f'/Volumes/sel_external/ethiopia_irrigation/images_for_inference/{region}/{year}/merged'
    tif_ext = f'{region}_{year}_merged_predictions_segmentation_output_queens_{min_px}px_min.tif'
    tif_fn = f'{tif_dir}/{tif_ext}'
    poly_fn = f'/Volumes/sel_external/ethiopia_irrigation/additional_files/admin_shapefiles/{region}_admin1.geojson'
    poly = gpd.read_file(poly_fn).to_crs('EPSG:32637')['geometry'].iloc[0]

    print(f'Reading img for {region}, {year}')
    with rasterio.open(tif_fn, 'r') as src:

        nodata_val = 4294967295
        img, trans = mask(src, [poly], nodata=nodata_val, crop=True)
        meta = src.meta
        print(meta)

        meta['transform'] = trans
        meta['height'] = img.shape[1]
        meta['width'] = img.shape[2]
        meta["compress"] = 'lzw'
        meta["nodata"] = nodata_val
        meta["BIGTIFF"] = 'YES'
        meta['driver'] = 'GTiff'

    out_tif_dir = f'{tif_dir}/compressed'
    if not os.path.exists(out_tif_dir):
        os.mkdir(out_tif_dir)

    out_file = f'{out_tif_dir}/{tif_ext}'

    print('Write out')
    with rasterio.open(out_file, 'w', **meta) as dest:
        dest.write(img)


def histogram_of_envi_segmented_output(year, region):
    min_px = 1

    tif_dir = f'/Volumes/sel_external/ethiopia_irrigation/images_for_inference/{region}/{year}/merged/compressed'
    tif_fn = f'{tif_dir}/{region}_{year}_merged_predictions_segmentation_output_queens_{min_px}px_min.tif'
    poly_fn = f'/Volumes/sel_external/ethiopia_irrigation/additional_files/admin_shapefiles/{region}_admin1.geojson'
    poly = gpd.read_file(poly_fn).to_crs('EPSG:32637')['geometry'].iloc[0]

    print(f'Reading img for {region}, {year}')
    with rasterio.open(tif_fn, 'r') as src:

        nodata_val = 0
        img, trans = mask(src, [poly], nodata=nodata_val, crop=True)
        meta = src.meta

        print('Finding max value')
        img = img[0]

        valid_px = np.where(img != nodata_val)
        valid_img = img[valid_px]
        max_val = np.max(valid_img)

        print(f'Max segment value: {max_val}')
        print(f'Number of unique segment value: {len(np.unique(valid_img))}')


        print(f'Valid image shape: {valid_img.shape}')

        print('Bin counting')
        bin_counter = np.bincount(valid_img)

        print('Non-zero bin values')

        nonzero_counter_ha = sorted(bin_counter[np.where(bin_counter != 0)]/100)

        print(f'bin_counter: {bin_counter}')
        print(f'non_zero_counter: {nonzero_counter_ha}')
        out_name = f'additional_data/prediction_histogram_results/{region}_{year}_predictions_bincount_area_ha.csv'

        df_bc = pd.DataFrame(nonzero_counter_ha, columns=['plot_size']).to_csv(out_name)
        # df_nzc = pd.DataFrame(bin_counter).to_csv(f'additional_data/prediction_histogram_results/{region}_nonzero_counter_ha.csv')


        return nonzero_counter_ha


def plot_segmentation_area_cdf():

    fig, [ax1, ax0] = plt.subplots(nrows=2, ncols=1, figsize=(8,8))
    years = [2020, 2021]

    regions = ['amhara', 'tigray']

    for region in regions:

        for year in years:
            print(f'{region}, {year}')

            nonzero_area_counter_ha = pd.read_csv(f'additional_data/prediction_histogram_results/'
                                                  f'{region}_{year}_predictions_bincount_area_ha.csv')

            nonzero_area_counter_ha = np.array(nonzero_area_counter_ha['plot_size'])
            # valid_ix = np.where(nonzero_area_counter_ha >= 0.1)
            # nonzero_area_counter_ha = nonzero_area_counter_ha[valid_ix]

            print(nonzero_area_counter_ha[-1])

            total_segments = len(nonzero_area_counter_ha)
            frac_area_above_ten = np.sum(nonzero_area_counter_ha[np.where(nonzero_area_counter_ha>=0.1)]) / \
                             np.sum(nonzero_area_counter_ha)

            print('area')
            print(1-frac_area_above_ten)

            total_area = int(np.round(np.sum(nonzero_area_counter_ha)))
            total_segments_above_ten = np.count_nonzero(nonzero_area_counter_ha>=0.1) / len(nonzero_area_counter_ha)

            print('segments')
            print(1-total_segments_above_ten)



            ax0.plot(np.log10(sorted(nonzero_area_counter_ha)), np.linspace(0,1,len(nonzero_area_counter_ha)) ,
                    label=f'{region.capitalize()} {year} (Total irrigated segments: {total_segments})')
                          # f'Fraction of area above 0.1 Ha: {np.round(frac_area_above_ten, 2)})')


            ax1.plot(np.log10(sorted(nonzero_area_counter_ha)),
                     np.cumsum(sorted(nonzero_area_counter_ha))/np.sum(nonzero_area_counter_ha),
                     label=f'{region.capitalize()} {year} (Total irrigated area: {total_area} Ha)')
                           # f'Fraction of segments above 0.1 Ha: {np.round(total_segments_above_ten, 2)})')



    for ax in [ax0, ax1]:

        ax.grid(True)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xlabel('Log10 of Irrigated Segment Area (Ha)', fontsize=12)

    ax0.set_ylabel('Fraction of Total Irrigated Segments', fontsize=12)
    ax1.set_ylabel('Fraction of Total Irrigated Area', fontsize=12)


    fig.tight_layout()
    plt.show()

if __name__ == '__main__':

    region = 'tigray'
    years=[2020, 2021]

    plot_segmentation_area_cdf()


    # region = 'tigray'
    # year = 2021
    # histogram_of_envi_segmented_output(year, region)
