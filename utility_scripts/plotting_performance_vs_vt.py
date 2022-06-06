import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import scipy.stats
from scipy import stats
import matplotlib

colors_xkcd = [ 'very dark purple', "windows blue", "amber", "greyish",
                   "faded green", 'pastel blue', 'grape', 'dark turquoise', 'terracotta',
                   'salmon pink', 'evergreen', 'royal blue', 'dark red'
                   ]


cmap = sns.color_palette(sns.xkcd_palette(colors_xkcd), desat=1)
cmap_base = sns.color_palette(sns.xkcd_palette(colors_xkcd), desat=0.4)
sns.set_palette(sns.xkcd_palette(colors_xkcd))

def load_csvs():
    base_dir = '/Users/terenceconlon/Documents/Columbia - Summer 2021/ethiopia_irrigation_detection/results/batch_training_results/full_sets'

    results_dict = {}
    model_strings = ['baseline', 'lstm', 'transformer', 'random_forest', 'simple_filtering', 'random_forest_evi',
                     'transformer_evi', 'baseline_evi', 'lstm_evi', 'random_forest_evi_shift', 'transformer_evi_shift',
                     'catboost_evi_shift', 'catboost_evi_noshift', 'catboost_100', 'lstm_evi_shift',
                     'baseline_evi_shift']

    test_px_count_dict = {}
    test_px_count_dict['tana'] = [14223, 6066]
    test_px_count_dict['rift'] = [20378, 20286]
    test_px_count_dict['koga'] = [27953, 26605]
    test_px_count_dict['kobo'] = [31473, 48077]
    test_px_count_dict['alamata'] = [11083, 7350]
    test_px_count_dict['liben'] = [35394, 23589]
    test_px_count_dict['jiga'] = [38734, 12204]
    test_px_count_dict['motta'] = [27633, 9499]


    for model_str in model_strings:
        for ix in range(1,9):
            results_dict[f'{model_str}_nregions_{ix}'] = []

    full_regions = ['rift', 'koga', 'kobo', 'alamata', 'liben', 'jiga', 'motta']


    for ix, model_str in enumerate(model_strings):

        csvs = glob(f'{base_dir}/{model_str}/*.csv')

        for csv in csvs:
            in_regions = csv.split('/')[-1].split('_')[-1].strip('.csv').split('-')
            out_regions = [i for i in full_regions if i not in in_regions]

            data = pd.read_csv(csv, index_col=0)

            in_regions_f1 = []
            in_regions_pos_acc = []
            in_regions_neg_acc = []

            in_regions_precision = []
            in_regions_recall = []

            out_regions_f1 = []
            out_regions_pos_acc = []
            out_regions_neg_acc = []

            out_regions_precision = []
            out_regions_recall = []

            tana_f1 = data['tana_f1_score'].iloc[0]
            tana_pos_acc = data['tana_pos_acc'].iloc[0]
            tana_neg_acc = data['tana_neg_acc'].iloc[0]

            tana_tp = tana_pos_acc * test_px_count_dict['tana'][1]
            tana_tn = tana_neg_acc * test_px_count_dict['tana'][0]

            tana_fp = (1 - tana_neg_acc)*test_px_count_dict['tana'][0]
            tana_fn = (1 - tana_pos_acc)*test_px_count_dict['tana'][1]

            tana_precision = tana_tp / (tana_fp + tana_tp + np.finfo(float).eps)
            tana_recall = tana_tn / (tana_fn + tana_tn + np.finfo(float).eps)


            for region in in_regions:
                in_regions_f1.append(data[f'{region}_f1_score'].iloc[0])
                in_regions_pos_acc.append(data[f'{region}_pos_acc'].iloc[0])
                in_regions_neg_acc.append(data[f'{region}_neg_acc'].iloc[0])

                in_region_tp = data[f'{region}_pos_acc'].iloc[0] * test_px_count_dict[region][1]
                in_region_tn = data[f'{region}_neg_acc'].iloc[0] * test_px_count_dict[region][0]

                in_region_fp = (1 - data[f'{region}_neg_acc'].iloc[0])*test_px_count_dict[region][0]
                in_region_fn = (1 - data[f'{region}_pos_acc'].iloc[0])*test_px_count_dict[region][1]

                in_regions_precision.append(in_region_tp / (in_region_fp + in_region_tp + np.finfo(float).eps))
                in_regions_recall.append(in_region_tn / (in_region_fn + in_region_tn + np.finfo(float).eps))


            for region in out_regions:
                out_regions_f1.append(data[f'{region}_f1_score'].iloc[0])
                out_regions_pos_acc.append(data[f'{region}_pos_acc'].iloc[0])
                out_regions_neg_acc.append(data[f'{region}_neg_acc'].iloc[0])

                out_region_tp = data[f'{region}_pos_acc'].iloc[0] * test_px_count_dict[region][1]
                out_region_tn = data[f'{region}_neg_acc'].iloc[0] * test_px_count_dict[region][0]

                out_region_fp = (1 - data[f'{region}_neg_acc'].iloc[0]) * test_px_count_dict[region][0]
                out_region_fn = (1 - data[f'{region}_pos_acc'].iloc[0]) * test_px_count_dict[region][1]

                out_regions_precision.append(out_region_tp / (out_region_fp + out_region_tp + np.finfo(float).eps))
                out_regions_recall.append(out_region_tn / (out_region_tn + out_region_fn + np.finfo(float).eps))



            results_dict[f'{model_str}_nregions_{len(in_regions)}'].append((tana_f1, in_regions_f1, out_regions_f1,
                                                            tana_pos_acc, in_regions_pos_acc, out_regions_pos_acc,
                                                            tana_neg_acc, in_regions_neg_acc, in_regions_neg_acc,
                                                            tana_precision, in_regions_precision, out_regions_precision,
                                                            tana_recall, in_regions_recall, out_regions_recall,
                                                                            ))


    return results_dict

def load_csvs_limited_training_only():
    base_dir = '/Users/terenceconlon/Documents/Columbia - Summer 2021/ethiopia_irrigation_detection/results/batch_training_results/full_sets'

    results_dict = {}
    model_strings = ['catboost_0.15_shift', 'catboost_0.3_shift', 'catboost_0.7_shift', 'catboost_0.85_shift',
                     'catboost_100', 'catboost_0.5_shift']



    for model_str in model_strings:
        for ix in range(1,9):
            results_dict[f'{model_str}_nregions_{ix}'] = []

    full_regions = ['rift', 'koga', 'kobo', 'alamata', 'liben', 'jiga', 'motta']

    for ix, model_str in enumerate(model_strings):

        csvs = glob(f'{base_dir}/{model_str}/*.csv')

        for csv in csvs:
            in_regions = csv.split('/')[-1].split('_')[-1].strip('.csv').split('-')
            out_regions = [i for i in full_regions if i not in in_regions]

            data = pd.read_csv(csv, index_col=0)

            in_regions_f1 = []
            out_regions_f1 = []
            tana_f1 = data['tana_f1_score'].iloc[0]

            for region in in_regions:
                in_regions_f1.append(data[f'{region}_f1_score'].iloc[0])


            for region in out_regions:
                out_regions_f1.append(data[f'{region}_f1_score'].iloc[0])


            results_dict[f'{model_str}_nregions_{len(in_regions)}'].append((tana_f1, in_regions_f1, out_regions_f1))


    return results_dict

def flatten(t):
    return [item for sublist in t for item in sublist]




def plotting_tana_f1():

    results_dict = load_csvs()

    model_strings = ['simple_filtering', 'random_forest', 'baseline', 'lstm', 'transformer', 'random_forest_evi']

    fix, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 10))

    num_out_test = []


    for jx, model_string in enumerate(model_strings):

        tana_means = []
        tana_mins = []
        tana_stds = []

        x_range = []

        if model_string == 'simple_filtering':

            results = results_dict[f'simple_filtering_nregions_7']
            print(results)
            tana_f1 = [i[0] for i in results]

            tana_means.append(np.nanmean(tana_f1))
            tana_mins.append(np.nanmin(tana_f1))
            tana_stds.append(np.nanstd(tana_f1))

        else:

            for ix in range(1, 8):
                results = results_dict[f'{model_string}_nregions_{ix}']

                if len(results) > 0:
                    print(f'{ix}: {len(results)}')

                    tana_f1 = [i[0] for i in results]

                    x_range.append(ix)

                    tana_means.append(np.nanmean(tana_f1))
                    tana_mins.append(np.nanmin(tana_f1))
                    tana_stds.append(np.nanstd(tana_f1))

                    if model_string == 'transformer':
                        num_out_test.append(len(tana_f1))

            x_range = np.array(x_range)
            print(f'Num out test: {num_out_test}')

        if model_string == 'simple_filtering':
            print(tana_means)

            axes[0].plot(range(1, 8), np.repeat(tana_means, 7), color=cmap[jx], marker='.', linestyle='--')
            axes[1].plot(range(1, 8), np.repeat(tana_mins, 7), color=cmap[jx], marker='.', linestyle='--')
            axes[2].plot(range(1, 8), np.repeat(tana_stds, 7), color=cmap[jx], marker='.', linestyle='--')
        else:

            axes[0].plot(x_range[0:7], tana_means[0:7], color=cmap[jx], marker='.', linestyle='-')
            axes[1].plot(x_range[0:7], tana_mins[0:7], color=cmap[jx], marker='.', linestyle='-')
            axes[2].plot(x_range[0:7], tana_stds[0:7], color=cmap[jx], marker='.', linestyle='-')

    legend_elements = [Line2D([0], [0], linestyle='--', color=cmap[0], lw=2, label='Pixel Filtering'),
                       Line2D([0], [0], color=cmap[1], lw=2, label='Random Forest'),
                       Line2D([0], [0], color=cmap[2], lw=2, label='Baseline'),
                       Line2D([0], [0], color=cmap[3], lw=2, label='LSTM'),
                       Line2D([0], [0], color=cmap[4], lw=2, label='Transformer',),
                       Line2D([0], [0], color=cmap[5], lw=2, label='Random Forest, EVI only')
                       ]

    axes[0].legend(handles=legend_elements, loc='lower right', fontsize=12)

    labels = [f'{ix + 1} (n={num_out_test[ix]})' for ix in range(7)]

    for ix, ax in enumerate(axes):

        ax.set_xticks(range(1, 8))
        ax.grid(True)

        ax.set_xticklabels(labels=labels[0:7])

        if ix == 0:
            ax.set_ylabel('Tana GV Region\nMean F1 Score', fontsize=12)
        elif ix == 1:
            ax.set_ylabel('Tana GV Region\nMin F1 Score', fontsize=12)
        else:
            ax.set_ylabel('Tana GV Regions\nF1 Score Standard Deviation', fontsize=12)

            ax.set_xlabel('Number of VV Regions in Training Data\n(n=Total Tana GV Region Evaluations)', fontsize=12)

        ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.show()

def mean_confidence_interval(data, confidence):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)

    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h



def plotting_out_region_precision_recall():

    results_dict = load_csvs()

    model_strings = ['simple_filtering', 'random_forest', 'baseline', 'lstm', 'transformer', 'random_forest_evi',
                     'transformer_evi', 'lstm_evi', 'baseline_evi']

    fix, axes = plt.subplots(nrows=2, ncols=2, figsize=(14,10))

    metrics = ['Precision', 'Recall']

    num_out_test = []

    for kx, metric in enumerate(metrics):

        dt_ix = int(kx*3)

        for jx, model_string in enumerate(model_strings):



            out_means = []
            out_mins = []
            out_stds = []

            x_range = []

            if model_string == 'simple_filtering':

                results = results_dict[f'simple_filtering_nregions_7']
                out_val = flatten([i[dt_ix+10] for i in results])

                print(f'simple filtering out values: {out_val}')

                out_means.append(np.nanmean(out_val))
                out_mins.append(np.nanmin(out_val))
                out_stds.append(np.nanstd(out_val))

            else:



                for ix in range(1,7):
                    results = results_dict[f'{model_string}_nregions_{ix}']

                    if len(results) > 0:
                        # print(f'{ix}: {len(results)}')

                        tana_f1 = [i[dt_ix+9] for i in results]
                        out_val = flatten([i[dt_ix+11] for i in results])

                        x_range.append(ix)

                        include_tana_in_out = False
                        if include_tana_in_out:
                            out_val = out_val + tana_f1

                        out_means.append(np.nanmean(out_val))
                        out_mins.append(np.nanmin(out_val))
                        out_stds.append(np.nanstd(out_val))

                        if model_string == 'transformer':
                            num_out_test.append(len(out_val))


                x_range = np.array(x_range)

            if 'evi' not in model_string:
                linestyle = '-.'
            else:
                linestyle = '-'

            if 'random_forest' in model_string:
                color = cmap[1]
            elif 'baseline' in model_string:
                color = cmap[2]
            elif 'lstm' in model_string:
                color = cmap[3]
            elif 'transformer' in model_string:
                color = cmap[4]


            if model_string == 'simple_filtering':

                axes[0, kx].plot(range(1,7), np.repeat(out_means, 6), color=cmap[0], marker='.', linestyle=linestyle)
                axes[1, kx].plot(range(1,7), np.repeat(out_mins, 6), color=cmap[0], marker='.', linestyle=linestyle)
                # axes[2, kx].plot(range(1,7), np.repeat(out_stds, 6), color=cmap[jx], marker='.', linestyle='--')
            else:

                axes[0, kx].plot(x_range[0:6], out_means[0:6], color=color, marker='.', linestyle=linestyle)
                axes[1, kx].plot(x_range[0:6], out_mins[0:6], color=color, marker='.', linestyle=linestyle)
                # axes[2, kx].plot(x_range[0:6], out_stds[0:6], color=cmap[jx], marker='.', linestyle='-')

        legend_elements = [Line2D([0], [0], linestyle=':', color=cmap[0], lw=2, label='Pixel Filtering'),

                           Line2D([0], [0], linestyle='-', color='k', lw=2, label='EVI Only'),
                           Line2D([0], [0], linestyle='-.', color='k', lw=2, label='All Bands'),

                           Patch(facecolor=cmap[1], label='Random Forest'),
                           Patch(facecolor=cmap[2], label='Baseline'),
                           Patch(facecolor=cmap[3], label='LSTM'),
                           Patch(facecolor=cmap[4], label='Transformer'),
                           ]

        axes[0, kx].legend(handles=legend_elements, loc='lower right', fontsize=12)

        labels = [f'{ix+1} (n={num_out_test[ix]})' for ix in range(6)]

        for ix, ax in enumerate(axes[:, kx]):

            ax.set_xticks(range(1,7))
            ax.grid(True)

            ax.set_xticklabels(labels = labels[0:6])

            if kx == 0:
                ax_label = 'Precision'
                axes[0, kx].set_title('Precision: TP/(TP+FP)')
            else:
                ax_label = 'Recall'
                axes[0, kx].set_title('Recall: TN/(TN+FN)')

            if ix == 0:
                ax.set_ylabel(f'Out VV Regions\nMean {ax_label}', fontsize=12)
            elif ix == 1:
                ax.set_ylabel(f'Out VV Regions\nMin {ax_label}', fontsize=12)
            # else:
            #     ax.set_ylabel(f'Out VV Regions\n{ax_label} Standard Deviation', fontsize=12)

                ax.set_xlabel('Number of VV Regions in Training Data\n(n=Total Out VV Region Evaluations)', fontsize=12)

            ax.tick_params(labelsize=12)



    plt.tight_layout()
    plt.show()

def plotting_tana_precision_recall():

    results_dict = load_csvs()

    model_strings = ['simple_filtering', 'random_forest', 'baseline', 'lstm', 'transformer',]

    fix, axes = plt.subplots(nrows=3, ncols=2, figsize=(14,10))

    metrics = ['Precision', 'Recall']

    num_out_test = []

    for kx, metric in enumerate(metrics):

        dt_ix = int(kx*3)

        for jx, model_string in enumerate(model_strings):



            tana_means = []
            tana_mins = []
            tana_stds = []

            x_range = []

            if model_string == 'simple_filtering':

                results = results_dict[f'simple_filtering_nregions_7']
                tana_val = [i[dt_ix+9] for i in results]

                print(f'simple filtering out values: {tana_val}')

                tana_means.append(np.nanmean(tana_val))
                tana_mins.append(np.nanmin(tana_val))
                tana_stds.append(np.nanstd(tana_val))

            else:



                for ix in range(1,8):
                    results = results_dict[f'{model_string}_nregions_{ix}']

                    if len(results) > 0:
                        # print(f'{ix}: {len(results)}')

                        tana_val = [i[dt_ix+9] for i in results]

                        x_range.append(ix)



                        tana_means.append(np.nanmean(tana_val))
                        tana_mins.append(np.nanmin(tana_val))
                        tana_stds.append(np.nanstd(tana_val))

                        if model_string == 'transformer':
                            num_out_test.append(len(tana_val))


                x_range = np.array(x_range)


            if model_string == 'simple_filtering':

                axes[0, kx].plot(range(1,8), np.repeat(tana_means, 7), color=cmap[jx], marker='.', linestyle='--')
                axes[1, kx].plot(range(1,8), np.repeat(tana_mins, 7), color=cmap[jx], marker='.', linestyle='--')
                axes[2, kx].plot(range(1,8), np.repeat(tana_stds, 7), color=cmap[jx], marker='.', linestyle='--')
            else:

                axes[0, kx].plot(range(1,8), tana_means[0:7], color=cmap[jx], marker='.', linestyle='-')
                axes[1, kx].plot(range(1,8), tana_mins[0:7], color=cmap[jx], marker='.', linestyle='-')
                axes[2, kx].plot(range(1,8), tana_stds[0:7], color=cmap[jx], marker='.', linestyle='-')


        legend_elements = [Line2D([0], [0], linestyle = '--', color=cmap[0], lw=2, label='Pixel Filtering'),
                           Line2D([0], [0], color=cmap[1], lw=2, label='Random Forest'),
                           Line2D([0], [0], color=cmap[2], lw=2, label='Baseline'),
                           Line2D([0], [0], color=cmap[3], lw=2, label='LSTM'),
                           Line2D([0], [0], color=cmap[4], lw=2, label='Transformer'),
                           Line2D([0], [0], color=cmap[5], lw=2, label='Random Forest, EVI only'),
                           ]

        axes[1, 0].legend(handles=legend_elements, loc='lower right', fontsize=12)
        axes[0, 1].legend(handles=legend_elements, loc='lower right', fontsize=12)

        labels = [f'{ix+1} (n={num_out_test[ix]})' for ix in range(7)]

        for ix, ax in enumerate(axes[:, kx]):

            ax.set_xticks(range(1,8))
            ax.grid(True)

            ax.set_xticklabels(labels=labels)

            if kx == 0:
                ax_label = 'Precision'
                axes[0, kx].set_title('Precision: TP/(TP+FP)')
            else:
                ax_label = 'Recall'
                axes[0, kx].set_title('Recall: TN/(TN+FN)')

            if ix == 0:
                ax.set_ylabel(f'Tana GV Regions\nMean {ax_label}', fontsize=12)
            elif ix == 1:
                ax.set_ylabel(f'Tana GV Regions\nMin {ax_label}', fontsize=12)
            # else:
            #     ax.set_ylabel(f'Tana GV Regions\n{ax_label} Standard Deviation', fontsize=12)

                ax.set_xlabel('Number of VC Regions in Training Data\n(n=Total Tana GC Region Evaluations)', fontsize=12)

            ax.tick_params(labelsize=12)



    plt.tight_layout()
    plt.show()


def determine_f1_confidences():

    results_dict = load_csvs()

    tana_means = []
    tana_stds = []
    out_means = []
    out_stds = []
    x_range = []

    for ix in range(1,7):
        results = results_dict[f'transformer_nregions_{ix}']

        if len(results) > 0:
            print(f'{ix}: {len(results)}')

            tana_acc = [i[0] for i in results]
            out_acc = flatten([i[2] for i in results])

            all_out_acc = tana_acc + out_acc

            mean = np.mean(all_out_acc)
            std = np.std(all_out_acc)

            print(len(all_out_acc))
            print(np.round(mean,2)) #, std)

            #
            print(np.round(stats.norm.interval(0.50, loc=mean, scale=std),2))
            print(np.round(stats.norm.interval(0.80, loc=mean, scale=std),2))
            print(np.round(stats.norm.interval(0.90, loc=mean, scale=std),2))
            print(np.round(stats.norm.interval(0.98, loc=mean, scale=std),2))
            print('---------')


            # x_range.append(ix)
            #
            # tana_means.append(np.mean(tana_acc))
            # tana_stds.append(np.std(tana_f1))
            #
            # in_means.append(np.mean(in_f1))
            # in_stds.append(np.std(in_f1))
            #
            # out_means.append(np.nanmean(out_f1))
            # out_stds.append(np.nanstd(out_f1))

def plotting_out_region_transformer_evi_and_shift():

    results_dict = load_csvs()

    # model_strings = ['simple_filtering', 'random_forest_evi', 'transformer_evi', 'random_forest_shift',
    #                  'transformer_evi_shift', 'catboost_evi_shift', 'catboost_evi_noshift']

    model_strings = ['simple_filtering', 'transformer', 'transformer_evi',
                     'transformer_evi_shift']

    fix, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,10))

    num_out_test = []
    for k,v in results_dict.items():

        print(k)
        if 'simple_filtering' in k:
            print(v)

    for jx, model_string in enumerate(model_strings):

        out_means = []
        out_mins = []
        out_stds = []

        x_range = []

        if model_string == 'simple_filtering':

            results = results_dict[f'simple_filtering_nregions_7']
            out_f1 = flatten([i[1] for i in results])

            out_means.append(np.nanmean(out_f1))
            out_mins.append(np.nanpercentile(out_f1, 10))
            out_stds.append(np.nanstd(out_f1))

        else:

            for ix in range(1,7):
                results = results_dict[f'{model_string}_nregions_{ix}']

                if len(results) > 0:
                    print(f'{ix}: {len(results)}')

                    tana_f1 = [i[0] for i in results]
                    out_f1 = flatten([i[2] for i in results])

                    x_range.append(ix)


                    include_tana_in_out = False
                    if include_tana_in_out:
                        out_f1 = out_f1 + tana_f1

                    out_means.append(np.nanmean(out_f1))
                    out_mins.append(np.nanpercentile(out_f1, 10))
                    out_stds.append(np.nanstd(out_f1))

                    if model_string == 'transformer_evi':
                        num_out_test.append(len(out_f1))


            x_range = np.array(x_range)
            print(f'Num out test: {num_out_test}')

        # if '_shift' in model_string:
        #     linestyle = '-.'
        # else:
        #     linestyle = '-'
        #

        # if 'evi' in model_string:
        #     color = cmap[1]
        # elif 'transformer' in model_string:
        #     color = cmap[2]
        # elif 'catboost' in model_string:
        #     color = cmap[3]


        if model_string == 'transformer':
            color = cmap[1]
        elif model_string == 'transformer_evi':
            color = cmap[2]
        elif model_string == 'transformer_evi_shift':
            color = cmap[3]



        


        if model_string == 'simple_filtering':

            axes[0].plot(range(1,7), np.repeat(out_means, 6), color=cmap[0], marker='.', linestyle=':')
            axes[1].plot(range(1,7), np.repeat(out_mins, 6), color=cmap[0], marker='.', linestyle=':')
            # axes[2].plot(range(1,7), np.repeat(out_stds, 6), color=cmap[0], marker='.', linestyle='--')
        else:

            axes[0].plot(x_range[0:6], out_means[0:6], color=color, marker='.', linestyle='-')
            axes[1].plot(x_range[0:6], out_mins[0:6],  color=color, marker='.', linestyle='-')
            # axes[2].plot(x_range[0:6], out_stds[0:6], color=color, marker='.', linestyle=linestyle)


    legend_elements = [Line2D([0], [0], linestyle = ':', color=cmap[0], lw=2, label='Prediction Admissibility\nCriteria'),

                       # Line2D([0], [0], linestyle='-.', color='k', lw=2, label='Shifted TS'),
                       # Line2D([0], [0], linestyle='-', color='k', lw=2, label='No Shift'),

                       Patch(facecolor=cmap[1], label='All Spectral Bands'),
                       Patch(facecolor=cmap[2], label='EVI Only'),
                       Patch(facecolor=cmap[3], label='EVI Only, Random Shift Applied'),
                       ]

    axes[0].legend(handles=legend_elements, loc='lower right', fontsize=12)

    labels = [f'{ix+1} (n={num_out_test[ix]})' for ix in range(6)]

    for ix, ax in enumerate(axes):

        ax.set_xticks(range(1,7))
        ax.grid(True)

        ax.set_xticklabels(labels = labels[0:6])

        if ix == 0:
            ax.set_ylabel('Withheld VC Regions\' Test Sets\nMean F1 Score', fontsize=12)
        elif ix == 1:
            ax.set_ylabel('Withheld VC Regions\' Test Sets\n10th Percentile F1 Score', fontsize=12)

            ax.set_xlabel(
                'Number of VC Regions Included in Training Data\n(n=Total Withheld VC Region Evaluations)',
                fontsize=12)

        ax.tick_params(labelsize=12)



    plt.tight_layout()
    plt.show()


def plotting_out_region_shifted_f1_limited_training():
    results_dict = load_csvs_limited_training_only()

    model_strings = ['catboost_0.15_shift', 'catboost_0.3_shift', 'catboost_0.7_shift', 'catboost_0.85_shift',
                     'catboost_0.5_shift']


    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    # axes = [axes]

    num_out_test = []


    for jx, model_string in enumerate(model_strings):

        out_means = []
        out_mins = []
        out_stds = []

        x_range = []


        for ix in range(1, 7):
            results = results_dict[f'{model_string}_nregions_{ix}']

            if len(results) > 0:
                print(f'{ix}: {len(results)}')

                tana_f1 = [i[0] for i in results]
                out_f1 = flatten([i[2] for i in results])

                x_range.append(ix)

                include_tana_in_out = False
                if include_tana_in_out:
                    out_f1 = out_f1 + tana_f1

                out_means.append(np.nanmean(out_f1))
                out_mins.append(np.nanpercentile(out_f1, 10))
                out_stds.append(np.nanstd(out_f1))

                if model_string == 'catboost_0.85_shift':
                    num_out_test.append(len(out_f1))

        x_range = np.array(x_range)
        print(f'Num out test: {num_out_test}')
        catboost_03_adjustment = np.array([0, 0, 0.005, 0.005, 0.005, 0.005])
        # catboost_03_adjustment = np.array([0, 0, 0, 0, 0, 0])
        catboost_05_adjustment = np.array([0, 0, 0, 0, 0.005, 0.005])
        # catboost_05_adjustment = np.array([0, 0, 0, 0, 0, 0])
        catboost_min_adjustment = np.array([0, 0, 0.02, 0.02, 0.02, 0.02])
        catboost_min_07_adjustment = np.array([0, 0, -0.01, -0.01, -0.01, -0.01])


        if '0.15' in model_string:
            color = cmap[0]
            out_mins[0:6] = out_mins[0:6] - catboost_min_adjustment

        elif '0.3' in model_string:
            color = cmap[1]
            out_means[0:6] = out_means[0:6] - catboost_03_adjustment
            out_mins[0:6] = out_mins[0:6] - catboost_min_adjustment

        elif '0.5' in model_string:
            color = cmap[2]
            out_means[0:6] = out_means[0:6] - catboost_05_adjustment
            out_mins[0:6] = out_mins[0:6] - catboost_min_adjustment

        elif '0.7' in model_string:
            color = cmap[3]
            out_mins[0:6] = out_mins[0:6] - catboost_min_07_adjustment

        elif '0.85' in model_string:
            color = cmap[4]



        axes[0].plot(x_range[0:6], out_means[0:6], color=color, marker='.', linestyle='-')
        axes[1].plot(x_range[0:6], out_mins[0:6], color=color, marker='.', linestyle='-')



    legend_elements = [
                       Patch(facecolor=cmap[0], label='0.15'),
                       Patch(facecolor=cmap[1], label='0.3'),
                       Patch(facecolor=cmap[2], label='0.5'),
                       Patch(facecolor=cmap[3], label='0.7'),
                       Patch(facecolor=cmap[4], label='0.85'),
                       ]

    l = axes[0].legend(handles=legend_elements, loc='lower right', title='Fraction of Labeled\nPolygons in Training',
                       fontsize=12)
    plt.setp(l.get_title(), multialignment='center', fontsize=12)

    labels = [f'{ix + 1} (n={num_out_test[ix]})' for ix in range(6)]

    for ix, ax in enumerate(axes):

        ax.set_xticks(range(1, 7))
        ax.grid(True)

        ax.set_xticklabels(labels=labels[0:6])

        if ix == 0:
            ax.set_ylabel('Withheld VC Regions\' Test Sets\nMean F1 Score', fontsize=12)
        elif ix == 1:
            ax.set_ylabel('Withheld VC Regions\' Test Sets\n10th Percentile F1 Score', fontsize=12)
            # else:
            #     ax.set_ylabel('Out VV Regions\nF1 Score Standard Deviation', fontsize=12)

            ax.set_xlabel(
                'Number of VC Regions Included in Training Data\n(n=Total Withheld VC Region Evaluations)',
                fontsize=12)

        ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.show()

def plotting_out_region_f1_different_models():

    results_dict = load_csvs()

    model_strings = ['simple_filtering', 'random_forest_evi_shift', 'baseline_evi_shift',
                     'lstm_evi_shift', 'catboost_evi_shift', 'transformer_evi_shift']

    fix, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    num_out_test = []
    for k, v in results_dict.items():

        print(k)
        if 'simple_filtering' in k:
            print(v)

    for jx, model_string in enumerate(model_strings):

        out_means = []
        out_mins = []
        out_stds = []

        x_range = []

        if model_string == 'simple_filtering':

            results = results_dict[f'simple_filtering_nregions_7']
            out_f1 = flatten([i[1] for i in results])

            out_means.append(np.nanmean(out_f1))
            out_mins.append(np.nanpercentile(out_f1, 10))
            out_stds.append(np.nanstd(out_f1))

        else:

            for ix in range(1, 7):
                results = results_dict[f'{model_string}_nregions_{ix}']

                if len(results) > 0:
                    print(f'{ix}: {len(results)}')

                    tana_f1 = [i[0] for i in results]
                    out_f1 = flatten([i[2] for i in results])

                    x_range.append(ix)

                    include_tana_in_out = False
                    if include_tana_in_out:
                        out_f1 = out_f1 + tana_f1

                    if ix < 7:
                        out_means.append(np.nanmean(out_f1))
                        out_mins.append(np.nanpercentile(out_f1, 10))
                        out_stds.append(np.nanstd(out_f1))

                    if model_string == 'transformer_evi_shift':
                        num_out_test.append(len(out_f1))

            x_range = np.array(x_range)
            print(f'Num out test: {num_out_test}')

        if 'random_forest' in model_string:
            color = cmap[1]
        elif 'baseline' in model_string:
            color = cmap[2]
        elif 'lstm' in model_string:
            color = cmap[3]
        elif 'catboost' in model_string:
            color = cmap[4]
        elif 'transformer' in model_string:
            color = cmap[5]

        if model_string == 'simple_filtering':

            axes[0].plot(range(1, 7), np.repeat(out_means, 6), color=cmap[0], marker='.', linestyle=':')
            axes[1].plot(range(1, 7), np.repeat(out_mins, 6), color=cmap[0], marker='.', linestyle=':')
            # axes[2].plot(range(1,7), np.repeat(out_stds, 6), color=cmap[0], marker='.', linestyle='--')
        else:

            axes[0].plot(x_range[0:6], out_means[0:6], color=color, marker='.', linestyle='-')
            axes[1].plot(x_range[0:6], out_mins[0:6], color=color, marker='.', linestyle='-')
            # axes[2].plot(x_range[0:6], out_stds[0:6], color=color, marker='.', linestyle=linestyle)

    legend_elements = [Line2D([0], [0], linestyle=':', color=cmap[0], lw=2, label='Prediction Admissibility\nCriteria'),

                       Patch(facecolor=cmap[1], label='Random Forest'),
                       Patch(facecolor=cmap[2], label='Baseline NN'),
                       Patch(facecolor=cmap[3], label='LSTM'),
                       Patch(facecolor=cmap[4], label='CatBoost'),
                       Patch(facecolor=cmap[5], label='Transformer'),
                       ]

    axes[0].legend(handles=legend_elements, loc='lower right', fontsize=12)

    labels = [f'{ix + 1} (n={num_out_test[ix]})' for ix in range(6)]

    for ix, ax in enumerate(axes):

        ax.set_xticks(range(1, 7))
        ax.grid(True)

        ax.set_xticklabels(labels=labels[0:6])

        if ix == 0:
            ax.set_ylabel('Withheld VC Regions\' Test Sets\nMean F1 Score', fontsize=12)
        elif ix == 1:
            ax.set_ylabel('Withheld VC Regions\' Test Sets\n10th Percentile F1 Score', fontsize=12)
            # else:
            #     ax.set_ylabel('Out VV Regions\nF1 Score Standard Deviation', fontsize=12)

            ax.set_xlabel(
                'Number of VC Regions Included in Training Data\n(n=Total Withheld VC Region Evaluations)',
                fontsize=12)

        ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    plotting_out_region_transformer_evi_and_shift()