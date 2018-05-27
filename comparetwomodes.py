import numpy as np
import matplotlib.pyplot as plt
from pitchdistamr_utils import load_data
from pitchdistamr_utils import get_args_compare_two_modes


def main(first_mode,
         second_mode,
         number_of_bins,
         first_last_pct,
         features_csv,
         classes_csv):
    """
        This method plots the average pitch histograms of two given modes

        Parameters
        ----------
        first_mode : str
            Name of the first mode to plot pitch histogram
        second_mode : str
            Name of the second mode to plot pitch histogram
        number_of_bins : int
            Number of comma values to divide between 0 and 1200 cent
        first_last_pct : int
            Whether to include the first and the last sections
        features_csv : str
            Name of the csv file containing feature values of instances
        classes_csv : str
            Name of the csv file containing class values of instances
    """
    data = load_data(features_csv=features_csv,
                     classes_csv=classes_csv)
    features = data['features']
    classes = data['classes']

    legend_properties = {'weight': 'bold'}

    fig = plt.figure(figsize=(15, 4))

    ax1 = fig.add_subplot(131)
    avg_f_e = np.mean(features[np.where((classes == first_mode))],
                      axis=0)[:number_of_bins]
    ax1.plot(avg_f_e, color='r', linestyle='--', label=first_mode)
    avg_s_e = np.mean(features[np.where((classes == second_mode))],
                      axis=0)[:number_of_bins]
    ax1.plot(avg_s_e, color='b', label=second_mode)
    ax1.tick_params(axis='both', which='both', bottom='off',
                    labelbottom='off', right='off', left='off',
                    labelleft='off')
    ax1.set_ylabel('Relative Occurances', labelpad=10, weight='bold')
    ax1.set_xlabel('Pitch class with respect to tonic (cents)',
                   labelpad=10, weight='bold')
    ax1.legend(prop=legend_properties, loc='best')
    ax1.set_title('Entire Recording', weight='bold')

    if first_last_pct == 1:
        ax2 = fig.add_subplot(132)
        avg_f_f = np.mean(features[np.where((classes == first_mode))],
                          axis=0)[number_of_bins:number_of_bins * 2]
        ax2.plot(avg_f_f, color='r', linestyle='--', label=first_mode)
        avg_s_f = np.mean(features[np.where((classes == second_mode))],
                          axis=0)[number_of_bins:number_of_bins * 2]
        ax2.plot(avg_s_f, color='b', label=second_mode)
        plt.xticks([])
        ax2.tick_params(axis='both', which='both', bottom='off',
                        labelbottom='off', right='off', left='off',
                        labelleft='off')
        ax2.set_ylabel('Relative Occurances', labelpad=10, weight='bold')
        ax2.set_xlabel('Pitch class with respect to tonic (cents)',
                       labelpad=10, weight='bold')
        ax2.legend(prop=legend_properties, loc='best')
        ax2.set_title('First Section', weight='bold')

        ax3 = fig.add_subplot(133)
        avg_f_l = np.mean(features[np.where((classes == first_mode))],
                          axis=0)[number_of_bins * 2:]
        ax3.plot(avg_f_l, color='r', linestyle='--', label=first_mode)
        avg_s_l = np.mean(features[np.where((classes == second_mode))],
                          axis=0)[number_of_bins * 2:]
        ax3.plot(avg_s_l, color='b', label=second_mode)
        ax3.tick_params(axis='both', which='both', bottom='off',
                        labelbottom='off', right='off', left='off',
                        labelleft='off')
        ax3.set_ylabel('Relative Occurances', labelpad=10, weight='bold')
        ax3.set_xlabel('Pitch class with respect to tonic (cents)',
                       labelpad=10, weight='bold')
        ax3.legend(prop=legend_properties, loc='best')
        ax3.set_title('Last Section', weight='bold')

    plt.show()


if __name__ == "__main__":
    args = get_args_compare_two_modes()
    main(first_mode=args['first_mode'],
         second_mode=args['second_mode'],
         number_of_bins=args['number_of_bins'],
         first_last_pct=args['first_last_pct'],
         features_csv=args['features_csv'],
         classes_csv=args['classes_csv']
         )
