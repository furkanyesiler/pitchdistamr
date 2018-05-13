import os
import csv
import argparse
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd


def get_args_download_makam_dataset():
    """ Argument parser for downloadmakamdataset.py

        Returns
        -------
        dictionary : dict
            directory : str
                Target directory to download the dataset into

    """
    parser = argparse.ArgumentParser(description='Downloads OTMM Makam '
                                                 'Recognition Dataset from '
                                                 'https://github.com/MTG/'
                                                 'otmm_makam_recognition_'
                                                 'dataset/')
    parser.add_argument('-d',
                        '--directory',
                        type=str,
                        help='Target directory for the downloaded files.'
                             'Type is string. Default is data/.',
                        default='data/'
                        )
    args = parser.parse_args()

    return {'directory': args.directory}


def get_args_compute_pitch_distribution():
    """ Argument parser for computepitchdistribution.py

        Returns
        -------
        dictionary : dict
            number_of_bins : int
                Number of comma values to divide between 0 and 1200 cent
            first_last_pct : int
                Whether to include the first and the last sections
            pct : float
                Percentage value for the first and the last sections
            annot : int
                Whether to use the annotations file
            pitch_files : int
                Whether to use the already extracted pitch files
            annot_tonic : int
                Whether to use the estimated tonic frequencies from
                the annotations file
            folder_dir : str
                Directory of the recordings
            features_save_name : str
                File name for storing feature values
            classes_save_name : str
                File name for storing class values
    """
    parser = argparse.ArgumentParser(description='This method computes pitch '
                                                 'distributions of target '
                                                 'recordings and aligns the '
                                                 'obtained distributions with'
                                                 ' respect to the tonic '
                                                 'frequency. The obtained '
                                                 'distributions are saved as '
                                                 'a csv file. To use the mode '
                                                 'information specified in '
                                                 'annotations file, annot '
                                                 'should be 1. To use the '
                                                 'already extracted pitch '
                                                 'values from the directory,'
                                                 ' pitch_files should be 1. '
                                                 'To use the already '
                                                 'estimated tonic frequencies'
                                                 ' included in annotations '
                                                 'file, annot_tonic should be'
                                                 ' 1. For the required folder'
                                                 ' structure, README file can '
                                                 'be referred.')
    parser.add_argument('-n',
                        '--number_of_bins',
                        type=int,
                        help='Number of comma values to divide between '
                             '0 and 1200 cent. Type is integer. '
                             'Default is 53.',
                        default=53)
    parser.add_argument('-f',
                        '--first_last_pct',
                        type=int,
                        default=1,
                        choices=(0, 1),
                        help='If 1, use the first and the last sections. '
                             'If 0, only the entire recording. '
                             'Default is 1.')
    parser.add_argument('-p',
                        '--pct',
                        type=int,
                        default=10,
                        help='Percentage value for the first and '
                             'the last sections. Type is integer.'
                             'Default is 10.')
    parser.add_argument('-a',
                        '--annot',
                        type=int,
                        default=1,
                        choices=(0, 1),
                        help='If 1, use the annotations file. If 0, '
                             'extract pitch values and estimate '
                             'tonic frequencies. Default is 1.')
    parser.add_argument('-pf',
                        '--pitch_files',
                        type=int,
                        default=1,
                        choices=(0, 1),
                        help='If 1, use the already extracted pitch files. '
                             'If 0, extract the pitch values. Default is 1.')
    parser.add_argument('-t',
                        '--annot_tonic',
                        type=int,
                        default=1,
                        choices=(0, 1),
                        help='If 1, use the tonic frequencies in the '
                             'annotations file. If 0, estimate the '
                             'tonic frequencies. Default is 1.')
    parser.add_argument('-d',
                        '--folder_dir',
                        type=str,
                        default='data/otmm_makam_recognition_dataset'
                                '-dlfm2016/',
                        help='Directory of the files. '
                             'Type is string. Default is '
                             'data/otmm_makam_recognition_dataset-dlfm2016/')
    parser.add_argument('-fs',
                        '--features_save_name',
                        type=str,
                        default='feature_values.csv',
                        help='File name for storing feature values. '
                             'Type is string. Default is feature_values.csv')
    parser.add_argument('-cs',
                        '--classes_save_name',
                        type=str,
                        default='class_values.csv',
                        help='File name for storing class values. '
                             'Type is string. Default is class_values.csv')

    args = parser.parse_args()

    return {'number_of_bins': args.number_of_bins,
            'first_last_pct': args.first_last_pct,
            'pct': args.pct,
            'annot': args.annot,
            'pitch_files': args.pitch_files,
            'annot_tonic': args.annot_tonic,
            'folder_dir': args.folder_dir,
            'features_save_name': args.features_save_name,
            'classes_save_name': args.classes_save_name
            }


def get_args_compute_dendrogram():
    """ Argument parser for computedendrogram.py

        Returns
        -------
        dictionary : dict
            features_csv : str
                Name of the csv file containing feature values of instances
            classes_csv : str
                Name of the csv file containing class values of instances
            distance_func : str
                Distance function to use
            number_of_bins : int
                Number of comma values to divide between 0 and 1200 cent

    """
    parser = argparse.ArgumentParser(description='This method computes the '
                                                 'distances of average pitch '
                                                 'distributions of modes and '
                                                 'forms hierarchical '
                                                 'clustering')
    parser.add_argument('-f',
                        '--features_csv',
                        type=str,
                        default='feature_values.csv',
                        help='Name of the csv file containing feature values '
                             'of instances. Type is string. Default is '
                             'feature_values.csv')
    parser.add_argument('-c',
                        '--classes_csv',
                        type=str,
                        default='class_values.csv',
                        help='Name of the csv file containing class values '
                             'of instances. '
                             'Type is string. Default is class_values.csv')
    parser.add_argument('-d',
                        '--distance_func',
                        type=str,
                        help='Distance function to use. Type is string.'
                             'Default is canberra.',
                        default='canberra')
    parser.add_argument('-n',
                        '--number_of_bins',
                        type=int,
                        help='Number of comma values to divide between '
                             '0 and 1200 cent. Type is integer. '
                             'Default is 53.',
                        default=53)
    args = parser.parse_args()

    return {'features_csv': args.features_csv,
            'classes_csv': args.classes_csv,
            'distance_func': args.distance_func,
            'number_of_bins': args.number_of_bins}


def get_args_compare_two_modes():
    """ Argument parser for comparetwomodes.py

        Returns
        -------
        dictionary : dict
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
    parser = argparse.ArgumentParser(description='This method plots the '
                                                 'average pitch histograms '
                                                 'of given modes in order '
                                                 'to compare them')
    parser.add_argument('-fm',
                        '--first_mode',
                        type=str,
                        help='Target directory for the downloaded files.'
                             'Type is string.',
                        required=True
                        )
    parser.add_argument('-sm',
                        '--second_mode',
                        type=str,
                        help='Target directory for the downloaded files.'
                             'Type is string.',
                        required=True
                        )
    parser.add_argument('-n',
                        '--number_of_bins',
                        type=int,
                        help='Number of comma values to divide between '
                             '0 and 1200 cent. Type is integer. '
                             'Default is 53.',
                        default=53)
    parser.add_argument('-f',
                        '--first_last_pct',
                        type=int,
                        default=1,
                        choices=(0, 1),
                        help='If 1, use the first and the last sections. '
                             'If 0, only the entire recording. '
                             'Default is 1.')
    parser.add_argument('-fn',
                        '--features_csv',
                        type=str,
                        default='feature_values.csv',
                        help='Name of the csv file containing feature values '
                             'of instances. Type is string. Default is '
                             'feature_values.csv')
    parser.add_argument('-cn',
                        '--classes_csv',
                        type=str,
                        default='class_values.csv',
                        help='Name of the csv file containing class values '
                             'of instances. '
                             'Type is string. Default is class_values.csv')
    args = parser.parse_args()

    return {'first_mode': args.first_mode,
            'second_mode': args.second_mode,
            'number_of_bins': args.number_of_bins,
            'first_last_pct': args.first_last_pct,
            'features_csv': args.features_csv,
            'classes_csv': args.classes_csv}


def get_args_automatic_classification():
    """Argument parser for automaticclasification.py

        Returns
        -------
        dictionary : dict
            use_model : int
                If 1, use the specified model; if 0, create and train a new model
            hyperparameter_tune : int
                If 1, perform hyperparameter classification for cross validation
                ; if 0, perform only cross validation (use_model has to be 0)
            features_csv : str
                Name of the csv file containing feature values of instances
            classes_csv : str
                Name of the csv file containing class values of
                instances (use_model has to be 0)
            hidden_layers : List[int]
                If hyperparameter_tune is 1, the list of number of nodes for
                one-hidden-layer MLP model to be used in hyperparameter tuning.
                If hyperparameter_tune is 0, list[0] is the number of nodes for
                one-hidden-layer MLP model to be used in cross validation.
            alphas : List[float]
                If hyperparameter_tune is 1, the list of alpha coefficients for
                the MLP model to be used in hyperparameter tuning.
                If hyperparameter_tune is 0, list[0] is the alpha coefficient for
                the MLP model to be used in cross validation.
            learning_rates : List[float]
                If hyperparameter_tune is 1, the list of learning rates for
                the MLP model to be used in hyperparameter tuning.
                If hyperparameter_tune is 0, list[0] is the learning rate for
                the MLP model to be used in cross validation.
            momenta : List[float]
                If hyperparameter_tune is 1, the list of momentum coefficients for
                the MLP model to be used in hyperparameter tuning.
                If hyperparameter_tune is 0, list[0] is momentum coefficient for
                the MLP model to be used in cross validation.
            iterations : int
                Number of iterations for cross validation and evaluation steps
            model_name : str
                Name of the model to load and use
    """
    parser = argparse.ArgumentParser(description='This method performs '
                                                 'automatic classification '
                                                 'with one-hidden-layer MLP '
                                                 'model. features_csv should '
                                                 'contain pitch distributions '
                                                 'and be formatted as each row'
                                                 ' is an instance, e.g. a '
                                                 'recording, and each column '
                                                 'is the respective bin of the'
                                                 ' distribution. classes_csv '
                                                 'should contain mode '
                                                 'information as each row is '
                                                 'the mode of an instance. To '
                                                 'use an existing model and '
                                                 'predict the modes of the '
                                                 'instances in features_csv '
                                                 'file, use_model should be 1.'
                                                 ' To perform cross validation'
                                                 ' and hyperparameter '
                                                 'tuning with the given '
                                                 'lists of parameters, '
                                                 'use_model should be 0 and '
                                                 'hyperparameter_tune should '
                                                 'be 1. To divide the dataset '
                                                 'into training and test '
                                                 'subsets without cross '
                                                 'validation, use_model and '
                                                 'hyperparameter_tune should '
                                                 'be 0. For the last case, '
                                                 'the first element of the '
                                                 'lists of parameters will '
                                                 'be considered for the MLP '
                                                 'model.')
    parser.add_argument('-u',
                        '--use_model',
                        type=int,
                        help='If 1, use the specified model; if 0, '
                             'create and train a new model. Default is 0',
                        default=0,
                        choices=(0, 1))
    parser.add_argument('-hpt',
                        '--hyperparameter_tune',
                        type=int,
                        default=1,
                        choices=(0, 1),
                        help='If 1, perform hyperparameter tuning '
                             'with cross validation (use_model has to be 0). '
                             'Default is 1.')
    parser.add_argument('-f',
                        '--features_csv',
                        type=str,
                        default='feature_values.csv',
                        help='Name of the csv file containing feature values '
                             'of instances. Type is string. Default is '
                             'feature_values.csv')
    parser.add_argument('-c',
                        '--classes_csv',
                        type=str,
                        default='class_values.csv',
                        help='Name of the csv file containing class values '
                             'of instances (use_model has to be 0). '
                             'Type is string. Default is class_values.csv')
    parser.add_argument('-hl',
                        '--hidden_layers',
                        type=int,
                        nargs="*",
                        default=[30, 50, 70, 90, 110, 130, 150, 170, 190],
                        help='If hyperparameter_tune is 1, the list of '
                             'number of nodes for one-hidden-layer MLP model '
                             'to be used in hyperparameter tuning. '
                             'If hyperparameter_tune is 0, list[0] is the '
                             'number of nodes for one-hidden-layer MLP model '
                             'to be used in cross validation. '
                             'Type is list of integers. Default is'
                             '[30, 50, 70, 90, 110, 130, 150, 170, 190].')
    parser.add_argument('-a',
                        '--alphas',
                        type=float,
                        nargs="*",
                        default=[0.1, 0.01, 0.001],
                        help='If hyperparameter_tune is 1, the list of '
                             'alpha coefficients for the MLP model to be '
                             'used in hyperparameter tuning. '
                             'If hyperparameter_tune is 0, list[0] is '
                             'the alpha coefficient for the MLP model '
                             'to be used in cross validation.'
                             'Type is list of floats. Default is'
                             '[0.1, 0.01, 0.001].')
    parser.add_argument('-lr',
                        '--learning_rates',
                        type=float,
                        nargs="*",
                        default=[0.1, 0.01, 0.001],
                        help='If hyperparameter_tune is 1, the list of '
                             'learning rates for the MLP model to '
                             'be used in hyperparameter tuning. '
                             'If hyperparameter_tune is 0, list[0] is '
                             'the learning rate for the MLP model to be '
                             'used in cross validation.'
                             'Type is list of floats. Default is'
                             '[0.1, 0.01, 0.001].')
    parser.add_argument('-m',
                        '--momenta',
                        type=float,
                        nargs="*",
                        default=[0.5, 0.7, 0.9],
                        help='If hyperparameter_tune is 1, the list of '
                             'momentum coefficients for the MLP model '
                             'to be used in hyperparameter tuning. '
                             'If hyperparameter_tune is 0, list[0] is '
                             'momentum coefficient for the MLP model to be '
                             'used in cross validation.'
                             'Type is list of floats. Default is'
                             '[0.5, 0.7, 0.9].')
    parser.add_argument('-i',
                        '--iterations',
                        type=int,
                        default=10,
                        help='Number of iterations for cross validation '
                             'and evaluation steps. Type is integer. '
                             'Default is 10.')
    parser.add_argument('-mn',
                        '--model_name',
                        type=str,
                        default='model.sav',
                        help='Name of the model to load and use. '
                             'Type is string.'
                             'Default is model.')

    args = parser.parse_args()

    return {'use_model': args.use_model,
            'hyperparameter_tune': args.hyperparameter_tune,
            'features_csv': args.features_csv,
            'classes_csv': args.classes_csv,
            'hidden_layers': args.hidden_layers,
            'alphas': args.alphas,
            'learning_rates': args.learning_rates,
            'momenta': args.momenta,
            'iterations': args.iterations,
            'model_name': args.model_name
            }


def convert_to_cent(pitch_hz, tonic):
    """Converting the pitch values from frequency to cent

        Parameters
        ----------
        pitch_hz : numpy.array
            Frequency values of the pitch
        tonic : float or str
            Tonic frequency to use

        Returns
        -------
        cent_val : numpy.array
            Cent values of the pitch relative to the tonic
    """
    cent_val = 1200 * np.log2(pitch_hz / np.float(tonic))

    # folding the values into the range (0, 1200)
    for k in range(0, cent_val.size):
        while cent_val[k] < 0:
            cent_val[k] = cent_val[k] + 1200
        while cent_val[k] >= 1200:
            cent_val[k] = cent_val[k] - 1200

    return {'cent_val': cent_val}


def write_to_csv(file_name, column_names, values):
    """Writing the input data into a csv file and store it
       in 'data/csv_file/' directory

        Parameters
        ----------
        file_name : str
            Name of the csv file
        column_names : numpy.ndarray
            Names of the columns
        values : numpy.ndarray
            Values to write

    """
    target_dir = 'data/csv_files/'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    with open(target_dir + file_name, 'w') as csv_:
        writer = csv.writer(csv_, dialect='excel')

        writer.writerow(column_names[0])

        if type(values.shape) is tuple:
            (n, _) = values.shape
        else:
            n = values.size
        for k in range(0, n):
            writer.writerow(values[k])


def load_data(features_csv='feature_values.csv',
              classes_csv='class_values.csv'):
    """Loading data from csv files stored in 'data/csv_files/' directory

        Parameters
        ----------
        features_csv : str
            Name of the file that contains feature values
        classes_csv : str
            Name of the file that contains class values

        Returns
        -------
        features : numpy.ndarray
            Feature values obtained from the file
        classes : numpy.ndarray
            Class values obtained from the file
    """
    features_dir = 'data/csv_files/' + features_csv
    classes_dir = 'data/csv_files/' + classes_csv
    features = \
        Normalizer(norm='max').fit_transform(pd.read_csv(features_dir))
    classes = np.ravel(pd.read_csv(classes_dir))

    return {'features': features,
            'classes': classes}


def plot_confusion_matrix(cm, classes,
                          cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix

        Parameters
        ----------
        cm : numpy.ndarray
            Confusion matrix to plot
        classes : numpy.ndarray
            All class values
        cmap : plt.cm
            Color map to use for the plot

    """
    class_names = np.unique(classes)
    plt.figure(figsize=(9, 9),)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Aggregated Confusion Matrix')
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, weight='bold')
    plt.yticks(tick_marks, class_names, weight='bold')

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 verticalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black',
                 weight='bold')

    plt.ylabel('True label', weight='bold')
    plt.xlabel('Predicted label', weight='bold')
    plt.show()
