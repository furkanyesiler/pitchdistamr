import json
import os
import numpy as np
from external_utilities.predominantmelodymakam import PredominantMelodyMakam
from external_utilities.pitchfilter import PitchFilter
from external_utilities.toniclastnote import TonicLastNote
from makamnn_utils import convert_to_cent
from makamnn_utils import write_to_csv
from makamnn_utils import get_args_compute_pitch_histograms


def main(number_of_bins,
         first_last_pct,
         pct,
         annot,
         pitch_files,
         annot_tonic,
         folder_dir,
         features_save_name,
         classes_save_name):
    """ This method computes pitch distributions of given recordings
        by aligning them with respect to the tonic frequency. The obtained
        distributions are saved as a csv file. There are several options
        on how to use this file. To use the mode information specified in
        annotations file, 'annot' should be 1. To use the already extracted
        pitch values from the directory, 'pitch_files' should be 1. To use
        the already estimated tonic frequencies included in annotations
        file, 'annot_tonic' should be 1. If the mode information is to be
        used, the directory for 'folder_dir' should include the recordings
        separated by their modes in the following way:
        'your_dir/mode_A/record_1.wav'

        Parameters
        ----------
        number_of_bins : int
            Number of comma values to divide between 0 and 1200 cent
        first_last_pct : int
            Whether to include the first and the last x% sections
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
            Path to the directory of the files
        features_save_name : str
            File name for storing feature values
        classes_save_name : str
            File name for storing class values

    """

    # arranging the number of columns of the feature data
    if first_last_pct == 1:
        number_of_columns = number_of_bins * 3
    else:
        number_of_columns = number_of_bins

    # creating an array for the column names
    pitch_column_names = np.ndarray(shape=(1, number_of_columns),
                                    dtype='|U16')

    # if the analysis includes the first and the last sections,
    # column names are arranged accordingly
    for k in range(0, number_of_bins):
        pitch_column_names[0][k] = 'FreqBin_' + str(k + 1)
        if first_last_pct == 1:
            pitch_column_names[0][k + number_of_bins] \
                = 'FreqBin_FS_' + str(k + 1)
            pitch_column_names[0][k + number_of_bins * 2] \
                = 'FreqBin_LS_' + str(k + 1)

    # if the user has an annotations file
    if annot == 1:
        with open(folder_dir + 'annotations.json') as json_data:
            files = json.load(json_data)
        number_of_files = len(files)

        # initializing arrays
        pitch_val = np.ndarray(shape=(number_of_files,
                                      number_of_columns))
        class_val = np.ndarray(shape=(number_of_files, 1),
                               dtype='|U16')
        class_column_names = np.ndarray(shape=(1, 1),
                                        dtype='|U16')
        class_column_names[0][0] = 'Class'

        count = 0

        # iterate over files
        for file in files:
            # checking whether the files are named with MusicBrainz ID
            if 'mbid' in file:
                name = file['mbid'].split('http://musicbrainz.org/'
                                          'recording/')[-1]
            else:
                name = file['name']

            pitch_dir = folder_dir + 'data/' + file['makam'] + '/' + name

            # checking whether to use the already extracted pitch files
            if pitch_files == 1:
                pitch_hz = np.loadtxt(pitch_dir + '.pitch')
            else:
                if os.path.exists(pitch_dir + '.wav'):
                    file_ext = '.wav'
                elif os.path.exists(pitch_dir + '.mp3'):
                    file_ext = '.mp3'
                else:
                    file_ext = ''
                extractor = PredominantMelodyMakam(filter_pitch=False)
                pitch_hz = extractor.run(pitch_dir + file_ext)['pitch']
                pitch_filter = PitchFilter()
                pitch_hz = pitch_filter.run(pitch_hz)

            # checking whether to use the estimated tonic frequency
            # included in the annotations file
            if annot_tonic == 1:
                tonic = file['tonic']
            else:
                tonic_identifier = TonicLastNote()
                tonic, _, _, _ = tonic_identifier.identify(pitch_hz)
                tonic = tonic['value']

            # discarding the 0 values
            pitch_hz = pitch_hz[pitch_hz != 0]

            # converting the pitch values from frequency to cent
            cent_val = convert_to_cent(pitch_hz, tonic)['cent_val']

            # creating the histogram based on the number of bins
            hist_val, _ = np.histogram(cent_val,
                                       bins=number_of_bins)

            # storing the values of the histogram
            pitch_val[count][:number_of_bins] = hist_val

            # checking if the first and the last quarters are included
            if first_last_pct == 1:
                # taking values of the first and the last quarters
                pitch_hz_fs \
                    = pitch_hz[: np.int(pitch_hz.size * (pct / 100))]
                pitch_hz_ls \
                    = pitch_hz[np.int(pitch_hz.size * ((100 - pct) / 100)):]

                # converting the pitch values from frequency to cent
                cent_val_fs = convert_to_cent(pitch_hz_fs, tonic)['cent_val']
                cent_val_ls = convert_to_cent(pitch_hz_ls, tonic)['cent_val']

                # creating the histograms based on the number of bins
                hist_val_fs, _ = np.histogram(cent_val_fs,
                                              bins=number_of_bins)
                hist_val_ls, _ = np.histogram(cent_val_ls,
                                              bins=number_of_bins)

                # storing the values of the histograms
                pitch_val[count][number_of_bins: (number_of_bins * 2)] \
                    = hist_val_fs
                pitch_val[count][(number_of_bins * 2):(number_of_bins * 3)] \
                    = hist_val_ls

            class_val[count][0] = file['makam']

            if count % 50 == 0:
                file_interval = int(50 * np.floor(count / 50))
                print('Files ' + str(file_interval + 1) + '-'
                      + str(file_interval + 50)
                      + ' are being processed')
            count += 1

        # writing the values to csv files
        write_to_csv(features_save_name, pitch_column_names, pitch_val)
        write_to_csv(classes_save_name, class_column_names, class_val)

    # only extract the pitch if there is no annotation file
    else:
        # calculating the number of the files
        number_of_files = 0
        for file in os.listdir(folder_dir):
            if file.endswith('.wav') or file.endswith('.mp3'):
                number_of_files += 1

        # initializing the array
        pitch_val = np.ndarray(shape=(number_of_files,
                                      number_of_columns))

        count = 0
        for file in os.listdir(folder_dir):
            if file.endswith('.wav') or file.endswith('.mp3'):
                # extract the pitch values
                extractor = PredominantMelodyMakam(filter_pitch=False)
                pitch_hz = extractor.run(folder_dir + file)['pitch']
                pitch_filter = PitchFilter()
                pitch_hz = pitch_filter.run(pitch_hz)

                # compute the tonic value
                tonic_identifier = TonicLastNote()
                tonic, _, _, _ = tonic_identifier.identify(pitch_hz)
                tonic = tonic['value']

                # discarding the 0 values
                pitch_hz = pitch_hz[pitch_hz != 0]

                # converting the pitch values from frequency to cent
                cent_val = convert_to_cent(pitch_hz, tonic)['cent_val']

                # creating the histogram based on the number of bins
                hist_val, _ = np.histogram(cent_val,
                                           bins=number_of_bins)

                # storing the values of the histograms
                pitch_val[count][: number_of_bins] = hist_val

                count += 1

            else:
                print('No .wav or .mp3 files have found')

        # writing the values to the csv file
        write_to_csv(features_save_name, pitch_column_names, pitch_val)

    print('Csv files are created!')


if __name__ == "__main__":
    args = get_args_compute_pitch_histograms()
    main(number_of_bins=args['number_of_bins'],
         first_last_pct=args['first_last_pct'],
         pct=args['pct'],
         annot=args['annot'],
         pitch_files=args['pitch_files'],
         annot_tonic=args['annot_tonic'],
         folder_dir=args['folder_dir'],
         features_save_name=args['features_save_name'],
         classes_save_name=args['classes_save_name']
         )
