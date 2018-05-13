import os
import zipfile
import urllib.request
from pitchdistamr_utils import get_args_download_makam_dataset


def main(directory):
    """
        Downloads OTMM Makam Recognition Dataset from
        https://github.com/MTG/otmm_makam_recognition_dataset/

        Parameters
        ----------
        directory : str
            Target directory to download the dataset into
    """
    if not os.path.exists(directory):
        # initialization
        url = 'https://github.com/MTG/otmm_makam_recognition_dataset/' \
              'archive/dlfm2016.zip'
        filename = 'otmm_makam_recognition_dataset-dlfm2016.zip'
        target_dir = directory

        # creating the directory
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        final_data_dir = target_dir + filename.replace('.zip', '')

        # downloading the zip file from the url
        urllib.request.urlretrieve(url, filename)

        # unzipping to a specific folder
        zip_ref = zipfile.ZipFile(filename, 'r')
        zip_ref.extractall(target_dir)
        zip_ref.close()

        # removing the zip file
        os.remove(filename)
        print('Data downloaded and unzipped to: ', final_data_dir)
    else:
        print('Folder ',
              directory,
              ' already exists, delete it if you want to re-download data')


if __name__ == "__main__":
    args = get_args_download_makam_dataset()
    main(directory=args['directory'])
