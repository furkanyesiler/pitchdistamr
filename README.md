[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-ff69b4.svg)](http://creativecommons.org/licenses/by-nc-sa/4.0/) 
# Pitch Distribution Based Supervised Mode Recognition Toolbox

## Introduction
This repository contains python files to compute pitch distributions aligned with the respected tonic frequencies of recordings and to perform supervised mode recognition. It uses culture-specific algorithms for Turkish Makam Music to obtain pitch distributions. For supervised mode recognition, a Multi-Layer Perceptron (MLP) model is used. The proposed method outperforms previous works on Turkish Makam Recognition Dataset.

Please cite the publication below, if you use the toolbox in your work:

> F. Yesiler, B. Bozkurt and X. Serra. "Makam Recognition Using Extended Pitch Distribution Features and Multi-Layer Perceptrons". In 15th Sound and Music Computing Conference, Limassol, Cyprus, 2018.

This toolbox is mainly created to provide a straightforward access to pitch distribution computation and supervised mode recognition tasks. Currently, the algorithms for [pitch extraction](https://github.com/sertansenturk/predominantmelodymakam) and [tonic frequency estimation](https://github.com/hsercanatli/tonicidentifier_makam) are culture-specific for Turkish Makam Music. An extension that includes other music traditions such as Hindustani Music and Carnatic Music will be the next step of this toolbox.

## Usage
### Pitch Distribution Computation

To compute pitch distributions, following cases can be considered:

**1-Mode Information is Known**

If the mode information is known, it should be included in "annotations.json" file located as:
```
pitchdistamr/data/your_data/annotations.json
```
"annotations.json" file includes:
* Mode of the recording
* Name of the recording
* Tonic Frequency of the recording (Optional)

An example annotations.json file:
```
[
    {
        "mode": "mode_1",
        "name": "rec_1",
        "tonic": 440.0
    },
    {
        "mode": "mode_2",
        "name": "rec_2",
        "tonic": 220.0
    }
]
```

When the mode information is known, the pitch distributions can be computed with:
* Extracting the pitch values of predominant melodies in the recordings or using the already extracted pitch values
* Estimating the tonic frequency of the recording or using the already estimated tonic frequencies included in the "annotations.json" file

The recordings or the pitch files should be organized regarding their modes. An example for the required file structure can be:
```
pitchdistamr/data/your_data/annotations.json
pitchdistamr/data/your_data/data/mode_1/rec_1.wav(rec_1.pitch, if the pitch files are to be used)
pitchdistamr/data/your_data/data/mode_2/rec_2.wav(rec_2.pitch, if the pitch files are to be used)
```

**2-Mode Information is Unknown**

If the mode information is unknown, the data directory should contain all the files. An example for the required file structure can be:
```
pitchdistamr/data/your_data/rec_3.wav(rec_3.pitch, if the pitch files are to be used)
pitchdistamr/data/your_data/rec_4.wav(rec_4.pitch, if the pitch files are to be used)
```

### Supervised Mode Recognition

For supervised mode recognition, following cases can be consired:

**1-Testing the method on recordings with known modes**

After obtaining pitch distributions and modes as csv files, two use cases can be considered:
* Performing cross validation to tune the hyperparameters of the MLP model, using the obtained hyperparameters to create an MLP model, training the model with 90% of the data (training subset) and evaluating the trained model on 10% of the data (test subset).
* Creating an MLP model with specified parameters, training the model with 90% of the data (training subset) and evaluating the trained model on 10% of the data (test subset).

**2-Identifying the modes of recordings using the pre-created model**

After obtaining pitch distributions of the recordings as a csv file, the model trained with the entire dataset can be used to predict the modes of respective recordings.

Please refer to the Demo.ipynb for a demonstration of this toolbox.

## Installation

This toolbox is written with Python 3. The list of libraries that are required are below:
```
numpy
scikit-learn
matplotlib
scipy
Essentia
```
The instructions for installing Essentia can be found [here](http://essentia.upf.edu/documentation/installing.html)

**An alternative way** for using this toolbox is to install 'Docker'. Docker provides an easy access to a virtual environment with desired libraries. All the necessary libraries for this toolbox are installed and compiled in ['MIR-Toolbox'](https://github.com/MTG/MIR-toolbox-docker), a docker image created for Music Information Retrieval applications.

The required installation steps are:
* Install [Docker Compose](https://docs.docker.com/compose/install/).
* Clone this repository as follows:
```
git clone https://github.com/furkanyesiler/pitchdistamr.git
```
* In terminal, go to the directory of the repository.
* Initiate the Docker Image by using the following command (it may require admin permissions):
```
sudo docker-compose up
```
* Access localhost:8888 on your browser, and when asked for a password, use *mir*.

## Authors

Furkan Yesiler furkan.yesiler@gmail.com

## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
