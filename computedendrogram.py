import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from pitchdistamr_utils import load_data
from pitchdistamr_utils import get_args_compute_dendrogram


def main(features_csv,
         classes_csv,
         distance_func,
         number_of_bins):
    """
        This method computes the distances of average pitch distributions
        of modes and forms hierarchical clusters based on distances
        of pitch distribution templates

        Parameters
        -------
        features_csv : str
            Name of the csv file containing feature values of instances
        classes_csv : str
            Name of the csv file containing class values of instances
        distance_func : str
            Distance function to use
        number_of_bins : int
            Number of comma values to divide between 0 and 1200 cent

    """
    data = load_data(features_csv=features_csv,
                     classes_csv=classes_csv)
    features = data['features']
    classes = data['classes']

    templates = []
    modes = []
    class_names = np.unique(classes)
    number_of_classes = len(class_names)
    for i in range(0, number_of_classes):
        templates.append(np.mean(features[np.where((classes == class_names[i]
                                                    ))],
                                 axis=0)[:number_of_bins])
        modes.append(class_names[i])
    templates = np.array(templates)

    dist_all = []
    for i in range(0, number_of_classes):
        for j in range(i + 1, number_of_classes):
            dist_temp = []
            for k in range(0, number_of_bins):
                dist_temp.append(
                    pdist(np.vstack((templates[i],
                                     np.roll(templates[j],
                                             k,
                                             axis=0))),
                          distance_func))
            dist_all.append(min(dist_temp)[0])

    z = linkage(np.array(dist_all), optimal_ordering=True)
    plt.figure(figsize=(7, 7))
    dendrogram(z, labels=np.array(modes), orientation='left')
    plt.title('Hierarchical clustering dendrogram for modes based on histogram'
              ' distances')
    plt.yticks(weight='bold', size='12')
    plt.show()


if __name__ == "__main__":
    args = get_args_compute_dendrogram()
    main(features_csv=args['features_csv'],
         classes_csv=args['classes_csv'],
         distance_func=args['distance_func'],
         number_of_bins=args['number_of_bins']
         )
