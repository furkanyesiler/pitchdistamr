import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split as tts
from sklearn.neural_network import MLPClassifier
from makamnn_utils import load_data
from makamnn_utils import plot_confusion_matrix
from makamnn_utils import get_args_automatic_classification
import pickle


def main(use_model,
         hyperparameter_opt,
         features_csv,
         classes_csv,
         hidden_layers,
         alphas,
         learning_rates,
         momenta,
         model_name,
         iterations
         ):
    """ This method performs automatic classification with one-hidden-layer
        MLP model. features_csv should contain pitch distributions and be
        formatted as each row is an instance, e.g. a recording, and each
        column is the respective bin of the distribution. classes_csv should
        contain mode information as each row is the mode of an instance. To
        use an existing model and predict the modes of the instances in
        features_csv file, use_model should be 1. To perform cross validation
        and hyperparameter optimization with the given lists of parameters,
        use_model should be 0 and hyperparameter_opt should be 1. To divide
        the dataset into training and test subsets without cross validation,
        use_model and hyperparameter_opt should be 0. For the last case, the
        first element of the lists of parameters will be considered for the
        MLP model.

        Parameters
        ----------
        use_model : int
            If 1, use the specified model; if 0, create and train a new model
        hyperparameter_opt : int
            If 1, perform hyperparameter classification for cross validation
            ; if 0, perform only cross validation (use_model has to be 0)
        features_csv : str
            Name of the csv file containing feature values of instances
        classes_csv : str
            Name of the csv file containing class values of
            instances (use_model has to be 0)
        hidden_layers : List[int]
            If hyperparameter_opt is 1, the list of number of nodes for
            one-hidden-layer MLP model to be used in hyperparameter
            optimization. If hyperparameter_opt is 0, list[0] is the
            number of nodes for one-hidden-layer MLP model to be used
            in cross validation.
        alphas : List[float]
            If hyperparameter_opt is 1, the list of alpha coefficients for
            the MLP model to be used in hyperparameter optimization.
            If hyperparameter_opt is 0, list[0] is the alpha coefficient for
            the MLP model to be used in cross validation.
        learning_rates : List[float]
            If hyperparameter_opt is 1, the list of learning rates for
            the MLP model to be used in hyperparameter optimization.
            If hyperparameter_opt is 0, list[0] is the learning rate for
            the MLP model to be used in cross validation.
        momenta : List[float]
            If hyperparameter_opt is 1, the list of momentum coefficients for
            the MLP model to be used in hyperparameter optimization.
            If hyperparameter_opt is 0, list[0] is momentum coefficient for
            the MLP model to be used in cross validation.
        iterations : int
            Number of iterations for cross validation and evaluation steps
        model_name : str
            Name of the model to load and use

    """
    # loading data from the specified csv files
    data = load_data(features_csv=features_csv,
                     classes_csv=classes_csv)
    features = data['features']

    # checking whether to use the pre-created model
    if use_model == 1:
        # loading the specified model
        mlp_model = pickle.load(open('data/models/' + model_name, 'rb'))
        # predicting the classes of instances
        y_pred = mlp_model.predict(features)
        # printing results
        print('Predicted classes are: ')
        print(y_pred)
    # creating and training an MLP model
    else:
        classes = data['classes']
        # calculating the number of classes
        no_of_classes = np.unique(classes).size
        # defining a numpy ndarray for aggregated confusion matrix
        agg_confusion_matrix = np.zeros(shape=(no_of_classes, no_of_classes))
        # empty list to keep the scores
        scores = []
        # cross validation and evaluation steps
        for k in range(0, iterations):
            # dividing the data into train and test sets
            x_train, x_test, y_train, y_test = tts(features,
                                                   classes,
                                                   test_size=0.1,
                                                   stratify=classes,
                                                   random_state=k)
            # checking whether to perform hyperparameter optimization
            if hyperparameter_opt == 1:
                # getting hyperparameters from optimization
                params = hyperp_optimization(train_features=x_train,
                                             train_classes=y_train,
                                             hidden_layers=hidden_layers,
                                             alphas=alphas,
                                             learning_rates=learning_rates,
                                             momenta=momenta)
            else:
                # using the hyperparameters the user provides
                params = {'hl': hidden_layers[0],
                          'alpha': alphas[0],
                          'lr': learning_rates[0],
                          'momentum': momenta[0]
                          }
            # creating an MLP model with the selected hyperparameters
            mlp_model = MLPClassifier(hidden_layer_sizes=params['hl'],
                                      alpha=params['alpha'],
                                      learning_rate_init=params['lr'],
                                      max_iter=10000,
                                      momentum=params['momentum'])
            # training the MLP model
            mlp_model = mlp_model.fit(x_train, y_train)
            # predicting the classes of instances in the test subset
            y_pred = mlp_model.predict(x_test)
            # calculating the accuracy score
            acc = accuracy_score(y_test, y_pred)
            # organizing accuracy score and hyperparameter values
            scores.append((acc,
                           params['hl'],
                           params['alpha'],
                           params['lr'],
                           params['momentum']))
            # calculating the confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            # adding the obtained confusion matrix to aggregated matrix
            agg_confusion_matrix = agg_confusion_matrix + cm
            # printing the results
            print('The accuracy score and hyperparameters used for Iteration '
                  + str(k+1) + ': ')
            print('Accuracy Score: ' + str(acc * 100) + '%')
            print('Hidden Layer Size: ' + str(params['hl']))
            print('Alpha: ' + str(params['alpha']))
            print('Learning Rate: ' + str(params['lr']))
            print('Momentum: ' + str(params['momentum']))

        # casting the values of aggregated confusion matrix as integers
        agg_confusion_matrix = agg_confusion_matrix.astype(int)
        # plotting the obtained aggregated confusion matrix
        plot_confusion_matrix(agg_confusion_matrix,
                              classes=np.unique(classes))
        print('The mean accuracy score of ' + str(iterations)
              + ' iterations: ' + str(100*np.mean(np.array(scores)[:, 0]))
              + '%')


def hyperp_optimization(train_features,
                        train_classes,
                        hidden_layers,
                        alphas,
                        learning_rates,
                        momenta):
    """

        Parameters
        ----------
        train_features : numpy.ndarray
            Number of comma values to divide between 0 and 1200 cent
        train_classes : numpy.ndarray
            Whether to include the first and the last x% values
        hidden_layers : List[int]
            Whether to use the already extracted pitch files
        alphas : List[float]
            Whether to use the estimated tonic frequencies from
            the annotations file
        learning_rates : List[float]
            Path to the directory of the files
        momenta : List[float]
            File name for the stored feature values

    """
    # variable for iteration number
    iteration = 0
    # creating a numpy ndarray for storing accuracy scores
    cvl_r = np.zeros(shape=(len(hidden_layers),
                            len(alphas),
                            len(learning_rates),
                            len(momenta)))
    # index for hidden layer dimension
    h_index = 0
    for hl in hidden_layers:
        # index for alpha dimension
        a_index = 0
        for alp in alphas:
            # index for learning rate dimension
            l_index = 0
            for lr in learning_rates:
                # index for momentum dimension
                m_index = 0
                for mo in momenta:
                    # creating an MLP model
                    model = MLPClassifier(hidden_layer_sizes=(hl,),
                                          alpha=alp,
                                          learning_rate_init=lr,
                                          max_iter=10000,
                                          momentum=mo)
                    # specifying the type of cross validation
                    cv = StratifiedKFold(n_splits=10, random_state=iteration)
                    # obtaining the mean accuracy of k-fold cross validation
                    cvl = np.mean(np.array(cross_val_score(model,
                                                           train_features,
                                                           train_classes,
                                                           cv=cv)))
                    cvl_r[h_index][a_index][l_index][m_index] = cvl
                    iteration = iteration + 1

                    print('Iteration ' + str(iteration) + ' is being performed'
                                                          ' for hyperparameter'
                                                          ' optimization.')
                    m_index += 1
                l_index += 1
            a_index += 1
        h_index += 1
    # getting the index of the element with the highest accuracy score
    hl_c, alp_c, lr_c, mo_c = np.unravel_index(np.argmax(cvl_r), cvl_r.shape)

    return {'hl': hidden_layers[hl_c],
            'alpha': alphas[alp_c],
            'lr': learning_rates[lr_c],
            'momentum': momenta[mo_c]
            }


if __name__ == "__main__":
    args = get_args_automatic_classification()
    main(use_model=args['use_model'],
         hyperparameter_opt=args['hyperparameter_opt'],
         features_csv=args['features_csv'],
         classes_csv=args['classes_csv'],
         hidden_layers=args['hidden_layers'],
         alphas=args['alphas'],
         learning_rates=args['learning_rates'],
         momenta=args['momenta'],
         iterations=args['iterations'],
         model_name=args['model_name']
         )
