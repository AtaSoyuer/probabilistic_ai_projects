import os
import typing

from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.kernel_approximation import Nystroem


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation


# Cost function constants
THRESHOLD = 35.5
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 5.0
COST_W_THRESHOLD = 20.0


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self,kernel=ConstantKernel(constant_value=0.173**2 , constant_value_bounds="fixed") * RBF(0.0183, length_scale_bounds="fixed") + ConstantKernel(constant_value=0, constant_value_bounds="fixed")):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)
        self.kernel = kernel

        # TODO: Add custom initialization for your model here if necessary
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=0.001 , random_state=0)
        #self.feature_map = Nystroem(kernel='rbf', gamma=0.2, random_state=0, n_components=2)



    def predict(self, x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param x: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # TODO: Use your GP to estimate the posterior mean and stddev for each location here
        # gp_mean = np.zeros(x.shape[0], dtype=float)
        # gp_std = np.zeros(x.shape[0], dtype=float)
        gp_mean, gp_std = self.gpr.predict(x, return_std=True)
        print('stde_dev:',gp_std)
        c_95_low = gp_mean-2*gp_std
        c_95_high = gp_mean+2*gp_std
        
        predictions = np.zeros_like(gp_mean)
        for ind, _ in enumerate(gp_mean):
            if gp_mean[ind] >= THRESHOLD:
                predictions[ind] = gp_mean[ind]

            elif gp_mean[ind] < THRESHOLD and c_95_high[ind]>=THRESHOLD:
                predictions[ind] = THRESHOLD

            else:
                predictions[ind] = gp_mean[ind] - 1*gp_std[ind]

        print(np.sum(gp_mean>=THRESHOLD))
        print(gp_mean.shape)

        # print('prediction began')
        
        # predictions = np.zeros_like(gp_mean)

        # n_bins = 1000
        # n_samples = 4000
        # y_posterior_samples = self.gpr.sample_y(x,n_samples=n_samples,random_state = 42)

        # y_start = np.min(y_posterior_samples)
        # y_stop = np.max(y_posterior_samples)

        # y_values = np.linspace(y_start, y_stop, n_bins)

        # #Notice the posteriors generated for each test pt in x is different!!!

        # #np.savetxt("samples_drawn.csv", y_posterior_samples, delimiter=",")
        # #y_posterior_emprical = np.hist(y_posterior_samples, bins = n_bins) / n_bins
        # y_posterior_emprical = np.apply_along_axis(lambda a: np.histogram(a, bins=n_bins)[0], 1, y_posterior_samples)/float(n_samples)
        # #print(y_posterior_samples.shape)
        # #np.savetxt("posterior.csv", y_posterior_emprical, delimiter=",")
        # plt.figure()
        # plt.bar(y_values, y_posterior_emprical[36,:],)
        # plt.title('100000 samples',)
        # plt.savefig('posterior.png')
        # y_matrix = np.tile(y_values, (int(n_bins),1))
        # y_matrix_minus_binned_values = (y_matrix.transpose() - y_values).transpose()
        # #print(y_matrix_minus_binned_values)

        # y_values_true = y_matrix.copy()
        # y_values_pred = y_matrix.copy().transpose()

        # # Unweighted cost
        # cost = y_matrix_minus_binned_values ** 2
        # weights = np.zeros_like(cost)

        # # Case i): overprediction
        # mask_1 = y_values_pred >= y_values_true
        # weights[mask_1] = COST_W_OVERPREDICT

        # # Case ii): true is above threshold, prediction below
        # mask_2 = (y_values_true >= THRESHOLD) & (y_values_pred <= THRESHOLD)
        # weights[mask_2] = COST_W_THRESHOLD

        # # Case iii): everything else
        # mask_3 = ~(mask_1 | mask_2)
        # weights[mask_3] = COST_W_NORMAL

        # #print('Mask',np.sum(mask_1+mask_2+mask_3))
        # #print('weights',weights.shape[0]*weights.shape[1])


        # loss_matrix = np.multiply(cost,weights)

        # for i in range(len(x)):     #Repeat for each test pt and its posterior

        #     posterior_expectations = np.sum(loss_matrix * y_posterior_emprical[i,:], axis=1)

        #     optimal_index = np.argmin(posterior_expectations)

        #     predictions[i] = y_values[optimal_index]
        


        # TODO: Use the GP posterior to form your predictions here
        #predictions = gp_mean
        
        #x_transformed = self.feature_map.transform(x)
        

        return predictions, gp_mean, gp_std

    def fit_model(self, train_x: np.ndarray, train_y: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """

        # TODO: Fit your model here
        #transformed_x = self.feature_map.fit_transform(train_x)
        sample_indices = np.arange(np.shape(train_x)[0])
        np.random.seed(36)
        num_selected_indices = 6500
        selected_indices = np.random.choice(sample_indices, num_selected_indices, replace=False)
        train_x = train_x[selected_indices,:]
        train_y = train_y[selected_indices]
        # train_x = train_x[::2,:]
        # train_y = train_y[::2]
        self.gpr.fit(train_x, train_y)
        pass


def cost_function(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param y_true: Ground truth pollution levels as a 1d NumPy float array
    :param y_predicted: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert y_true.ndim == 1 and y_predicted.ndim == 1 and y_true.shape == y_predicted.shape

    # Unweighted cost
    cost = (y_true - y_predicted) ** 2
    weights = np.zeros_like(cost)

    # Case i): overprediction
    mask_1 = y_predicted > y_true
    weights[mask_1] = COST_W_OVERPREDICT

    # Case ii): true is above threshold, prediction below
    mask_2 = (y_true >= THRESHOLD) & (y_predicted < THRESHOLD)
    weights[mask_2] = COST_W_THRESHOLD

    # Case iii): everything else
    mask_3 = ~(mask_1 | mask_2)
    weights[mask_3] = COST_W_NORMAL

    # Weigh the cost and return the average
    return np.mean(cost * weights)

def select_model(train_x, train_y):

    score = make_scorer(cost_function, greater_is_better=False)


    # a_array = np.logspace(0, 6, num=2)
    # b_array = np.logspace(-1, 2, num=2)
    # l_array = np.logspace(-2,0, num=2)
    
    a_array = np.linspace(5e-3, 5e-2, 10)
    b_array = np.array([0])
    #l_array = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100 , 1000])
    l_array = np.linspace(0.01, 0.025, 10)

    # a_array = np.array([0.1, 10])
    # b_array = np.array([0, 5.75, 33])
    # l_array = np.array([1e-3, 10])

    a_array, b_array, l_array = np.meshgrid(a_array, b_array, l_array)
    cartesian_product = np.concatenate([a_array.flatten().reshape(-1,1), b_array.flatten().reshape(-1,1), l_array.flatten().reshape(-1,1)], axis=1)

    #[print(l,a,b) for l,a,b in cartesian_product]

    param_grid = [{
        "alpha":  [1e-3, 5e-4, 1e-4, 5e-5],
        "kernel": [ConstantKernel(constant_value=a, constant_value_bounds="fixed")*RBF(l, length_scale_bounds="fixed")+ConstantKernel(constant_value=b, constant_value_bounds="fixed") for a,b,l in cartesian_product]
    }]

    #print(param_grid)

    gp = GaussianProcessRegressor()

    clf = GridSearchCV(estimator=gp, param_grid=param_grid, cv=4, 
                       scoring=score)

    clf.fit(train_x, train_y)

    print(clf.best_params_)

    return clf.best_params_['kernel']


def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.predict(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()

def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    print("Model Selection")

    #kernel = select_model(train_x[::3,:], train_y[::3])

    # validation partition
    val_x = train_x[1::3,:]
    val_y = train_y[1::3]
    #val_x = val_x[::2,:]
    #val_y = val_y[::2]
    sample_indices = np.arange(np.shape(val_x)[0])
    np.random.seed(36)
    num_selected_indices = 2000
    selected_indices = np.random.choice(sample_indices, num_selected_indices, replace=False)
    val_x = val_x[selected_indices,:]
    val_y = val_y[selected_indices]


    #observation on training points
    plt.figure()
    plt.scatter(train_x[:,0], train_x[:,1], s=0.1)
    plt.savefig('training_points.png')

    # Fit the model
    print('Fitting model')
    #model = Model(kernel=kernel)
    model = Model()
    model.fit_model(train_x, train_y)

    # Predict on the test features
    print('Predicting on test features')
    predicted_y = model.predict(test_x)
    print(predicted_y[0])
    print(predicted_y[1])
    print(predicted_y[2])

    predicted_val_y, _ , _  = model.predict(val_x)
    print("val_score: " + str(cost_function(val_y, predicted_val_y)) )

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
