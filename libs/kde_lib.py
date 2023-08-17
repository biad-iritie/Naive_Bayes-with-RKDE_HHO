import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import rbf_kernel
from cvxopt import matrix, solvers
from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold
from mealpy.swarm_based.HHO import OriginalHHO
from mealpy.utils.problem import Problem


def gaussian_kernel(X, h, d):
    """
    Apply gaussian kernel to the input distance X
    """
    K = np.exp(-X / (2 * (h**2))) / ((2 * np.pi * (h**2))**(d / 2))
    return K


def rho(x, typ='hampel', a=0, b=0, c=0):
    if typ == 'huber':
        in1 = (x <= a)
        in2 = (x > a)
        in1_t = x[in1]**2 / 2
        in2_t = x[in2] * a - a**2 / 2
        L = np.sum(in1_t) + np.sum(in2_t)
    if typ == 'hampel':
        in1 = (x < a)
        in2 = np.logical_and(a <= x, x < b)
        in3 = np.logical_and(b <= x, x < c)
        in4 = (c <= x)
        in1_t = (x[in1]**2) / 2
        in2_t = a * x[in2] - a**2 / 2
        in3_t = (a * (x[in3] - c)**2 / (2 * (b - c))) + a * (b + c - a) / 2
        in4_t = np.ones(x[in4].shape) * a * (b + c - a) / 2
        L = np.sum(in1_t) + np.sum(in2_t) + np.sum(in3_t) + np.sum(in4_t)
    if typ == 'square':
        t = x**2
    if typ == 'abs':
        t = np.abs(x)
        L = np.sum(t)

    return L / x.shape[0]


def loss(x, typ='hampel', a=0, b=0, c=0):
    return rho(x, typ=typ, a=a, b=b, c=c) / x.shape[0]


def psi(x, typ='hampel', a=0, b=0, c=0):
    if typ == 'huber':
        return np.minimum(x, a)
    if typ == 'hampel':
        in1 = (x < a)
        in2 = np.logical_and(a <= x, x < b)
        in3 = np.logical_and(b <= x, x < c)
        in4 = (c <= x)
        in1_t = x[in1]
        in2_t = np.ones(x[in2].shape) * a
        in3_t = a * (c - x[in3]) / (c - b)
        in4_t = np.zeros(x[in4].shape)
        return np.concatenate((in1_t, in2_t, in3_t, in4_t)).reshape((-1, x.shape[1]))
    if typ == 'square':
        return 2 * x
    if typ == 'abs':
        return 1


def phi(x, typ='hampel', a=0, b=0, c=0):
    x[x == 0] = 10e-6
    return psi(x, typ=typ, a=a, b=b, c=c) / x


def irls(Km, type_rho, n, a, b, c, alpha=10e-8, max_it=100):
    """
    Iterative reweighted least-square
    """
    # init weights
    w = np.ones((n, 1)) / n
    #  first pass
    t1 = np.diag(Km).reshape((-1, 1))  #  (-1, dimension)
    t2 = - 2 * np.dot(Km, w)
    t3 = np.dot(np.dot(w.T, Km), w)
    t = t1 + t2 + t3
    norm = np.sqrt(t)
    J = loss(norm, typ=type_rho, a=a, b=b, c=c)
    stop = 0
    count = 0
    losses = [J]
    while not stop:
        count += 1
        # print("i: {}  loss: {}".format(count, J))
        J_old = J
        # update weights
        w = phi(norm, typ=type_rho, a=a, b=b, c=c)
        w = w / np.sum(w)
        t1 = np.diag(Km).reshape((-1, 1))  #  (-1, dimension)
        t2 = - 2 * np.dot(Km, w)
        t3 = np.dot(np.dot(w.T, Km), w)
        t = t1 + t2 + t3
        norm = np.sqrt(t)
        # update loss
        J = loss(norm, typ=type_rho, a=a, b=b, c=c)
        losses.append(J)
        if ((np.abs(J - J_old) < (J_old * alpha)) or (count == max_it)):
            print("Stop at {} iterations".format(count))
            stop = 1
    return w, norm, losses


def kde(X_data, X_plot, h, kernel='gaussian', return_model=False):
    kde_fit = KernelDensity(kernel=kernel, bandwidth=h).fit(X_data)
    if return_model:
        return np.exp(kde_fit.score_samples(X_plot)), kde_fit
    else:
        return np.exp(kde_fit.score_samples(X_plot))


def area_density(z, grid):
    if grid is None:
        print('\nWARNING: no grid ==> return area = 1')
        return 1
    shapes = [el.shape[0] for el in grid]
    area = np.trapz(z.reshape(shapes), grid[0], axis=0)
    for i_grid, ax in enumerate(grid[1:]):
        area = np.trapz(area, ax, axis=0)
    return area


def rkde(X_data, X_plot, h, type_rho='hampel', return_model=False):
    # kernel matrix
    n_samples, d = X_data.shape
    gamma = 1. / (2 * (h**2))
    Km = rbf_kernel(X_data, X_data, gamma=gamma) * \
        (2 * np.pi * h**2)**(-d / 2.)
    # find a, b, c via iterative reweighted least square
    a = b = c = 0
    alpha = 10e-8
    max_it = 100
    #  first it. reweighted least ssquare with rho = absolute function
    w, norm, losses = irls(Km, 'abs', n_samples, a, b, c, alpha, max_it)
    a = np.median(norm)
    b = np.percentile(norm, 75)
    c = np.percentile(norm, 95)
    # find weights via second iterative reweighted least square with input rho
    w, norm, losses = irls(Km, type_rho, n_samples, a, b, c, alpha, max_it)
    #  kernel evaluated on plot data
    gamma = 1. / (2 * (h**2))
    K_plot = rbf_kernel(X_plot, X_data, gamma=gamma) * \
        (2 * np.pi * h**2)**(-d / 2.)
    #  final density
    z = np.dot(K_plot, w)
    if return_model:
        return z, w
    else:
        return z


def spkde(X_data, X_plot, h, outliers_fraction, return_model=False):
    d = X_data.shape[1]
    beta = 1. / (1 - outliers_fraction)
    gamma = 1. / (2 * (h**2))
    G = rbf_kernel(X_data, X_data, gamma=gamma) * (2 * np.pi * h**2)**(-d / 2.)

    P = matrix(G)
    q = matrix(-beta / X_data.shape[0] * np.sum(G, axis=0))
    G = matrix(-np.identity(X_data.shape[0]))
    h_solver = matrix(np.zeros(X_data.shape[0]))
    A = matrix(np.ones((1, X_data.shape[0])))
    b = matrix(1.0)
    sol = solvers.qp(P, q, G, h_solver, A, b)
    a = np.array(sol['x']).reshape((-1, ))
    # final density
    GG = rbf_kernel(X_data, X_plot, gamma=gamma) * \
        (2 * np.pi * h**2)**(-d / 2.)
    z = np.zeros((X_plot.shape[0]))
    for j in range(X_plot.shape[0]):
        for i in range(len(a)):
            z[j] += a[i] * GG[i, j]
    if return_model:
        return z, a
    else:
        return z


def bandwidth_cvgrid(X_data, loo=False, kfold=5):
    """
    Compute the best bandwidth along a grid search.

    Parameters
    -----
    X_data : input data

    Returns
    -----
    h : the best bandwidth
    sigma : the search grid
    losses : the scores along the grid
    """
    print("Finding best bandwidth...")
    sigma = np.logspace(-1.5, 0.5, 80)  # grid for method 2 et 3
    if loo:
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': sigma},
                            cv=LeaveOneOut())
    else:
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': sigma},
                            cv=KFold(n_splits=kfold))
    grid.fit(X_data)
    h = grid.best_params_['bandwidth']
    losses = grid.cv_results_['mean_test_score']
    # print('best h: ', h)
    return h, sigma, losses


def ucv_objective(bandwidth, data, X_plot):
    n = len(data)
    kde_values = rkde(data, X_plot, bandwidth)
    error = np.sum((1 / (n * bandwidth)) - kde_values)**2
    return error


def bcv_objective(bandwidth, data, X_plot):
    n = len(data)
    h = 1.06 * np.std(data) * (n**(-1/5))  # Silverman's rule of thumb
    kde_values = rkde(data, X_plot, bandwidth)
    error = np.sum((1 / (n * h)) - kde_values)**2
    return error


def pso_bandwidth_selection(data, X_plot, objective=ucv_objective, inertia=.8, cognitive_weight=2.8):
    num_particles = 5
    num_iterations = 10
    bandwidth_min = 0.1
    bandwidth_max = 1.
    # x= np.linspace(min(data), max(data), len(data))

    def fitness(position):
        return objective(position, data, X_plot)

    best_global_position = None
    best_global_fitness = float('inf')

    swarm = np.random.uniform(bandwidth_min, bandwidth_max, (num_particles, 1))
    velocities = np.zeros_like(swarm)
    for _ in range(num_iterations):
        for i in range(num_particles):
            fitness_i = fitness(swarm[i])
            if fitness_i < best_global_fitness:
                best_global_position = swarm[i].copy()
                best_global_fitness = fitness_i

            # Update velocity
            velocities[i] = inertia * velocities[i] + cognitive_weight * \
                np.random.rand() * (best_global_position - swarm[i])

            # Update position
            swarm[i] += velocities[i]
            swarm[i] = np.clip(swarm[i], bandwidth_min, bandwidth_max)

    return best_global_position[0]

# Step 3: Implement HHO for bandwidth selection in RKDE

#!
# Created by "Boli" at 14:51, 17/08/2023 ----------%
#       Email: biadboze@gmail.com@gmail.com            %
#       Github: https://github.com/biad-iritie        %
# --------------------------------------------------%


class HHO_BandwidthSelection(Problem):

    def __init__(self, lb, ub, minmax, **kwargs):
        super().__init__(lb, ub, minmax, **kwargs)
        self.lb = lb
        self.ub = ub
        self.minmax = minmax
        self.data = kwargs["data"]
        self.X_plot = kwargs["X_plot"]

    def bcv_objective(self, bandwidth):
        data = self.data
        X_plot = self.X_plot
        n = len(data)
        h = 1.06 * np.std(data) * (n**(-1/5))  # Silverman's rule of thumb
        kde_values = rkde(data, X_plot, bandwidth)
        error = np.sum((1 / (n * h)) - kde_values)**2
        return error

    def ucv_objective(self, bandwidth):
        data = self.data
        X_plot = self.X_plot
        n = len(data)
        kde_values = rkde(data, X_plot, bandwidth)
        error = np.sum((1 / (n * bandwidth)) - kde_values)**2
        return error

    def fit_func(self, solution):
        return [self.bcv_objective(solution), self.ucv_objective(solution)]

    """ def fitness(self, bandwidth, X_plot):
        # Implement the fitness function using least-square UCV or BCV
        # (You can use the RKDE and IRLS functions here)
        return self.fitness_func(bandwidth, self.data, self.X_plot)

    def get_bounds(self):
        return ([self.lower_bound], [self.upper_bound])
 """

    def get_name(self):
        return "HHO for Bandwidth Selection"


def hho_bandwith_selection(data, X_plot):

    problem_multi = HHO_BandwidthSelection(
        lb=np.array([.1]), ub=np.array([.9]), minmax="min", obj_weights=[1, 1], data=data, X_plot=X_plot)
    # Define the model and solve the problem
    # epoch = 1000
    epoch = 10  # maximum number of iterations
    # pop_size = 50
    pop_size = 10  # number of population size
    model = OriginalHHO(epoch, pop_size)
    best_position, best_fitness = model.solve(problem_multi)
    print("Result hho_bandwith_selection: {}".format(best_position))
    return best_position[0]
