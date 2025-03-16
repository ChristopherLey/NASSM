import numpy as np
import multiprocessing as mp
import timeit


def Rastringin_Function(x, args=[]):
    return float(300 + np.sum(np.power(x, 2.0) - 10.0 * np.cos(2 * np.pi * x)))


def functional_test(max_iterations=5000, number_of_cores=1, verbose=False):
    obj = Objective(Rastringin_Function, (-5.12, 5.12), 30, number_of_cores=number_of_cores)
    topology = VonNeumannTopology()
    pso = FIPS_PSO(obj, topology)
    best = pso.optimise(max_iterations=max_iterations, verbose=True)
    print('Best Position:', str(best))
    print('Best Score:', str(pso.global_best_score))


def sudo_objective(particle_positions, objective_fcn, args):
    return args[0], objective_fcn(particle_positions, args[1:])


class Objective(object):
    """" Objective Function"""
    def __init__(self, objective_function, init_range, input_rows, boundaries=False, additional_args=[], number_of_cores=1):
        self.input_rows = input_rows
        self.output_rows = 1
        self.objective_function = objective_function
        self.init_range = init_range
        self.boundaries = boundaries
        self.additional_args=additional_args
        if self.boundaries:
            self.lower_bound = np.ones((input_rows, 1)) * init_range[0]
            self.upper_bound = np.ones((input_rows, 1)) * init_range[1]
        self.number_of_cores = number_of_cores
        if number_of_cores > 1:
            self.multiprocessing = True
        else:
            self.multiprocessing = False


    def evaluate_objectives(self, particle_positions):
        pool_size = particle_positions.shape[1]
        results = np.matrix(np.zeros((1, pool_size)))
        if self.multiprocessing:
            pool = mp.Pool(processes=self.number_of_cores)
            temp = [pool.apply_async(sudo_objective,
                                     args=(particle_positions[:, i],
                                           self.objective_function,
                                           [i] + self.additional_args,))
                    for i in range(pool_size)]
            # temp = pool.map_async(sudo_objective,
            #                        range(pool_size),
            #                          args=(particle_positions[:, i],
            #                                self.objective_function,
            #                                [i] + self.additional_args,))
            for processes in temp:
                thread = processes.get()
                results[0, thread[0]] = thread[1]
            # clean up
            pool.close()
            pool.join()
            return results
        else:
            for i in range(pool_size):
                (index, results[0, i]) = sudo_objective(particle_positions[:, i],
                                                        self.objective_function,
                                                        [i] + self.additional_args)
            return results

    def init_x(self, particle_population_size):
        return np.matrix(np.random.uniform(self.init_range[0], self.init_range[1],
                                           (self.input_rows, particle_population_size)))


class VonNeumannTopology(object):
    """ von Neumman Square Topology"""
    def __init__(self):
        self.neighbours = np.array([
            [15, 1, 5, 4],      # 0's neighbours
            [16, 2, 6, 0],      # 1's
            [17, 3, 7, 1],      # 2's
            [18, 4, 8, 2],      # 3's
            [19, 0, 9, 3],      # 4's
            [0, 6, 10, 9],      # 5's
            [1, 7, 11, 5],      # 6's
            [2, 8, 12, 6],      # 7's
            [3, 9, 13, 7],      # 8's
            [4, 5, 14, 8],      # 9's
            [5, 11, 15, 14],    # 10's
            [6, 12, 16, 10],    # 11's
            [7, 13, 17, 11],    # 12's
            [8, 14, 18, 12],    # 13's
            [9, 10, 19, 13],    # 14's
            [10, 16, 0, 19],    # 15's
            [11, 17, 1, 15],    # 16's
            [12, 18, 2, 16],    # 17's
            [13, 19, 3, 17],    # 18's
            [14, 15, 4, 18]     # 19's
        ])
        self.shape = (4, 20)


class CircularTopology(object):
    def __init__(self, num_neighbours):
        pass    # TODO Generate neigbours array


class FIPSParticle(object):
    def __init__(self, current_pos, objective_result, num_neighbours, objective_object):
        self.x = np.matrix(np.copy(current_pos))
        self.v = np.matrix(np.zeros(current_pos.shape))
        self.num_neighbors = num_neighbours + 1
        self.p_best = np.matrix(np.zeros((current_pos.shape[0], self.num_neighbors)))
        self.p_best[:, -1] = current_pos[:, 0]
        self.shape = self.x.shape[0]
        self.best_score = self.current_score = np.copy(objective_result)
        self.phi = 4.1
        self.chi = 0.729844
        self.boundaries = objective_object.boundaries
        if self.boundaries:
            self.lower_bound = objective_object.lower_bound
            self.upper_bound = objective_object.upper_bound

    def evolve(self):
        neighbourhood = np.matrix(np.copy(self.p_best))
        for i in range(0, self.num_neighbors):
            neighbourhood[:, i] -= self.x
        self.v = self.chi*(self.v +
                           np.sum(np.multiply(np.random.uniform(0, self.phi, (self.shape, self.num_neighbors)),
                                  neighbourhood) /
                                  self.num_neighbors,
                                  axis=1))
        self.x += self.v
        if self.boundaries:
            if (self.x > self.upper_bound).any():
                self.x + 2.0 * np.multiply(self.upper_bound - self.x, self.x > self.upper_bound)
            elif (self.x < self.lower_bound).any():
                self.x + 2.0 * np.multiply(self.lower_bound - self.x, self.x < self.lower_bound)

    def update_objective(self, results):
        self.current_score = results
        if self.current_score < self.best_score:
            self.best_score = self.current_score
            self.p_best[:, -1] = self.x[:, 0]


class FIPS_PSO(object):
    """ Basic PSO implementation"""
    def __init__(self, objective_object, topology):
        self.obj = objective_object
        self.num_particles = topology.shape[1]
        # self.vector_dim = objective_function.input_matrix_rows
        self.topology = topology
        """ initiate particle population"""
        self.particles = []
        initial_pos = objective_object.init_x(self.num_particles)
        self.results = self.obj.evaluate_objectives(initial_pos)
        for i in range(0, self.num_particles):
            self.particles.append(FIPSParticle(initial_pos[:, i],
                                               self.results[0, i],
                                               self.topology.shape[0],
                                               objective_object))
        self.set_p_local()
        self.global_best_score = np.infty
        self.global_best_pos = []
        # print "Finished initialising"

    def optimise(self, max_iterations=1000, verbose=False):
        current_pos = np.matrix(np.zeros((self.obj.input_rows, self.num_particles)))
        initial_time = timeit.default_timer()  # Timer
        for i in range(0, max_iterations):
            start_time = timeit.default_timer()  # Timer
            for k in range(0, self.num_particles):
                self.particles[k].evolve()
                current_pos[:, k] = self.particles[k].x[:, 0]
            self.results[0, :] = self.obj.evaluate_objectives(current_pos)
            for k in range(0, self.num_particles):
                self.particles[k].update_objective(self.results[0, k])
            elapsed = timeit.default_timer() - start_time
            if verbose:
                print("Elapsed time:", str(elapsed), "sec @ iteration:", str(i))
            self.set_p_local()
            for k in range(0, self.num_particles):
                if self.particles[k].best_score < self.global_best_score:
                    self.global_best_score = self.particles[k].best_score
                    self.global_best_pos = np.copy(self.particles[k].p_best[:, -1])
                    if verbose:
                        print("FIPS: interation", str(i), " found a better score: ", str(self.global_best_score))
                        # print "with position: " + str(self.global_best_pos)
        final_time = timeit.default_timer() - initial_time
        print("Finished Optimising (", str(final_time/60), " min)")
        return self.global_best_pos

    def set_p_local(self):
        for i in range(0, self.num_particles):
            for j in range(0, self.topology.shape[0]):
                self.particles[i].p_best[:, j] = self.particles[self.topology.neighbours[i, j]].p_best[:, -1]


class CanonicalParticle(object):
    """ Basic Particle"""
    def __init__(self, current_pos, objective_function):
        self.x = np.copy(current_pos)
        self.p_best = np.copy(current_pos)
        self.v = np.zeros(current_pos.shape)
        self.p_local = np.zeros(current_pos.shape)
        self.shape = self.x.shape
        self.objective_function = objective_function
        self.best_score = self.current_score = objective_function(self.x)
        self.phi = 2.05
        self.chi = 0.729844

    def evolve(self):
        self.v = self.chi*(self.v +
                           np.random.uniform(0, self.phi, self.shape)*(self.p_best - self.x) +
                           np.random.uniform(0, self.phi, self.shape)*(self.p_local - self.x))
        self.x += self.v
        self.current_score = self.objective_function(self.x)
        if self.current_score < self.best_score:
            self.best_score = self.current_score
            self.p_best[:] = self.x[:]

    def set_v_init(self):
        self.v = self.chi * (np.random.uniform(0, self.phi, self.shape) * (self.p_local - self.x))


class PSO(object):
    """ Basic PSO implementation"""
    def __init__(self, objective_object, topology):
        self.num_particles = topology.shape[1]
        # self.vector_dim = objective_function.input_matrix_rows
        self.topology = topology
        """ initiate particle population"""
        self.particles = []
        initial_pos = objective_object.init_x(self.num_particles)
        for i in range(0, self.num_particles):
            self.particles.append(CanonicalParticle(initial_pos[:, i],
                                                    objective_object.objective_function))
        self.set_p_local()
        self.global_best_score = np.infty
        self.global_best_pos = []

    def optimise(self, max_iterations, number_of_cores=1):
        pool = mp.Pool(processes=number_of_cores)
        pool_size = self.topology.shape[1]
        for i in range(0, max_iterations):
            start_time = timeit.default_timer()  # Timer
            # [pool.apply_async(self.particles[j].evolve()) for j in range(pool_size)]
            for k in range(0, self.num_particles):
                self.particles[k].evolve()
            elapsed = timeit.default_timer() - start_time
            print("Elapsed time: ", str(elapsed), " sec @ iteration: ", str(i))
            self.set_p_local()
            for k in range(0, self.num_particles):
                if self.particles[k].best_score < self.global_best_score:
                    self.global_best_score = self.particles[k].best_score
                    self.global_best_pos = self.particles[k].p_best
                    msg = "Canonical: interation " + str(i) + " found a better score: " + str(self.global_best_score)
                    print(msg)
                    msg = "with position: " + str(self.global_best_pos)
                    print(msg)
        print("Finished Optimising")
        return self.global_best_pos

    def set_p_local(self):
        for i in range(0, self.num_particles):
            score = self.particles[self.topology.neighbours[i, 0]].best_score
            self.particles[i].p_local[:] = self.particles[self.topology.neighbours[i, 0]].p_best[:]
            for j in range(1, self.topology.shape[0]):
                if self.particles[self.topology.neighbours[i, j]].best_score < score:
                    score = self.particles[self.topology.neighbours[i, j]].best_score
                    self.particles[i].p_local[:] = self.particles[self.topology.neighbours[i, j]].p_best[:]


if __name__ == "__main__":
    functional_test()
