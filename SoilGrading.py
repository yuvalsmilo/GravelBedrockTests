
import numpy as np
from landlab import Component
import scipy.stats as stats
import time
from landlab.grid.nodestatus import NodeStatus


class SoilGrading(Component):
    """Simulate fragmentation of soil grains through time .

        Landlab component that simulates grading of soil particles through time
        based on mARM (Cohen et al., 2010) approach.

        The fragmentation process is controlled by weathering transition matrix which defines
        the relative mass change in each soil grain size class (grading class) as a result of the fracturing
        of particles in the weathering mechanism.

        The primary method of this class is :func:`run_one_step`.

        References
        ----------
        Cohen, S., Willgoose, G., & Hancock, G. (2009). The mARM spatially distributed soil evolution model:
        A computationally efficient modeling framework and analysis of hillslope soil surface organization.
        Journal of Geophysical Research: Earth Surface, 114(F3).

        Cohen, S., Willgoose, G., & Hancock, G. (2010). The mARM3D spatially distributed soil evolution model:
        Three‐dimensional model framework and analysis of hillslope and landform responses.
        Journal of Geophysical Research: Earth Surface, 115(F4).
        """


    _name = 'SoilGrading'
    _unit_agnostic = True
    _info = {
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional":True,
            "units": "m",
            "mapping": "node",
            "doc": "Topographic elevation at node",
        },
        "bedrock__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": True,
            "units": "m",
            "mapping": "node",
            "doc": "Bedrock elevation at node",
        },
        "soil__depth": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Soil depth at node",
        },
        "grains__weight": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "Weight of grains in each size fractions",
        },
        "median_size__weight": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "The median grain size in each node based on grains weight distribution",
        },
        "grains_fractions__size": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "The size of soil grain fractions",
        },
        "bed_grains__proportions":
            {
                "dtype": float,
                "intent": "out",
                "optional": True,
                "units": "m",
                "mapping": "node",
                "doc": "Weight proportion of each grain size fractions in the bed",
            },
        }


    def __init__(self,
                 grid,
                 grading_name = 'p2-0-100',         # Fragmentation pattern (string)
                 n_of_grainsize_classes = 10,       # Number of size classes
                 alpha = 1,                         # Fragmentation rate
                 A_factor = 0.001,
                 soil_density = 2650,               # Soil density [kg/m3]
                 phi = 0.4,                         # Soil porosity [-]
                 grain_max_size = 0.02,
                 power_of = 1/3,
                 initial_median_size = 0.002,       # Initial median grain size [m]
                 initial_total_soil_weight = 2650, 
                 std = None,
                 CV = 0.6,
                 is_bedrock_distribution_flag = False,
                 precent_of_volume_in_spread = 10,
                 
                 
                 
    ):
        super(SoilGrading, self).__init__(grid)

        self._grading_name = grading_name
        self._alpha = alpha
        self._A_factor = A_factor
        self._n_sizes = n_of_grainsize_classes
        self._N = int(self._grading_name.split('-')[0][1:])
        self._soil_density = float(soil_density)
        self._phi = phi
        self._grain_max_size = grain_max_size
        self._power_of = power_of
        self._initial_median_size = initial_median_size
        self._CV = CV
        self._initial_total_soil_weight = initial_total_soil_weight
        if std == None:
            std = median_size * CV
        self._std = std
        self._is_bedrock_distribution_flag  = is_bedrock_distribution_flag
        self._precent_of_volume_in_spread = precent_of_volume_in_spread
        # Create out fields.
        # Note: Landlabs' init_out_field procedure will not work for the 'grains__weight' and 'grains_fractions__size' fields
        # because the shape of these fields is: n_nodes x n_grain_sizes.
        grid.add_field("soil__depth", np.zeros((grid.shape[0], grid.shape[1])), at="node", dtype=float)
        grid.add_field('median_size__weight',np.zeros((grid.shape[0], grid.shape[1])), at='node', dtype=float)
        grid.add_field("grains_fractions__size", np.ones((grid.shape[0], grid.shape[1], self._n_sizes )), at="node", dtype=float)
        grid.add_field("grains__weight", np.ones((grid.shape[0], grid.shape[1], self._n_sizes)), at="node", dtype=float)
        grid.add_field("bed_grains__proportions", np.ones((self._grid.shape[0], self._grid.shape[1], self._n_sizes)), at="node", dtype=float)
        
        if "topographic__elevation" not in grid.at_node:
            grid.add_field("topographic__elevation",
                           np.zeros_like(self._grid.nodes.flatten()), at="node", dtype=float)

        if "bedrock__elevation" not in grid.at_node:
            grid.add_field("bedrock__elevation",
                           np.zeros_like(self._grid.nodes.flatten()), at="node", dtype=float)
    
        
        self.create_transition_mat()
        self.set_grading_classes()
        self.create_dist()

    def create_transition_mat(self):

        # A matrix is this is the volume/weight that weathered from each size fraction in each step
        # A matrix controls the fragmentation pattern
        # A_factor matrix is the fragmentation factor / rate

        self._A = np.zeros((self._n_sizes, self._n_sizes))
        precents = np.array([float(s) for s in self._grading_name.split('-') if s.replace('.', '', 1).isdigit()])
        if 'spread' in self._grading_name:
            precents_to_add = np.ones((1, self._n_sizes)) * self._precent_of_volume_in_spread
            precents = np.append(precents, precents_to_add)
        alphas_fractios = precents / 100
        self._A_factor = np.ones_like(self._A) * self._A_factor

        for i in range(self._n_sizes):

            if i == 0:
                self._A[i, i] = 0
            elif i == self._n_sizes:
                self._A[i, i] = -(self._alpha - (
                            self._alpha * alphas_fractios[0])
                                 )
            else:
                self._A[i, i] = -(self._alpha - (
                        self._alpha * alphas_fractios[0])
                                 )
                cnti = i - 1 # rows,
                cnt = 1
                while cnti >= 0 and cnt <= (len(alphas_fractios) - 1):
                    self._A[cnti, i] = (self._alpha * alphas_fractios[cnt])
                    if cnti == 0 and cnt <= (len(alphas_fractios) - 1):
                        self._A[cnti, i] = (1 - alphas_fractios[0]) - np.sum(alphas_fractios[1:cnt])
                        cnt += 1
                        cnti -= 1
                    cnt += 1
                    cnti -= 1

    def set_grading_classes(self, meansizes = None,
                            input_sizes_flag = False):
        
        maxsize = self._grain_max_size
        power_of = self._power_of
        if meansizes is not None:
            input_sizes_flag = True
        
        # The grain-size classes could be a-priori set based on
        # a given geometery relations and known fragmentation pattern

        def lower_limit_of(maxsize):
            lower_limit = maxsize * (1 / self._N) ** (power_of)
            return lower_limit

        upperlimits = []
        lowerlimits = []
        num_of_size_classes_plusone = self._n_sizes + 1
        if input_sizes_flag:
            self._meansizes = np.array(meansizes)
            self._upperlims = np.array(meansizes)
            self._lowerlims = np.insert(np.array(meansizes),0,0)
        else:
            for _ in range(num_of_size_classes_plusone-1):
                upperlimits.append(maxsize)
                maxsize = lower_limit_of(maxsize)
                lowerlimits.append(maxsize)

            self._upperlims = np.sort(upperlimits)
            self._lowerlims = np.sort(lowerlimits)
            self._meansizes = ( np.array(self._upperlims) + np.array(self._lowerlims) )/ 2

        self.grid.at_node["grains_fractions__size"] *= self._meansizes
        self.update_median__size()

    def create_dist(self, median_size = None, std = None,
                    is_bedrock_distribution_flag = False):

        # Pointers
        total_soil_weight = self._initial_total_soil_weight
        if not is_bedrock_distribution_flag:
            median_size = self._initial_median_size
            std = self._std
            is_bedrock_distribution_flag = self._is_bedrock_distribution_flag
        
        else:
            if median_size == None:
                median_size = self._initial_median_size
                std = self._std
            else:
                median_size = median_size 
                if std == None:
                    std = self._CV * median_size
                else:
                    std =std
                    
        lower = np.min(self._lowerlims)
        upper = np.max(self._upperlims)
        print(median_size)
        b = stats.truncnorm(
            (lower - median_size) / std, (upper - median_size) / std, loc = median_size, scale = std)
        b = b.rvs(total_soil_weight)
        print(b )

        grains_weight__distribution = np.histogram(b, np.insert(self._upperlims, 0, 0))[0]
        
        if not is_bedrock_distribution_flag:
            self.g_state = np.ones(
                (int(np.shape(self.grid)[0]), int(np.shape(self.grid)[1]), int(len(grains_weight__distribution)))) * grains_weight__distribution
            
            self.g_state0 = grains_weight__distribution
            self.grid.at_node['grains__weight'] *= grains_weight__distribution
            layer_depth = np.sum(self.g_state0) / (self._soil_density * self.grid.dx * self.grid.dx )  # meter
            layer_depth /= (1-self._phi) # Porosity correction

            self.grid.at_node['soil__depth'] +=  layer_depth
            self.grid.at_node['topographic__elevation'] = self.grid.at_node['soil__depth'] + self.grid.at_node['bedrock__elevation']

            self.g_state_bedrock = grains_weight__distribution
            self.grid.at_node["bed_grains__proportions"][:] = 1
            self.grid.at_node["bed_grains__proportions"] *= np.divide(self.g_state_bedrock, np.sum(self.g_state_bedrock))

        else:

            self.g_state_bedrock = grains_weight__distribution
            self.grid.at_node["bed_grains__proportions"][:] = 1
            self.grid.at_node["bed_grains__proportions"] *= np.divide(self.g_state_bedrock,np.sum(self.g_state_bedrock))

    def update_median__size(self):

        # Median size based on weight distribution
        cumsum_gs = np.cumsum(self.grid.at_node['grains__weight'], axis=1)
        sum_gs = np.sum(self.grid.at_node['grains__weight'], axis=1)
        self.grid.at_node['median_size__weight'][sum_gs <= 0] = 0
        sum_gs_exp = np.expand_dims(sum_gs, -1)

        median_val_indx = np.argmax(np.where(
                np.divide(
                    cumsum_gs,
                    sum_gs_exp,
                    out=np.zeros_like(cumsum_gs),
                    where=sum_gs_exp != 0)>=0.5, 1, 0)
            ,axis=1
        )

        self.grid.at_node['median_size__weight'] = self._meansizes[median_val_indx[:]]

    def run_one_step(self, A_factor=None):
        # Run one step of fragmentation
        # 'update_median__size' procedure is called after

        if np.any(A_factor == None):
            A_factor = self._A_factor

        temp_g_weight = np.moveaxis(
            np.dot(
                self._A * A_factor, np.swapaxes(
                    np.reshape(self.grid.at_node['grains__weight'], (self.grid.shape[0],
                                                        self.grid.shape[1],
                                                        self._n_sizes)),
                    1,2)),
            0, -1)
        
        
        self.grid.at_node['grains__weight'] += np.reshape(temp_g_weight,
                                                         (self.grid.shape[0] * self.grid.shape[1],
                                                          self._n_sizes))
        self.update_median__size()