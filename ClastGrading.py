
import numpy as np
from landlab import Component
import scipy.stats as stats
import time
from landlab.grid.nodestatus import NodeStatus


class ClastGrading(Component):
    """Simulate fragmentation of soil grains through time .

        Landlab component that simulates soil grading based on mARM (Cohen et al.,
        2010) approach. 

        The fragmentation process is controlled by weathering transition matrix which defines 
        the relative mass change in each soil grain size class (grading class) as a result of the fracturing 
        of particles in the weathering mechanism.

        The primary method of this class is :func:`run_one_step`.

        References
        ----------
        **Required Software Citation(s) Specific to this Component**

        Adams, J., Gasparini, N., Hobley, D., Tucker, G., Hutton, E., Nudurupati,
        S., Istanbulluoglu, E. (2017). The Landlab v1. 0 OverlandFlow component:
        a Python tool for computing shallow-water flow across watersheds.
        Geoscientific Model Development  10(4), 1645.
        https://dx.doi.org/10.5194/gmd-10-1645-2017

        **Additional References**

        de Almeida, G., Bates, P., Freer, J., Souvignet, M. (2012). Improving the
        stability of a simple formulation of the shallow water equations for 2-D
        flood modeling. Water Resources Research 48(5)
        https://dx.doi.org/10.1029/2011wr011570

        """


    _name = 'ClastGrading'
    _unit_agnostic = True
    _info = {
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Topographic elevation at node",
        },
        "soil__depth":{
          "dtype":float,
          "intent":"out",
          "optional":False,
          "units": "m",
          "mapping":"node",
          "doc":"Soil depth at node",
      },
        "bedrock__elevation": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Bedrock elevation at node",
        },
        "grain__weight": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "Sediment weight for all size fractions",
        },
        "median__size_weight": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "The median grain size in each node based on grain size weight",
        },
        "fraction_sizes": {
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
                 grading_name =  'p2-0-100',  # Fragmentation pattern (string)
                 n_size_classes = 10,         # Number of size classes
                 alpha = 1,                   # Fragmentation rate
                 clast_density = 2000,        # Particles density [kg/m3]
                 phi = 0.4,                   # Porosity [-]
    ):
        super(ClastGrading, self).__init__(grid)

        self._grading_name = grading_name
        self._alpha = alpha
        self._n_sizes = n_size_classes
        self._N = int(self._grading_name.split('-')[0][1:])
        self._clast_density = float(clast_density)
        self._phi = phi

        # Create out fields.
        # Note: Landlabs' init_out_field procedure will not work for the 'grain__weight' and 'fraction_sizes' fields
        # because the shape of these fields is: n_nodes x n_grain_sizes.
        grid.add_field("soil__depth", np.zeros((grid.shape[0], grid.shape[1])), at="node", dtype=float)
        grid.add_field('median__size_weight',np.zeros((grid.shape[0], grid.shape[1])), at='node', dtype=float)
        grid.add_field("bedrock__elevation", np.zeros((grid.shape[0], grid.shape[1])), at="node", dtype=float)
        grid.add_field("fraction_sizes", np.ones((grid.shape[0], grid.shape[1], self._n_sizes )), at="node", dtype=float)
        grid.add_field("grain__weight", np.ones((grid.shape[0], grid.shape[1], self._n_sizes)), at="node", dtype=float)


    def create_transition_mat(self,
                            n_fragments=2,
                            A_factor = 1,
                            volume_precent_in_spread = 10):

        # A matrix is this is the volume/weight that weathered from each size fraction in each step
        # A matrix controls the fragmentation pattern
        # A_factor matrix is the fragmentation factor / rate

        self._A = np.zeros((self._n_sizes, self._n_sizes))
        precents = np.array([float(s) for s in self._grading_name.split('-') if s.replace('.', '', 1).isdigit()])
        if 'spread' in self._grading_name:
            precents_to_add = np.ones((1, n_fragments)) * volume_precent_in_spread
            precents = np.append(precents, precents_to_add)
        alphas_fractios = precents / 100
        self._A_factor = np.ones_like(self._A) * A_factor

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

    def set_grading_classes(self, maxsize = None,
                            power_of = 1 / 3,
                            input_sizes_flag = False,
                            meansizes =None):

        # The grain-size classes could be a-priori set based on
        # geomotery relations and known fragmentation pattern
        # (following Cohen et al., 2010)

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

        self._clast_volumes = (self._meansizes / 2) ** 3 * np.pi * (4 / 3)  # [L^3]
        self.grid.at_node["fraction_sizes"] *= self._meansizes
        self.update_sizes()

    def create_dist(self,
                    median_size = 0.05,
                    std = None,
                    num_of_clasts = 10000,
                    grading_name = 'g_state0',
                    init_val_flag = False,
                    CV = 0.8):

        n_classes = self._n_sizes
        median_size = median_size
        num_of_clasts = num_of_clasts
        if std == None:
            std = median_size * CV

        lower = np.min(self._lowerlims)
        upper = np.max(self._upperlims)
        b = stats.truncnorm(
            (lower - median_size) / std, (upper - median_size) / std, loc=median_size, scale=std)
        b = b.rvs(num_of_clasts)
        locals()[grading_name] = np.histogram(b, np.insert(self._upperlims, 0, 0))[0]
        if init_val_flag:
            self.g_state = np.ones(
                (int(np.shape(self.grid)[0]), int(np.shape(self.grid)[1]), int(len(locals()[grading_name])))) * locals()[grading_name]
            self.g_state0 = locals()[grading_name]
            self.grid.at_node['grain__weight'] *= locals()[grading_name]
            layer_depth = np.sum(self.g_state0) / (self._clast_density * self.grid.dx * self.grid.dx )  # meter
            layer_depth /= (1-self._phi) # Porosity correction
            self.grid.at_node['soil__depth'] +=  layer_depth
            self.grid.at_node['bedrock__elevation'] = np.copy(self.grid.at_node['topographic__elevation'])
            self.grid.at_node['topographic__elevation'] = self.grid.at_node['bedrock__elevation']  + self.grid.at_node['soil__depth']

        else:
            self.g_state_slide = locals()[grading_name]
            self.grid.add_field("bed_grains__proportions", np.ones((self.grid.shape[0], self.grid.shape[1], self._n_sizes)), at="node",
                           dtype=float)
            self.grid.at_node["bed_grains__proportions"] *= np.divide(self.g_state_slide,np.sum(self.g_state_slide))

    def update_sizes(self):

        grain_number = self.grid.at_node['grain__weight'] / (self._clast_volumes * self._clast_density)
        grain_volume = grain_number * (self._clast_volumes)  # Volume

        # Median size based on weight
        cumsum_gs = np.cumsum(self.grid.at_node['grain__weight'], axis=1)
        sum_gs = np.sum(self.grid.at_node['grain__weight'], axis=1)
        self.grid.at_node['median__size_weight'][sum_gs <= 0] = 0
        sum_gs_exp = np.expand_dims(sum_gs, -1)

        median_val_indx = np.argmax(np.where(
                np.divide(
                    cumsum_gs,
                    sum_gs_exp,
                    out=np.zeros_like(cumsum_gs),
                    where=sum_gs_exp != 0)>=0.5, 1, 0)
            ,axis=1
        )

        self.grid.at_node['median__size_weight'] = self._meansizes[median_val_indx[:]]

    def run_one_step(self, A_factor=None):
        # Run one step of fragmentation
        # 'update_sizes' procedure is called after

        if np.any(A_factor == None):
            A_factor = self._A_factor
        temp_g_weight = np.moveaxis(np.dot(self._A * A_factor, np.swapaxes(
            np.reshape(self.grid.at_node['grain__weight'], (self.grid.shape[0],
                                                            self.grid.shape[1],
                                                            self._n_sizes)), 1,
            2)),
                                    0, -1)
        self.grid.at_node['grain__weight'] += np.reshape(temp_g_weight,
                                                         (self.grid.shape[0] * self.grid.shape[1],
                                                          self._n_sizes))
        self.update_sizes()