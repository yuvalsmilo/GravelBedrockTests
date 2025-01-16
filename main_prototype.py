

from landlab import RasterModelGrid
from gravel_bedrock_eroder.gravel_bedrock_eroder import GravelBedrockEroder
from landlab.components import FlowAccumulator
from soil_grading import SoilGrading
from matplotlib import pyplot as plt
import numpy as np
from landlab import imshow_grid
import noise
from landlab.components import PriorityFloodFlowRouter
from landlab.components import LinearDiffuser

""" Create grading features with different fragmentation models. 
Fragmentation models are describe by the following string:
      !!   pX-AAA-BBB-CCC-DDD'  !!       
while
X = number of daughter particles
AAA = weight % that *REMAINS* in  parent after fragmentation
BBB / CCC / DDD ... = proportion of daughters '
"""

## Flag for an apriori grain size disribution OR for generating a distribution
known_dist_flag = False

# Global parameters
porosity = 0.4                              # [-]
soil_density = 2650                         # [kg/m^3]
initial_sediment_weight_at_node = 100000    # [kg]
grading_name = 'p2-0-100'  # ,'p2-10-10-spread'  #,'p2-0-100'

xy_spacing=500
#grid = RasterModelGrid((200, 3), xy_spacing=xy_spacing)
grid = RasterModelGrid((50, 50), xy_spacing=xy_spacing)
elev = grid.add_zeros("topographic__elevation", at="node")

if known_dist_flag:
    mean_grainsize_per_class = [0.002, 0.02, 0.2, 2, 10]  # [0.002, 0.2, 1 ]  # [m]
    init_median_grainsize_at_node = 0.2           # [m]
    n_size_classes = np.size(mean_grainsize_per_class )
    A_factor = 0.001
else:
    init_median_grainsize_at_node = 0.01   # [m]
    n_size_classes = 2
    maxsize = init_median_grainsize_at_node+(init_median_grainsize_at_node*2)
    A_factor = 0.001

# sg = SoilGrading(grid,
#                  initial_median_size = init_median_grainsize_at_node,
#                  grain_max_size=maxsize,
#                  n_of_grainsize_classes = n_size_classes, phi = porosity,
# initial_total_soil_weight = 2650*1000000
# )

sg = SoilGrading(grid,
                        meansizes=[0.005, 0.01, 0.05],
                        grains_weight=[2650*1000000*0.3, 2650*1000000*0.3, 2650*1000000*0.3],
                        phi=porosity)

# #10-20
# fa = PriorityFloodFlowRouter(
#     grid,
#     flow_metric="D8",
#     separate_hill_flow=True,
#     hill_flow_metric="Quinn",
#     update_hill_flow_instantaneous=True, depression_handler='fill',
#     runoff_rate=1
# )


fa = FlowAccumulator(grid, runoff_rate=1)

fa.run_one_step()
eroder = GravelBedrockEroder(
    grid, sediment_porosity = porosity, abrasion_coefficients = 0.0005,
    number_of_sediment_classes  = 3,
    plucking_coefficient= 10**-7,
    depth_decay_scale=0.5)#    plucking_coefficient= 10**-3

diffuser = LinearDiffuser(grid, linear_diffusivity=0.001)

# eroder = GravelBedrockEroder(
#     grid,
#     plucking_coefficient= 10**-2,
#     depth_decay_scale=0.5)#    plucking_coefficient= 10**-3


rock_elev = grid.at_node["bedrock__elevation"]
rock_elev[:] +=100
#rock_elev[grid.nodes[:,1]] += grid.node_y[grid.nodes[:,1]]*(0.001)
elev[:] = rock_elev[:] + grid.at_node['soil__depth']
# grid.status_at_node[grid.perimeter_nodes] = grid.BC_NODE_IS_CLOSED
# grid.status_at_node[5] = grid.BC_NODE_IS_FIXED_VALUE


grid.set_closed_boundaries_at_grid_edges(False, False, False, False)

noise_mat = np.zeros_like(grid.at_node['soil__depth'][grid.nodes])
scale = 200
octaves = 20
persistence = 0.5
lacunarity =55

for i in range(np.shape(noise_mat)[0]):
    for j in range(np.shape(noise_mat)[1]):
        noise_mat[i][j] = noise.pnoise2(i / scale,
                                        j / scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=np.shape(noise_mat)[1],
                                        repeaty=np.shape(noise_mat)[0],
                                        base=2
                                        )

rock_elev[:] += np.abs(noise_mat[:].flatten())

# rock_elev[:] += np.random.rand(np.shape(rock_elev)[0])
elev[:] = rock_elev[:] + grid.at_node['soil__depth']

# save inital weight per grain size
#init_weight_per_grainsize = np.copy(grid.at_node['grains__weight'][4,:])

# MAIN LOOP
n_steps = 1050000
fig_cnt = 0
xvec  = np.arange(0, np.size(grid.at_node['topographic__elevation'][grid.core_nodes]))*grid.dx
for i,_ in enumerate(range(n_steps )):
    rock_elev[grid.core_nodes] += 0.001
    elev[grid.core_nodes] =rock_elev[grid.core_nodes] + grid.at_node['soil__depth'][grid.core_nodes]
    fa.run_one_step()
    eroder.run_one_step(1)
    diffuser.run_one_step(1)
    if i>= fig_cnt:
        fig, ax = plt.subplots()
        ax.plot(xvec, grid.at_node['topographic__elevation'][grid.core_nodes], color='black')
        ax.plot(xvec, grid.at_node['bedrock__elevation'][grid.core_nodes], color='blue')
        ax2 = ax.twinx()
        ax2.plot(xvec, grid.at_node['median_size__weight'][grid.core_nodes], color='salmon')
        ax.set_xlabel('Distnace upstream')
        ax.set_ylabel('Elevation [m]')
        ax2.set_ylabel('Grainsize [m]')
        plt.title(f'time step = {i}')
        #ax.axis('equal')
        ax.set_ylim([0,np.max(grid.at_node['topographic__elevation'])+100])
        plt.tight_layout()
        plt.show()
        fig_cnt+=5500
    sg.update_median_grain_size()

fig, ax = plt.subplots()
ax.plot(xvec, grid.at_node['topographic__elevation'][grid.core_nodes], color='black')
ax.plot(xvec, grid.at_node['bedrock__elevation'][grid.core_nodes], color='blue')
ax2 = ax.twinx()
ax2.plot(xvec, grid.at_node['median_size__weight'][grid.core_nodes], color='salmon')
ax.set_xlabel('Distnace upstream')
ax.set_ylabel('Elevation [m]')
ax2.set_ylabel('Grainsize [m]')
plt.title(f'time step = {i}')
# ax.axis('equal')
ax.set_ylim([np.min(grid.at_node['bedrock__elevation'][grid.core_nodes]),np.max(grid.at_node['bedrock__elevation'][grid.core_nodes])+50])
plt.tight_layout()
plt.show()
fig_cnt += 500
    #sg.run_one_step(A_factor = 0.01)

plt.plot(grid.at_node['topographic__elevation'][grid.core_nodes])
plt.plot(grid.at_node['bedrock__elevation'][grid.core_nodes])
plt.show()
# # Figure
# total_weight = np.sum(grid.at_node['grains__weight'][4,:])
# fig,ax = plt.subplots(1,1)
# ax.plot(-np.log2(sg._meansizes*1000),
#         np.cumsum(init_weight_per_grainsize)/total_weight,
#         color='black',
#        label='t = 0')
# ax.plot(-np.log2(sg._meansizes*1000),np.cumsum(grid.at_node['grains__weight'][4,:])/total_weight,color='blue',label='t = ' + str(n_steps ))
# ax.set_xlabel('Grain size [$\phi$]')
# ax.set_ylabel('Weight fraction')
# ax.invert_xaxis() # phi scale
# ax.legend()
# plt.show()




#
# if known_dist_flag:
#     grading.set_grading_classes(input_sizes_flag=True,
#                                 meansizes=mean_grainsize_per_class,
#                                 )
#     grading.create_transion_mat(A_factor=A_factor)
#     # Create grain size distribution for initial debris layer
#     grading.create_dist(
#         median_size=init_median_grainsize_at_node,
#         num_of_clasts=initial_sediment_weight_at_node,
#         init_val_flag=True,
#     std=1)
#
# else:
#
#     grading.set_grading_classes(maxsize = maxsize)
#     grading.create_transion_mat(A_factor=A_factor)
#     # Create grain size distribution for initial debris layer
#     grading.create_dist(
#         median_size=init_median_grainsize_at_node,
#         num_of_clasts=initial_sediment_weight_at_node,
#         init_val_flag=True, )





