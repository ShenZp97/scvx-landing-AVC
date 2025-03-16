import numpy as np
from scripts.twophase_landing_6dof_server import scvx_landing
from scripts.twophase_landing_6dof_server import scvx_landing_new
from scripts.twophase_landing_6dof_server import scvx_landing_moving_exp
from scripts.twophase_landing_6dof_server import scvx_landing_no_3dof
from scripts.twophase_landing_6dof_server import scvx_landing_moving_exp_no_3dof

plot = True
# scvx_landing("", np.array([1,1.2,4.2]), 3.5, np.array([1,1,2]), plot=plot)
# scvx_landing("", np.array([4,4,2.2]), 3.5, np.array([1,1,2]), plot=plot)

# scvx_landing("iris", np.array([4,4,2.2]), 4.5, np.array([1,1,2]), plot=plot)

# scvx_landing("aims1", np.array([4,4,2.2]), 4.5, np.array([1,1,2]), plot=plot)

# scvx_landing("aims1", np.array([1.7-0.013,0.0+0.005,0.5+0.079+0.02]), 2.0, np.array([-1.5,0.,0.5]), plot=plot)
# scvx_landing("aims1", np.array([1.706,-0.708+0.026,0.5+0.077+0.02]), 3.8, np.array([-1.5,1.5,0.5]), plot=plot, goal_v=np.array([0.7102, 0.7084, 0.0]))
# scvx_landing("aims1", np.array([1.706,-0.708+0.026,0.5+0.077+0.02]), 3.8, np.array([-1.5,1.5,0.5]), plot=plot)

# scvx_landing("hummingbird", np.array([4,4,2.2]), 3.5, np.array([1,1,2]), plot=plot)
# scvx_landing("hummingbird", np.array([1,1.2,4.2]), 2.5, np.array([1,1,2]), plot=plot)


# scvx_landing_moving_exp("aims1", np.array([1.706,-0.708+0.026,0.5+0.077+0.02]), 3.8, np.array([-1.5,1.5,0.5]), plot=plot, goal_v=np.array([0.7102, 0.7084, 0.0]))
# scvx_landing("aims1", np.array([-0.98,0.012+0.05,0.5+0.077+0.10]), 2.5, np.array([2.22, 0.33, 0.4]), plot=plot)

# scvx_landing_moving_exp("aims1", np.array([1.706,-0.708+0.026,0.5+0.077+0.02]), 3.5, np.array([1.5,1.5,0.5]), plot=plot, goal_v=np.array([0.7102, 0.7084, 0.0]))
# scvx_landing_moving_exp("aims1", np.array([1.706,-0.708+0.026,0.5+0.077+0.02]), 3.5, np.array([2.5,1.5,0.5]), plot=plot, goal_v=np.array([0.7102, 0.7084, 0.0]))
# scvx_landing_moving_exp("aims1", np.array([1.706,-0.708+0.026,0.5+0.077+0.02]), 3.5, np.array([3.5,1.5,0.5]), plot=plot, goal_v=np.array([0.7102, 0.7084, 0.0]))
# scvx_landing_moving_exp("aims1", np.array([1.706,-0.708+0.026,0.5+0.077+0.02]), 3.5, np.array([2.5,2.5,0.5]), plot=plot, goal_v=np.array([0.7102, 0.7084, 0.0]))
# scvx_landing_moving_exp("aims1", np.array([1.706-0.05,-0.708+0.05,0.5+0.077+0.02]), 3.8, np.array([-1.5,1.5,0.5]), plot=plot, goal_v=np.array([0.7102, 0.7084, 0.0]))

# scvx_landing_moving_exp("aims1", np.array([1.706-0.05,-0.708+0.05,0.5+0.077+0.02]), 3.5, np.array([2.75,2.3,0.5]), plot=plot, goal_v=np.array([0.7102, 0.7084, 0.0]))



#collision avoidance
# scvx_landing("aims1", np.array([0.0,0.0+0.05,1.5+0.077+0.10]), 3.0, np.array([-0.5,0.,0.3]), plot=plot)

# exp1 new
# scvx_landing("aims1", np.array([1.7,0.0+0.05,1.0+0.077+0.10]), 2.5, np.array([-1.5,0.,0.5]), plot=plot)

# exp light
# scvx_landing("aims1", np.array([0.,0.0+0.026,1.0+0.077+0.10]), 2.5, np.array([-2.,0.,0.5]), plot=plot)
# scvx_landing("aims1", np.array([0.,0.0+0.1,1.0+0.077+0.10]), 2.5, np.array([-1.5,1.5,0.5]), plot=plot)
# scvx_landing("aims1", np.array([0.,0.0+0.05,1.0+0.077+0.10]), 2.5, np.array([0.,-1.95,0.5]), plot=plot)

# scvx_landing("aims1", np.array([0.6,0.0+0.05,1.0+0.077+0.10]), 2.5, np.array([2.96, 0.,0.5]), plot=plot)
# scvx_landing("aims1", np.array([0.6,0.0+0.05,1.0+0.077+0.10]), 2.5, np.array([0.6, -2., 0.5]), plot=plot)
# scvx_landing("aims1", np.array([-0.98,0.012+0.05,0.5+0.077+0.10]), 2.5, np.array([2.22, 0.33, 0.4]), plot=plot)



# For paper REVISION

# scvx_landing_no_3dof("", np.array([4,4,2.2]), 3.5, np.array([1,1,2]), plot=plot)
# scvx_landing_no_3dof("", np.array([1,1.2,4.2]), 3.5, np.array([1,1,2]), plot=plot)

# scvx_landing_no_3dof("aims1", np.array([0.0,0.0+0.05,1.5+0.077+0.10]), 3.0, np.array([-0.5,0.,0.3]), plot=plot)
# scvx_landing_no_3dof("aims1", np.array([1.7,0.0+0.05,1.0+0.077+0.10]), 2.5, np.array([-1.5,0.,0.5]), plot=plot)
# scvx_landing_no_3dof("aims1", np.array([0.0,0.0+0.05,1.5+0.077+0.10]), 3.0, np.array([-1.0,0.,0.3]), plot=plot)


# scvx_landing_moving_exp("aims1", np.array([1.706,-0.708+0.026,0.5+0.077+0.02]), 3.8, np.array([-1.5,1.5,0.5]), plot=plot, goal_v=np.array([0.7102, 0.7084, 0.0]))
# scvx_landing_moving_exp_no_3dof("aims1", np.array([1.706,-0.708+0.026,0.5+0.077+0.02]), 3.8, np.array([-1.5,1.5,0.5]), plot=plot, goal_v=np.array([0.7102, 0.7084, 0.0]))

# scvx_landing_moving_exp("aims1", np.array([1.706-0.05,-0.708+0.05,0.5+0.077+0.02]), 3.5, np.array([2.75,2.3,0.5]), plot=plot, goal_v=np.array([0.7102, 0.7084, 0.0]))
# scvx_landing_moving_exp_no_3dof("aims1", np.array([1.706-0.05,-0.708+0.05,0.5+0.077+0.02]), 3.5, np.array([2.75,2.3,0.5]), plot=plot, goal_v=np.array([0.7102, 0.7084, 0.0]))

# Takeoff test
# scvx_landing_moving_exp("aims1", np.array([1.991,-0.128,0.5+0.077+0.02]), 3.8, np.array([-1.5,1.5, 1.0]), plot=plot, goal_v=np.array([0.998, -0.007, -0.051]))
# scvx_landing("aims1", np.array([1.7,0.,0.5]), 2.0, np.array([-1.5, 0., 1.]), plot=plot)


# EHang visit

# scvx_landing_moving_exp("aims3", np.array([1.991,-0.128,0.5+0.077+0.02]), 3.8, np.array([-1.5,1.5, 1.0]), plot=plot, goal_v=np.array([0.998, -0.007, -0.051]))
# scvx_landing_moving_exp("aims3", np.array([1.706,-0.708+0.026,0.5+0.077+0.02]), 3.8, np.array([-1.5,1.5,0.5]), plot=plot, goal_v=np.array([0.7102, 0.7084, 0.0]))
# scvx_landing("aims3", np.array([1.7,0.0+0.05,1.0+0.077+0.10]), 2.5, np.array([-1.5,0.,0.5]), plot=plot)
# scvx_landing("aims3", np.array([-1.9,-1.0+0.05,1.0+0.077+0.10]), 2.5, np.array([2.9,2.2,0.5]), plot=plot)
# scvx_landing_moving_exp("aims3", np.array([1.19,-0.98+0.10,0.45]), 3.8, np.array([-1.5,1.5, 1.0]), plot=plot, goal_v=np.array([-0.36, -0.23, -0.0]))


scvx_landing("aims3", np.array([0.0,0.0+0.05,1.5+0.077+0.10]), 3.0, np.array([-0.5,0.,0.3]), plot=plot)
# scvx_landing_moving_exp("aims3", np.array([1.18,-0.98,0.55]), 3.8, np.array([-1.5,1.5, 1.0]), plot=plot, goal_v=np.array([1.04, 0.198, -0.0]))
