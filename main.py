# %% ====== Imports ======

import importlib
import numpy as np

import plot_funcs

file_prelim = r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI"



# %% ====== Compare ======
importlib.reload(plot_funcs)
final_t = 25
save = True

share_t = False

# # pos 0001->1000
mod = "pos"
init_buckle = "0001"
desired_buckle = "1000"
exp_file_prelim = r"..\\Meca500\\data\\training\\June21_fromPos\\{}to{}\\".format(init_buckle, desired_buckle)
sim_file_prelim = "..\\Bistable shape acquisition jax\\Training\\May17Pos_symmetrical_delta\\"
exp_file_path = exp_file_prelim + r"{}to{}_correctedLoss.csv".format(init_buckle, desired_buckle)
sim_file_path = sim_file_prelim + r"final_loss_0_init_{}_desired_{}_extended_correctedLoss.csv".format(init_buckle, desired_buckle)

plot_funcs.plot_compare_sim_exp_training(exp_file_path, sim_file_path, final_t, save, mod = mod, share_t = share_t)

# force 0011->1000
mod = "force"
init_buckle = "0011"
desired_buckle = "1000"
exp_file_prelim = r"..\\Meca500\\data\\training\\June20_fullTrainingContd\\0011to1000pos2\\"
sim_file_prelim = "..\\Bistable shape acquisition jax\\Training\June6_May22singleHinge2ndEnd_May27shortArcTraj\\stiffk\\1stEnd\\"
exp_file_path = exp_file_prelim + r"{}to{}.csv".format(init_buckle, desired_buckle)
sim_file_path = sim_file_prelim + r"final_loss_0_init_{}_desired_{}.csv".format(init_buckle, desired_buckle)

exp_force_1_path = file_prelim + r"\Meca500\data\measurements\Feb26\1001\buckle=1001.csv"
sim_force_1_path = file_prelim + r"\Bistable shape acquisition jax\Predetermined trajectory\Mar2\L=pt047 tip 2 mm left\1001\L=0.047_buckle1001.csv"
exp_force_2_path = file_prelim + r"\Meca500\data\measurements\Feb26\0001\buckle=0001_2.csv"
sim_force_2_path = file_prelim + r"\Bistable shape acquisition jax\Predetermined trajectory\Mar2\L=pt047 tip 2 mm left\0001\L=0.047_buckle0001.csv"

# sim_file_prelim = "Bistable shape acquisition jax\\Training\\\June6_May22singleHinge2ndEnd_May27shortArcTraj\\"
# exp_file_path = exp_file_prelim + r"combined.csv".format(init_buckle, desired_buckle)

plot_funcs.plot_compare_sim_exp_training(
    exp_file_path, sim_file_path, final_t, save, mod=mod, share_t=share_t,
    force_traj_files=[
        (exp_force_1_path, sim_force_1_path),
        (exp_force_2_path, sim_force_2_path),
    ],
)

# %% ====== Single ======

final_t = None
save = True

file_path = r"..\Bistable shape acquisition jax\Training\June15_H10_pos\1stAndLast\0010100001to1010100000.csv"
plot_funcs.plot_sim_or_exp(file_path, mod="pos", final_t=final_t, save=save)

# %% ====== Training through force video ======

importlib.reload(plot_funcs)
csv_path = r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI\Meca500\data\training\June19_fullTraining\0001to0000TrainingFull_pos1_Good\0001to0000_pos1.csv"
images_dir = r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI\Meca500\data\training\June19_fullTraining\0001to0000TrainingFull_pos1_Good\0001to0000Training_justUpdates" 
plot_funcs.training_force_data_and_vid(csv_file_path=csv_path, image_dir=images_dir, fps=2)


# %% ====== Training through pos video ======

importlib.reload(plot_funcs)

csv_file_path=r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI\Meca500\data\training\June21_fromPos\0001to1000\0001to1000.csv"
pics_dir=r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI\Meca500\data\training\June21_fromPos\0001to1000\pics"
plot_funcs.training_pos_data_and_vid(csv_file_path=csv_file_path, pics_dir=pics_dir, infer_image_sequence=True, fps=2)

# %% ====== Force along trajectory ======

importlib.reload(plot_funcs)

csv_file_path_des = r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI\Meca500\data\training\June19_fullTraining\0001to0000TrainingFull_pos1_Good\buckle=0000_fromSims.csv"
vid_path_des = r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI\Meca500\data\training\June19_fullTraining\0001to0000TrainingFull_pos1_Good\0001to0000_justMeas\0000.mp4"
# plot_funcs.plot_force_along_traj(csv_file_path=csv_file_path_des, vid_path=vid_path_des, initial_time_s=3.0,
#                                  final_time_s=11.0, fps=5, mean_line_mode = "des", save=True)


csv_file_path_meas = r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI\Meca500\data\training\June19_fullTraining\0001to0000TrainingFull_pos1_Good\buckle=0001_fromSims.csv"
vid_path_meas = r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI\Meca500\data\training\June19_fullTraining\0001to0000TrainingFull_pos1_Good\0001to0000_justMeas\0001.mp4"
plot_funcs.plot_force_along_traj(csv_file_path=csv_file_path_meas, vid_path=vid_path_meas, initial_time_s=9.0,
                                 final_time_s=18.0, fps=5, mean_line_mode = "meas", csv_file_path_des = csv_file_path_des, 
                                 save=True)

csv_file_path_meas = r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI\Meca500\data\training\June19_fullTraining\0001to0000TrainingFull_pos1_Good\buckle=0000_measuredEnd.csv"
vid_path_meas = r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI\Meca500\data\training\June19_fullTraining\0001to0000TrainingFull_pos1_Good\0001to0000_justMeas\0000_meas.mp4"
plot_funcs.plot_force_along_traj(csv_file_path=csv_file_path_meas, vid_path=vid_path_meas, initial_time_s=4.0,
                                 final_time_s=13.0, fps=5, mean_line_mode = "meas", csv_file_path_des = csv_file_path_des, 
                                 save=True)

# %% ====== Force along trajectory: graph only ======
importlib.reload(plot_funcs)
# Choose the experiment and simulation CSV files here.
csv_file_path_exp = file_prelim + r"\paper\Setup\Setup data\F along traj.csv"
csv_file_path_sim = file_prelim + r"\paper\Setup\Setup data\F along traj sim.csv"
plot_funcs.plot_force_along_traj(
    csv_file_path=csv_file_path_exp,
    csv_file_path_sim=csv_file_path_sim,
    graph_only=True,
    experiment_error=10.0,
    save=True,
    scale_y=False,
    range_y = True,
    y_lims=(-180, 420)
)

# csv_file_path_exp = file_prelim + r"\Meca500\data\measurements\June4_PredeterMay27ShortArcZeroDeg\1stEnd\0000.csv"
# csv_file_path_sim = file_prelim + r"\Bistable shape acquisition jax\Predetermined trajectory\May27\short_arc\May24Chain_1stEnd\buckle=0000.csv"
# plot_funcs.plot_force_along_traj(
#     csv_file_path=csv_file_path_exp,
#     csv_file_path_sim=csv_file_path_sim,
#     graph_only=True,
#     experiment_error=10.0,
#     save=True,
#     scale_y=True
# )

# csv_file_path_exp = file_prelim + r"\Meca500\data\measurements\Feb26\0001\buckle=0001_2.csv"
# csv_file_path_sim = file_prelim + r"\Bistable shape acquisition jax\Predetermined trajectory\Mar2\L=pt047 tip 2 mm left\0001\L=0.047_buckle0001.csv"
# plot_funcs.plot_force_along_traj(
#     csv_file_path=csv_file_path_exp,
#     csv_file_path_sim=csv_file_path_sim,
#     graph_only=True,
#     experiment_error=10.0,
#     save=True,
#     scale_y=False
# )

# %% ====== Trajectory positions ======
importlib.reload(plot_funcs)

csv_file_path_sim = file_prelim + r"\paper\Setup\Setup data\F along traj sim.csv"
plot_funcs.plot_trajectory_positions(csv_file_path_exp, save=True)

# %% ====== Force during zero Force ======
importlib.reload(plot_funcs)

csv_file_path_des=r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI\Meca500\data\training\June21_fromPos\0001to1000\1000_des_reach_zero.csv"
vid_path_des=r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI\Meca500\data\training\June21_fromPos\0001to1000\vids\1000zeroForce.mp4"

# plot_funcs.plot_force_during_zero_force(csv_file_path=csv_file_path_des, vid_path=vid_path_des, initial_time_vid=3.0,
#                                         final_time_vid=63.0, final_time_csv=10, fps=5, mean_line_mode="des")

csv_file_path_meas=r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI\Meca500\data\training\June21_fromPos\0001to1000\1001_meas_reach_zero.csv"
vid_path_meas=r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI\Meca500\data\training\June21_fromPos\0001to1000\vids\1001zeroForce.mp4"

plot_funcs.plot_force_during_zero_force(csv_file_path=csv_file_path_meas, vid_path=vid_path_meas, initial_time_vid=10.0,
                                        final_time_vid=83.0, final_time_csv=10, fps=5, mean_line_mode="meas", csv_file_path_des=csv_file_path_des)
# %%
