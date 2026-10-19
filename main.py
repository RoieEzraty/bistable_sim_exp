# %% ====== Imports ======

import importlib
import numpy as np

import plot_funcs, colors, file_helpers

# file_prelim = r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI"
file_prelim = r"C:\Users\roiee\OneDrive - huji.ac.il\ORIGAMI"

# %% ====== Compare ======
importlib.reload(plot_funcs)
font_size=12
save = "pdf"
share_t = False

# pos 0001->1000
mod = "pos"
final_t = 25
init_buckle = "0001"
desired_buckle = "1000"
exp_file_prelim = r"..\\Meca500\\data\\training\\June21_fromPos\\{}to{}\\".format(init_buckle, desired_buckle)
sim_file_prelim = "..\\Bistable shape acquisition jax\\Training\\May17Pos_symmetrical_delta\\"
exp_file_path = exp_file_prelim + r"{}to{}_correctedLoss.csv".format(init_buckle, desired_buckle)
sim_file_path = sim_file_prelim + r"final_loss_0_init_{}_desired_{}_extended_correctedLoss.csv".format(init_buckle, desired_buckle)

plot_funcs.plot_compare_sim_exp_training(exp_file_path, sim_file_path, final_t, save, mod = mod, share_t = share_t, font_size=font_size)

# force 0011->1000
final_t = 25
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

plot_funcs.plot_compare_sim_exp_training(exp_file_path, sim_file_path, final_t, save, mod=mod, share_t=share_t, font_size=font_size, force_traj_files=[(exp_force_1_path, sim_force_1_path),
                                                                                                                                                       (exp_force_2_path, sim_force_2_path),])

# %% ====== Single ======

final_t = None
save = True

file_path = r"..\Bistable shape acquisition jax\Training\June15_H10_pos\1stAndLast\0010100001to1010100000.csv"
plot_funcs.plot_sim_or_exp(file_path, mod="pos", final_t=final_t, save=save)

# %% ====== Training through force video ======

importlib.reload(plot_funcs)
csv_path = r"..\Meca500\data\training\June19_fullTraining\0001to0000TrainingFull_pos1_Good\0001to0000_pos1.csv"
images_dir = r"..\Meca500\data\training\June19_fullTraining\0001to0000TrainingFull_pos1_Good\0001to0000Training_justUpdates" 

# 0011-1000
# csv_path = "C:\\Users\\roiee\\OneDrive - huji.ac.il\\ORIGAMI\\Meca500\\data\\training\\June20_fullTrainingContd\\0011to1000pos2\\0011to1000.csv"
# images_dir = "C:\\Users\\roiee\\OneDrive - huji.ac.il\\ORIGAMI\\Meca500\\data\\training\\June20_fullTrainingContd\\0011to1000pos2\\pics"

# 1100-0101 
# csv_path = "C:\\Users\\roiee\\OneDrive - huji.ac.il\\ORIGAMI\\Meca500\\data\\training\\June19_fullTraining\\1100to0101Training_pos3_GoodButABitShort\\1100to0101_pos3.csv"
# images_dir = "C:\\Users\\roiee\\OneDrive - huji.ac.il\\ORIGAMI\\Meca500\\data\\training\\June19_fullTraining\\1100to0101Training_pos3_GoodButABitShort\\1100to0101Training_filmJustUpdate"

plot_funcs.training_force_data_and_vid(csv_file_path=csv_path, image_dir=images_dir, fps=2)


# %% ====== Training through pos video ======

importlib.reload(plot_funcs)

# 0001-1000 through pos
csv_path=r"..\Meca500\data\training\June21_fromPos\0001to1000\0001to1000.csv"
images_dir=r"..\Meca500\data\training\June21_fromPos\0001to1000\pics"

# # 1111-1101 through pos
# csv_path = "C:\\Users\\roiee\\OneDrive - huji.ac.il\\ORIGAMI\\Meca500\\data\\training\\June21_fromPos\\1111to1101\\1111to1101.csv"
# images_dir = "C:\\Users\\roiee\\OneDrive - huji.ac.il\\ORIGAMI\\Meca500\\data\\training\\June21_fromPos\\1111to1101\\pics"

plot_funcs.training_pos_data_and_vid(csv_file_path=csv_path, pics_dir=images_dir, infer_image_sequence=True, fps=2)

# %% ====== Force along trajectory ======

importlib.reload(plot_funcs)

csv_file_path_des = r"..\Meca500\data\training\June19_fullTraining\0001to0000TrainingFull_pos1_Good\buckle=0000_fromSims.csv"
# vid_path_des = r"..\Meca500\data\training\June19_fullTraining\0001to0000TrainingFull_pos1_Good\0001to0000_justMeas\0000.mp4"
# plot_funcs.plot_force_along_traj(csv_file_path=csv_file_path_des, vid_path=vid_path_des, initial_time_s=3.0,
#                                  final_time_s=11.0, fps=5, scale_y=True, mean_line_mode = "des", save=True)

# csv_file_path_meas = r"..\Meca500\data\training\June19_fullTraining\0001to0000TrainingFull_pos1_Good\buckle=0001_fromSims.csv"
# vid_path_meas = r"..\Meca500\data\training\June19_fullTraining\0001to0000TrainingFull_pos1_Good\0001to0000_justMeas\0001.mp4"
# plot_funcs.plot_force_along_traj(csv_file_path=csv_file_path_meas, vid_path=vid_path_meas, initial_time_s=9.0,
#                                  final_time_s=18.0, fps=5, scale_y=True,mean_line_mode = "meas", csv_file_path_des = csv_file_path_des, 
#                                  save=True)

csv_file_path_meas = r"..\Meca500\data\training\June19_fullTraining\0001to0000TrainingFull_pos1_Good\buckle=0000_measuredEnd.csv"
vid_path_meas = r"..\Meca500\data\training\June19_fullTraining\0001to0000TrainingFull_pos1_Good\0001to0000_justMeas\0000_meas.mp4"
plot_funcs.plot_force_along_traj(csv_file_path=csv_file_path_meas, vid_path=vid_path_meas, initial_time_s=4.0,
                                 final_time_s=13.0, fps=5, scale_y=True, mean_line_mode = "meas", csv_file_path_des = csv_file_path_des, 
                                 save=True)

# %% ====== Force along trajectory: graph only ======
importlib.reload(plot_funcs)

# # Choose the experiment and simulation CSV files here.
# csv_file_path_exp = file_prelim + r"\paper\Setup\Setup data\F along traj.csv"
# csv_file_path_sim = file_prelim + r"\paper\Setup\Setup data\F along traj sim.csv"
# plot_funcs.plot_force_along_traj(csv_file_path=csv_file_path_exp, csv_file_path_sim=csv_file_path_sim, graph_only=True, 
#                                  experiment_error=10.0, save="pdf", scale_y=True, range_y = True, y_lims=(-180, 420), font_size=20,
#                                  plot_final_force_lines=True, sims_as_markers=True, error_style = "None")

# y_lims=(-120, 350)

# csv_file_path_exp = file_prelim + r"\Meca500\data\measurements\June4_PredeterMay27ShortArcZeroDeg\1stEnd\0000.csv"
# csv_file_path_sim = file_prelim + r"\Bistable shape acquisition jax\Predetermined trajectory\May27\short_arc\May24Chain_1stEnd\buckle=0000.csv"
# plot_funcs.plot_force_along_traj(csv_file_path=csv_file_path_exp, csv_file_path_sim=csv_file_path_sim, graph_only=True, experiment_error=10.0, 
#                                  save="pdf", scale_y=True, y_lims=y_lims)

# csv_file_path_exp = file_prelim + r"\Meca500\data\measurements\Feb26\0001\buckle=0001_2.csv"
# csv_file_path_sim = file_prelim + r"\Bistable shape acquisition jax\Predetermined trajectory\Mar2\L=pt047 tip 2 mm left\0001\L=0.047_buckle0001.csv"
# plot_funcs.plot_force_along_traj(
#     csv_file_path=csv_file_path_exp,
#     csv_file_path_sim=csv_file_path_sim,
#     graph_only=True,
#     experiment_error=10.0,
#     save="pdf",
#     scale_y=False,
#     y_lims=y_lims
# )

# %% ====== Force along trajectory: average only ======

importlib.reload(plot_funcs)

# from 0011->1000 experiment, in \\Meca500\\data\\training\\June20_fullTrainingContd\\0011to1000pos2\\0011to1000.csv
buckle = "1011"
average_fx_1011 = 129.1
average_fy_1011 = -18.6
plot_funcs.plot_average_force_along_traj(
    average_fx_1011, average_fy_1011, buckle=buckle, save=True
)

buckle = "1010"
average_fx_1010 = 166.7
average_fy_1010 = 10.6
plot_funcs.plot_average_force_along_traj(
    average_fx_1010, average_fy_1010, buckle=buckle, save=True
)

# %% ====== Trajectory positions ======
importlib.reload(plot_funcs)
importlib.reload(colors)

csv_file_path_sim = file_prelim + r"\paper\Setup\Setup data\F along traj sim.csv"
plot_funcs.plot_trajectory_positions(csv_file_path_sim, save="pdf", font_size=20)

# %% ====== Force during zero Force ======
importlib.reload(plot_funcs)

# csv_file_path_des=r"..\Meca500\data\training\June21_fromPos\0001to1000\1000_des_reach_zero.csv"
# vid_path_des=r"..\Meca500\data\training\June21_fromPos\0001to1000\vids\1000zeroForce.mp4"

# plot_funcs.plot_force_during_zero_force(csv_file_path=csv_file_path_des, vid_path=vid_path_des, initial_time_vid=2.5,
#                                         final_time_vid=64.0, final_time_csv=10, fps=5, mean_line_mode="des")

csv_file_path_meas=r"..\Meca500\data\training\June21_fromPos\0001to1000\0001_meas_reach_zero.csv"
vid_path_meas=r"..\Meca500\data\training\June21_fromPos\0001to1000\vids\0001zeroForce.mp4"

plot_funcs.plot_force_during_zero_force(csv_file_path=csv_file_path_meas, vid_path=vid_path_meas, initial_time_vid=10.0,
                                        final_time_vid=68.0, final_time_csv=10, fps=5, mean_line_mode="meas", csv_file_path_des=csv_file_path_des)

# csv_file_path_meas=r"..\Meca500\data\training\June21_fromPos\0001to1000\1001_meas_reach_zero.csv"
# vid_path_meas=r"..\Meca500\data\training\June21_fromPos\0001to1000\vids\1001zeroForce.mp4"

# plot_funcs.plot_force_during_zero_force(csv_file_path=csv_file_path_meas, vid_path=vid_path_meas, initial_time_vid=10.0,
#                                         final_time_vid=83.0, final_time_csv=10, fps=5, mean_line_mode="meas", csv_file_path_des=csv_file_path_des)
# %% ====== Force along trajectory Sim Exp comparison ======

importlib.reload(plot_funcs)
importlib.reload(file_helpers)

# buckle = 1101

buckle_str = "1101"

path_exp_1 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Meca500/data/measurements/Feb26/1101/buckle=1101.csv'
path_exp_2 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Meca500/data/measurements/Mar9/filled chain/buckle=1101.csv'
path_exp_3 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Meca500/data/measurements/Mar12/dL90/buckle=1101.csv'

exp_labels = [r'$d/L=0.75$ 1st exp', r'$d/L=0.75$ 2nd exp', r'$d/L=0.90$ exp' ]

err=0.01*1e3

x_y_theta_vec_exp = []
F_vec_exp = []
for i, path in enumerate([path_exp_1, path_exp_2, path_exp_3]):
    rows = file_helpers.load_pos_force(path)  # uses your CSV format
    x_y_theta_vec_exp.append(np.array([r["pos"] for r in rows], dtype=float))
    F_vec_exp.append(np.array([r["force"] for r in rows], dtype=float))
    if path == 'data/measurements/Feb17/x=160_buckle=0001/x=160_buckle=0001.csv':
        F_vec_exp[i] = 1e3 * F_vec_exp[i]

y_pos = y_pos = x_y_theta_vec_exp[0][:, 1]


path_sim_1 = 'C:\\Users\\roiee\\OneDrive - huji.ac.il\\ORIGAMI\\Bistable shape acquisition jax\\Predetermined trajectory\\Feb26\\L=pt046 tip 5mm left\\L=0.046_buckle[ 1  1 -1  1].csv'
path_sim_2 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Bistable shape acquisition jax/Predetermined trajectory/Mar9/9Mar filled shim/buckle=1101.csv'
path_sim_3 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Bistable shape acquisition jax/Predetermined trajectory/Mar12/dl=90_3/buckle=1101.csv'
# path_sim_3 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Bistable shape acquisition jax/Predetermined trajectory/Mar12/dl=90_3/buckle=1101_2.csv'


sim_labels = [r'$d/L=0.75$ 1st sim', r'$d/L=0.75$ 2nd sim', r'$d/L=0.90$ sim']

F_vec_sim = []
for i, path in enumerate([path_sim_1, path_sim_2, path_sim_3]):
    rows = file_helpers.load_pos_force(path)  # uses your CSV format
    F_vec_sim.append(np.array([r["force"] for r in rows], dtype=float))

plot_funcs.plot_force_along_traj_comparsion(F_vec_exp, F_vec_sim, x_y_theta_vec_exp, exp_labels, sim_labels, buckle_str, err=err)

# %%

importlib.reload(plot_funcs)
importlib.reload(file_helpers)

# buckle = 0001

buckle_str = "0001"

path_exp_1 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Meca500/data/measurements/Feb26/0001/buckle=0001.csv'
path_exp_2 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Meca500/data/measurements/Mar9/filled chain/buckle=0001.csv'
path_exp_3 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Meca500/data/measurements/Mar12/dL90/buckle=0001.csv'

exp_labels = [r'$d/L=0.75$ 1st exp', r'$d/L=0.75$ 2nd exp', r'$d/L=0.90$ exp' ]

err=0.01*1e3

x_y_theta_vec_exp = []
F_vec_exp = []
for i, path in enumerate([path_exp_1, path_exp_2, path_exp_3]):
    rows = file_helpers.load_pos_force(path)  # uses your CSV format
    x_y_theta_vec_exp.append(np.array([r["pos"] for r in rows], dtype=float))
    F_vec_exp.append(np.array([r["force"] for r in rows], dtype=float))
    if path == 'data/measurements/Feb17/x=160_buckle=0001/x=160_buckle=0001.csv':
        F_vec_exp[i] = 1e3 * F_vec_exp[i]

y_pos = y_pos = x_y_theta_vec_exp[0][:, 1]


path_sim_1 = 'C:\\Users\\roiee\\OneDrive - huji.ac.il\\ORIGAMI\\Bistable shape acquisition jax\\Predetermined trajectory\\Feb27\\0001\\L=0.045_buckle[-1 -1 -1  1].csv'
path_sim_2 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Bistable shape acquisition jax/Predetermined trajectory/Mar9/9Mar filled shim/buckle=0001.csv'
path_sim_3 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Bistable shape acquisition jax/Predetermined trajectory/Mar12/dl=90_3/buckle=0001.csv'
# path_sim_3 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Bistable shape acquisition jax/Predetermined trajectory/Mar12/dl=90_3/buckle=0001_2.csv'


sim_labels = [r'$d/L=0.75$ 1st sim', r'$d/L=0.75$ 2nd sim', r'$d/L=0.90$ sim']

F_vec_sim = []
for i, path in enumerate([path_sim_1, path_sim_2, path_sim_3]):
    rows = file_helpers.load_pos_force(path)  # uses your CSV format
    F_vec_sim.append(np.array([r["force"] for r in rows], dtype=float))

plot_funcs.plot_force_along_traj_comparsion(F_vec_exp, F_vec_sim, x_y_theta_vec_exp, exp_labels, sim_labels, buckle_str, err=err)

# %%

# buckle = 1001

buckle_str = "1001"

path_exp_1 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Meca500/data/measurements/Feb26/1001/buckle=1001.csv'
path_exp_2 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Meca500/data/measurements/Mar9/filled chain/buckle=1001.csv'
path_exp_3 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Meca500/data/measurements/Mar12/dL90/buckle=1001.csv'

exp_labels = [r'$d/L=0.75$ 1st exp', r'$d/L=0.75$ 2nd exp', r'$d/L=0.90$ exp' ]

err=0.01*1e3

x_y_theta_vec_exp = []
F_vec_exp = []
for i, path in enumerate([path_exp_1, path_exp_2, path_exp_3]):
    rows = file_helpers.load_pos_force(path)  # uses your CSV format
    x_y_theta_vec_exp.append(np.array([r["pos"] for r in rows], dtype=float))
    F_vec_exp.append(np.array([r["force"] for r in rows], dtype=float))
    if path == 'data/measurements/Feb17/x=160_buckle=0001/x=160_buckle=0001.csv':
        F_vec_exp[i] = 1e3 * F_vec_exp[i]

y_pos = y_pos = x_y_theta_vec_exp[0][:, 1]


path_sim_1 = 'C:\\Users\\roiee\\OneDrive - huji.ac.il\\ORIGAMI\\Bistable shape acquisition jax\\Predetermined trajectory\\Feb27\\1001\\L=0.045_buckle[ 1 -1 -1  1].csv'
path_sim_2 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Bistable shape acquisition jax/Predetermined trajectory/Mar9/9Mar filled shim/buckle=1001.csv'
# path_sim_3 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Bistable shape acquisition jax/Predetermined trajectory/Mar12/dl=90_3/buckle=1001.csv'
path_sim_3 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Bistable shape acquisition jax/Predetermined trajectory/Mar12/dl=90_3/buckle=1001_2.csv'


sim_labels = [r'$d/L=0.75$ 1st sim', r'$d/L=0.75$ 2nd sim', r'$d/L=0.90$ sim']

F_vec_sim = []
for i, path in enumerate([path_sim_1, path_sim_2, path_sim_3]):
    rows = file_helpers.load_pos_force(path)  # uses your CSV format
    F_vec_sim.append(np.array([r["force"] for r in rows], dtype=float))

plot_funcs.plot_force_along_traj_comparsion(F_vec_exp, F_vec_sim, x_y_theta_vec_exp, exp_labels, sim_labels, buckle_str, err=err)
# %%

# buckle = 0010

buckle_str = "0010"

path_exp_1 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Meca500/data/measurements/Feb26/0010/buckle=0010.csv'
path_exp_2 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Meca500/data/measurements/Mar9/filled chain/buckle=0010.csv'
path_exp_3 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Meca500/data/measurements/Mar12/dL90/buckle=0010.csv'

exp_labels = [r'$d/L=0.75$ 1st exp', r'$d/L=0.75$ 2nd exp', r'$d/L=0.90$ exp' ]

err=0.01*1e3

x_y_theta_vec_exp = []
F_vec_exp = []
for i, path in enumerate([path_exp_1, path_exp_2, path_exp_3]):
    rows = file_helpers.load_pos_force(path)  # uses your CSV format
    x_y_theta_vec_exp.append(np.array([r["pos"] for r in rows], dtype=float))
    F_vec_exp.append(np.array([r["force"] for r in rows], dtype=float))
    if path == 'data/measurements/Feb17/x=160_buckle=0001/x=160_buckle=0001.csv':
        F_vec_exp[i] = 1e3 * F_vec_exp[i]

y_pos = y_pos = x_y_theta_vec_exp[0][:, 1]


path_sim_1 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Bistable shape acquisition jax/Predetermined trajectory/Mar2/L=pt047 tip 2 mm left/0010/L=0.047_buckle0010.csv'
path_sim_2 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Bistable shape acquisition jax/Predetermined trajectory/Mar9/9Mar filled shim/buckle=0010.csv'
# path_sim_3 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Bistable shape acquisition jax/Predetermined trajectory/Mar12/dl=90_3/buckle=0010.csv'
path_sim_3 = 'C:/Users/roiee/OneDrive - huji.ac.il/ORIGAMI/Bistable shape acquisition jax/Predetermined trajectory/Mar12/dl=90_3/buckle=0010_2.csv'


sim_labels = [r'$d/L=0.75$ 1st sim', r'$d/L=0.75$ 2nd sim', r'$d/L=0.90$ sim']

F_vec_sim = []
for i, path in enumerate([path_sim_1, path_sim_2, path_sim_3]):
    rows = file_helpers.load_pos_force(path)  # uses your CSV format
    F_vec_sim.append(np.array([r["force"] for r in rows], dtype=float))

plot_funcs.plot_force_along_traj_comparsion(F_vec_exp, F_vec_sim, x_y_theta_vec_exp, exp_labels, sim_labels, buckle_str, err=err)
# %%
