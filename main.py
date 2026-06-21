# %% Imports

import importlib
import numpy as np

import plot_funcs

importlib.reload(plot_funcs)

# %% Compare

final_t = None
save = True

exp_file_prelim = "..\\Meca500\\data\\training\\\\May28_UpdateFromSim\\2ndTip\\"
sim_file_prelim = "..\\Bistable shape acquisition jax\\Training\\June6_May22singleHinge2ndEnd_May27shortArcTraj\\stiffk\\1stEnd\\"
# sim_file_prelim = "Bistable shape acquisition jax\\Training\\\June6_May22singleHinge2ndEnd_May27shortArcTraj\\"

init_buckle = "0001"
desired_buckle = "0000"

exp_file_path = exp_file_prelim + r"{}to{}.csv".format(init_buckle, desired_buckle)
# exp_file_path = exp_file_prelim + r"combined.csv".format(init_buckle, desired_buckle)
sim_file_path = sim_file_prelim + r"final_loss_0_init_{}_desired_{}.csv".format(init_buckle, desired_buckle)

plot_funcs.plot_compare_sim_exp_training(exp_file_path, sim_file_path, final_t, save)

# %% Single

final_t = None
save = True

file_path = r"..\Bistable shape acquisition jax\Training\June15_H10_pos\1stAndLast\0010100001to1010100000.csv"
plot_funcs.plot_sim_or_exp(file_path, mod="pos", final_t=final_t, save=save)

# %%

csv_path = r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI\Meca500\data\training\June19_fullTraining\0001to0000TrainingFull_pos1_Good\0001to0000_pos1.csv"
images_dir = r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI\Meca500\data\training\June19_fullTraining\0001to0000TrainingFull_pos1_Good\0001to0000Training_justUpdates" 
plot_funcs.training_data_and_vid(csv_file_path=csv_path, image_dir=images_dir, fps=2)
# %%

# %% Position-state training video

csv_file_path=r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI\Meca500\data\training\June21_fromPos\0001to1000\0001to1000.csv"
pics_dir=r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI\Meca500\data\training\June21_fromPos\0001to1000\pics"
plot_funcs.training_position_states_and_vid(csv_file_path=csv_file_path, pics_dir=pics_dir, infer_image_sequence=True, fps=2)
# %%
