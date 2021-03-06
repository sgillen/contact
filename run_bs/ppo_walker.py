# %%

from seagul.rl.run_utils import run_and_save_bs
import numpy as np
from multiprocessing import Process

trial_name = input("trial name:")

proc_list = []
for seed in np.random.randint(0,2**16,8):
    arg_dict = {
        'env' : 'Walker2d-v2',
        'alg' : 'ppo2',
        'num_timesteps' : '2e6',
        'seed' : str(seed),
    }

    p = Process(
        target=run_and_save_bs,
        args=(arg_dict, "15_" + str(seed), "many seeds, default params", f"/data_bs/{trial_name}/"),
    )

    p.start()
    proc_list.append(p)

for p in proc_list:
    p.join()

#run_and_save_bs(arg_dict)
        
