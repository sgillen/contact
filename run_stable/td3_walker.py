import gym
import numpy as np
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TD3
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import pybullet_envs
import time
import os
from stable_baselines.common import make_vec_env
from multiprocessing import Process
import seagul.envs.bullet
import json


num_steps = int(1e6)
    
physics_params = {
    'fixedTimeStep': 0.008,
    'numSubSteps': 4,
    'numSolverIterations': 5,
    'useSplitImpulse': 1,
    'splitImpulsePenetrationThreshold': -0.03999999910593033,
    'contactBreakingThreshold': 0.02,
    'collisionFilterMode': 1,
    'enableFileCaching': 1,
    'restitutionVelocityThreshold': 0.20000000298023224,
    'erp': 0.0,
    'frictionERP': 0.0,
    'contactERP': 0.0,
    'globalCFM': 0.0,
    'enableConeFriction': 0,
    'deterministicOverlappingPairs': 1,
    'allowedCcdPenetration': 0.04,
    'jointFeedbackMode': 0,
    'solverResidualThreshold': 1e-07,
    'contactSlop': 1e-05,
    'enableSAT': 0,
    'constraintSolverType': 0,
    'reportSolverAnalytics': 1,
}

dynamics_params = {
    'lateralFriction': 0.7,
    'restitution': 0.0,
    'restitution': 0.0,
    'rollingFriction': 0.1,
    'spinningFriction': 0.1,
    'contactDamping': -1,
    'contactStiffness': -1,
    'contactDamping': 2e3,
    'contactStiffness': 6e5,
    #    'contactDamping': 948.95018837,
    #    'contactStiffness': 23201.60208068,
    'contactDamping': -1,
    'contactStiffness': -1,
    'collisionMargin': 0.0,
    'angularDamping': 0.0,
    'linearDamping': 0.0,
    'jointDamping': .1,
}

torque_limits = (100,100,100,100,100,100)

env_config = {"physics_params":physics_params,
              "dynamics_params":dynamics_params,
              "torque_limits"  :torque_limits
}


base_dir = "data_sgb/td3/"
trial_name = input("Trial name: ")

trial_dir = base_dir + trial_name + "/"
base_ok = input("run will be saved in " + trial_dir + " ok? y/n")

if base_ok == "n":
    exit()


def run_stable(num_steps, save_dir):

    env = make_vec_env(seagul.envs.bullet.PBMJWalker2dFCEnv, n_envs=1, monitor_dir=save_dir, env_kwargs=env_config)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    
    model = TD3(MlpPolicy,
                env,
                action_noise=action_noise,
                verbose=1,
                gamma = 0.99,
                buffer_size= 1000000,
                learning_starts= 10000,
                batch_size= 100,
                learning_rate= 1e-3,
                train_freq= 1000,
                gradient_steps= 1000,
                policy_kwargs={"layers":[400, 300]},
                n_cpu_tf_sess=1,
    )

    model.learn(total_timesteps=num_steps)
    model.save(save_dir + "/model.zip")


if __name__ == "__main__":
    
    start = time.time()

    proc_list = []

    os.makedirs(trial_dir, exist_ok=False)
    with open(trial_dir + "config.json", "w") as config_file:
        json.dump(env_config, config_file)

    
    for seed in np.random.randint(0, 2 ** 32, 8):
        #    run_stable(int(8e4), "./data/walker/" + trial_name + "_" + str(seed))

        save_dir = trial_dir + "/" + str(seed)
        os.makedirs(save_dir, exist_ok=False)

        
        p = Process(
            target=run_stable,
            args=(num_steps, save_dir)
        )

        p.start()
        proc_list.append(p)

    for p in proc_list:
        print("joining")
        p.join()

    print(f"experiment complete, total time: {time.time() - start}, saved in {save_dir}")

