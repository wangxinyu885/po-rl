import Grid2OpOld
from Grid2OpOld.grid2op.Environment.outage_env import OutageEnv
env = OutageEnv()
obs = env.reset()

import numpy as np
dim_action_space = len(env.action_space)
act = np.array([1 for _ in range(dim_action_space)])

for j in range(30):
    print('----------------------------------',j)
    done = False
    i=0
    obs = env.reset()
    if obs is not None:
        print(obs[0][6:26])
    else:
        continue
    while not done and i < 10:
        i=i+1
        obs, reward, done,terminated, info = env.step(act)
        if not done:
            print(obs[6:26])