import copy
from typing import Optional
import grid2op
import numpy as np

import gymnasium as gym

from grid2op.Backend.backend import Backend
from grid2op.Backend.pandaPowerBackend import PandaPowerBackend
from grid2op.Reward.baseReward import BaseReward
from grid2op.Reward.flatReward import FlatReward
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, DiscreteActSpace, MultiDiscreteActSpace
from grid2op.Chronics.gridStateFromFile import GridStateFromFile

class OutageEnv(gym.Env):

    def __init__(self, 
                 env_name: str="l2rpn_case14_sandbox",
                 backend: Optional[Backend]=PandaPowerBackend,
                 reward_class: Optional[BaseReward]=FlatReward,
                 test: bool=False,
                ) -> None:
        
        self._env = grid2op.make(
            env_name, 
            reward_class=reward_class, 
            backend=backend(), 
            test=test,
            data_feeding_kwargs={"gridvalueClass": GridStateFromFile}
        )
        param = self._env.parameters
        param.MAX_LINE_STATUS_CHANGED = 1000
        param.ENV_DC = False #不影响
        param.NO_OVERFLOW_DISCONNECTION = True # must ture
        param.NB_TIMESTEP_OVERFLOW_ALLOWED = 0 #不影响
        self._env.change_parameters(param)

        self._gym_env = GymEnv(self._env)

        # TODO: move these to config
        self.obs_attr_to_keep = ["gen_p", "load_p","line_status","rho"]
        self._gym_env.observation_space = BoxGymObsSpace(
            self._env.observation_space, 
            attr_to_keep = self.obs_attr_to_keep
        )
        self.observation_space = self._gym_env.observation_space

        self.action_attr_to_keep = ["set_line_status", "set_bus"]
        self._gym_env.action_space = MultiDiscreteActSpace(
            self._env.action_space,
            attr_to_keep = self.action_attr_to_keep
        )
        # print(self._gym_env.action_space)
        self.action_space = self._gym_env.action_space

        self.num_outages = self._env.n_line
        self.outage_idx = 0
        def getInitialOutages(linename):
            linename1 = list(map(lambda x: [int(i) for i in x.split('_')], linename))
            busmat1 = np.array([[int(-1) for j in range(self._env.n_sub)] for i in range(self._env.n_sub)])
            for l in linename1:
                busmat1[l[0],l[1]] = l[2]
                busmat1[l[1],l[0]] = l[2]

            n1 = [[l[2]] for l in linename1]

            from itertools import combinations
            n2 = [[] for _ in range(len(busmat1))]
            for i in range(len(busmat1)):
                x = busmat1[i]
                x = x[x>-1]
                if len(x)>0:
                    n2[i] = list(combinations(x, 2))
            n2 = [item for sublist in n2 for item in sublist]
            n2 = list(map(list, n2))

            initialOutages =  n1 + n2

            return initialOutages
        
        self.initial_outages = getInitialOutages(self._env.backend.name_line)
        self.n_initial_outages = len(self.initial_outages)
        
        self.current_obs = None

    # def _get_obs(self):
    #     return {"agent": self._agent_location, "target": self._target_location}

    # def _get_info(self):
    #     return {
    #         "distance": np.linalg.norm(
    #             self._agent_location - self._target_location, ord=1
    #         )
    #     }

    # convert grid2op_action of disconnecting lines to gym_action
    # only tested for attr_to_keep=["set_bus", "set_line_status"]
    # dim_set_line_status ,dim_change_line_status,dim_set_bus,dim_change_bus,dim_raise_alarm,\
    #     dim_raise_alert,dim_sub_set_bus,dim_sub_change_bus,dim_one_sub_set,dim_one_sub_change = \
    # [env.n_line,env.n_line,env.dim_topo,env.dim_topo,env.dim_alarms,env.dim_alerts,env.n_sub,env.n_sub,1,1]

    def initial_outage_act_2_gym(self, initial_outages, dim_topo, dim_action_space):
        front_dim = 0
        sorted_attr_to_keep = sorted(self.action_attr_to_keep)
        for attr_name in sorted_attr_to_keep:
          # 如果遇到"set_line_status"，则停止遍历
            if attr_name == "set_line_status":
                break
         # 找到当前属性名称对应的dim值
            dim_value = {
                "change_line_status":self._env.n_line,
                "set_bus": self._env.n_line,
                "change_bus": self._env.dim_topo,
                "raise_alarm" : self._env.dim_alarms,
                "raise_alert" : self._env.dim_alerts,
                "sub_set_bus" : self._env.n_sub,
                "sub_change_bus": self._env.n_sub,
                "one_sub_set" : 1,
                "one_sub_change":1
            }.get(attr_name, 0)  # 默认为0
            
            # 累加dim值到total
            front_dim += dim_value
        gym_action = np.array([1 for _ in range(dim_action_space)])
        for i in initial_outages:
            gym_action[front_dim + i] = 0 
        return gym_action

    # def reset(self, **kwargs):
    #     self._gym_env.reset()
    #     self.old_line_status = copy.deepcopy(self._gym_env.init_env.backend.get_line_status())
    #     outage_line_id_to_set = self.initial_outages[self.outage_idx]
    #     gym_action = self.initial_outage_act_2_gym(outage_line_id_to_set, self._env.dim_topo, len(self._gym_env.action_space))
    #     obs, reward, done, truncated, info = self._gym_env.step(gym_action)
    #     self.outage_idx = (self.outage_idx + 1) % self.n_initial_outages
    #     if done:
    #         print("system done after initial outages")
    #         return 
    #     return obs, info
    def reset(self, **kwargs):
        obs, info = self._gym_env.reset()
        self.old_line_status = copy.deepcopy(self._gym_env.init_env.backend.get_line_status())
        outage_line_id_to_set = self.initial_outages[self.outage_idx]
        gym_action = self.initial_outage_act_2_gym(outage_line_id_to_set, self._env.dim_topo, len(self._gym_env.action_space))
        
        obs, reward, done, truncated, info = self._gym_env.step(gym_action)
        self.outage_idx = (self.outage_idx + 1) % self.n_initial_outages
        if done:
            print("system done after initial outages")
            return self.reset(**kwargs)  # 尝试再次重置以获取有效的初始状态
        return obs, info

    def step(self, action):
        # import pdb;pdb.sent_trace()
        #####添加判断，系统done掉 先把obs, reward, done, truncated, info返回   蒙特卡洛
     
        # if action is not none: 
        #   apply action 
        # if no overflow:
        #   return
        # else:
        #   disconnect overflowed line
        obs, reward, done, truncated, info = self._gym_env.step(action)
        #line_status = obs[6:26]
        
        # TODO: how to find rho instead of using obs[37:57]
        to_disconnect = []
        front_dim = self.rho_pos() #ZWJ: find rho position
        for i in range(self._env.n_line):
            if obs[front_dim:front_dim+self._env.n_line][i] > 1:
                to_disconnect.append(i)
        # for i in range(len(obs[37:57])):
        #     if obs[37:57][i] > 1:
        #         to_disconnect.append(i)
        #print(to_disconnect)
        # breakpoint()

        if len(to_disconnect) > 0: 
            gym_action = self.initial_outage_act_2_gym(to_disconnect, self._env.dim_topo, len(self._gym_env.action_space))
            obs, reward, done, truncated, info = self._gym_env.step(gym_action)
        else:
            done = True

        # new_line_status = obs[6:26]
        #if np.array_equal(line_status, new_line_status):
        #    done = True       
        info['failed_nodes'] = to_disconnect
        return obs, reward, done, truncated, info
        # return observation, reward, terminated, False, info

    def render(self):
        pass
        
    def close(self):
        pass
    
    def rho_pos(self):
        front_dim = 0
        sorted_attr_to_keep = sorted(self.obs_attr_to_keep)
        for attr_name in sorted_attr_to_keep:
        # 如果遇到"set_line_status"，则停止遍历
            if attr_name == "rho":
                break
        # 找到当前属性名称对应的dim值
            dim_value = {
                "year" :1, "month":1, "day": 1 ,"hour_of_day":1,  "minute_of_hour":1, "day_of_week":1,"current_step":1,\
                "gen_p":self._env.n_gen, "gen_q":self._env.n_gen, "gen_v":self._env.n_gen, "gen_margin_up":self._env.n_gen,"gen_margin_down":self._env.n_gen,\
                "gen_theta":self._env.n_gen,"load_p":self._env.n_load,"load_q":self._env.n_load,"load_v":self._env.n_load, "load_theta":self._env.n_load,\
                "p_or":self._env.n_line,"q_or":self._env.n_line,"a_or":self._env.n_line ,"v_or":self._env.n_line,"theta_or":self._env.n_line,"p_ex":self._env.n_line,\
                "q_ex":self._env.n_line,"a_ex":self._env.n_line ,"v_ex":self._env.n_line,"theta_ex":self._env.n_line,"line_status":self._env.n_line ,\
                "timestep_overflow": self._env.n_line,"topo_vect":self._env.dim_topo,"time_before_cooldown_line":self._env.n_line,"time_before_cooldown_sub":self._env.n_sub ,\
                "time_next_maintenance":self._env.n_line, "duration_next_maintenance":self._env.n_line,  "target_dispatch":self._env.n_gen, "actual_dispatch":self._env.n_gen,\
                "storage_charge":self._env.n_storage,"storage_power_target":self._env.n_storage,"storage_power": self._env.n_storage,"storage_theta":self._env.n_storage,\
                "curtailment": self._env.n_gen,"curtailment_limit":self._env.n_gen, "curtailment_mw":self._env.n_gen,"curtailment_limit_mw":self._env.n_gen,\
                "thermal_limit": self._env.n_line,"is_alarm_illegal":1,"time_since_last_alarm":1,"last_alarm":self._env.dim_alarms,"attention_budget":1,\
                "was_alarm_used_after_game_over":1,"delta_time":1, "active_alert":self._env.dim_alerts,"time_since_last_alert":self._env.dim_alarms,\
                "alert_duration": self._env.dim_alarms,"time_since_last_attack":self._env.dim_alarms,"was_alert_used_after_attack":self._env.dim_alarms,\
                "attack_under_alert":self._env.dim_alarms,"max_step":1,"delta_time":1,"prod_p":self._env.n_gen,"prod_q":self._env.n_gen,"prod_v":self._env.n_gen,\
                "gen_p_before_curtail":self._env.n_gen,"curtailment_limit_effective":self._env.n_gen
                }.get(attr_name, 0)  # 默认为0
            # 累加dim值到total
            front_dim += dim_value
        return front_dim