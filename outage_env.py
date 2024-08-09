import copy
from typing import Optional
import grid2op
import numpy as np
import networkx as nx
import gymnasium as gym
import torch
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
                 enable_action_mapping: bool=False) -> None:
        
        self._env = grid2op.make(
            env_name, 
            reward_class=reward_class, 
            backend=backend(), 
            test=test,
            data_feeding_kwargs={"gridvalueClass": GridStateFromFile}
        )
        param = self._env.parameters
        param.MAX_LINE_STATUS_CHANGED = 1000
        param.ENV_DC = False 
        param.NO_OVERFLOW_DISCONNECTION = True 
        param.NB_TIMESTEP_OVERFLOW_ALLOWED = 0 
        self._env.change_parameters(param)

        self._gym_env = GymEnv(self._env)

        # 确保定义了 obs_attr_to_keep
        self.obs_attr_to_keep = ["gen_p", "load_p", "line_status", "rho", "topo_vect", "connectivity_matrix"]
        self._gym_env.observation_space = BoxGymObsSpace(
            self._env.observation_space,
            attr_to_keep=self.obs_attr_to_keep,
            functs={"connectivity_matrix": (lambda grid2opobs: grid2opobs.connectivity_matrix().flatten(), 0., 1.0, None, None)}
        )
        self.observation_space = self._gym_env.observation_space
        # print(self.observation_space)
        self.action_attr_to_keep = [ "change_bus","set_line_status_simple"]
        self._gym_env.action_space= DiscreteActSpace(self._env.action_space,attr_to_keep=self.action_attr_to_keep)
        self.enable_action_mapping = enable_action_mapping
        self.action_space = self._gym_env.action_space
        # print(self.action_space)
        if self.enable_action_mapping:
            self.action_space = gym.spaces.Discrete(3)
        self.num_outages = self._env.n_line
        self.outage_idx = 0 + 23

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
        
        self.g2op_obs = None
        self.executed_actions = set()  # 初始化已执行的动作
    def action_mapping(self, action: np.ndarray) -> np.ndarray:
        """
        map an int to nd_array (grid2op can use)
        """
        pass
        # mapping = {1: np.array([1 for _ in range(len(self._gym_env.action_space))]}
                   
        # return mapping[action]
        

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
    def initial_outage_act_2_gym(self, initial_outages, dim_topo, dim_action_space): # this works for multidiscrete action space
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
    def line_disconnect_act_2_gym(self, initial_outages,dim_action_space): # this works for discrete action space
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
    #     # print(outage_line_id_to_set)        
        
    #     #gym_action = self.initial_outage_act_2_gym(outage_line_id_to_set, self._env.dim_topo, len(self._gym_env.action_space))
    #     # obs, reward, done, truncated, info = self._gym_env.step(gym_action)
    #     g2op_act = self._env.action_space({"set_line_status":[[l_id, status] for l_id, status in zip(outage_line_id_to_set, [-1 for _ in range(len(outage_line_id_to_set))])]})
    #     g2op_obs, reward, done, info = self._gym_env.init_env.step(g2op_act)
    #     self.g2op_obs = g2op_obs
    #     obs = self._gym_env.observation_space.to_gym(g2op_obs)
    #     truncated = False
    #     info['line_outages'] = np.where(self.g2op_obs.line_status == False)[0]

    #     self.outage_idx = (self.outage_idx + 1) % self.n_initial_outages
    #     if done:
    #         print("system done after initial outages")
    #         return None 
    #     # breakpoint()
    #     return obs, reward, done, truncated, info
    
    # def step(self, action):

    #     tran_action = action
    #     if self.enable_action_mapping:
    #         tran_action = self.action_mapping(action)

    #     g2op_act = self._gym_env.action_space.from_gym(tran_action)
    #     g2op_obs, reward, done, info = self._gym_env.init_env.step(g2op_act)
    #     obs = self._gym_env.observation_space.to_gym(g2op_obs)
    #     truncated = False # see https://github.com/openai/gym/pull/2752
    #     self.g2op_obs = g2op_obs
        
    #     to_disconnect = np.where(self.g2op_obs.rho > 1)[0]
    #     if len(to_disconnect) > 0: 
          
    #         g2op_act = self._env.action_space({"set_line_status":[[l_id, status] for l_id, status in zip(to_disconnect, [-1 for _ in range(len(to_disconnect))])]})
    #         g2op_obs, _, done, info = self._gym_env.init_env.step(g2op_act)
    #         truncated = False
    #         self.g2op_obs = g2op_obs
    #         obs = self._gym_env.observation_space.to_gym(g2op_obs)
    #     else:
    #         done = True

    #     info['line_outages'] = np.where(self.g2op_obs.line_status == False)[0]
    #     return obs, reward, done, truncated, info
 
    def reset(self, **kwargs):
        self._gym_env.reset()
        self.old_line_status = copy.deepcopy(self._gym_env.init_env.backend.get_line_status())
        outage_line_id_to_set = self.initial_outages[self.outage_idx]

        g2op_act = self._env.action_space({"set_line_status":[[l_id, status] for l_id, status in zip(outage_line_id_to_set, [-1 for _ in range(len(outage_line_id_to_set))])]})
        print(f"Initial Action for Grid2Op: {g2op_act}")  # 打印初始动作
        g2op_obs, reward, done, info = self._gym_env.init_env.step(g2op_act)
        self.g2op_obs = g2op_obs
        obs = self._gym_env.observation_space.to_gym(g2op_obs)
        truncated = False
        info['line_outages'] = np.where(self.g2op_obs.line_status == False)[0]

        print(f"Initial Line Status After Reset: {self.g2op_obs.line_status}")  # 打印初始线路状态

        self.outage_idx = (self.outage_idx + 1) % self.n_initial_outages
        if done:
            print("system done after initial outages")
            return None 

        return obs, reward, done, truncated, info

    def step(self, action):
        tran_action = action
        if self.enable_action_mapping:
            tran_action = self.action_mapping(action)

        g2op_act = self._gym_env.action_space.from_gym(tran_action)
        print(f"Generated Action for Grid2Op: {g2op_act}")  # 打印生成的动作
        g2op_obs, reward, done, info = self._gym_env.init_env.step(g2op_act)
        obs = self._gym_env.observation_space.to_gym(g2op_obs)
        truncated = False
        self.g2op_obs = g2op_obs
        
        # 打印当前线路状态
        print(f"Current Line Status: {self.g2op_obs.line_status}")

        to_disconnect = np.where(self.g2op_obs.rho > 1)[0]
        if len(to_disconnect) > 0: 
            g2op_act = self._env.action_space({"set_line_status":[[l_id, status] for l_id, status in zip(to_disconnect, [-1 for _ in range(len(to_disconnect))])]})
            print(f"Disconnecting lines: {to_disconnect}")  # 打印将断开的线路
            g2op_obs, _, done, info = self._gym_env.init_env.step(g2op_act)
            self.g2op_obs = g2op_obs
            obs = self._gym_env.observation_space.to_gym(g2op_obs)

        info['line_outages'] = np.where(self.g2op_obs.line_status == False)[0]
        return obs, reward, done, truncated, info


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
    def load_p_pos(self):
        front_dim = 0
        sorted_attr_to_keep = sorted(self.obs_attr_to_keep)
        for attr_name in sorted_attr_to_keep:
            if attr_name == "load_p":
                break
            dim_value = {
                "gen_p": self._env.n_gen,
                "load_p": self._env.n_load,
                "line_status": self._env.n_line,
                "rho": self._env.n_line,
                "topo_vect": self._env.dim_topo,
                "connectivity_matrix": self._env.n_sub * self._env.n_sub
            }.get(attr_name, 0)
            front_dim += dim_value
        return front_dim
    def line_status_pos(self):
        front_dim = 0
        sorted_attr_to_keep = sorted(self.obs_attr_to_keep)
        for attr_name in sorted_attr_to_keep:
            if attr_name == "line_status":
                break
            dim_value = {
                "gen_p": self._env.n_gen,
                "load_p": self._env.n_load,
                "line_status": self._env.n_line,
                "rho": self._env.n_line,
                "topo_vect": self._env.dim_topo,
                "connectivity_matrix": self._env.n_sub * self._env.n_sub
            }.get(attr_name, 0)
            front_dim += dim_value
        return front_dim
