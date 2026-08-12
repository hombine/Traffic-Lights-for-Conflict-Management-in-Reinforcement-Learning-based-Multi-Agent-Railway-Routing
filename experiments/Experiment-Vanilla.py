

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import ray
import gymnasium as gym
import os
import matplotlib.pyplot as plt

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from flatland.envs.rail_env import RailEnv
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.agent_utils import TrainState

# loggin setup
import logging
from datetime import datetime

# metrics set up 
import csv
from collections import defaultdict, Counter

log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}_exp1.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    filemode="w"
)

# another metrics helper
def append_rows_to_csv(path, rows):
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    file_exists = os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)

# observation flattening

CHILDREN = ["L", "F", "R", "B"]

def node_to_features(node):
    if node is None:
        return np.zeros(6)

    return np.array([
        float(getattr(node, "dist_own_target_encountered", 0)),
        float(getattr(node, "dist_other_target_encountered", 0)),
        float(getattr(node, "dist_other_agent_encountered", 0)),
        float(getattr(node, "dist_potential_conflict", 0)),
        float(getattr(node, "dist_unusable_switch", 0)),
        float(getattr(node, "dist_to_next_branch", 0)),
    ])

def flatten_tree(node, depth, max_depth):
    feat = node_to_features(node)

    if depth == max_depth:
        return feat

    children = []

    for direction in CHILDREN:
        if node is not None and hasattr(node, "childs"):
            child = node.childs.get(direction)
        else:
            child = None

        children.append(flatten_tree(child, depth + 1, max_depth))

    return np.concatenate([feat] + children)

def flatten_obs(obs, max_depth=2):
    if obs is None:
        return np.zeros((4 ** (max_depth + 1) - 1) * 6)

    return flatten_tree(obs, 0, max_depth)

# metric class
class EpisodeMetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_reward = 0.0
        self.living_reward_sum = 0.0
        self.goal_reward_sum = 0.0
        self.progress_reward_sum = 0.0
        self.conflict_penalty_sum = 0.0

        self.blocked_steps = 0
        self.block_events = []
        self.block_location_counts = Counter()
        self.waiting_steps_by_agent = defaultdict(int)

        self.arrival_steps = {}
        self.last_positions = {}

    def add_reward(self, total, living=0.0, goal=0.0, progress=0.0, conflict=0.0):
        self.total_reward += total
        self.living_reward_sum += living
        self.goal_reward_sum += goal
        self.progress_reward_sum += progress
        self.conflict_penalty_sum += conflict

    def add_blocked_step(self, agent, pos):
        self.blocked_steps += 1
        self.waiting_steps_by_agent[agent] += 1
        if pos is not None:
            self.block_location_counts[pos] += 1

    def add_block_event(self, event):
        self.block_events.append(event)

    def add_arrival(self, agent, step):
        if agent not in self.arrival_steps:
            self.arrival_steps[agent] = step

    def episode_summary(self, experiment, iteration, episode_id, completion_rate, max_agents, elapsed_steps):
        durations = [e["duration"] for e in self.block_events]

        return {
            "experiment": experiment,
            "iteration": iteration,
            "episode_id": episode_id,
            "elapsed_steps": elapsed_steps,
            "total_reward": self.total_reward,
            "living_reward_sum": self.living_reward_sum,
            "goal_reward_sum": self.goal_reward_sum,
            "progress_reward_sum": self.progress_reward_sum,
            "conflict_penalty_sum": self.conflict_penalty_sum,
            "completion_rate": completion_rate,
            "agents_completed": len(self.arrival_steps),
            "mean_arrival_step": np.mean(list(self.arrival_steps.values())) if self.arrival_steps else np.nan,
            "throughput": len(self.arrival_steps) / max(elapsed_steps, 1),
            "blocked_steps": self.blocked_steps,
            "num_block_events": len(self.block_events),
            "mean_block_duration": np.mean(durations) if durations else 0.0,
            "max_block_duration": np.max(durations) if durations else 0.0,
            "mean_waiting_steps_per_agent": np.mean(list(self.waiting_steps_by_agent.values())) if self.waiting_steps_by_agent else 0.0,
        }

    def location_summary(self, experiment, iteration, episode_id):
        rows = []
        for pos, count in self.block_location_counts.items():
            x, y = pos
            rows.append({
                "experiment": experiment,
                "iteration": iteration,
                "episode_id": episode_id,
                "x": x,
                "y": y,
                "blocked_step_count": count,
            })
        return rows


# flatland env wrapper for rllib

class FlatlandRllibEnv(MultiAgentEnv):

    def __init__(self, config):

        import logging
        import os

        log_dir = os.path.abspath("worker_logs")
        os.makedirs(log_dir, exist_ok=True)

        log_filename = os.path.join(
            log_dir,
            f"deadlock_penalty_worker_{os.getpid()}_fixed.log"
        )

        # Create worker-specific logger
        self.logger = logging.getLogger(f"flatland_worker_{os.getpid()}")

        # Prevent duplicate handlers
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)

            handler = logging.FileHandler(log_filename, mode="w")
            formatter = logging.Formatter("%(asctime)s | %(message)s")

            handler.setFormatter(formatter)

            self.logger.addHandler(handler)

            

            self.logger.propagate = False

        super().__init__()

        self.max_depth = config.get("max_depth", 2)
        self.seed = config.get("seed", 42)

        self.finished_agents = set()
        self.last_completion_rate = 0.0

        self.prev_dist = {}

        self.blocked_since = {}
        self.blocked_events = []

        # metric init
        self.experiment_name = config.get("experiment_name", "exp1_vanilla_ppo")
        self.current_iteration = 0
        self.episode_id = 0

        self.metric_tracker = EpisodeMetricTracker()
        self.completed_episode_summaries = []
        self.completed_location_summaries = []

        # debugging
        self.episode_id_debug = 0
        self.local_step_debug = 0

        obs_builder = TreeObsForRailEnv(max_depth=self.max_depth)
        num_agents = 4
        
        

        self.env = RailEnv(
            width=100,
            height=100,
            rail_generator=sparse_rail_generator(
                max_num_cities=3,
                grid_mode=False,
                max_rails_between_cities=2,
                max_rail_pairs_in_city=3,
                seed = self.seed,
            ),
            line_generator=sparse_line_generator(seed=self.seed),
            number_of_agents=num_agents,
            obs_builder_object=obs_builder
        )

        
        num_agents_env = num_agents
        self.possible_agents = list(range(num_agents_env))
        self.agents = self.possible_agents.copy()
        print(self.possible_agents)
        print(self.agents)

        

        # fixed observation mismatch
        self.obs_size = sum(4 ** i for i in range(self.max_depth + 1)) * 6

        obs_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_size,),
            dtype=np.float32
        )

        act_space = gym.spaces.Discrete(5)

        self.observation_space = gym.spaces.Dict({
        i: obs_space for i in range(num_agents_env)
        })

        print(self.observation_space)

        self.action_space = gym.spaces.Dict({
        i: act_space for i in range(num_agents_env)
        })

        print(self.action_space)

    # metrics helper
    def drain_completed_metrics(self):
        episode_rows = self.completed_episode_summaries
        location_rows = self.completed_location_summaries

        self.completed_episode_summaries = []
        self.completed_location_summaries = []

        return episode_rows, location_rows

    def reset(self, *, seed=None, options=None):

        raw_obs, info = self.env.reset()
        self.finished_agents = set()

        self.prev_dist = {}

        self.blocked_since = {}
        self.blocked_events = []

        self.metric_tracker.reset()
        self.episode_id += 1

        self.agents = self.possible_agents.copy()

        obs = {
            agent: np.nan_to_num(
                flatten_obs(raw_obs.get(agent, None), self.max_depth),
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            ).astype(np.float32)
            for agent in self.agents
        }

        self.episode_id_debug += 1
        self.local_step_debug = 0

        #print(
         #   f"[RESET] episode={self.episode_id_debug} "
          #  f"obs_keys={sorted(obs.keys())} "
           # f"self_agents={sorted(self.agents)}"
        #)

        for agent, o in obs.items():
            if not np.all(np.isfinite(o)):
                print(f"[BAD OBS] agent={agent}")
                print("shape:", o.shape)
                print("obs:", o)
                print("has_nan:", np.isnan(o).any(), "has_inf:", np.isinf(o).any())
                raise ValueError("Non-finite observation detected")

        infos = {agent: {} for agent in self.agents}

        return obs, infos

    def step(self, action_dict):
        try:
            active_agents = self.agents.copy()
            flatland_actions = {}

            for agent, action in action_dict.items():
                if isinstance(action, dict):
                    if len(action) == 0:
                        action = 0
                    else:
                        action = next(iter(action.values()))
                flatland_actions[agent] = int(action)

            obs, rewards, dones, flatland_info = self.env.step(flatland_actions)

            # info conversion
            infos = {agent: {} for agent in self.possible_agents}

            if isinstance(flatland_info, dict):
                for field_name, per_agent_values in flatland_info.items():
                    if isinstance(per_agent_values, dict):
                        for agent, value in per_agent_values.items():
                            infos.setdefault(agent, {})
                            infos[agent][field_name] = value

            # blocks

            for agent in self.possible_agents:
                agent_info = infos.get(agent, {})
                state = agent_info.get("state", None)

                # only stopped trains on the map
                if state == TrainState.STOPPED and self.env.agents[agent].position is not None:
                    pos = self.env.agents[agent].position
                    direction = self.env.agents[agent].direction
                    target = self.env.agents[agent].target

                    

            current_step = self.env._elapsed_steps

            for agent in self.possible_agents:
                state = infos.get(agent, {}).get("state", None)
                pos = self.env.agents[agent].position

                # check that train actually on map
                is_blocked = (
                        state == TrainState.STOPPED
                        and pos is not None
                )
                if is_blocked:
                    self.metric_tracker.add_blocked_step(agent, pos)

                if is_blocked and agent not in self.blocked_since:
                    self.blocked_since[agent] = {
                        "start_step": current_step,
                        "start_pos": pos,
                        "start_direction": self.env.agents[agent].direction,
                        "target": self.env.agents[agent].target
                    }

                    

                elif not is_blocked and agent in self.blocked_since:
                    event = self.blocked_since.pop(agent)
                    duration = current_step - event["start_step"]

                    self.blocked_events.append({
                        "agent": agent,
                        "start_step": event["start_step"],
                        "end_step": current_step,
                        "duration": duration,
                        "start_pos": event["start_pos"],
                        "resolved": True
                    })

                    self.metric_tracker.add_block_event(self.blocked_events[-1])

                    

            for agent in rewards:
                base_reward = rewards.get(agent, 0.0)
                living_penalty = -0.01
                progress_term = 0.0
                conflict_penalty = 0.0

                r = base_reward + living_penalty

                self.metric_tracker.add_reward(
                    total=r,
                    living=living_penalty,
                    goal=base_reward,
                    progress=progress_term,
                    conflict=conflict_penalty
                )
                rewards[agent] = r



            

            terminateds = {}
            truncateds = {}

            for agent in active_agents:
                state = infos.get(agent, {}).get("state", None)
                terminateds[agent] = state == TrainState.DONE
                truncateds[agent] = False

            terminateds["__all__"] = bool(dones.get("__all__", False))
            truncateds["__all__"] = False

            for agent in active_agents:
                if terminateds.get(agent, False):
                    self.finished_agents.add(agent)
                    self.metric_tracker.add_arrival(agent, self.env._elapsed_steps)





            obs = {
                agent: np.nan_to_num(
                    flatten_obs(o, self.max_depth),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                ).astype(np.float32)
                for agent, o in obs.items()
                if not terminateds.get(agent, False)
            }

            for agent, o in obs.items():
                if not np.all(np.isfinite(o)):
                    print(f"[BAD OBS] agent={agent}")
                    print("shape:", o.shape)
                    print("obs:", o)
                    print("has_nan:", np.isnan(o).any(), "has_inf:", np.isinf(o).any())
                    raise ValueError("Non-finite observation detected")

                    
            if np.random.rand() < 0.1:  

                conflict_summary = {
                    "blocked": 0,
                    "done": 0,
                    "malfunction": 0,
                    "active": 0
                }

                for agent in self.possible_agents:
                    agent_info = infos.get(agent, {})
                    state = agent_info.get("state", None)

                    if state == TrainState.DONE:
                        conflict_summary["done"] += 1
                    elif state == TrainState.STOPPED:
                        conflict_summary["blocked"] += 1
                    elif state == TrainState.MALFUNCTION:
                        conflict_summary["malfunction"] += 1
                    else:
                        conflict_summary["active"] += 1

                

            self.agents = list(obs.keys())

            
            rewards = {a: float(rewards.get(a, 0.0)) for a in active_agents}
            infos = {a: infos.get(a, {}) for a in active_agents}
            terminateds = {a: bool(terminateds.get(a, False)) for a in active_agents}
            truncateds = {a: bool(truncateds.get(a, False)) for a in active_agents}

            terminateds["__all__"] = bool(dones.get("__all__", False))
            if terminateds["__all__"]:
                for a in active_agents:
                    terminateds[a] = True
            truncateds["__all__"] = False

            self.agents = [
                a for a in active_agents
                if not terminateds.get(a, False)
            ]

            obs = {
                a: obs[a]
                for a in self.agents
                if a in obs
            }

            if dones["__all__"]:
                completion = len(self.finished_agents) / len(self.possible_agents)
                #print(completion)
                self.last_completion_rate = completion

                current_step = self.env._elapsed_steps

                for agent, event in self.blocked_since.items():
                    duration = current_step - event["start_step"]

                    self.blocked_events.append({
                        "agent": agent,
                        "start_step": event["start_step"],
                        "end_step": current_step,
                        "duration": duration,
                        "start_pos": event["start_pos"],
                        "resolved": False
                    })

                    self.metric_tracker.add_block_event(self.blocked_events[-1])

                    

                self.blocked_since = {}

                for agent in infos:
                    infos[agent]["episode_extra_metrics"] = {
                        "completion_rate": completion
                    }

                episode_row = self.metric_tracker.episode_summary(
                    experiment=self.experiment_name,
                    iteration=self.current_iteration,
                    episode_id=self.episode_id,
                    completion_rate=completion,
                    max_agents=len(self.possible_agents),
                    elapsed_steps=self.env._elapsed_steps
                )

                location_rows = self.metric_tracker.location_summary(
                    experiment=self.experiment_name,
                    iteration=self.current_iteration,
                    episode_id=self.episode_id
                )

                self.completed_episode_summaries.append(episode_row)
                self.completed_location_summaries.extend(location_rows)

                # asserts to catch errors
                assert set(obs.keys()).issubset(set(self.agents))
                assert set(rewards.keys()) == set(active_agents)
                assert set(infos.keys()) == set(active_agents)
                assert set(k for k in terminateds if k != "__all__") == set(active_agents)
                assert set(k for k in truncateds if k != "__all__") == set(active_agents)

                

            return obs, rewards, terminateds, truncateds, infos

        except Exception as e:
            
            print(f"[ENV STEP CRASH] {e}")
            raise


# connecting implementation to rllib

def env_creator(config):
    return FlatlandRllibEnv(config)

register_env("flatland_env", env_creator)


# plotting utils
def plot_training_curves(reward_history, completion_history, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    # Plt 1 reward
    plt.figure()
    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Reward Over Time")
    plt.savefig(os.path.join(save_dir, "reward_curve_exp1.png"))
    plt.close()

    # Plt 2 completion 
    plt.figure()
    plt.plot(completion_history)
    plt.xlabel("Episode")
    plt.ylabel("Completion Rate")
    plt.title("Completion Rate Over Time")
    plt.savefig(os.path.join(save_dir, "completion_curve_exp1.png"))
    plt.close()

    print(f"Plots saved to: {save_dir}")


# training

if __name__ == "__main__":

    ray.init(
    num_cpus=2,
    include_dashboard=False,
    object_store_memory=512 * 1024 * 1024) # trying to managa computer resources

    dummy_env = FlatlandRllibEnv({"max_depth": 2})
    print(dummy_env.observation_space[0])
    obs_space = dummy_env.observation_space[0]
    act_space = dummy_env.action_space[0]

    policies = {
        "shared_policy": (
            None,
            obs_space,
            act_space,
            {}
        )
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return "shared_policy"

    config = (
        PPOConfig()
        .environment(
            env="flatland_env",
            env_config={"max_depth": 2, "seed": 42}
        )
        .framework("torch")
        .env_runners(
            num_env_runners=1,
            num_envs_per_env_runner=1,
            sample_timeout_s=600,
            rollout_fragment_length=200,
        )
        .training(
            gamma=0.99,
            lr=5e-5,
            train_batch_size=4000, 
            model={"fcnet_hiddens": [256, 256]}
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn
        )

    )

    algo = config.build()

    reward_history = []
    completion_history = []

    logging.info("Starting training")
    logging.info(f"PPO Config: {config.to_dict()}")

    metrics_dir = "metrics_exp1"
    episode_metrics_path = os.path.join(metrics_dir, "episode_metrics_exp1.csv")
    location_metrics_path = os.path.join(metrics_dir, "location_metrics_exp1.csv")
    iteration_metrics_path = os.path.join(metrics_dir, "iteration_metrics_exp1.csv")

    for i in range(400):

        result = algo.train()


        def collect_worker_metrics(r):
            if r.env is None:
                return [], []

            env = r.env.envs[0].env
            env.current_iteration = i
            return env.drain_completed_metrics()


        worker_outputs = algo.env_runner_group.foreach_env_runner(collect_worker_metrics)

        episode_rows = []
        location_rows = []

        for ep_rows, loc_rows in worker_outputs:
            episode_rows.extend(ep_rows)
            location_rows.extend(loc_rows)

        append_rows_to_csv(episode_metrics_path, episode_rows)
        append_rows_to_csv(location_metrics_path, location_rows)

        

        reward = result["env_runners"]["episode_return_mean"]
        episodes = result["env_runners"]["num_episodes"]

        

        env_runner = algo.env_runner_group.local_env_runner
        env = env_runner.env
        
        def inspect_runner(r):
            if r.env is None:
                return "LOCAL RUNNER (no env)"

            e = r.env.envs[0]   
            stack = []

            
            while True:
                stack.append(type(e).__name__)
                if hasattr(e, "env"):
                    e = e.env
                else:
                    break

            return stack

        
        completion_values = algo.env_runner_group.foreach_env_runner(
            lambda r: r.env.envs[0].env.last_completion_rate
            if r.env is not None else None
        )

        completion_values = [c for c in completion_values if c is not None]
        completion = np.mean(completion_values) if completion_values else 0.0

        print(
            f"Iter {i}: "
            f"reward_mean = {reward:.2f} | "
            f"episodes = {episodes} |"
            f"completion rate = {completion}"
        )

        reward_history.append(reward)
        completion_history.append(completion)

        logging.info(
            f"Iter {i} | reward_mean={reward:.2f} | episodes={episodes} | completion={completion:.3f}"
        )

        

    checkpoint_dir = os.path.abspath("exp1")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = algo.save(checkpoint_dir)
    print(f"Saved checkpoint to: {checkpoint}")    
    plot_training_curves(reward_history, completion_history)    
    ray.shutdown()
