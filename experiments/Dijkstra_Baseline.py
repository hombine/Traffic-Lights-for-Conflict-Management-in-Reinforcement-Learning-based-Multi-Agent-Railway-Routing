

import warnings
warnings.filterwarnings("ignore")

import os
import csv
import heapq
import logging
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.agent_utils import TrainState

# define action
DO_NOTHING = 0
MOVE_LEFT = 1
MOVE_FORWARD = 2
MOVE_RIGHT = 3
STOP_MOVING = 4

# directions
DIR_TO_DELTA = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}

log_filename = f"dijkstra_{datetime.now().strftime('%Y%m%d_%H%M%S')}_exp1_21.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    filemode="w"
)


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


def create_flatland_env(seed=21, num_agents=4, max_depth=2):

    obs_builder = TreeObsForRailEnv(max_depth=max_depth)
    return RailEnv(
        width=100,
        height=100,
        rail_generator=sparse_rail_generator(
            max_num_cities=5,
            grid_mode=False,
            max_rails_between_cities=2,
            max_rail_pairs_in_city=3,
            seed=seed,
        ),
        line_generator=sparse_line_generator(seed=seed),
        number_of_agents=num_agents,
        obs_builder_object=obs_builder,
    )


def move_position(pos, direction):
    dr, dc = DIR_TO_DELTA[direction]
    return pos[0] + dr, pos[1] + dc


def get_valid_successors(env, position, direction):

    row, col = position

    transitions = env.rail.get_transitions(((row, col), direction))

    successors = []
    for new_direction, allowed in enumerate(transitions):
        if not allowed:
            continue

        next_pos = move_position(position, new_direction)
        successors.append((next_pos, new_direction))

    return successors


def dijkstra_shortest_path(env, start_pos, start_dir, target_pos):

    start_state = (start_pos, start_dir)
    pq = [(0, start_state)]
    dist = {start_state: 0}
    prev = {}
    visited = set()
    best_target_state = None

    while pq:
        cost, state = heapq.heappop(pq)
        if state in visited:
            continue
        visited.add(state)

        position, direction = state
        if position == target_pos:
            best_target_state = state
            break

        for next_pos, next_dir in get_valid_successors(env, position, direction):
            next_state = (next_pos, next_dir)
            new_cost = cost + 1
            if new_cost < dist.get(next_state, float("inf")):
                dist[next_state] = new_cost
                prev[next_state] = state
                heapq.heappush(pq, (new_cost, next_state))

    if best_target_state is None:
        return []

    path = []
    current = best_target_state
    while current != start_state:
        path.append(current)
        current = prev[current]
    path.append(start_state)
    path.reverse()
    return path


def direction_to_action(current_dir, next_dir):

    if next_dir == current_dir:
        return MOVE_FORWARD
    if next_dir == (current_dir - 1) % 4:
        return MOVE_LEFT
    if next_dir == (current_dir + 1) % 4:
        return MOVE_RIGHT

    return DO_NOTHING


def agent_current_state(agent):

    if agent.position is None:
        return agent.initial_position, agent.initial_direction
    return agent.position, agent.direction


def compute_all_paths(env):
    paths = {}
    for agent_id, agent in enumerate(env.agents):
        start_pos = agent.initial_position
        start_dir = agent.initial_direction
        target_pos = agent.target
        path = dijkstra_shortest_path(env, start_pos, start_dir, target_pos)
        paths[agent_id] = path
        logging.info(
            f"agent={agent_id} start={start_pos} dir={start_dir} target={target_pos} "
            f"path_length={len(path)}"
        )
    return paths


def choose_actions_from_paths(env, paths, finished_agents):

    actions = {}
    reserved_next_cells = set()

    occupied_cells = {
        agent.position
        for agent in env.agents
        if agent.position is not None
    }

    for agent_id, agent in enumerate(env.agents):
        if agent_id in finished_agents:
            continue


        if getattr(agent, "state", None) == TrainState.DONE:
            actions[agent_id] = DO_NOTHING
            continue

        path = paths.get(agent_id, [])
        if not path:
            actions[agent_id] = DO_NOTHING
            continue

        current_pos, current_dir = agent_current_state(agent)


        current_index = None
        for i, (path_pos, path_dir) in enumerate(path):
            if path_pos == current_pos and path_dir == current_dir:
                current_index = i
                break

        if current_index is None:
            for i, (path_pos, path_dir) in enumerate(path):
                if path_pos == current_pos:
                    current_index = i
                    break

        if current_index is None or current_index >= len(path) - 1:
            actions[agent_id] = DO_NOTHING
            continue

        next_pos, next_dir = path[current_index + 1]


        blocked_by_current_occupancy = next_pos in occupied_cells and next_pos != agent.position
        blocked_by_reservation = next_pos in reserved_next_cells

        if blocked_by_current_occupancy or blocked_by_reservation:
            actions[agent_id] = STOP_MOVING
        else:
            actions[agent_id] = direction_to_action(current_dir, next_dir)
            reserved_next_cells.add(next_pos)

    return actions


def update_block_metrics(env, metric_tracker, blocked_since, current_step, infos):
    for agent_id, agent in enumerate(env.agents):
        state = None
        if isinstance(infos, dict):
            state = infos.get("state", {}).get(agent_id, None)

        pos = agent.position
        is_blocked = state == TrainState.STOPPED and pos is not None

        if is_blocked:
            metric_tracker.add_blocked_step(agent_id, pos)

        if is_blocked and agent_id not in blocked_since:
            blocked_since[agent_id] = {
                "start_step": current_step,
                "start_pos": pos,
                "start_direction": agent.direction,
                "target": agent.target,
            }
        elif not is_blocked and agent_id in blocked_since:
            event = blocked_since.pop(agent_id)
            duration = current_step - event["start_step"]
            metric_tracker.add_block_event({
                "agent": agent_id,
                "start_step": event["start_step"],
                "end_step": current_step,
                "duration": duration,
                "start_pos": event["start_pos"],
                "resolved": True,
            })


def run_dijkstra_episode(seed, episode_id, metrics_dir="metrics_dijkstra_exp1_21", max_steps=None):
    env = create_flatland_env(seed=seed, num_agents=4, max_depth=2)
    obs, info = env.reset()

    paths = compute_all_paths(env)
    metric_tracker = EpisodeMetricTracker()
    finished_agents = set()
    blocked_since = {}

    done_all = False

    while not done_all:
        actions = choose_actions_from_paths(env, paths, finished_agents)
        obs, rewards, dones, infos = env.step(actions)

        current_step = env._elapsed_steps

        for agent_id, reward in rewards.items():
            # track reward
            living_penalty = -0.01
            total_reward = float(reward) + living_penalty
            metric_tracker.add_reward(
                total=total_reward,
                living=living_penalty,
                goal=float(reward),
                progress=0.0,
                conflict=0.0,
            )

        update_block_metrics(env, metric_tracker, blocked_since, current_step, infos)

        for agent_id, agent in enumerate(env.agents):
            if agent_id not in finished_agents and agent.state == TrainState.DONE:
                finished_agents.add(agent_id)
                metric_tracker.add_arrival(agent_id, current_step)

        done_all = bool(dones.get("__all__", False))
        if max_steps is not None and current_step >= max_steps:
            done_all = True


    current_step = env._elapsed_steps
    for agent_id, event in list(blocked_since.items()):
        metric_tracker.add_block_event({
            "agent": agent_id,
            "start_step": event["start_step"],
            "end_step": current_step,
            "duration": current_step - event["start_step"],
            "start_pos": event["start_pos"],
            "resolved": False,
        })
    blocked_since.clear()

    completion_rate = len(finished_agents) / len(env.agents)

    episode_row = metric_tracker.episode_summary(
        experiment="exp1_dijkstra_baseline_21",
        iteration=0,
        episode_id=episode_id,
        completion_rate=completion_rate,
        max_agents=len(env.agents),
        elapsed_steps=env._elapsed_steps,
    )

    location_rows = metric_tracker.location_summary(
        experiment="exp1_dijkstra_baseline_21",
        iteration=0,
        episode_id=episode_id,
    )

    append_rows_to_csv(
        os.path.join(metrics_dir, "episode_metrics_dijkstra_exp1_21.csv"),
        [episode_row],
    )
    append_rows_to_csv(
        os.path.join(metrics_dir, "location_metrics_dijkstra_exp1_21.csv"),
        location_rows,
    )

    return episode_row


def run_experiment(num_episodes=100, base_seed=42):
    metrics_dir = "metrics_dijkstra_exp1_21"


    os.makedirs(metrics_dir, exist_ok=True)
    for filename in ["episode_metrics_dijkstra_exp1_21.csv", "location_metrics_dijkstra_exp1_21.csv"]:
        path = os.path.join(metrics_dir, filename)
        if os.path.exists(path):
            os.remove(path)

    all_episode_rows = []

    for episode_id in range(1, num_episodes + 1):

        seed = base_seed

        row = run_dijkstra_episode(
            seed=seed,
            episode_id=episode_id,
            metrics_dir=metrics_dir,
        )
        all_episode_rows.append(row)

        print(
            f"Episode {episode_id}: "
            f"completion={row['completion_rate']:.2f} | "
            f"agents_completed={row['agents_completed']} | "
            f"elapsed_steps={row['elapsed_steps']} | "
            f"blocked_steps={row['blocked_steps']} | "
            f"throughput={row['throughput']:.4f}"
        )

    mean_completion = np.mean([r["completion_rate"] for r in all_episode_rows])
    mean_throughput = np.mean([r["throughput"] for r in all_episode_rows])
    mean_blocked_steps = np.mean([r["blocked_steps"] for r in all_episode_rows])

    print("\nSummary")
    print(f"episodes: {num_episodes}")
    print(f"mean completion rate: {mean_completion:.3f}")
    print(f"mean throughput: {mean_throughput:.4f}")
    print(f"mean blocked steps: {mean_blocked_steps:.2f}")
    print(f"CSV files saved in: {metrics_dir}")


if __name__ == "__main__":
    run_experiment(num_episodes=5000, base_seed=21)
