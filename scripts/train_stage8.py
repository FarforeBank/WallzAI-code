import sys

import train as train_module
from envs.quoridor.quoridor_env import MOVE_ACTIONS, QuoridorEnv as BaseQuoridorEnv


class RaceFinishQuoridorEnv(BaseQuoridorEnv):
    """Reward wrapper for late-game race discipline.

    This does not add any browser/runtime guard. It changes training feedback so the
    policy learns to finish races itself:
    - reward direct progress more when close to the finish;
    - punish sideways/backward moves near the finish;
    - punish non-urgent walls when our path is already short;
    - reward emergency walls only when they actually increase opponent BFS distance.
    """

    def step(self, action):
        action = int(action)
        prev_p1_dist = self.engine.get_bfs_distance(self.engine.p1_pos, 0)
        prev_p2_dist = self.engine.get_bfs_distance(self.engine.p2_pos, 8)
        was_wall = action >= MOVE_ACTIONS

        obs, reward, terminated, truncated, info = super().step(action)

        new_p1_dist = self.engine.get_bfs_distance(self.engine.p1_pos, 0)
        new_p2_dist = self.engine.get_bfs_distance(self.engine.p2_pos, 8)
        p1_progress = prev_p1_dist - new_p1_dist
        p2_slowdown = new_p2_dist - prev_p2_dist

        # Endgame: if we can finish soon, teach the policy to actually run.
        if prev_p1_dist <= 5:
            if not was_wall:
                if p1_progress > 0:
                    reward += 0.28 + 0.45 * p1_progress
                else:
                    reward -= 0.45
            else:
                # Near finish, walls are only worth it if opponent is also dangerous
                # and the wall measurably slows them.
                if prev_p2_dist > 2:
                    reward -= 0.55
                if p2_slowdown <= 0:
                    reward -= 0.35
                if new_p1_dist > prev_p1_dist:
                    reward -= 0.25

        # Emergency defense: reward walls only when they really delay an opponent
        # who is close to winning. This should reduce random wall spam.
        if was_wall and prev_p2_dist <= 3:
            if p2_slowdown > 0:
                reward += 0.45 + 0.35 * p2_slowdown
            else:
                reward -= 0.35

        # Race pressure: do not let the opponent walk freely into the finish.
        if not was_wall and prev_p2_dist <= 2 and new_p2_dist <= prev_p2_dist:
            reward -= 0.20

        return obs, reward, terminated, truncated, info


train_module.QuoridorEnv = RaceFinishQuoridorEnv
train_module.STAGES["8"] = {
    "name": "race-finish-wall-timing-finetune",
    "model_dir": "best_model",
    "random_walls_range": (0, 2),
    "move_only": False,
    "repeat_penalty": True,
    "opponent_policy": "greedy",
    "opponent_randomness": 0.08,
    "wall_reward": True,
    "wall_candidate_limit": 40,
    "opponent_start_advantage_range": (0, 2),
    "defensive_wall_reward": True,
    "opponent_wall_probability": 0.25,
    "self_trap_penalty": True,
    "timesteps": 1_500_000,
    "n_eval_episodes": 80,
    "description": "fine-tune without browser guards: finish winning races and use walls only when they really slow opponent",
}


if __name__ == "__main__":
    if "--stage" not in sys.argv:
        sys.argv.extend(["--stage", "8"])
    train_module.main()
