from envs.quoridor.quoridor_env import (
    H_WALL_OFFSET,
    MOVE_ACTIONS,
    MOVES,
    V_WALL_OFFSET,
    QuoridorEnv,
)


class StrategicQuoridorEnv(QuoridorEnv):
    """Quoridor/Wallz environment with less greedy, less wall-spammy rewards.

    The old reward made the agent overvalue "my shortest path got smaller" and
    undervalue tactical traps.  This env trains on advantage instead:

        advantage = opponent_distance_to_finish - own_distance_to_finish

    It also masks most low-value wall actions before the policy can select them.
    Walls are only usually legal to the policy when they actually slow the
    opponent more than they hurt us, or when the opponent is close to winning.
    """

    def __init__(
        self,
        *args,
        wall_cost=0.22,
        useless_wall_penalty=0.85,
        self_harm_wall_penalty=0.65,
        min_wall_margin_delta=1,
        reserve_walls=3,
        emergency_p2_dist=4,
        mobility_penalty=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.wall_cost = float(wall_cost)
        self.useless_wall_penalty = float(useless_wall_penalty)
        self.self_harm_wall_penalty = float(self_harm_wall_penalty)
        self.min_wall_margin_delta = int(min_wall_margin_delta)
        self.reserve_walls = int(reserve_walls)
        self.emergency_p2_dist = int(emergency_p2_dist)
        self.mobility_penalty = bool(mobility_penalty)

    def _evaluate_own_wall_action(self, action, prev_p1_dist=None, prev_p2_dist=None):
        parts = self._wall_action_to_parts(action)
        if parts is None:
            return None
        r, c, orientation = parts
        if not self.engine.can_place_wall(1, r, c, orientation):
            return None

        prev_p1_dist = self.engine.get_bfs_distance(self.engine.p1_pos, 0) if prev_p1_dist is None else prev_p1_dist
        prev_p2_dist = self.engine.get_bfs_distance(self.engine.p2_pos, 8) if prev_p2_dist is None else prev_p2_dist

        wall_array = self.engine.horizontal_walls if orientation == "H" else self.engine.vertical_walls
        wall_array[r, c] = True
        try:
            new_p1_dist = self.engine.get_bfs_distance(self.engine.p1_pos, 0)
            new_p2_dist = self.engine.get_bfs_distance(self.engine.p2_pos, 8)
        finally:
            wall_array[r, c] = False

        if new_p1_dist == 999 or new_p2_dist == 999:
            return None

        own_delta = new_p1_dist - prev_p1_dist
        opponent_delta = new_p2_dist - prev_p2_dist
        margin_delta = opponent_delta - own_delta
        return {
            "r": r,
            "c": c,
            "orientation": orientation,
            "own_delta": own_delta,
            "opponent_delta": opponent_delta,
            "margin_delta": margin_delta,
        }

    def action_masks(self):
        mask = super().action_masks()
        if self.move_only or self.engine.walls_left[1] <= 0:
            return mask

        prev_p1_dist = self.engine.get_bfs_distance(self.engine.p1_pos, 0)
        prev_p2_dist = self.engine.get_bfs_distance(self.engine.p2_pos, 8)
        low_wall_budget = self.engine.walls_left[1] <= self.reserve_walls
        emergency = prev_p2_dist <= self.emergency_p2_dist

        for action in range(H_WALL_OFFSET, len(mask)):
            if not mask[action]:
                continue
            evaluated = self._evaluate_own_wall_action(action, prev_p1_dist, prev_p2_dist)
            if evaluated is None:
                mask[action] = False
                continue

            opponent_delta = evaluated["opponent_delta"]
            own_delta = evaluated["own_delta"]
            margin_delta = evaluated["margin_delta"]

            # Do not let the policy even see walls that do not slow the opponent.
            if opponent_delta <= 0:
                mask[action] = False
                continue

            # Save the last walls unless the opponent is already dangerous.
            if low_wall_budget and not emergency:
                mask[action] = False
                continue

            # Avoid self-blocking walls unless they clearly hurt the opponent more.
            if own_delta > 0 and margin_delta < self.min_wall_margin_delta:
                mask[action] = False
                continue

            # Outside emergencies, wall must improve the race margin.
            if not emergency and margin_delta < self.min_wall_margin_delta:
                mask[action] = False
                continue

        # Always keep legal moves available. If every wall got filtered out, that is fine.
        return mask

    def _mobility_penalty_value(self):
        if not self.mobility_penalty:
            return 0.0
        valid_moves = len(self.engine.get_valid_moves(1))
        if valid_moves <= 1:
            return -0.45
        if valid_moves == 2:
            return -0.12
        return 0.0

    def _strategic_wall_reward(self, prev_p1_dist, prev_p2_dist):
        new_p1_dist = self.engine.get_bfs_distance(self.engine.p1_pos, 0)
        new_p2_dist = self.engine.get_bfs_distance(self.engine.p2_pos, 8)
        own_delta = new_p1_dist - prev_p1_dist
        opponent_delta = new_p2_dist - prev_p2_dist
        margin_delta = opponent_delta - own_delta

        reward = -self.wall_cost
        reward += max(0, opponent_delta) * 0.70
        reward += margin_delta * 0.25
        reward -= max(0, own_delta) * 0.45

        if opponent_delta <= 0:
            reward -= self.useless_wall_penalty
        if own_delta > 0 and opponent_delta <= own_delta:
            reward -= self.self_harm_wall_penalty

        # Last walls are valuable. Spending them must be justified by real tempo.
        if self.engine.walls_left[1] < self.reserve_walls:
            reward -= (self.reserve_walls - self.engine.walls_left[1]) * 0.12

        # Emergency walling is good only when it actually slows the opponent.
        if prev_p2_dist <= self.emergency_p2_dist:
            if opponent_delta > 0:
                reward += 0.25 + (self.emergency_p2_dist - prev_p2_dist) * 0.10
            else:
                reward -= 0.40

        return reward

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected 0..{self.action_space.n - 1}")

        self.current_step += 1
        p1_target_row = 0
        p2_target_row = 8
        prev_p1_pos = self.engine.p1_pos
        prev_p1_dist = self.engine.get_bfs_distance(prev_p1_pos, p1_target_row)
        prev_p2_dist = self.engine.get_bfs_distance(self.engine.p2_pos, p2_target_row)
        prev_margin = prev_p2_dist - prev_p1_dist
        valid_action = self._is_action_valid(action)

        reward = -0.03

        if not valid_action:
            reward -= 1.5
        elif action < MOVE_ACTIONS:
            dx, dy = MOVES[action]
            cx, cy = self.engine.p1_pos
            new_pos = (cx + dx, cy + dy)
            self._move_player_to(1, new_pos)
            new_p1_dist = self.engine.get_bfs_distance(new_pos, p1_target_row)
            new_margin = prev_p2_dist - new_p1_dist
            progress = prev_p1_dist - new_p1_dist

            # Race advantage matters more than raw closeness to finish.
            reward += progress * 0.18
            reward += (new_margin - prev_margin) * 0.24

            if progress < 0:
                reward -= 0.25
            if abs(dx) + abs(dy) >= 2:
                reward += 0.06

            reward += self._repeat_penalty(new_pos) * 1.35
            reward += self._self_trap_penalty_value(new_p1_dist) * 1.35
            reward += self._mobility_penalty_value()

            if self.defensive_wall_reward and prev_p2_dist <= 3 and self.engine.walls_left[1] > 0 and new_p1_dist > 0:
                reward -= 0.60 + (3 - prev_p2_dist) * 0.22

            self.position_history.append(new_pos)
        elif action < V_WALL_OFFSET:
            idx = action - H_WALL_OFFSET
            if self.engine.place_wall(1, idx // 8, idx % 8, "H"):
                reward += self._strategic_wall_reward(prev_p1_dist, prev_p2_dist)
            else:
                reward -= 1.5
        else:
            idx = action - V_WALL_OFFSET
            if self.engine.place_wall(1, idx // 8, idx % 8, "V"):
                reward += self._strategic_wall_reward(prev_p1_dist, prev_p2_dist)
            else:
                reward -= 1.5

        terminated = False
        if self._is_p1_win():
            reward += 14.0
            terminated = True

        if not terminated:
            moved = self._opponent_step()
            if moved:
                new_p2_dist = self.engine.get_bfs_distance(self.engine.p2_pos, p2_target_row)
                opponent_progress = prev_p2_dist - new_p2_dist
                reward -= max(0, opponent_progress) * 0.18

                current_p1_dist = self.engine.get_bfs_distance(self.engine.p1_pos, p1_target_row)
                current_margin = new_p2_dist - current_p1_dist
                reward += (current_margin - prev_margin) * 0.08

                if self.defensive_wall_reward and new_p2_dist <= 2 and self.engine.walls_left[1] > 0:
                    reward -= 0.35

            if self._is_p2_win():
                reward -= 14.0
                terminated = True

        truncated = False
        if self.current_step >= self.max_steps and not terminated:
            truncated = True
            reward -= 7.0

        return self._get_obs(), reward, terminated, truncated, {}
