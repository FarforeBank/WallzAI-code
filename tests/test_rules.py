import pytest

from wallz_ai.env.action_space import ACTION_SIZE, WallOrientation, square_to_action, wall_to_action
from wallz_ai.env.rules import WallzState, apply_action, can_place_wall, has_path, legal_action_mask, legal_pawn_targets, shortest_path_distance
from wallz_ai.env.wallz_env import WallzEnv


def test_initial_position_has_valid_legal_moves_and_action_size():
    env = WallzEnv()
    obs, _ = env.reset(seed=123)
    mask = env.legal_action_mask()
    assert obs.shape == (11, 9, 9)
    assert mask.shape == (ACTION_SIZE,)
    assert mask.sum() > 0
    assert mask[square_to_action((7, 4))]
    assert shortest_path_distance(env.state, 0) == 8
    assert shortest_path_distance(env.state, 1) == 8


def test_walls_cannot_overlap_or_cross():
    state = WallzState()
    state = apply_action(state, wall_to_action(3, 3, WallOrientation.HORIZONTAL))
    state.current_player = 0
    assert not can_place_wall(state, 0, 3, 3, WallOrientation.HORIZONTAL)
    assert not can_place_wall(state, 0, 3, 2, WallOrientation.HORIZONTAL)
    assert not can_place_wall(state, 0, 3, 4, WallOrientation.HORIZONTAL)
    assert not can_place_wall(state, 0, 3, 3, WallOrientation.VERTICAL)


def test_wall_placement_requires_remaining_walls():
    state = WallzState(walls_remaining=[0, 10])
    assert not can_place_wall(state, 0, 4, 4, WallOrientation.HORIZONTAL)


def test_walls_cannot_seal_off_player_path_after_candidate():
    state = WallzState()
    state.horizontal_walls[:, :] = True
    state.horizontal_walls[0, 0] = False
    assert not has_path(state, 0) or not has_path(state, 1)


def test_straight_jump_rule():
    state = WallzState(pawn_positions=[(4, 4), (3, 4)])
    targets = legal_pawn_targets(state, 0)
    assert (2, 4) in targets
    assert (3, 4) not in targets


def test_diagonal_jump_rule_when_jump_blocked_by_edge():
    state = WallzState(pawn_positions=[(1, 4), (0, 4)])
    targets = legal_pawn_targets(state, 0)
    assert (0, 3) in targets
    assert (0, 5) in targets


def test_diagonal_jump_rule_when_wall_blocks_straight_jump():
    state = WallzState(pawn_positions=[(4, 4), (3, 4)])
    state.horizontal_walls[2, 4] = True
    targets = legal_pawn_targets(state, 0)
    assert (2, 4) not in targets
    assert (3, 3) in targets
    assert (3, 5) in targets


def test_game_ends_when_pawn_reaches_goal_row():
    state = WallzState(pawn_positions=[(1, 4), (8, 4)], current_player=0)
    next_state = apply_action(state, square_to_action((0, 4)))
    assert next_state.terminal
    assert next_state.winner == 0


def test_env_rejects_illegal_action():
    env = WallzEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step(square_to_action((0, 0)))
