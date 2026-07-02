from wallz_ai.env.action_space import WallOrientation, wall_to_action
from wallz_ai.env.rules import WallzState, apply_action, has_path, shortest_path, shortest_path_distance


def test_shortest_path_initial_is_eight_steps():
    state = WallzState()
    assert shortest_path_distance(state, 0) == 8
    assert shortest_path_distance(state, 1) == 8
    assert shortest_path(state, 0)[0] == (8, 4)
    assert shortest_path(state, 0)[-1][0] == 0


def test_path_distance_increases_after_relevant_wall():
    state = WallzState()
    before = shortest_path_distance(state, 0)
    state = apply_action(state, wall_to_action(7, 4, WallOrientation.HORIZONTAL))
    assert has_path(state, 0)
    assert shortest_path_distance(state, 0) > before
