from wallz_ai.agents.heuristic_bot import GreedyShortestPathBot, WallHeuristicBot
from wallz_ai.agents.random_bot import RandomLegalBot
from wallz_ai.env.action_space import ACTION_SIZE, square_to_action
from wallz_ai.env.rules import is_legal_action
from wallz_ai.env.wallz_env import WallzEnv


def test_legal_mask_has_fixed_shape_and_no_terminal_actions_after_win():
    env = WallzEnv()
    env.state.pawn_positions = [(1, 4), (8, 4)]
    env.state.current_player = 0
    mask = env.legal_action_mask()
    assert mask.shape == (ACTION_SIZE,)
    assert mask[square_to_action((0, 4))]
    env.step(square_to_action((0, 4)))
    assert env.state.terminal
    assert env.legal_action_mask().sum() == 0


def test_baseline_bots_only_choose_legal_actions():
    env = WallzEnv()
    env.reset(seed=5)
    for bot in [RandomLegalBot(1), GreedyShortestPathBot(), WallHeuristicBot()]:
        action = bot.act(env.state.clone())
        assert is_legal_action(env.state, action)


def test_random_legal_games_never_crash():
    bot = RandomLegalBot(123)
    for _ in range(3):
        env = WallzEnv(max_moves=30)
        env.reset()
        while not env.state.terminal:
            action = bot.act(env.state.clone())
            assert is_legal_action(env.state, action)
            env.step(action)
