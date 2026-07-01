import argparse
import time
from pathlib import Path

import numpy as np

import browser_agent as browser_agent_module


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_ALIASES = {
    "best": ROOT_DIR / "models" / "best_model" / "best_model.zip",
    "empty": ROOT_DIR / "models" / "empty_model" / "best_model.zip",
}

_original_sync_walls_from_screen = browser_agent_module.BrowserAgent._sync_walls_from_screen


def resolve_model_path(value: str) -> Path:
    if value in MODEL_ALIASES:
        return MODEL_ALIASES[value]

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def safe_wall_click_point(self, action, centers):
    """Click away from the ambiguous H/V intersection.

    The original browser_agent clicks the exact center shared by horizontal and
    vertical wall slots. Wallz may then choose the wrong orientation. This keeps
    the same slot math but biases the click along the requested wall direction:
    horizontal walls are clicked slightly left/right of the crossing, vertical
    walls slightly up/down of it.
    """
    xs, ys = centers
    if len(xs) < 9 or len(ys) < 9:
        return None

    gap = min(browser_agent_module._median_gap(xs), browser_agent_module._median_gap(ys))
    bias = max(10.0, min(18.0, gap * 0.22))

    if browser_agent_module.H_WALL_OFFSET <= action < browser_agent_module.V_WALL_OFFSET:
        idx = action - browser_agent_module.H_WALL_OFFSET
        r, c = divmod(idx, 8)
        x = (xs[c] + xs[c + 1]) / 2.0
        y = (ys[r] + ys[r + 1]) / 2.0
        if c < 7:
            x += bias
        else:
            x -= bias
        return {
            "x": x,
            "y": y,
            "r": 0.0,
            "synthetic": False,
            "kind": "wall",
            "orientation": "H",
            "wall_rc": (r, c),
            "safe_bias": bias,
        }

    if browser_agent_module.V_WALL_OFFSET <= action < browser_agent_module.TOTAL_ACTIONS:
        idx = action - browser_agent_module.V_WALL_OFFSET
        r, c = divmod(idx, 8)
        x = (xs[c] + xs[c + 1]) / 2.0
        y = (ys[r] + ys[r + 1]) / 2.0
        if r < 7:
            y += bias
        else:
            y -= bias
        return {
            "x": x,
            "y": y,
            "r": 0.0,
            "synthetic": False,
            "kind": "wall",
            "orientation": "V",
            "wall_rc": (r, c),
            "safe_bias": bias,
        }

    return None


def safe_sync_walls_from_screen(self, state, centers):
    prev_horizontal = set(getattr(self, "screen_horizontal", set()))
    prev_vertical = set(getattr(self, "screen_vertical", set()))
    prev_h_array = self.local_env.engine.horizontal_walls.copy()
    prev_v_array = self.local_env.engine.vertical_walls.copy()

    counts = _original_sync_walls_from_screen(self, state, centers)
    total = counts[0] + counts[1]
    if total > 20:
        self.screen_horizontal = prev_horizontal
        self.screen_vertical = prev_vertical
        self.local_env.engine.horizontal_walls[:, :] = prev_h_array
        self.local_env.engine.vertical_walls[:, :] = prev_v_array
        self.wall_debug = (
            f"ignored impossible parse H/V={counts}; "
            f"kept H={sorted(prev_horizontal)} V={sorted(prev_vertical)}"
        )
        print(f"[Стены] Подозрительный SVG parse: H/V={counts} > 20, оставил прошлое состояние")
        return len(prev_horizontal), len(prev_vertical)

    return counts


def safe_verify_wall_click(self, board, action, old_wall_total):
    """Accept a wall click only if the exact expected slot appears on screen."""
    time.sleep(0.9)
    try:
        new_state = self._read_screen_state(board)
        new_centers = self._cell_centers(new_state)
        if len(new_centers[0]) < 2 or len(new_centers[1]) < 2:
            return
        new_counts = self._sync_walls_from_screen(new_state, new_centers)
        new_wall_total = new_counts[0] + new_counts[1]
    except Exception:
        return

    parts = browser_agent_module._wall_action_parts(action)
    if parts is None:
        return

    r, c, orientation = parts
    if orientation == "H":
        expected_present = (r, c) in self.screen_horizontal
    else:
        expected_present = (r, c) in self.screen_vertical

    if expected_present:
        self.failed_wall_actions.clear()
        self.local_env.engine.walls_left[1] = max(0, self.local_env.engine.walls_left[1] - 1)
        return

    self.failed_wall_actions.add(action)
    expected = f"{orientation}({r}, {c})"
    actual = f"H={sorted(self.screen_horizontal)} V={sorted(self.screen_vertical)}"
    if new_wall_total > old_wall_total:
        print(
            f"[Стена] Сайт поставил не ожидаемый слот для {browser_agent_module.action_name(action)}; "
            f"ждали {expected}, увидели {actual}. Убрал действие из mask."
        )
    else:
        print(
            f"[Стена] Сайт не принял {browser_agent_module.action_name(action)} — "
            f"ждали {expected}, увидели {actual}. Убрал действие из mask."
        )


def install_safe_wall_patches():
    browser_agent_module.BrowserAgent._wall_click_point = safe_wall_click_point
    browser_agent_module.BrowserAgent._verify_wall_click = safe_verify_wall_click
    browser_agent_module.BrowserAgent._sync_walls_from_screen = safe_sync_walls_from_screen


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Wallz browser agent with safer wall clicks and exact wall verification."
    )
    parser.add_argument(
        "--model",
        default="best",
        help=(
            "Model alias or path. Aliases: best=models/best_model/best_model.zip, "
            "empty=models/empty_model/best_model.zip. Relative paths are resolved from repo root."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model)
    browser_agent_module.MODEL_PATH = model_path
    install_safe_wall_patches()
    print(f"[System] Выбрана модель: {model_path}")
    print("[System] Safe wall mode: offset H/V clicks + exact expected-slot verification")
    browser_agent_module.BrowserAgent().run()


if __name__ == "__main__":
    main()
