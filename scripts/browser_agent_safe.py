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

_original_svg_wall_from_item = browser_agent_module.BrowserAgent._svg_wall_from_item
_original_sync_walls_from_screen = browser_agent_module.BrowserAgent._sync_walls_from_screen
WALL_FAILURE_LIMIT = 3


def resolve_model_path(value: str) -> Path:
    if value in MODEL_ALIASES:
        return MODEL_ALIASES[value]

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def safe_svg_wall_from_item(self, item):
    parsed = _original_svg_wall_from_item(self, item)
    if parsed is None:
        return None

    orientation, r, c = parsed
    if orientation == "H":
        r = 7 - r
    return orientation, r, c


def safe_wall_click_point(self, action, centers):
    """Return the visual drop point for a wall action.

    Wallz exposes a wall tray with draggable H/V wall pieces. For drag-and-drop
    the orientation comes from the tray source, so the target should be the slot
    center rather than a biased click point.
    """
    xs, ys = centers
    if len(xs) < 9 or len(ys) < 9:
        return None

    if browser_agent_module.H_WALL_OFFSET <= action < browser_agent_module.V_WALL_OFFSET:
        idx = action - browser_agent_module.H_WALL_OFFSET
        r, c = divmod(idx, 8)
        x = (xs[c] + xs[c + 1]) / 2.0
        y = (ys[r] + ys[r + 1]) / 2.0
        return {"x": x, "y": y, "r": 0.0, "synthetic": False, "kind": "wall-drag", "orientation": "H", "wall_rc": (r, c)}

    if browser_agent_module.V_WALL_OFFSET <= action < browser_agent_module.TOTAL_ACTIONS:
        idx = action - browser_agent_module.V_WALL_OFFSET
        r, c = divmod(idx, 8)
        x = (xs[c] + xs[c + 1]) / 2.0
        y = (ys[r] + ys[r + 1]) / 2.0
        return {"x": x, "y": y, "r": 0.0, "synthetic": False, "kind": "wall-drag", "orientation": "V", "wall_rc": (r, c)}

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
        self.wall_debug = f"ignored impossible parse H/V={counts}; kept H={sorted(prev_horizontal)} V={sorted(prev_vertical)}"
        print(f"[Стены] Подозрительный SVG parse: H/V={counts} > 20, оставил прошлое состояние")
        return len(prev_horizontal), len(prev_vertical)

    return counts


def safe_verify_wall_click(self, board, action, old_wall_total):
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
    expected_present = (r, c) in (self.screen_horizontal if orientation == "H" else self.screen_vertical)

    if expected_present:
        self.failed_wall_actions.clear()
        self.local_env.engine.walls_left[1] = max(0, self.local_env.engine.walls_left[1] - 1)
        return

    self.failed_wall_actions.add(action)
    expected = f"{orientation}({r}, {c})"
    actual = f"H={sorted(self.screen_horizontal)} V={sorted(self.screen_vertical)}"
    if new_wall_total > old_wall_total:
        print(f"[Стена] Сайт поставил не ожидаемый слот для {browser_agent_module.action_name(action)}; ждали {expected}, увидели {actual}. Убрал действие из mask.")
    else:
        print(f"[Стена] Сайт не принял {browser_agent_module.action_name(action)} — ждали {expected}, увидели {actual}. Убрал действие из mask.")

    if len(self.failed_wall_actions) >= WALL_FAILURE_LIMIT:
        browser_agent_module.ALLOW_WALL_ACTIONS = False
        self.failed_wall_actions.clear()
        print(f"[Стена] {WALL_FAILURE_LIMIT} ошибки стен подряд. Wall-actions отключены до перезапуска агента.")


def _tray_locator(page, orientation):
    if orientation == "H":
        return page.locator("button[aria-label*='horizontal wall']").first
    return page.locator("button[aria-label*='vertical wall']").first


def drag_wall_from_tray(page, orientation, target):
    locator = _tray_locator(page, orientation)
    box = locator.bounding_box(timeout=1500)
    if box is None:
        raise RuntimeError(f"Wall tray button not found for orientation={orientation}")

    start_x = box["x"] + box["width"] / 2.0
    start_y = box["y"] + box["height"] / 2.0
    end_x = target["x"]
    end_y = target["y"]

    page.mouse.move(start_x, start_y)
    page.mouse.down()
    page.mouse.move(end_x, end_y, steps=18)
    time.sleep(0.08)
    page.mouse.up()


def safe_play_loop(self, page, board):
    while True:
        try:
            state = self._read_screen_state(board)
            centers = self._cell_centers(state)
            if len(centers[0]) < 2 or len(centers[1]) < 2:
                print("[Поиск] Не смог определить сетку доски")
                time.sleep(0.5)
                continue

            own, opponent = self._pick_pawns(state)
            if own is None:
                print("[Поиск] Не нашёл фишку на доске")
                time.sleep(0.5)
                continue

            p1_pos, p2_pos, wall_counts = self._sync_env_from_screen(own, opponent, state, centers)
            self._sync_walls_left_from_page(page)
            walls_text = self._walls_text(wall_counts)
            if p1_pos[1] == 0 or p2_pos[1] == 8:
                print(f"[Ожидание] Матч, похоже, завершён | P1={p1_pos} P2={p2_pos} | {walls_text}")
                self._reset_identity()
                time.sleep(1.0)
                continue

            action_options = self._screen_action_options(own, state, centers)
            if not action_options:
                print(f"[Ожидание] Жду свой ход | P1={p1_pos} P2={p2_pos} | {walls_text}")
                time.sleep(0.7)
                continue

            masks = np.zeros(self.local_env.action_space.n, dtype=bool)
            for action in action_options:
                masks[action] = True

            predicted_action, _ = self.model.predict(self.obs, action_masks=masks, deterministic=True)
            predicted_action = int(predicted_action)
            action, overridden = self._choose_action(predicted_action, action_options, p1_pos)

            target = action_options.get(action)
            if target is None:
                print(f"[Пропуск] Недоступное действие модели: {action} {browser_agent_module.action_name(action)} | options={sorted(action_options)}")
                time.sleep(0.5)
                continue

            source = "synthetic" if target.get("synthetic") else target.get("kind", "screen")
            guard = f" | guard {browser_agent_module.action_name(predicted_action)}->{browser_agent_module.action_name(action)}" if overridden else ""

            old_wall_total = wall_counts[0] + wall_counts[1]
            if action < browser_agent_module.MOVE_ACTIONS:
                target_pos = self._target_pos_for_action(p1_pos, action)
                print(
                    f"[Действие] P1={p1_pos} P2={p2_pos} | {walls_text} | "
                    f"ход {action} {browser_agent_module.action_name(action)} -> {target_pos} | {source} | "
                    f"клик ({target['x']:.1f}, {target['y']:.1f}){guard}"
                )
                page.mouse.move(target["x"], target["y"])
                page.mouse.click(target["x"], target["y"])
                self.position_history.append(target_pos)
                self.position_history = self.position_history[-12:]
                time.sleep(1.2)
            else:
                parts = browser_agent_module._wall_action_parts(action)
                if parts is None:
                    self.failed_wall_actions.add(action)
                    time.sleep(0.5)
                    continue
                r, c, orientation = parts
                print(
                    f"[Стена] P1={p1_pos} P2={p2_pos} | {walls_text} | "
                    f"действие {action} {browser_agent_module.action_name(action)} | drag-{orientation} | "
                    f"drop ({target['x']:.1f}, {target['y']:.1f})"
                )
                drag_wall_from_tray(page, orientation, target)
                self._verify_wall_click(board, action, old_wall_total)
                self._sync_walls_left_from_page(page)
                time.sleep(0.6)
        except KeyboardInterrupt:
            print("\n[System] Остановлено пользователем.")
            break
        except Exception as e:
            print(f"[Поиск] {e}")
            time.sleep(1)


def install_safe_wall_patches():
    browser_agent_module.BrowserAgent._svg_wall_from_item = safe_svg_wall_from_item
    browser_agent_module.BrowserAgent._wall_click_point = safe_wall_click_point
    browser_agent_module.BrowserAgent._verify_wall_click = safe_verify_wall_click
    browser_agent_module.BrowserAgent._sync_walls_from_screen = safe_sync_walls_from_screen
    browser_agent_module.BrowserAgent._play_loop = safe_play_loop


def parse_args():
    parser = argparse.ArgumentParser(description="Run Wallz browser agent with tray drag wall actions and exact verification.")
    parser.add_argument(
        "--model",
        default="best",
        help="Model alias or path. Aliases: best=models/best_model/best_model.zip, empty=models/empty_model/best_model.zip. Relative paths are resolved from repo root.",
    )
    parser.add_argument(
        "--no-walls",
        action="store_true",
        help="Disable wall actions from the start. Useful for isolating movement quality.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model)
    browser_agent_module.MODEL_PATH = model_path
    browser_agent_module.ALLOW_WALL_ACTIONS = not args.no_walls
    install_safe_wall_patches()
    print(f"[System] Выбрана модель: {model_path}")
    print("[System] Safe wall mode: tray drag H/V + exact slot verification + wall-failure fallback")
    if args.no_walls:
        print("[System] Wall-actions отключены с запуска (--no-walls)")
    browser_agent_module.BrowserAgent().run()


if __name__ == "__main__":
    main()
