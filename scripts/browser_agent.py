import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from sb3_contrib import MaskablePPO

from envs.quoridor.quoridor_env import QuoridorEnv

PROFILE_DIR = ROOT_DIR / "browser_profile"
MODEL_PATH = ROOT_DIR / "models" / "best_model" / "best_model.zip"
WALLZ_URL = "https://wallz.gg/"

MOVE_DELTAS = {
    0: (0, -1),   # Up
    1: (0, 1),    # Down
    2: (-1, 0),   # Left
    3: (1, 0),    # Right
}
ACTION_BY_DELTA = {delta: action for action, delta in MOVE_DELTAS.items()}
RGB_RE = re.compile(r"rgba?\((\d+),\s*(\d+),\s*(\d+)")


def load_maskable_model(model_path: Path, env: QuoridorEnv):
    """Load old checkpoints even if only Box bounds changed."""
    return MaskablePPO.load(
        str(model_path),
        env=env,
        device="cpu",
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        },
    )


def _rgb(fill: str):
    match = RGB_RE.search(fill or "")
    if not match:
        return None
    return tuple(int(value) for value in match.groups())


def _looks_teal(item: dict) -> bool:
    text = f"{item.get('fill', '')} {item.get('stroke', '')} {item.get('className', '')}".lower()
    if any(token in text for token in ("cyan", "teal", "turquoise", "emerald")):
        return True

    for key in ("fill", "stroke"):
        rgb = _rgb(item.get(key, ""))
        if rgb is None:
            continue
        red, green, blue = rgb
        if green >= red + 15 and blue >= red + 15:
            return True
    return False


def _looks_red(item: dict) -> bool:
    for key in ("fill", "stroke"):
        rgb = _rgb(item.get(key, ""))
        if rgb is None:
            continue
        red, green, blue = rgb
        if red >= green + 15 and red >= blue - 10:
            return True
    return False


def _cluster_axis(values, expected=9):
    if not values:
        return []

    values = sorted(values)
    if len(values) <= expected:
        return values

    span = values[-1] - values[0]
    threshold = max(8.0, span / 80.0)
    clusters = []
    current = [values[0]]

    for value in values[1:]:
        if abs(value - current[-1]) <= threshold:
            current.append(value)
        else:
            clusters.append(sum(current) / len(current))
            current = [value]
    clusters.append(sum(current) / len(current))

    if len(clusters) > expected:
        best = None
        best_score = None
        for start in range(0, len(clusters) - expected + 1):
            candidate = clusters[start:start + expected]
            gaps = [candidate[i + 1] - candidate[i] for i in range(expected - 1)]
            avg_gap = sum(gaps) / len(gaps)
            score = sum(abs(gap - avg_gap) for gap in gaps)
            if best is None or score < best_score:
                best = candidate
                best_score = score
        clusters = best

    return clusters


def _nearest_index(value, centers):
    if not centers:
        return 0
    return min(range(len(centers)), key=lambda i: abs(value - centers[i]))


def _boundaries(centers):
    return [(centers[i] + centers[i + 1]) / 2.0 for i in range(len(centers) - 1)]


def _median_gap(centers, default=70.0):
    if len(centers) < 2:
        return default
    gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
    return float(np.median(gaps))


class BrowserAgent:
    def __init__(self):
        self.local_env = QuoridorEnv()
        self.obs, _ = self.local_env.reset()

        if MODEL_PATH.exists():
            self.model = load_maskable_model(MODEL_PATH, self.local_env)
            print(f"[System] Модель загружена: {MODEL_PATH}")
        else:
            self.model = MaskablePPO("MlpPolicy", self.local_env, device="cpu")
            print("[System] Модель не найдена, используются случайные веса.")

    def run(self):
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(PROFILE_DIR),
                headless=False,
                slow_mo=200,
            )
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(WALLZ_URL, wait_until="domcontentloaded")

            input("\n[Управление] Начни матч и нажми ENTER, когда увидишь доску...")

            board = page.locator(".w-full.h-full").first
            try:
                board.wait_for(state="visible", timeout=40_000)
            except PlaywrightTimeoutError:
                print("[Ошибка] Доска не найдена. Проверь, что матч запущен.")
                context.close()
                return

            try:
                self._play_loop(page, board)
            finally:
                context.close()

    def _read_screen_state(self, board):
        return board.evaluate(
            """
            (el) => {
                const svg = el.tagName.toLowerCase() === 'svg' ? el : el.querySelector('svg');
                const source = svg || el;

                const circles = Array.from(source.querySelectorAll('circle')).map((circle) => {
                    const rect = circle.getBoundingClientRect();
                    const style = window.getComputedStyle(circle);
                    return {
                        x: rect.x + rect.width / 2,
                        y: rect.y + rect.height / 2,
                        r: Math.max(rect.width, rect.height) / 2,
                        fill: style.fill || circle.getAttribute('fill') || '',
                        stroke: style.stroke || circle.getAttribute('stroke') || '',
                        className: circle.getAttribute('class') || '',
                    };
                }).filter((circle) => circle.r > 0 && Number.isFinite(circle.x) && Number.isFinite(circle.y));

                const rects = Array.from(source.querySelectorAll('rect')).map((rectEl) => {
                    const rect = rectEl.getBoundingClientRect();
                    const style = window.getComputedStyle(rectEl);
                    return {
                        x: rect.x + rect.width / 2,
                        y: rect.y + rect.height / 2,
                        w: rect.width,
                        h: rect.height,
                        fill: style.fill || rectEl.getAttribute('fill') || '',
                        stroke: style.stroke || rectEl.getAttribute('stroke') || '',
                        className: rectEl.getAttribute('class') || '',
                    };
                }).filter((rect) => rect.w > 4 && rect.h > 4 && Number.isFinite(rect.x) && Number.isFinite(rect.y));

                return { circles, rects };
            }
            """
        )

    def _cell_centers(self, state):
        square_rects = []
        for rect in state["rects"]:
            w, h = rect["w"], rect["h"]
            ratio = w / h if h else 0
            if 0.65 <= ratio <= 1.55 and w >= 25 and h >= 25:
                square_rects.append(rect)

        if len(square_rects) >= 20:
            xs = _cluster_axis([rect["x"] for rect in square_rects], expected=9)
            ys = _cluster_axis([rect["y"] for rect in square_rects], expected=9)
            if len(xs) >= 2 and len(ys) >= 2:
                return xs[:9], ys[:9]

        xs = _cluster_axis([circle["x"] for circle in state["circles"]], expected=9)
        ys = _cluster_axis([circle["y"] for circle in state["circles"]], expected=9)
        return xs[:9], ys[:9]

    def _grid_pos(self, item, centers):
        xs, ys = centers
        return _nearest_index(item["x"], xs), _nearest_index(item["y"], ys)

    def _wall_index(self, item, centers):
        xs, ys = centers
        xb = _boundaries(xs)
        yb = _boundaries(ys)
        c = _nearest_index(item["x"], xb)
        r = _nearest_index(item["y"], yb)
        return max(0, min(7, r)), max(0, min(7, c))

    def _is_wall_rect(self, rect, centers):
        xs, ys = centers
        gap = min(_median_gap(xs), _median_gap(ys))
        w, h = rect["w"], rect["h"]

        is_vertical = h >= gap * 1.25 and h >= w * 2.2 and w <= gap * 0.45
        is_horizontal = w >= gap * 1.25 and w >= h * 2.2 and h <= gap * 0.45
        if not (is_vertical or is_horizontal):
            return None

        # Walls can be teal/red/gray depending on owner and history. We mostly rely on geometry,
        # but ignore very dark background pieces by requiring a visible fill/stroke token or color.
        color_text = f"{rect.get('fill', '')} {rect.get('stroke', '')} {rect.get('className', '')}".lower()
        has_color = bool(color_text.strip()) and "none" not in color_text
        if not has_color:
            return None

        return "V" if is_vertical else "H"

    def _sync_walls_from_screen(self, state, centers):
        engine = self.local_env.engine
        engine.horizontal_walls[:, :] = False
        engine.vertical_walls[:, :] = False

        horizontal = 0
        vertical = 0
        for rect in state["rects"]:
            orientation = self._is_wall_rect(rect, centers)
            if orientation is None:
                continue

            r, c = self._wall_index(rect, centers)
            if orientation == "H":
                engine.horizontal_walls[r, c] = True
                horizontal += 1
            else:
                engine.vertical_walls[r, c] = True
                vertical += 1

        return horizontal, vertical

    def _pick_pawns(self, state):
        circles = state["circles"]
        if not circles:
            return None, None

        max_radius = max(circle["r"] for circle in circles)
        pawns = [circle for circle in circles if circle["r"] >= max(10.0, max_radius * 0.65)]
        if not pawns:
            pawns = sorted(circles, key=lambda circle: circle["r"], reverse=True)[:2]

        teal_pawns = [circle for circle in pawns if _looks_teal(circle)]
        red_pawns = [circle for circle in pawns if _looks_red(circle)]

        if teal_pawns:
            own = max(teal_pawns, key=lambda circle: (circle["r"], circle["y"]))
        else:
            own = max(pawns, key=lambda circle: circle["y"])

        opponent_pool = [circle for circle in red_pawns if circle is not own]
        if opponent_pool:
            opponent = max(opponent_pool, key=lambda circle: circle["r"])
        else:
            other_pawns = [circle for circle in pawns if circle is not own]
            opponent = min(other_pawns, key=lambda circle: circle["y"]) if other_pawns else None

        return own, opponent

    def _sync_env_from_screen(self, own, opponent, state, centers):
        p1_pos = self._grid_pos(own, centers)
        p2_pos = self._grid_pos(opponent, centers) if opponent else self.local_env.engine.p2_pos

        engine = self.local_env.engine
        wall_counts = self._sync_walls_from_screen(state, centers)
        engine.board[:, :] = 0
        engine.p1_pos = p1_pos
        engine.p2_pos = p2_pos
        engine.board[p1_pos[1], p1_pos[0]] = 1
        engine.board[p2_pos[1], p2_pos[0]] = 2
        self.obs = self.local_env._get_obs()
        return p1_pos, p2_pos, wall_counts

    def _screen_move_options(self, own, state, centers):
        own_pos = self._grid_pos(own, centers)
        own_radius = own["r"]
        options = {}

        for circle in state["circles"]:
            if circle is own:
                continue
            if circle["r"] >= own_radius * 0.75:
                continue
            if not _looks_teal(circle):
                continue

            dot_pos = self._grid_pos(circle, centers)
            delta = (dot_pos[0] - own_pos[0], dot_pos[1] - own_pos[1])
            action = ACTION_BY_DELTA.get(delta)
            if action is not None:
                options[action] = circle

        return options

    def _play_loop(self, page, board):
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
                move_options = self._screen_move_options(own, state, centers)

                if not move_options:
                    print(f"[Ожидание] Нет доступных точек хода | P1={p1_pos} P2={p2_pos} | walls H/V={wall_counts}")
                    time.sleep(0.7)
                    continue

                masks = self.local_env.action_masks()
                masks[:] = False
                for action in move_options:
                    masks[action] = True
                masks[4:] = False

                action, _ = self.model.predict(self.obs, action_masks=masks, deterministic=True)
                action = int(action)

                target = move_options.get(action)
                if target is None:
                    print(f"[Пропуск] Недоступный ход модели: {action} | options={sorted(move_options)}")
                    time.sleep(0.5)
                    continue

                target_pos = self._grid_pos(target, centers)
                print(
                    f"[Действие] P1={p1_pos} P2={p2_pos} | walls H/V={wall_counts} | "
                    f"ход {action} -> {target_pos} | клик ({target['x']:.1f}, {target['y']:.1f})"
                )
                page.mouse.click(target["x"], target["y"])
                time.sleep(1.2)
            except KeyboardInterrupt:
                print("\n[System] Остановлено пользователем.")
                break
            except Exception as e:
                print(f"[Поиск] {e}")
                time.sleep(1)


if __name__ == "__main__":
    BrowserAgent().run()
