import re
import sys
import time
from pathlib import Path

import numpy as np
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from sb3_contrib import MaskablePPO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from envs.quoridor.quoridor_env import (
    H_WALL_OFFSET,
    MOVE_ACTIONS,
    MOVES,
    TOTAL_ACTIONS,
    V_WALL_OFFSET,
    QuoridorEnv,
)

PROFILE_DIR = ROOT_DIR / "browser_profile"
MODEL_PATH = ROOT_DIR / "models" / "best_model" / "best_model.zip"
WALLZ_URL = "https://wallz.gg/"

MOVE_DELTAS = {i: delta for i, delta in enumerate(MOVES)}
ACTION_BY_DELTA = {delta: action for action, delta in MOVE_DELTAS.items()}
MOVE_ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "JUMP_UP",
    5: "JUMP_DOWN",
    6: "JUMP_LEFT",
    7: "JUMP_RIGHT",
    8: "UP_LEFT",
    9: "UP_RIGHT",
    10: "DOWN_LEFT",
    11: "DOWN_RIGHT",
}
RGB_RE = re.compile(r"rgba?\((\d+),\s*(\d+),\s*(\d+)")
CYCLE_GUARD = True
DEBUG_WALLS = True
USE_SYNTHETIC_MOVES = False
ALLOW_WALL_ACTIONS = True


def action_name(action: int) -> str:
    if action < MOVE_ACTIONS:
        return MOVE_ACTION_NAMES.get(action, f"MOVE_{action}")
    if H_WALL_OFFSET <= action < V_WALL_OFFSET:
        idx = action - H_WALL_OFFSET
        return f"WALL_H_{idx // 8}_{idx % 8}"
    if V_WALL_OFFSET <= action < TOTAL_ACTIONS:
        idx = action - V_WALL_OFFSET
        return f"WALL_V_{idx // 8}_{idx % 8}"
    return f"ACTION_{action}"


def load_maskable_model(model_path: Path, env: QuoridorEnv):
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
    text = f"{item.get('fill', '')} {item.get('stroke', '')} {item.get('className', '')}".lower()
    if any(token in text for token in ("pink", "rose", "red", "crimson", "salmon")):
        return True
    for key in ("fill", "stroke"):
        rgb = _rgb(item.get(key, ""))
        if rgb is None:
            continue
        red, green, blue = rgb
        if red >= green + 15 and red >= blue - 10:
            return True
    return False


def _pawn_color_key(item: dict) -> str:
    if _looks_teal(item):
        return "teal"
    if _looks_red(item):
        return "red"
    return "unknown"


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


def _clean_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


class BrowserAgent:
    def __init__(self):
        # Browser play uses the same smart observation / wall-capable env as Stage 5.
        self.local_env = QuoridorEnv(wall_candidate_limit=40)
        self.obs, _ = self.local_env.reset()
        self.position_history = []
        self.own_color_key = None
        self.last_own_screen_xy = None
        self.wall_debug = ""
        self.last_ui_walls_left = 10

        if MODEL_PATH.exists():
            try:
                self.model = load_maskable_model(MODEL_PATH, self.local_env)
                print(f"[System] Модель загружена: {MODEL_PATH}")
            except Exception as exc:
                print(f"[System] Модель несовместима с текущим env: {type(exc).__name__}")
                print("[System] Сначала дообучи актуальную smart-модель через python scripts/train.py --stage 5")
                self.model = MaskablePPO("MlpPolicy", self.local_env, device="cpu")
        else:
            self.model = MaskablePPO("MlpPolicy", self.local_env, device="cpu")
            print("[System] Модель не найдена, используются случайные веса.")

    def run(self):
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(PROFILE_DIR),
                headless=False,
                slow_mo=120,
            )
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(WALLZ_URL, wait_until="domcontentloaded")

            input("\n[Управление] Начни матч и нажми ENTER, когда увидишь доску...")

            board = page.locator(".w-full.h-full").first
            try:
                board.wait_for(state="visible", timeout=40_000)
            except PlaywrightTimeoutError:
                print("[Ошибка] Доска не найдена. Проверь, что матч запущен.")
                try:
                    context.close()
                except Exception:
                    pass
                return

            try:
                self._play_loop(page, board)
            finally:
                try:
                    context.close()
                except Exception:
                    pass

    def _read_screen_state(self, board):
        return board.evaluate(
            """
            (el) => {
                const svg = el.tagName.toLowerCase() === 'svg' ? el : el.querySelector('svg');
                const source = svg || el;

                function numAttr(node, name) {
                    const value = node.getAttribute(name);
                    if (value === null) return null;
                    const parsed = Number(value);
                    return Number.isFinite(parsed) ? parsed : null;
                }

                function readNode(node) {
                    const rect = node.getBoundingClientRect();
                    const style = window.getComputedStyle(node);
                    const opacity = Number(style.opacity || 1);
                    return {
                        tag: node.tagName.toLowerCase(),
                        x: rect.x + rect.width / 2,
                        y: rect.y + rect.height / 2,
                        w: rect.width,
                        h: rect.height,
                        r: Math.max(rect.width, rect.height) / 2,
                        rawX: numAttr(node, 'x'),
                        rawY: numAttr(node, 'y'),
                        rawW: numAttr(node, 'width'),
                        rawH: numAttr(node, 'height'),
                        rawCx: numAttr(node, 'cx'),
                        rawCy: numAttr(node, 'cy'),
                        rawR: numAttr(node, 'r'),
                        fill: style.fill || node.getAttribute('fill') || '',
                        attrFill: node.getAttribute('fill') || '',
                        stroke: style.stroke || node.getAttribute('stroke') || '',
                        opacity: Number.isFinite(opacity) ? opacity : 1,
                        className: node.getAttribute('class') || '',
                    };
                }

                const all = Array.from(source.querySelectorAll('*'))
                    .map(readNode)
                    .filter((item) => (
                        (Math.max(item.w, item.h) > 3 || Math.max(item.rawW || 0, item.rawH || 0, item.rawR || 0) > 3) &&
                        Number.isFinite(item.x) && Number.isFinite(item.y) &&
                        item.opacity > 0.03 && item.tag !== 'svg'
                    ));

                const circles = all.filter((item) => item.tag === 'circle' && item.r > 0);
                const shapes = all.filter((item) => item.tag !== 'circle');
                return { circles, shapes, rects: shapes };
            }
            """
        )

    def _sync_walls_left_from_page(self, page):
        """Read the user's remaining wall count from the Wallz sidebar.

        The board SVG tells us which walls exist, but not how many of them are ours.
        Without this, the local env keeps walls_left=10 forever and still offers
        wall actions after the website shows WALLS · 0.
        """
        try:
            value = page.evaluate(
                """
                () => {
                    const text = document.body.innerText || '';
                    const lower = text.toLowerCase();
                    const youIndex = lower.indexOf('you');
                    if (youIndex >= 0) {
                        const chunk = text.slice(youIndex, youIndex + 350);
                        const match = chunk.match(/WALLS\s*[·:.\-]?\s*(\d+)/i);
                        if (match) return Number(match[1]);
                    }
                    const matches = [...text.matchAll(/WALLS\s*[·:.\-]?\s*(\d+)/gi)].map((m) => Number(m[1]));
                    if (matches.length) return matches[matches.length - 1];
                    return null;
                }
                """
            )
        except Exception:
            return

        if isinstance(value, (int, float)) and np.isfinite(value):
            value = int(max(0, min(10, value)))
            self.last_ui_walls_left = value
            self.local_env.engine.walls_left[1] = value

    def _cell_centers(self, state):
        square_rects = []
        for rect in state["shapes"]:
            w, h = rect["w"], rect["h"]
            raw_w = rect.get("rawW")
            raw_h = rect.get("rawH")
            ratio = w / h if h else 0
            raw_cell = raw_w == 60 and raw_h == 60
            screen_cell = 0.65 <= ratio <= 1.55 and 25 <= w <= 140 and 25 <= h <= 140
            if raw_cell or screen_cell:
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

    def _cell_point(self, pos, centers):
        xs, ys = centers
        x, y = pos
        return {"x": xs[x], "y": ys[y], "r": 0.0, "synthetic": True, "kind": "move"}

    def _wall_index(self, item, centers):
        xs, ys = centers
        xb = _boundaries(xs)
        yb = _boundaries(ys)
        c = _nearest_index(item["x"], xb)
        r = _nearest_index(item["y"], yb)
        return max(0, min(7, r)), max(0, min(7, c))

    def _svg_wall_from_item(self, item):
        raw_x = _clean_float(item.get("rawX"))
        raw_y = _clean_float(item.get("rawY"))
        raw_w = _clean_float(item.get("rawW"))
        raw_h = _clean_float(item.get("rawH"))
        if raw_x is None or raw_y is None or raw_w is None or raw_h is None:
            return None

        is_horizontal = abs(raw_w - 132) <= 18 and abs(raw_h - 12) <= 8
        is_vertical = abs(raw_w - 12) <= 8 and abs(raw_h - 132) <= 18
        if not is_horizontal and not is_vertical:
            return None

        fill_text = f"{item.get('attrFill', '')} {item.get('fill', '')}".lower()
        if "color-p" not in fill_text and "color-wall" not in fill_text:
            return None

        if is_horizontal:
            c = round(raw_x / 72)
            r = round(((raw_y + raw_h / 2) - 66) / 72)
            return "H", max(0, min(7, r)), max(0, min(7, c))

        c = round(((raw_x + raw_w / 2) - 66) / 72)
        r = round(raw_y / 72)
        return "V", max(0, min(7, r)), max(0, min(7, c))

    def _is_wall_bar(self, item, centers):
        if item.get("tag") == "circle":
            return None
        xs, ys = centers
        if len(xs) < 2 or len(ys) < 2:
            return None

        gap = min(_median_gap(xs), _median_gap(ys))
        w, h = item["w"], item["h"]
        long_side = max(w, h)
        short_side = min(w, h)
        ratio = long_side / max(1.0, short_side)
        squareish = 0.65 <= (w / h if h else 0) <= 1.55
        if squareish and w >= gap * 0.45 and h >= gap * 0.45:
            return None
        if ratio < 1.7 or short_side > gap * 0.72 or long_side < gap * 0.38 or long_side > gap * 3.5:
            return None

        color_text = f"{item.get('fill', '')} {item.get('stroke', '')} {item.get('className', '')}".lower()
        if "transparent" in color_text or "none none" in color_text:
            return None
        return "H" if w >= h else "V"

    def _sync_walls_from_screen(self, state, centers):
        engine = self.local_env.engine
        engine.horizontal_walls[:, :] = False
        engine.vertical_walls[:, :] = False

        horizontal = set()
        vertical = set()
        for item in state["shapes"]:
            parsed = self._svg_wall_from_item(item)
            if parsed is not None:
                orientation, r, c = parsed
            else:
                orientation = self._is_wall_bar(item, centers)
                if orientation is None:
                    continue
                r, c = self._wall_index(item, centers)

            if orientation == "H":
                horizontal.add((r, c))
            else:
                vertical.add((r, c))

        for r, c in horizontal:
            engine.horizontal_walls[r, c] = True
        for r, c in vertical:
            engine.vertical_walls[r, c] = True

        self.wall_debug = f"H={sorted(horizontal)} V={sorted(vertical)}" if DEBUG_WALLS else ""
        return len(horizontal), len(vertical)

    def _pick_pawns(self, state):
        circles = state["circles"]
        if not circles:
            return None, None

        max_radius = max(circle["r"] for circle in circles)
        pawns = [circle for circle in circles if circle["r"] >= max(10.0, max_radius * 0.65)]
        if not pawns:
            pawns = sorted(circles, key=lambda circle: circle["r"], reverse=True)[:2]

        def distance_to_last(circle):
            if self.last_own_screen_xy is None:
                return 0.0
            lx, ly = self.last_own_screen_xy
            return abs(circle["x"] - lx) + abs(circle["y"] - ly)

        if self.own_color_key is None:
            own = max(pawns, key=lambda circle: (circle["y"], circle["r"]))
            self.own_color_key = _pawn_color_key(own)
            print(f"[Зрение] Привязал свою фишку: color={self.own_color_key}")
        else:
            same_color = [circle for circle in pawns if _pawn_color_key(circle) == self.own_color_key]
            if same_color:
                own = min(same_color, key=distance_to_last)
            else:
                own = min(pawns, key=distance_to_last) if self.last_own_screen_xy else max(pawns, key=lambda circle: (circle["y"], circle["r"]))

        self.last_own_screen_xy = (own["x"], own["y"])
        opponent_candidates = [circle for circle in pawns if circle is not own]
        opponent = max(opponent_candidates, key=lambda circle: distance_to_last(circle)) if opponent_candidates else None
        return own, opponent

    def _reset_identity(self):
        self.position_history.clear()
        self.own_color_key = None
        self.last_own_screen_xy = None
        self.local_env.engine.walls_left[1] = 10
        self.last_ui_walls_left = 10

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
            if circle is own or circle["r"] >= own_radius * 0.75:
                continue
            dot_pos = self._grid_pos(circle, centers)
            delta = (dot_pos[0] - own_pos[0], dot_pos[1] - own_pos[1])
            action = ACTION_BY_DELTA.get(delta)
            if action is not None:
                circle["kind"] = "move"
                options[action] = circle

        if USE_SYNTHETIC_MOVES and options:
            valid_moves = self.local_env.engine.get_valid_moves(1)
            for action, (dx, dy) in MOVE_DELTAS.items():
                target_pos = (own_pos[0] + dx, own_pos[1] + dy)
                if target_pos in valid_moves and action not in options:
                    options[action] = self._cell_point(target_pos, centers)
        return options

    def _wall_click_point(self, action, centers):
        xs, ys = centers
        if len(xs) < 9 or len(ys) < 9:
            return None

        if H_WALL_OFFSET <= action < V_WALL_OFFSET:
            idx = action - H_WALL_OFFSET
            r, c = divmod(idx, 8)
            gap_x = xs[c + 1] - xs[c]
            x = xs[c] + gap_x * 0.25
            y = (ys[r] + ys[r + 1]) / 2.0
            return {"x": x, "y": y, "r": 0.0, "synthetic": False, "kind": "wall", "orientation": "H", "wall_rc": (r, c)}

        if V_WALL_OFFSET <= action < TOTAL_ACTIONS:
            idx = action - V_WALL_OFFSET
            r, c = divmod(idx, 8)
            gap_y = ys[r + 1] - ys[r]
            x = (xs[c] + xs[c + 1]) / 2.0
            y = ys[r] + gap_y * 0.25
            return {"x": x, "y": y, "r": 0.0, "synthetic": False, "kind": "wall", "orientation": "V", "wall_rc": (r, c)}
        return None

    def _wall_action_options(self, centers):
        if not ALLOW_WALL_ACTIONS or self.local_env.engine.walls_left[1] <= 0:
            return {}
        masks = self.local_env.action_masks()
        options = {}
        for action in range(MOVE_ACTIONS, TOTAL_ACTIONS):
            if not masks[action]:
                continue
            target = self._wall_click_point(action, centers)
            if target is not None:
                options[action] = target
        return options

    def _screen_action_options(self, own, state, centers):
        options = self._screen_move_options(own, state, centers)
        options.update(self._wall_action_options(centers))
        return options

    def _target_pos_for_action(self, p1_pos, action):
        dx, dy = MOVE_DELTAS[action]
        return p1_pos[0] + dx, p1_pos[1] + dy

    def _choose_action(self, predicted_action, action_options, p1_pos):
        if predicted_action >= MOVE_ACTIONS:
            return predicted_action, False
        if not CYCLE_GUARD or predicted_action not in action_options:
            return predicted_action, False

        predicted_pos = self._target_pos_for_action(p1_pos, predicted_action)
        current_dist = self.local_env.engine.get_bfs_distance(p1_pos, 0)
        predicted_dist = self.local_env.engine.get_bfs_distance(predicted_pos, 0)
        if predicted_dist < current_dist:
            return predicted_action, False

        recent = set(self.position_history[-4:])
        immediate_backtrack = len(self.position_history) >= 2 and predicted_pos == self.position_history[-2]
        short_loop = predicted_pos in recent
        if not immediate_backtrack and not short_loop:
            return predicted_action, False

        move_actions = [action for action in action_options if action < MOVE_ACTIONS]
        if not move_actions:
            return predicted_action, False

        def score(action):
            target_pos = self._target_pos_for_action(p1_pos, action)
            dist = self.local_env.engine.get_bfs_distance(target_pos, 0)
            repeat = 2.5 if target_pos in recent else 0.0
            side = abs(target_pos[0] - 4) * 0.04
            return dist + repeat + side

        best_action = min(move_actions, key=score)
        return best_action, best_action != predicted_action

    def _walls_text(self, wall_counts):
        debug = f" | {self.wall_debug}" if DEBUG_WALLS and self.wall_debug else ""
        return f"walls H/V={wall_counts} left={self.local_env.engine.walls_left[1]}{debug}"

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
                    print(f"[Пропуск] Недоступное действие модели: {action} {action_name(action)} | options={sorted(action_options)}")
                    time.sleep(0.5)
                    continue

                source = "synthetic" if target.get("synthetic") else target.get("kind", "screen")
                guard = f" | guard {action_name(predicted_action)}->{action_name(action)}" if overridden else ""
                if action < MOVE_ACTIONS:
                    target_pos = self._target_pos_for_action(p1_pos, action)
                    print(
                        f"[Действие] P1={p1_pos} P2={p2_pos} | {walls_text} | "
                        f"ход {action} {action_name(action)} -> {target_pos} | {source} | "
                        f"клик ({target['x']:.1f}, {target['y']:.1f}){guard}"
                    )
                else:
                    print(
                        f"[Стена] P1={p1_pos} P2={p2_pos} | {walls_text} | "
                        f"действие {action} {action_name(action)} | {source} | "
                        f"клик ({target['x']:.1f}, {target['y']:.1f})"
                    )

                page.mouse.move(target["x"], target["y"])
                page.mouse.click(target["x"], target["y"])
                if action < MOVE_ACTIONS:
                    self.position_history.append(self._target_pos_for_action(p1_pos, action))
                    self.position_history = self.position_history[-12:]
                time.sleep(1.2)
            except KeyboardInterrupt:
                print("\n[System] Остановлено пользователем.")
                break
            except Exception as e:
                print(f"[Поиск] {e}")
                time.sleep(1)


if __name__ == "__main__":
    BrowserAgent().run()
