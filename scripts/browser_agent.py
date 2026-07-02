import argparse
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
WALLZ_URL = "https://wallz.gg/"
MODEL_ALIASES = {
    "best": ROOT_DIR / "models" / "best_model" / "best_model.zip",
    "stage8": ROOT_DIR / "models" / "best_model_stage8" / "best_model.zip",
    "empty": ROOT_DIR / "models" / "empty_model" / "best_model.zip",
}

MOVE_DELTAS = {i: delta for i, delta in enumerate(MOVES)}
ACTION_BY_DELTA = {delta: action for action, delta in MOVE_DELTAS.items()}
MOVE_NAMES = {
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


def resolve_model_path(value: str) -> Path:
    if value in MODEL_ALIASES:
        return MODEL_ALIASES[value]
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT_DIR / path


def action_name(action: int) -> str:
    if action < MOVE_ACTIONS:
        return MOVE_NAMES.get(action, f"MOVE_{action}")
    if H_WALL_OFFSET <= action < V_WALL_OFFSET:
        idx = action - H_WALL_OFFSET
        return f"WALL_H_{idx // 8}_{idx % 8}"
    if V_WALL_OFFSET <= action < TOTAL_ACTIONS:
        idx = action - V_WALL_OFFSET
        return f"WALL_V_{idx // 8}_{idx % 8}"
    return f"ACTION_{action}"


def wall_action_parts(action: int):
    if H_WALL_OFFSET <= action < V_WALL_OFFSET:
        idx = action - H_WALL_OFFSET
        return idx // 8, idx % 8, "H"
    if V_WALL_OFFSET <= action < TOTAL_ACTIONS:
        idx = action - V_WALL_OFFSET
        return idx // 8, idx % 8, "V"
    return None


def load_model(path: Path, env: QuoridorEnv):
    return MaskablePPO.load(
        str(path),
        env=env,
        device="cpu",
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        },
    )


def rgb(value: str):
    match = RGB_RE.search(value or "")
    if not match:
        return None
    return tuple(int(x) for x in match.groups())


def color_key(item: dict) -> str:
    text = f"{item.get('fill', '')} {item.get('stroke', '')} {item.get('className', '')}".lower()
    if any(token in text for token in ("cyan", "teal", "turquoise", "emerald")):
        return "teal"
    if any(token in text for token in ("pink", "rose", "red", "crimson", "salmon")):
        return "red"
    for key in ("fill", "stroke"):
        value = rgb(item.get(key, ""))
        if value is None:
            continue
        r, g, b = value
        if g >= r + 15 and b >= r + 15:
            return "teal"
        if r >= g + 15 and r >= b - 10:
            return "red"
    return "unknown"


def cluster_axis(values, expected=9):
    values = sorted(v for v in values if np.isfinite(v))
    if not values:
        return []
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
            clusters.append(current)
            current = [value]
    clusters.append(current)
    centers = [sum(group) / len(group) for group in clusters]
    if len(centers) > expected:
        centers = sorted(centers, key=lambda c: sum(abs(v - c) for v in values))[:expected]
    return sorted(centers)


def nearest_index(value, centers):
    return min(range(len(centers)), key=lambda i: abs(value - centers[i]))


def clamp_slot(value):
    return max(0, min(7, int(round(value))))


def wall_boundaries(centers):
    xs, ys = centers
    if len(xs) < 9 or len(ys) < 9:
        return [], []
    bx = [(xs[i] + xs[i + 1]) / 2.0 for i in range(8)]
    by = [(ys[i] + ys[i + 1]) / 2.0 for i in range(8)]
    return bx, by


class BrowserAgent:
    def __init__(self, model_path: Path, allow_walls: bool, wall_fail_limit: int):
        self.env = QuoridorEnv(wall_candidate_limit=40)
        self.obs, _ = self.env.reset()
        self.model_path = model_path
        self.allow_walls = allow_walls
        self.wall_fail_limit = max(0, int(wall_fail_limit))
        self.wall_failures = 0
        self.failed_wall_actions = set()
        self.own_color = None
        self.last_own_xy = None
        self.last_centers = None
        self.position_history = []
        self.screen_horizontal = set()
        self.screen_vertical = set()
        self.walls_left = 10

        if model_path.exists():
            try:
                self.model = load_model(model_path, self.env)
                print(f"[System] Model loaded: {model_path}")
            except Exception as exc:
                print(f"[System] Model load failed ({type(exc).__name__}); using random policy")
                self.model = MaskablePPO("MlpPolicy", self.env, device="cpu")
        else:
            print(f"[System] Model not found: {model_path}; using random policy")
            self.model = MaskablePPO("MlpPolicy", self.env, device="cpu")

    def run(self, url: str):
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(PROFILE_DIR),
                headless=False,
                slow_mo=90,
            )
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(url, wait_until="domcontentloaded")
            input("\n[Control] Start a match and press ENTER when the board is visible...")

            try:
                page.locator('svg[aria-label="Wallz board"]').first.wait_for(state="visible", timeout=40_000)
            except PlaywrightTimeoutError:
                print("[Error] Board SVG not found")
                context.close()
                return

            try:
                self.play_loop(page)
            finally:
                context.close()

    def read_board(self, page):
        return page.evaluate(
            """
            () => {
                const svg = document.querySelector('svg[aria-label="Wallz board"]');
                if (!svg) return null;

                function num(node, name) {
                    const value = node.getAttribute(name);
                    if (value === null) return null;
                    const parsed = Number(value);
                    return Number.isFinite(parsed) ? parsed : null;
                }

                function read(node) {
                    const rect = node.getBoundingClientRect();
                    const style = window.getComputedStyle(node);
                    return {
                        tag: node.tagName.toLowerCase(),
                        x: rect.x + rect.width / 2,
                        y: rect.y + rect.height / 2,
                        w: rect.width,
                        h: rect.height,
                        r: Math.max(rect.width, rect.height) / 2,
                        rawX: num(node, 'x'),
                        rawY: num(node, 'y'),
                        rawW: num(node, 'width'),
                        rawH: num(node, 'height'),
                        rawCx: num(node, 'cx'),
                        rawCy: num(node, 'cy'),
                        rawR: num(node, 'r'),
                        fill: style.fill || node.getAttribute('fill') || '',
                        attrFill: node.getAttribute('fill') || '',
                        stroke: style.stroke || node.getAttribute('stroke') || '',
                        opacity: Number(style.opacity || 1),
                        className: node.getAttribute('class') || '',
                    };
                }

                const items = Array.from(svg.querySelectorAll('rect,circle'))
                    .map(read)
                    .filter((item) => Number.isFinite(item.x) && Number.isFinite(item.y) && item.opacity > 0.03);
                return {
                    circles: items.filter((item) => item.tag === 'circle'),
                    rects: items.filter((item) => item.tag === 'rect'),
                };
            }
            """
        )

    def cell_centers(self, state):
        cells = []
        for item in state["rects"]:
            raw_cell = item.get("rawW") == 60 and item.get("rawH") == 60
            ratio = item["w"] / item["h"] if item["h"] else 0
            screen_cell = 0.65 <= ratio <= 1.55 and 25 <= item["w"] <= 140 and 25 <= item["h"] <= 140
            if raw_cell or screen_cell:
                cells.append(item)
        xs = cluster_axis([item["x"] for item in cells], expected=9)
        ys = cluster_axis([item["y"] for item in cells], expected=9)
        return xs[:9], ys[:9]

    def grid_pos(self, item, centers):
        xs, ys = centers
        return nearest_index(item["x"], xs), nearest_index(item["y"], ys)

    def wall_slot_from_screen(self, item, centers):
        bx, by = wall_boundaries(centers)
        if len(bx) < 8 or len(by) < 8:
            return None
        return nearest_index(item["y"], by), nearest_index(item["x"], bx)

    def wall_point_from_slot(self, centers, r, c):
        bx, by = wall_boundaries(centers)
        if len(bx) < 8 or len(by) < 8:
            return None
        return {"x": float(bx[c]), "y": float(by[r])}

    def parse_walls(self, state, centers=None):
        horizontal = set()
        vertical = set()
        for item in state["rects"]:
            raw_x = item.get("rawX")
            raw_y = item.get("rawY")
            raw_w = item.get("rawW")
            raw_h = item.get("rawH")

            raw_h_wall = raw_w is not None and raw_h is not None and abs(raw_w - 132) <= 18 and abs(raw_h - 12) <= 8
            raw_v_wall = raw_w is not None and raw_h is not None and abs(raw_w - 12) <= 8 and abs(raw_h - 132) <= 18
            ratio = item["w"] / item["h"] if item["h"] else 0
            screen_h_wall = ratio >= 3.0 and item["w"] >= 30 and item["h"] <= 24
            screen_v_wall = ratio <= 0.33 and item["h"] >= 30 and item["w"] <= 24
            is_h = raw_h_wall or screen_h_wall
            is_v = raw_v_wall or screen_v_wall
            if not is_h and not is_v:
                continue

            text = f"{item.get('attrFill', '')} {item.get('fill', '')}".lower()
            # color-wall is the tray/hover preview. Count only committed player-colored walls.
            if "color-p" not in text:
                continue

            slot = self.wall_slot_from_screen(item, centers) if centers is not None else None
            if slot is None:
                if raw_x is None or raw_y is None or raw_w is None or raw_h is None:
                    continue
                if is_h:
                    slot = (clamp_slot(((raw_y + raw_h / 2) - 66) / 72), clamp_slot(raw_x / 72))
                else:
                    slot = (clamp_slot(raw_y / 72), clamp_slot(((raw_x + raw_w / 2) - 66) / 72))

            r, c = slot
            if is_h:
                horizontal.add((r, c))
            else:
                vertical.add((r, c))
        return horizontal, vertical

    def pick_pawns(self, state):
        circles = state["circles"]
        if not circles:
            return None, None
        max_radius = max(item["r"] for item in circles)
        pawns = []
        for item in circles:
            fill_text = f"{item.get('fill', '')} {item.get('stroke', '')}".lower()
            if item["r"] >= max(10.0, max_radius * 0.65) and "none" not in fill_text:
                pawns.append(item)
        if len(pawns) < 2:
            pawns = sorted(circles, key=lambda item: item["r"], reverse=True)[:2]
        if not pawns:
            return None, None

        def dist_last(item):
            if self.last_own_xy is None:
                return 0.0
            return abs(item["x"] - self.last_own_xy[0]) + abs(item["y"] - self.last_own_xy[1])

        if self.own_color is None:
            own = max(pawns, key=lambda item: (item["y"], item["r"]))
            self.own_color = color_key(own)
            print(f"[Vision] Bound own pawn color={self.own_color}")
        else:
            same_color = [item for item in pawns if color_key(item) == self.own_color]
            own = min(same_color, key=dist_last) if same_color else min(pawns, key=dist_last)

        self.last_own_xy = (own["x"], own["y"])
        opponents = [item for item in pawns if item is not own]
        opponent = max(opponents, key=dist_last) if opponents else None
        return own, opponent

    def read_walls_left(self, page):
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
                    return matches.length ? matches[matches.length - 1] : null;
                }
                """
            )
        except Exception:
            return
        if isinstance(value, (int, float)) and np.isfinite(value):
            self.walls_left = int(max(0, min(10, value)))
            self.env.engine.walls_left[1] = self.walls_left

    def sync_env(self, page, own, opponent, state, centers):
        self.last_centers = centers
        p1_pos = self.grid_pos(own, centers)
        p2_pos = self.grid_pos(opponent, centers) if opponent else self.env.engine.p2_pos
        horizontal, vertical = self.parse_walls(state, centers)
        self.screen_horizontal = horizontal
        self.screen_vertical = vertical

        engine = self.env.engine
        engine.board[:, :] = 0
        engine.horizontal_walls[:, :] = False
        engine.vertical_walls[:, :] = False
        for r, c in horizontal:
            engine.horizontal_walls[r, c] = True
        for r, c in vertical:
            engine.vertical_walls[r, c] = True
        engine.p1_pos = p1_pos
        engine.p2_pos = p2_pos
        engine.board[p1_pos[1], p1_pos[0]] = 1
        engine.board[p2_pos[1], p2_pos[0]] = 2
        self.read_walls_left(page)
        self.obs = self.env._get_obs()
        return p1_pos, p2_pos

    def move_options(self, own, state, centers):
        own_pos = self.grid_pos(own, centers)
        own_radius = own["r"]
        options = {}
        for circle in state["circles"]:
            if circle is own or circle["r"] >= own_radius * 0.75:
                continue
            target_pos = self.grid_pos(circle, centers)
            delta = (target_pos[0] - own_pos[0], target_pos[1] - own_pos[1])
            action = ACTION_BY_DELTA.get(delta)
            if action is not None:
                options[action] = {"kind": "move", "x": circle["x"], "y": circle["y"], "target_pos": target_pos}
        return options

    def wall_conflicts_screen(self, action):
        parts = wall_action_parts(action)
        if parts is None:
            return True
        r, c, orientation = parts
        if orientation == "H":
            return (
                (r, c) in self.screen_horizontal
                or (r, c - 1) in self.screen_horizontal
                or (r, c + 1) in self.screen_horizontal
                or (r, c) in self.screen_vertical
            )
        return (
            (r, c) in self.screen_vertical
            or (r - 1, c) in self.screen_vertical
            or (r + 1, c) in self.screen_vertical
            or (r, c) in self.screen_horizontal
        )

    def wall_options(self, centers):
        if not self.allow_walls or self.walls_left <= 0:
            return {}
        masks = self.env.action_masks()
        options = {}
        for action in range(MOVE_ACTIONS, TOTAL_ACTIONS):
            if action in self.failed_wall_actions or not masks[action] or self.wall_conflicts_screen(action):
                continue
            r, c, orientation = wall_action_parts(action)
            point = self.wall_point_from_slot(centers, r, c)
            if point is None:
                continue
            options[action] = {
                "kind": "wall",
                "x": float(point["x"]),
                "y": float(point["y"]),
                "orientation": orientation,
                "wall_rc": (r, c),
            }
        return options

    def choose_action(self, predicted, options, p1_pos):
        if predicted in options:
            return predicted, False
        move_actions = [a for a in options if a < MOVE_ACTIONS]
        if move_actions:
            def score(action):
                target = options[action].get("target_pos", p1_pos)
                dist = self.env.engine.get_bfs_distance(target, 0)
                repeat = 2.0 if target in self.position_history[-4:] else 0.0
                return dist + repeat + abs(target[0] - 4) * 0.05
            return min(move_actions, key=score), True
        if options:
            return next(iter(options)), True
        return None, False

    def tray_button(self, page, orientation):
        if orientation == "H":
            return page.locator("button[aria-label*='horizontal wall']").first
        return page.locator("button[aria-label*='vertical wall']").first

    def clear_wall_preview(self, page):
        try:
            page.mouse.move(8, 8, steps=4)
            time.sleep(0.25)
        except Exception:
            pass

    def drag_wall(self, page, action, target):
        locator = self.tray_button(page, target["orientation"])
        box = locator.bounding_box(timeout=1500)
        if box is None:
            raise RuntimeError(f"wall tray button not found for {target['orientation']}")
        sx = box["x"] + box["width"] / 2.0
        sy = box["y"] + box["height"] / 2.0
        ex = target["x"]
        ey = target["y"]
        print(f"[Wall] {action_name(action)} env={target['wall_rc']} drop=({ex:.1f},{ey:.1f})")
        page.bring_to_front()
        page.evaluate("() => window.focus()")

        board = page.locator('svg[aria-label="Wallz board"]').first
        board_box = board.bounding_box(timeout=1500)
        if board_box is not None:
            try:
                locator.drag_to(
                    board,
                    source_position={"x": box["width"] / 2.0, "y": box["height"] / 2.0},
                    target_position={"x": ex - board_box["x"], "y": ey - board_box["y"]},
                    timeout=3500,
                    force=True,
                )
                time.sleep(0.45)
                self.clear_wall_preview(page)
                return
            except Exception as exc:
                print(f"[WallDragFallback] locator.drag_to failed: {type(exc).__name__}: {exc}")

        page.mouse.move(sx, sy)
        time.sleep(0.18)
        page.mouse.down(button="left")
        time.sleep(0.35)
        page.mouse.move(sx, sy - 8, steps=4)
        page.mouse.move((sx + ex) / 2.0, (sy + ey) / 2.0, steps=24)
        page.mouse.move(ex, ey, steps=32)
        page.mouse.move(ex + 2.0, ey + 2.0, steps=4)
        page.mouse.move(ex, ey, steps=4)
        time.sleep(0.25)
        page.mouse.up(button="left")
        time.sleep(0.45)
        self.clear_wall_preview(page)

    def verify_wall(self, page, action, old_walls, old_left):
        parts = wall_action_parts(action)
        if parts is None:
            return False
        r, c, orientation = parts
        wall_set_name = "H" if orientation == "H" else "V"

        for _ in range(6):
            self.clear_wall_preview(page)
            state = self.read_board(page)
            if state is None:
                time.sleep(0.2)
                continue
            horizontal, vertical = self.parse_walls(state, self.last_centers)
            self.screen_horizontal = horizontal
            self.screen_vertical = vertical
            expected = (r, c) in (horizontal if orientation == "H" else vertical)
            if expected:
                self.wall_failures = 0
                self.failed_wall_actions.clear()
                self.walls_left = max(0, min(self.walls_left, old_left - 1))
                self.env.engine.walls_left[1] = self.walls_left
                print(f"[WallOK] committed {wall_set_name}({r},{c}), local walls={self.walls_left}")
                return True
            time.sleep(0.2)

        state = self.read_board(page)
        horizontal, vertical = self.parse_walls(state, self.last_centers) if state else (set(), set())
        self.screen_horizontal = horizontal
        self.screen_vertical = vertical
        self.failed_wall_actions.add(action)
        self.wall_failures += 1
        print(
            f"[WallFail] no committed {wall_set_name}({r},{c}); got H={sorted(horizontal)} V={sorted(vertical)}, "
            f"text-left was {old_left}->{self.walls_left}, failures={self.wall_failures}"
        )
        if self.wall_fail_limit and self.wall_failures >= self.wall_fail_limit:
            self.allow_walls = False
            print("[WallFail] wall actions disabled for this run")
        return False

    def reset_identity(self):
        self.own_color = None
        self.last_own_xy = None
        self.last_centers = None
        self.position_history.clear()
        self.failed_wall_actions.clear()
        self.wall_failures = 0
        self.env.engine.walls_left[1] = 10
        self.walls_left = 10

    def play_loop(self, page):
        while True:
            try:
                state = self.read_board(page)
                if state is None:
                    print("[Wait] board not found")
                    time.sleep(0.7)
                    continue
                centers = self.cell_centers(state)
                if len(centers[0]) < 9 or len(centers[1]) < 9:
                    print("[Wait] grid not detected")
                    time.sleep(0.7)
                    continue
                own, opponent = self.pick_pawns(state)
                if own is None:
                    print("[Wait] pawns not detected")
                    time.sleep(0.7)
                    continue

                p1_pos, p2_pos = self.sync_env(page, own, opponent, state, centers)
                if p1_pos[1] == 0 or p2_pos[1] == 8:
                    print(f"[Wait] match ended P1={p1_pos} P2={p2_pos}")
                    self.reset_identity()
                    time.sleep(1.0)
                    continue

                options = self.move_options(own, state, centers)
                if self.allow_walls:
                    options.update(self.wall_options(centers))
                if not options:
                    print(f"[Wait] not our turn P1={p1_pos} P2={p2_pos} walls={self.walls_left}")
                    time.sleep(0.7)
                    continue

                masks = np.zeros(self.env.action_space.n, dtype=bool)
                for action in options:
                    masks[action] = True
                predicted, _ = self.model.predict(self.obs, action_masks=masks, deterministic=True)
                predicted = int(predicted)
                action, overridden = self.choose_action(predicted, options, p1_pos)
                if action is None:
                    time.sleep(0.5)
                    continue

                target = options[action]
                guard = f" guard {action_name(predicted)}->{action_name(action)}" if overridden else ""
                old_walls = (len(self.screen_horizontal), len(self.screen_vertical))
                old_left = self.walls_left

                if action < MOVE_ACTIONS:
                    target_pos = target["target_pos"]
                    print(f"[Move] P1={p1_pos} P2={p2_pos} {action_name(action)}->{target_pos}{guard}")
                    page.mouse.click(target["x"], target["y"])
                    self.position_history.append(target_pos)
                    self.position_history = self.position_history[-12:]
                    time.sleep(1.1)
                else:
                    self.drag_wall(page, action, target)
                    self.verify_wall(page, action, old_walls, old_left)
                    time.sleep(0.4)
            except KeyboardInterrupt:
                print("\n[System] stopped")
                break
            except Exception as exc:
                print(f"[Error] {type(exc).__name__}: {exc}")
                time.sleep(1.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Stable Wallz browser agent")
    parser.add_argument("--model", default="stage8", help="Model alias or path: best, stage8, empty, or file path")
    parser.add_argument("--url", default=WALLZ_URL)
    parser.add_argument("--allow-walls", action="store_true", help="Enable experimental wall dragging")
    parser.add_argument("--wall-fail-limit", type=int, default=2, help="Disable walls after this many failed wall drops")
    return parser.parse_args()


def main():
    args = parse_args()
    allow_walls = bool(args.allow_walls)
    agent = BrowserAgent(
        model_path=resolve_model_path(args.model),
        allow_walls=allow_walls,
        wall_fail_limit=args.wall_fail_limit if allow_walls else 0,
    )
    mode = "experimental walls enabled" if allow_walls else "move-only stable mode"
    print(f"[System] Browser agent mode: {mode}")
    agent.run(args.url)


if __name__ == "__main__":
    main()
