import os
import re
import sys
import time
from pathlib import Path

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
    """Load old checkpoints even if only Box bounds changed.

    Older checkpoints were saved with Box(-1, 2, (9, 9, 3), int8). The env shape
    stayed the same, but the bounds were corrected to Box(0, 10, ...). SB3 checks
    the full space object on load, so we override the saved metadata.
    """
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


def _looks_like_own_color(circle: dict) -> bool:
    text = f"{circle.get('fill', '')} {circle.get('className', '')}".lower()
    if any(token in text for token in ("cyan", "teal", "turquoise", "emerald")):
        return True

    rgb = _rgb(circle.get("fill", ""))
    if rgb is None:
        return False

    red, green, blue = rgb
    # The user's pawn/legal-move dots on Wallz are cyan/teal.
    return green >= red + 20 and blue >= red + 20


def _looks_like_opponent_color(circle: dict) -> bool:
    rgb = _rgb(circle.get("fill", ""))
    if rgb is None:
        return False
    red, green, blue = rgb
    return red >= green + 20 and red >= blue - 20


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

    def _read_svg_state(self, board):
        return board.evaluate(
            """
            (el) => {
                const svg = el.tagName.toLowerCase() === 'svg' ? el : el.querySelector('svg');
                const source = svg || el;
                const rect = source.getBoundingClientRect();
                const viewBox = svg && svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.width
                    ? {
                        x: svg.viewBox.baseVal.x,
                        y: svg.viewBox.baseVal.y,
                        width: svg.viewBox.baseVal.width,
                        height: svg.viewBox.baseVal.height,
                    }
                    : { x: 0, y: 0, width: 660, height: 660 };

                const circles = Array.from(source.querySelectorAll('circle')).map((circle) => {
                    const style = window.getComputedStyle(circle);
                    return {
                        cx: Number(circle.getAttribute('cx')),
                        cy: Number(circle.getAttribute('cy')),
                        r: Number(circle.getAttribute('r') || 0),
                        fill: style.fill || circle.getAttribute('fill') || '',
                        className: circle.getAttribute('class') || '',
                    };
                }).filter((circle) => Number.isFinite(circle.cx) && Number.isFinite(circle.cy));

                return {
                    viewBox,
                    rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
                    circles,
                };
            }
            """
        )

    def _pick_pawns(self, circles):
        if not circles:
            return None, None

        max_radius = max(circle.get("r", 0) for circle in circles)
        min_pawn_radius = max(8, max_radius * 0.65)
        pawns = [circle for circle in circles if circle.get("r", 0) >= min_pawn_radius]
        if not pawns:
            pawns = sorted(circles, key=lambda circle: circle.get("r", 0), reverse=True)[:2]

        own_candidates = [circle for circle in pawns if _looks_like_own_color(circle)]
        opponent_candidates = [circle for circle in pawns if _looks_like_opponent_color(circle)]

        if own_candidates:
            own = max(own_candidates, key=lambda circle: circle.get("r", 0))
        else:
            # Fallback: at game start our pawn is the lower large pawn.
            own = max(pawns, key=lambda circle: circle.get("cy", 0))

        opponent_pool = [circle for circle in opponent_candidates if circle is not own]
        if opponent_pool:
            opponent = max(opponent_pool, key=lambda circle: circle.get("r", 0))
        else:
            other_pawns = [circle for circle in pawns if circle is not own]
            opponent = min(other_pawns, key=lambda circle: circle.get("cy", 0)) if other_pawns else None

        return own, opponent

    def _grid_pos(self, circle, state):
        view_box = state["viewBox"]
        cell_w = view_box["width"] / 9.0
        cell_h = view_box["height"] / 9.0
        x = int((circle["cx"] - view_box["x"]) // cell_w)
        y = int((circle["cy"] - view_box["y"]) // cell_h)
        return max(0, min(8, x)), max(0, min(8, y))

    def _svg_to_page_point(self, circle, state):
        view_box = state["viewBox"]
        rect = state["rect"]
        x = rect["x"] + ((circle["cx"] - view_box["x"]) / view_box["width"]) * rect["width"]
        y = rect["y"] + ((circle["cy"] - view_box["y"]) / view_box["height"]) * rect["height"]
        return x, y

    def _sync_env_from_screen(self, own, opponent, state):
        p1_pos = self._grid_pos(own, state)
        p2_pos = self._grid_pos(opponent, state) if opponent else self.local_env.engine.p2_pos

        engine = self.local_env.engine
        engine.board[:, :] = 0
        engine.p1_pos = p1_pos
        engine.p2_pos = p2_pos
        engine.board[p1_pos[1], p1_pos[0]] = 1
        engine.board[p2_pos[1], p2_pos[0]] = 2
        self.obs = self.local_env._get_obs()
        return p1_pos, p2_pos

    def _screen_move_options(self, own, state):
        own_pos = self._grid_pos(own, state)
        own_radius = own.get("r", 0)
        options = {}

        for circle in state["circles"]:
            if circle is own:
                continue
            # Legal move dots are smaller cyan/teal circles near the pawn.
            if circle.get("r", 0) >= max(own_radius * 0.85, own_radius - 2):
                continue
            if not _looks_like_own_color(circle):
                continue

            dot_pos = self._grid_pos(circle, state)
            delta = (dot_pos[0] - own_pos[0], dot_pos[1] - own_pos[1])
            action = ACTION_BY_DELTA.get(delta)
            if action is not None:
                options[action] = circle

        return options

    def _play_loop(self, page, board):
        while True:
            try:
                state = self._read_svg_state(board)
                own, opponent = self._pick_pawns(state["circles"])
                if own is None:
                    print("[Поиск] Не нашёл фишку на доске")
                    time.sleep(0.5)
                    continue

                p1_pos, p2_pos = self._sync_env_from_screen(own, opponent, state)
                move_options = self._screen_move_options(own, state)

                if not move_options:
                    print(f"[Ожидание] Нет доступных точек хода | P1={p1_pos} P2={p2_pos}")
                    time.sleep(0.7)
                    continue

                masks = self.local_env.action_masks()
                masks[:] = False
                for action in move_options:
                    masks[action] = True
                masks[4:] = False  # browser test mode: move only, no walls

                action, _ = self.model.predict(self.obs, action_masks=masks, deterministic=True)
                action = int(action)

                target_circle = move_options.get(action)
                if target_circle is None:
                    # Should not happen because the mask only allows move_options.
                    print(f"[Пропуск] Недоступный ход модели: {action} | options={sorted(move_options)}")
                    time.sleep(0.5)
                    continue

                click_x, click_y = self._svg_to_page_point(target_circle, state)
                print(
                    f"[Действие] P1={p1_pos} P2={p2_pos} | ход {action} "
                    f"-> {self._grid_pos(target_circle, state)} | клик ({click_x:.1f}, {click_y:.1f})"
                )
                page.mouse.click(click_x, click_y)
                time.sleep(1.2)
            except KeyboardInterrupt:
                print("\n[System] Остановлено пользователем.")
                break
            except Exception as e:
                print(f"[Поиск] {e}")
                time.sleep(1)


if __name__ == "__main__":
    BrowserAgent().run()
