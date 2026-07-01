import os
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
                self._play_loop(board)
            finally:
                context.close()

    def _play_loop(self, board):
        while True:
            try:
                chips = board.locator("circle")
                if chips.count() < 2:
                    time.sleep(0.5)
                    continue

                cx_raw = chips.nth(0).get_attribute("cx")
                cy_raw = chips.nth(0).get_attribute("cy")
                if cx_raw is None or cy_raw is None:
                    time.sleep(0.5)
                    continue

                cx = float(cx_raw)
                cy = float(cy_raw)

                masks = self.local_env.action_masks()
                masks[4:] = False  # test mode: move only, no walls
                action, _ = self.model.predict(self.obs, action_masks=masks, deterministic=True)
                action = int(action)

                if action not in MOVE_DELTAS:
                    print(f"[Пропуск] Модель выбрала не-ход: {action}")
                    time.sleep(1)
                    continue

                svg_box = board.bounding_box()
                if svg_box is None:
                    time.sleep(0.5)
                    continue

                dx, dy = MOVE_DELTAS[action]
                board_units = 660.0
                cell_step = board_units / 9.0
                scale = svg_box["width"] / board_units
                target_x = (cx + dx * cell_step) * scale
                target_y = (cy + dy * cell_step) * scale

                print(f"[Действие] Ход {action} | Клик: ({target_x:.1f}, {target_y:.1f})")
                board.click(position={"x": target_x, "y": target_y}, force=True)

                self.obs, _, terminated, truncated, _ = self.local_env.step(action)
                if terminated or truncated:
                    self.obs, _ = self.local_env.reset()

                time.sleep(2)
            except KeyboardInterrupt:
                print("\n[System] Остановлено пользователем.")
                break
            except Exception as e:
                print(f"[Поиск] {e}")
                time.sleep(1)


if __name__ == "__main__":
    BrowserAgent().run()
