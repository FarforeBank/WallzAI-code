import time
import numpy as np
from playwright.sync_api import sync_playwright
from sb3_contrib import MaskablePPO
from envs.quoridor.engine import QuoridorEngine
from envs.quoridor.quoridor_env import QuoridorEnv

class WallzBrowserBot:
    def __init__(self, model_path: str = None):
        print("[System] Инициализация бота...")
        self.model = MaskablePPO.load(model_path, device="cpu") if model_path else None
        self.engine = QuoridorEngine()
        self.env = QuoridorEnv(engine=self.engine)
        self.board_start_x, self.board_start_y, self.scale = 0, 0, 1.0 

    def run(self):
        with sync_playwright() as p:
            browser = p.chromium.launch_persistent_context(
                user_data_dir="./browser_profile",
                headless=False,
                args=["--disable-blink-features=AutomationControlled"]
            )
            self.page = browser.pages[0]
            self.page.goto("https://www.wallz.gg/")
            print("[Управление] Зайди в игру. Нажми ENTER, когда увидишь доску...")
            input()
            self._calibrate_board()
            self._play_loop()

    def _calibrate_board(self):
        print("[Калибровка] Ждем загрузки доски...")
        # Увеличиваем таймаут ожидания до 10 секунд
        board_bg = self.page.locator('rect[fill="var(--color-board)"]')
        
        try:
            # Ждем, пока элемент реально появится в DOM
            board_bg.wait_for(state="attached", timeout=10000)
            
            if board_bg.count() > 0:
                box = board_bg.first.bounding_box()
                self.scale = box['width'] / 648.0
                self.board_start_x = box['x'] + (6 * self.scale)
                self.board_start_y = box['y'] + (6 * self.scale)
                print(f"[Калибровка] Успешно! Масштаб: {self.scale:.2f}")
            else:
                print("[Ошибка] Доска не найдена даже после ожидания.")
        except Exception as e:
            print(f"[Ошибка] Не удалось дождаться доски: {e}")

    def _parse_walls(self):
        """Парсит ВСЕ стены на доске и добавляет в движок."""
        self.engine.walls_h.fill(False)
        self.engine.walls_v.fill(False)
        walls = self.page.locator('rect[fill="var(--color-wall)"]')
        for i in range(walls.count()):
            box = walls.nth(i).bounding_box()
            if not box: continue
            wx = int(round(((box['x'] - self.board_start_x) / self.scale) / 72))
            wy = int(round(((box['y'] - self.board_start_y) / self.scale) / 72))
            if box['width'] > box['height']: self.engine.walls_h[wy, wx] = True
            else: self.engine.walls_v[wy, wx] = True

    def _sync_state_from_browser(self):
        """Синхронизация + проверка 'Чей ход'."""
        try:
            # Проверка хода: ищем текст "YOUR TURN"
            if self.page.locator('text="YOUR TURN"').count() == 0:
                return False 
            
            p1 = self.page.locator('circle[fill="var(--color-p1)"]').first
            p2 = self.page.locator('circle[fill="var(--color-p2)"]').first
            if p1.count() > 0 and p2.count() > 0:
                self.engine.p1_pos = np.array([int(round((float(p1.get_attribute("cy")) - 30) / 72)), int(round((float(p1.get_attribute("cx")) - 30) / 72))])
                self.engine.p2_pos = np.array([int(round((float(p2.get_attribute("cy")) - 30) / 72)), int(round((float(p2.get_attribute("cx")) - 30) / 72))])
            
            self._parse_walls()
            p1_w = self.page.locator('div[aria-label*="Wall tray"] span.tabular-nums').first
            if p1_w.count() > 0 and p1_w.inner_text().isdigit():
                self.engine.p1_walls = int(p1_w.inner_text())
            return True
        except: return False

    def _action_to_click(self, action: int):
        if action < 8:
            target_y, target_x = self.env._get_movement_destinations(0).get(action, self.engine.p1_pos)
            self.page.mouse.click(self.board_start_x + ((target_x * 72) + 30) * self.scale, 
                                  self.board_start_y + ((target_y * 72) + 30) * self.scale)
        else:
            wall_action = action - 8
            orientation = 'H' if wall_action < 64 else 'V'
            wy, wx = (wall_action % 64) // 8, (wall_action % 64) % 8
            btn = self.page.get_by_role("button", name=("Drag a horizontal wall" if orientation == 'H' else "Drag a vertical wall")).first
            box = btn.bounding_box()
            if box:
                self.page.mouse.move(box['x'] + box['width']/2, box['y'] + box['height']/2)
                self.page.mouse.down()
                time.sleep(0.2)
                self.page.mouse.move(self.board_start_x + ((wx * 72) + (72 if orientation == 'H' else 66)) * self.scale,
                                     self.board_start_y + ((wy * 72) + (66 if orientation == 'H' else 72)) * self.scale, steps=10)
                self.page.mouse.up()

    def _play_loop(self):
        while True:
            if not self._sync_state_from_browser():
                time.sleep(1)
                continue
            mask = self.env._get_info()["action_mask"]
            action = int(self.model.predict(self.env._get_obs(), action_masks=mask, deterministic=True)[0]) if self.model else int(np.random.choice(np.where(mask == 1)[0]))
            self._action_to_click(action)
            self.env.step(action)
            time.sleep(2.5)

if __name__ == "__main__":
    bot = WallzBrowserBot(model_path=None) 
    bot.run()