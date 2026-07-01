import time
import os
import sys

# Обеспечиваем доступ к корню проекта для импорта envs
BASE_DIR = os.getcwd()
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from playwright.sync_api import sync_playwright, TimeoutError
from sb3_contrib import MaskablePPO
from envs.quoridor.quoridor_env import QuoridorEnv

# Пути к файлам
PROFILE_DIR = os.path.join(BASE_DIR, "browser_profile")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model", "best_model.zip")

class BrowserBot:
    def __init__(self):
        self.env = QuoridorEnv()
        self.obs, _ = self.env.reset()
        
        # Загрузка модели
        if os.path.exists(MODEL_PATH):
            self.model = MaskablePPO.load(MODEL_PATH)
            print(f"[System] Модель загружена: {MODEL_PATH}")
        else:
            print("[System] Модель не найдена, используем случайные веса.")
            self.model = MaskablePPO("MlpPolicy", self.env)

    def run(self):
        with sync_playwright() as p:
            print(f"[System] Используем профиль: {PROFILE_DIR}")
            context = p.chromium.launch_persistent_context(
                user_data_dir=PROFILE_DIR, 
                headless=False, 
                slow_mo=200
            )
            
            page = context.pages[0] if context.pages else context.new_page()
            print("[Browser] Открываем Wallz.gg...")
            page.goto("https://wallz.gg/")
            
            input("[Управление] Зайди в игру (Local/Arcade). Нажми ENTER, когда увидишь доску...")

            print("\n[Калибровка] Ждем загрузки доски...")
            board_svg = page.locator(".w-full.h-full").first
            board_svg.wait_for(state="visible", timeout=40000)
            
            # Калибровка масштаба
            svg_box = board_svg.bounding_box()
            print(f"[Калибровка] Успешно! Масштаб: {svg_box['width']/660:.2f}")

            print("[System] Бот перехватывает управление!")
            self._play_loop(page, board_svg)
            
            context.close()

    def _play_loop(self, page, board_svg):
        while True:
            try:
                # Парсинг фишек
                chips = board_svg.locator("circle")
                if chips.count() < 2:
                    time.sleep(1)
                    continue

                # Координаты (первая фишка - наша)
                cx = float(chips.nth(0).get_attribute("cx"))
                cy = float(chips.nth(0).get_attribute("cy"))
                print(f"[Зрение] P1: [{int(cx/71.5)}, {int(cy/71.5)}]")

                # Предсказание действия
                masks = self.env.action_masks()
                masks[4:] = 0 # Запрещаем стены для теста
                action, _ = self.model.predict(self.obs, action_masks=masks, deterministic=True)
                
                # Клик
                if action < 4:
                    dx, dy = [(0,-71.5), (0,71.5), (-71.5,0), (71.5,0)][int(action)]
                    svg_box = board_svg.bounding_box()
                    page.mouse.click(svg_box["x"] + cx + dx, svg_box["y"] + cy + dy)
                    print(f"[Действие] Шагаю на {action}")
                
                # Обновление среды
                self.obs, _, term, trunc, _ = self.env.step(action)
                if term or trunc:
                    self.obs, _ = self.env.reset()
                
                time.sleep(1)
            except Exception as e:
                print(f"[Ошибка] {e}")
                time.sleep(2)

if __name__ == "__main__":
    bot = BrowserBot()
    bot.run()