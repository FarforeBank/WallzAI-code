# scripts/browser_agent.py
import time
import os
import sys

BASE_DIR = os.getcwd()
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from playwright.sync_api import sync_playwright
from sb3_contrib import MaskablePPO
from envs.quoridor.quoridor_env import QuoridorEnv

PROFILE_DIR = os.path.join(BASE_DIR, "browser_profile")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model", "best_model.zip")

class BrowserAgent:
    def __init__(self):
        self.local_env = QuoridorEnv()
        self.obs, _ = self.local_env.reset()
        self.model = MaskablePPO.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else MaskablePPO("MlpPolicy", self.local_env)

    def run(self):
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(user_data_dir=PROFILE_DIR, headless=False, slow_mo=200)
            page = context.pages[0] if context.pages else context.new_page()
            page.goto("https://wallz.gg/")
            
            print("\n[Управление] Начни матч и нажми ENTER, когда увидишь доску...")
            input()
            
            # ИСПОЛЬЗУЕМ КЛАССЫ, А НЕ СТРОГИЙ VIEWBOX
            board = page.locator(".w-full.h-full").first
            
            while True:
                try:
                    # Сканируем фишки
                    chips = board.locator("circle")
                    if chips.count() < 2: continue # Ждем обоих игроков
                    
                    # Берем P1
                    cx = float(chips.nth(0).get_attribute("cx"))
                    cy = float(chips.nth(0).get_attribute("cy"))
                    
                    # ПРЕДСКАЗАНИЕ
                    masks = self.local_env.action_masks()
                    masks[4:] = 0 # ТОЛЬКО ХОДЬБА для теста
                    action, _ = self.model.predict(self.obs, action_masks=masks, deterministic=True)
                    
                    # КЛИК
                    svg_box = board.bounding_box()
                    scale = svg_box["width"] / 660.0
                    
                    # Расчет целевой точки
                    target_x, target_y = cx, cy
                    step = 73.33
                    if action == 0: target_y -= step
                    elif action == 1: target_y += step
                    elif action == 2: target_x -= step
                    elif action == 3: target_x += step
                    
                    click_x = (target_x + 12) * scale
                    click_y = (target_y + 12) * scale
                    
                    print(f"[Действие] Ход {int(action)} | Клик: ({click_x:.1f}, {click_y:.1f})")
                    board.click(position={"x": click_x, "y": click_y}, force=True)
                    
                    # ВАЖНО: обновляем состояние среды, чтобы бот видел, что он сходил
                    self.obs, _, _, _, _ = self.local_env.step(action)
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"Поиск... {e}")
                    time.sleep(1)

if __name__ == "__main__":
    BrowserAgent().run()