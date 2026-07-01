# scripts/browser_agent.py
import time
from playwright.sync_api import sync_playwright, TimeoutError
from sb3_contrib import MaskablePPO
import numpy as np

# ВАЖНО: Загрузи свою уже обученную модель
# model = MaskablePPO.load("models/best_model/best_model.zip")

def run_browser_agent():
    with sync_playwright() as p:
        # slow_mo=300 дает сайту время на отрисовку анимаций
        browser = p.chromium.launch(headless=False, slow_mo=300)
        context = browser.new_context()
        page = context.new_page()
        
        print("Открываем Wallz.gg...")
        page.goto("https://wallz.gg/") 
        
        # Ждем, пока пользователь вручную залогинится или зайдет в лобби
        print("У тебя есть 20 секунд, чтобы начать игру...")
        time.sleep(20)
        
        try:
            # Ожидание появления SVG-доски
            page.locator("svg").wait_for(state="visible", timeout=60000)
            print("Доска найдена. Агент подключен.")
        except TimeoutError:
            print("Не удалось найти доску. Скрипт остановлен.")
            return

        while True:
            try:
                # 1. ПРОВЕРКА ОЧЕРЕДИ ХОДА
                # ЗАМЕНИ ".my-turn" НА РЕАЛЬНЫЙ СЕЛЕКТОР САЙТА WALLZ.GG
                # Например, это может быть подсветка аватара или текст
                is_my_turn = page.locator(".my-turn-indicator").is_visible()
                
                if not is_my_turn:
                    time.sleep(0.5)
                    continue
                
                print("Наш ход! Парсим доску...")

                # 2. ПАРСИНГ
                # ЗАМЕНИ "svg circle.player" НА РЕАЛЬНЫЙ СЕЛЕКТОР ФИШЕК
                try:
                    players = page.locator("svg circle.player")
                    p1_cx = float(players.nth(0).get_attribute("cx"))
                    p1_cy = float(players.nth(0).get_attribute("cy"))
                except Exception as e:
                    print(f"Ошибка парсинга элементов (SVG обновился): {e}")
                    time.sleep(1)
                    continue # Пробуем заново в следующем цикле

                # --- ТУТ НУЖНО СКОНВЕРТИРОВАТЬ CX/CY В ИНДЕКСЫ 0-8 ---
                # matrix_x = int(p1_cx / cell_width) ...
                
                # 3. ИНФЕРЕНС АГЕНТА (Заглушка)
                # obs = get_obs_from_parsed_data()
                # action, _states = model.predict(obs, action_masks=get_masks())
                
                # 4. ДЕЙСТВИЕ (КЛИК)
                target_x_svg = 150 # Вычисленная координата клика
                target_y_svg = 200
                
                # Кликаем напрямую по координатам SVG-полотна, чтобы избежать TargetClosedError
                svg_box = page.locator("svg").bounding_box()
                if svg_box:
                    click_x = svg_box["x"] + target_x_svg
                    click_y = svg_box["y"] + target_y_svg
                    page.mouse.click(click_x, click_y)
                    print("Ход сделан.")
                
                # Ждем, пока ход засчитается сайтом
                time.sleep(2)

            except Exception as e:
                if "Target page, context or browser has been closed" in str(e):
                    print("Браузер закрыт пользователем.")
                    break
                print(f"Неожиданная ошибка. Восстановление: {e}")
                time.sleep(1)

if __name__ == "__main__":
    run_browser_agent()