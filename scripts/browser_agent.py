# scripts/browser_agent.py
import time
from playwright.sync_api import sync_playwright, TimeoutError
from sb3_contrib import MaskablePPO
from envs.quoridor.quoridor_env import QuoridorEnv

# Загружаем модель (или создаем пустышку для теста UI)
try:
    model = MaskablePPO.load("models/quoridor_latest.zip") # Берем самую свежую модель
    print("Загружена обученная модель!")
except:
    print("Сохраненная модель не найдена. Загружаем агента со случайными весами для теста UI.")
    env = QuoridorEnv()
    model = MaskablePPO("MlpPolicy", env)

def run_browser_agent():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=300)
        context = browser.new_context()
        page = context.new_page()
        
        print("Открываем Wallz.gg...")
        page.goto("https://wallz.gg/") 
        
        print("У тебя есть 20 секунд, чтобы начать игру или войти в лобби...")
        time.sleep(20)
        
        try:
            # ИСПРАВЛЕНИЕ: Ищем строго игровую доску по её уникальным координатам
            board_svg = page.locator("svg[viewBox='-12 -12 660 660']")
            board_svg.wait_for(state="visible", timeout=60000)
            print("Доска найдена. Агент подключен.")
        except TimeoutError:
            print("Не удалось найти доску. Скрипт остановлен.")
            return

        while True:
            try:
                # 1. ПРОВЕРКА ОЧЕРЕДИ ХОДА
                # ЗАМЕНИ ".my-turn-indicator" НА РЕАЛЬНЫЙ СЕЛЕКТОР (когда изучишь DOM во время хода)
                is_my_turn = page.locator(".my-turn-indicator").is_visible()
                
                if not is_my_turn:
                    time.sleep(0.5)
                    continue
                
                print("Наш ход! Парсим доску...")

                # 2. ПАРСИНГ
                try:
                    # Ищем кружки (фишки) ТОЛЬКО внутри найденной доски
                    players = board_svg.locator("circle")
                    
                    # Пытаемся получить координаты первой фишки (просто для теста парсинга)
                    if players.count() > 0:
                        p1_cx = float(players.nth(0).get_attribute("cx") or 0)
                        p1_cy = float(players.nth(0).get_attribute("cy") or 0)
                        print(f"Координаты фишки: cx={p1_cx}, cy={p1_cy}")
                    else:
                        print("Фишки не найдены.")
                        
                except Exception as e:
                    print(f"Ошибка парсинга элементов (SVG обновился): {e}")
                    time.sleep(1)
                    continue 

                # 3. ДЕЙСТВИЕ (КЛИК) - Тестовый клик по координатам внутри SVG
                target_x_svg = 150 
                target_y_svg = 200
                
                svg_box = board_svg.bounding_box()
                if svg_box:
                    click_x = svg_box["x"] + target_x_svg
                    click_y = svg_box["y"] + target_y_svg
                    page.mouse.click(click_x, click_y)
                    print(f"Выполнен клик по экрану: x={click_x}, y={click_y}")
                
                time.sleep(2)

            except Exception as e:
                if "Target page, context or browser has been closed" in str(e):
                    print("Браузер закрыт пользователем.")
                    break
                print(f"Неожиданная ошибка. Восстановление: {e}")
                time.sleep(1)

if __name__ == "__main__":
    run_browser_agent()