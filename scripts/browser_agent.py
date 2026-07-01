# scripts/browser_agent.py
# (Импорты и начало файла оставляем как было)

def run_browser_agent():
    with sync_playwright() as p:
        print(f" Используем профиль аккаунта: {PROFILE_DIR}")
        context = p.chromium.launch_persistent_context(
            user_data_dir=PROFILE_DIR, headless=False, slow_mo=200
        )
        page = context.pages[0] if context.pages else context.new_page()
        
        print(" Открываем Wallz.gg...")
        page.goto("https://wallz.gg/") 
        print(" Ждем запуска матча (у тебя есть 40 секунд)...")
        
        try:
            board_svg = page.locator(".w-full.h-full").first
            board_svg.wait_for(state="visible", timeout=40000)
            print(" Найдено игровое поле Quoridor! Агент подключился.")
        except TimeoutError:
            print("❌ Не удалось найти доску. Матч не начался вовремя.")
            context.close()
            return

        local_env = QuoridorEnv()
        obs, _ = local_env.reset()

        while True:
            try:
                chips = board_svg.locator("circle")
                if chips.count() == 0:
                    time.sleep(0.5)
                    continue

                chip = chips.nth(0)
                cx, cy = chip.get_attribute("cx"), chip.get_attribute("cy")
                
                if not cx or not cy:
                    time.sleep(0.5)
                    continue
                    
                raw_x, raw_y = float(cx), float(cy)
                print(f"📍 Координаты фишки: cx={raw_x}, cy={raw_y}")

                # --- 1. ВЫБОР ДЕЙСТВИЯ ---
                masks = local_env.action_masks()
                
                # ХАК ДЛЯ ТЕСТА: Отключаем все стены (индексы с 4 по 131)
                # Заставляем бота ТОЛЬКО ходить
                masks[4:] = 0 
                
                action, _ = model.predict(obs, action_masks=masks, deterministic=True)
                
                action_names = {0: "ВВЕРХ", 1: "ВНИЗ", 2: "ВЛЕВО", 3: "ВПРАВО"}
                print(f"🤖 Нейросеть решила идти: {action_names.get(int(action), str(action))}")

                # --- 2. ФИЗИЧЕСКИЙ КЛИК В БРАУЗЕРЕ ---
                svg_box = board_svg.bounding_box()
                if svg_box and action < 4:
                    dx, dy = 0, 0
                    step_px = 71.5 # Примерный шаг одной клетки в пикселях
                    
                    if action == 0: dy = -step_px
                    elif action == 1: dy = step_px
                    elif action == 2: dx = -step_px
                    elif action == 3: dx = step_px
                    
                    click_x = svg_box["x"] + raw_x + dx
                    click_y = svg_box["y"] + raw_y + dy
                    
                    print(f"🖱️ Клик мыши: X={click_x:.1f}, Y={click_y:.1f}")
                    page.mouse.click(click_x, click_y)
                
                # --- 3. ОБНОВЛЕНИЕ ПАМЯТИ (ИСПРАВЛЕНИЕ ЗАВИСАНИЯ) ---
                # Обязательно сообщаем локальной среде, что мы сделали шаг!
                obs, reward, terminated, truncated, _ = local_env.step(action)
                
                if terminated or truncated:
                    print("🏁 Бот дошел до цели (или уперся в лимит), сбрасываем память.")
                    obs, _ = local_env.reset()
                    time.sleep(3) # Пауза перед новой игрой
                    
                time.sleep(1.5) # Пауза между ходами

            except Exception as e:
                if "Target page, context or browser has been closed" in str(e):
                    print("❌ Браузер закрыт.")
                    break
                print(f"⚠️ Ошибка парсинга: {e}")
                time.sleep(2)

        context.close()

if __name__ == "__main__":
    run_browser_agent()