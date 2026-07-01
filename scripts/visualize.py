import os
import time
import pygame
import numpy as np
import gymnasium as gym

from sb3_contrib import MaskablePPO
from envs.quoridor.engine import QuoridorEngine
from envs.quoridor.quoridor_env import QuoridorEnv

class QuoridorVisualizer:
    """
    Легковесный UI на Pygame для рендеринга матчей AI vs AI.
    """
    CELL_SIZE = 60
    WALL_THICKNESS = 10
    MARGIN = 40
    BOARD_PIXELS = CELL_SIZE * 9
    WINDOW_SIZE = BOARD_PIXELS + 2 * MARGIN

    BG_COLOR = (240, 240, 230)
    GRID_COLOR = (200, 200, 200)
    P1_COLOR = (220, 50, 50)     
    P2_COLOR = (50, 100, 220)    
    WALL_COLOR = (100, 60, 40)   
    TEXT_COLOR = (50, 50, 50)

    def __init__(self, model_path: str = None):
        pygame.init()
        pygame.display.set_caption("Quoridor RL AI Inference")
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE + 50))
        self.font = pygame.font.SysFont("Arial", 24)
        self.clock = pygame.time.Clock()

        self.engine = QuoridorEngine()
        self.env = QuoridorEnv(engine=self.engine)
        
        if model_path and os.path.exists(model_path):
            print(f"Загрузка модели из {model_path}...")
            self.model = MaskablePPO.load(model_path, device="cpu")
        else:
            print("ВНИМАНИЕ: Модель не найдена. Агент будет делать случайные ходы.")
            self.model = None

    def _draw_board(self):
        self.screen.fill(self.BG_COLOR)
        state = self.engine.get_state()

        # 1. Отрисовка сетки
        for y in range(9):
            for x in range(9):
                rect = pygame.Rect(
                    self.MARGIN + x * self.CELL_SIZE,
                    self.MARGIN + y * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, self.GRID_COLOR, rect, 1)

        # 2. Отрисовка фишек (ФИКС: принудительный каст в int)
        p1_y = int(state['p1_pos'][0])
        p1_x = int(state['p1_pos'][1])
        p2_y = int(state['p2_pos'][0])
        p2_x = int(state['p2_pos'][1])
        
        pygame.draw.circle(
            self.screen, self.P1_COLOR,
            (self.MARGIN + p1_x * self.CELL_SIZE + self.CELL_SIZE // 2,
             self.MARGIN + p1_y * self.CELL_SIZE + self.CELL_SIZE // 2),
            self.CELL_SIZE // 3
        )
        pygame.draw.circle(
            self.screen, self.P2_COLOR,
            (self.MARGIN + p2_x * self.CELL_SIZE + self.CELL_SIZE // 2,
             self.MARGIN + p2_y * self.CELL_SIZE + self.CELL_SIZE // 2),
            self.CELL_SIZE // 3
        )

        # 3. Отрисовка стен
        walls_h = state['walls_h']
        walls_v = state['walls_v']

        for y in range(8):
            for x in range(8):
                if walls_h[y, x]:
                    rect = pygame.Rect(
                        self.MARGIN + x * self.CELL_SIZE,
                        self.MARGIN + (y + 1) * self.CELL_SIZE - self.WALL_THICKNESS // 2,
                        self.CELL_SIZE * 2,
                        self.WALL_THICKNESS
                    )
                    pygame.draw.rect(self.screen, self.WALL_COLOR, rect)
                
                if walls_v[y, x]:
                    rect = pygame.Rect(
                        self.MARGIN + (x + 1) * self.CELL_SIZE - self.WALL_THICKNESS // 2,
                        self.MARGIN + y * self.CELL_SIZE,
                        self.WALL_THICKNESS,
                        self.CELL_SIZE * 2
                    )
                    pygame.draw.rect(self.screen, self.WALL_COLOR, rect)

        # 4. Информация
        info_text = f"P1 (Красный) Стен: {state['p1_walls']} | P2 (Синий) Стен: {state['p2_walls']}"
        text_surf = self.font.render(info_text, True, self.TEXT_COLOR)
        self.screen.blit(text_surf, (self.MARGIN, self.WINDOW_SIZE - 10))

        pygame.display.flip()

    def run(self):
        obs, info = self.env.reset()
        terminated, truncated = False, False
        
        running = True
        paused = False

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused 
                    elif event.key == pygame.K_r:
                        obs, info = self.env.reset()
                        terminated, truncated = False, False

            self._draw_board()

            if not paused and not (terminated or truncated):
                time.sleep(0.5)

                mask = info["action_mask"]
                
                if self.model is not None:
                    action, _ = self.model.predict(
                        obs, 
                        action_masks=mask, 
                        deterministic=True 
                    )
                else:
                    valid_actions = np.where(mask == 1)[0]
                    action = np.random.choice(valid_actions)

                obs, reward, terminated, truncated, info = self.env.step(action)

                if terminated:
                    winner = "Красный (P1)" if reward > 0 and self.engine.current_player == 1 else "Синий (P2)"
                    print(f"Матч завершен! Победитель: {winner}")
                elif truncated:
                    print("Матч завершен: Лимит ходов (Ничья).")

            self.clock.tick(30)

        pygame.quit()

if __name__ == "__main__":
    MODEL_PATH = "./models/best_model/best_model.zip"
    visualizer = QuoridorVisualizer(model_path=MODEL_PATH)
    visualizer.run()