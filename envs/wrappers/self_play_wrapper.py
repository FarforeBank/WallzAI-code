import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple

class SelfPlayWrapper(gym.Wrapper):
    """
    Обертка, которая скрывает второго игрока от обучающегося алгоритма.
    Когда обучающийся Агент делает ход, обертка автоматически запрашивает 
    ход у модели Оппонента и применяет его, возвращая Агенту состояние после 
    ответного удара.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.opponent_model = None

    def load_opponent(self, model_path: str):
        """
        Загружает веса оппонента из файла.
        Вызывается через multiprocessing из главного скрипта.
        """
        from sb3_contrib import MaskablePPO
        # Грузим оппонента на CPU. Использование MPS/GPU в изолированных 
        # worker-процессах при инференсе часто вызывает deadlock (взаимоблокировку).
        self.opponent_model = MaskablePPO.load(model_path, device="cpu")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 1. Ход нашего обучающегося Агента (Игрок 1)
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Если игра закончилась сразу после хода агента (Победа или ошибка)
        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        # 2. Ход Оппонента (Игрок 2)
        # Наш QuoridorEnv эгоцентричен: после хода Агента он вернул obs с точки зрения Игрока 2!
        # Поэтому мы можем скормить этот obs модели оппонента без конвертаций.
        if self.opponent_model is None:
            # Если оппонент еще не загружен (самое начало обучения), делаем случайный валидный ход
            mask = info["action_mask"]
            valid_actions = np.where(mask == 1)[0]
            opp_action = np.random.choice(valid_actions)
        else:
            # Получаем действие от исторического чекпоинта
            opp_action, _ = self.opponent_model.predict(
                obs, 
                action_masks=info["action_mask"], 
                deterministic=True # Оппонент играет на победу, без исследований
            )

        # Применяем ход Оппонента
        obs, opp_reward, terminated, truncated, info = self.env.step(opp_action)

        # 3. Корректировка награды
        # Если оппонент победил своим ходом (terminated=True), opp_reward будет +1.
        # Для нашего Агента это поражение, поэтому мы инвертируем эту награду (-1).
        if terminated:
            reward = -opp_reward

        # Если игра продолжается, управление снова перешло к Агенту.
        # Возвращаем obs (который уже перевернут под Агента) и reward (за его предыдущий ход).
        return obs, reward, terminated, truncated, info