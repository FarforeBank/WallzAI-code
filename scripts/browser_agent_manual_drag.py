"""Run the safe Wallz agent with manual mouse wall dragging.

Use this when Playwright locator.drag_to shows a wall preview over the board but
Wallz does not commit the drop on mouse release.
"""

import time

import browser_agent_safe as safe_agent


safe_agent.MODEL_ALIASES["stage8"] = safe_agent.ROOT_DIR / "models" / "best_model_stage8" / "best_model.zip"


def _manual_drag_wall_from_tray(page, orientation, target):
    locator = safe_agent._tray_locator(page, orientation)
    box = locator.bounding_box(timeout=1500)
    if box is None:
        raise RuntimeError(f"Wall tray button not found for orientation={orientation}")

    start_x = box["x"] + box["width"] / 2.0
    start_y = box["y"] + box["height"] / 2.0
    end_x = target["x"]
    end_y = target["y"]

    page.mouse.move(start_x, start_y)
    time.sleep(0.08)
    page.mouse.down(button="left")
    time.sleep(0.20)
    page.mouse.move((start_x + end_x) / 2.0, (start_y + end_y) / 2.0, steps=18)
    time.sleep(0.05)
    page.mouse.move(end_x, end_y, steps=26)
    time.sleep(0.20)
    page.mouse.move(end_x + 1.0, end_y + 1.0, steps=3)
    page.mouse.move(end_x, end_y, steps=3)
    time.sleep(0.10)
    page.mouse.up(button="left")
    time.sleep(0.20)
    page.mouse.move(end_x, end_y - 18.0, steps=4)
    time.sleep(0.08)


def drag_wall_from_tray(page, board, orientation, target):
    _manual_drag_wall_from_tray(page, orientation, target)


safe_agent.drag_wall_from_tray = drag_wall_from_tray


if __name__ == "__main__":
    safe_agent.main()
