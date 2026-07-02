"""Run the safe Wallz agent with manual mouse wall dragging.

Use this when Playwright locator.drag_to shows a wall preview over the board but
Wallz does not commit the drop on mouse release.
"""

import time

import browser_agent_safe as safe_agent


safe_agent.MODEL_ALIASES["stage8"] = safe_agent.ROOT_DIR / "models" / "best_model_stage8" / "best_model.zip"


def _dispatch_release_events(page, x, y):
    page.evaluate(
        """
        ({ x, y }) => {
            const target = document.elementFromPoint(x, y) || document;
            const common = {
                bubbles: true,
                cancelable: true,
                clientX: x,
                clientY: y,
                screenX: x,
                screenY: y,
                button: 0,
                buttons: 0,
            };
            for (const eventTarget of [target, document, window]) {
                eventTarget.dispatchEvent(new PointerEvent('pointerup', {
                    ...common,
                    pointerId: 1,
                    pointerType: 'mouse',
                    isPrimary: true,
                }));
                eventTarget.dispatchEvent(new MouseEvent('mouseup', common));
            }
        }
        """,
        {"x": x, "y": y},
    )


def _manual_drag_wall_from_tray(page, orientation, target):
    page.bring_to_front()
    page.evaluate("() => window.focus()")

    locator = safe_agent._tray_locator(page, orientation)
    box = locator.bounding_box(timeout=1500)
    if box is None:
        raise RuntimeError(f"Wall tray button not found for orientation={orientation}")

    start_x = box["x"] + box["width"] / 2.0
    start_y = box["y"] + box["height"] / 2.0
    end_x = target["x"]
    end_y = target["y"]

    page.mouse.move(start_x, start_y)
    time.sleep(0.10)
    page.mouse.down(button="left")
    time.sleep(0.25)
    page.mouse.move((start_x + end_x) / 2.0, (start_y + end_y) / 2.0, steps=20)
    time.sleep(0.06)
    page.mouse.move(end_x, end_y, steps=30)
    time.sleep(0.25)
    page.mouse.move(end_x + 1.0, end_y + 1.0, steps=4)
    page.mouse.move(end_x, end_y, steps=4)
    time.sleep(0.12)
    page.mouse.up(button="left")
    _dispatch_release_events(page, end_x, end_y)
    time.sleep(0.18)

    # If the app stayed in a hover/drop state, this extra gap click is harmless
    # for pawn movement but often commits the currently hovered wall preview.
    page.mouse.click(end_x, end_y)
    _dispatch_release_events(page, end_x, end_y)
    time.sleep(0.20)
    page.mouse.move(end_x, end_y - 18.0, steps=4)
    time.sleep(0.08)


def drag_wall_from_tray(page, board, orientation, target):
    _manual_drag_wall_from_tray(page, orientation, target)


safe_agent.drag_wall_from_tray = drag_wall_from_tray


if __name__ == "__main__":
    safe_agent.main()
