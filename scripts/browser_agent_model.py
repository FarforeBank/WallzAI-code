import argparse
from pathlib import Path

import browser_agent as browser_agent_module
from browser_agent_safe import install_safe_wall_patches


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_ALIASES = {
    "best": ROOT_DIR / "models" / "best_model" / "best_model.zip",
    "stage8": ROOT_DIR / "models" / "best_model_stage8" / "best_model.zip",
    "empty": ROOT_DIR / "models" / "empty_model" / "best_model.zip",
}


def resolve_model_path(value: str) -> Path:
    if value in MODEL_ALIASES:
        return MODEL_ALIASES[value]

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Wallz browser agent with a selected MaskablePPO model and safe tray-drag walls."
    )
    parser.add_argument(
        "--model",
        default="best",
        help=(
            "Model alias or path. Aliases: best=models/best_model/best_model.zip, "
            "stage8=models/best_model_stage8/best_model.zip, "
            "empty=models/empty_model/best_model.zip. Relative paths are resolved from repo root."
        ),
    )
    parser.add_argument(
        "--no-walls",
        action="store_true",
        help="Disable wall actions from the start. Useful for isolating movement quality.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model)
    browser_agent_module.MODEL_PATH = model_path
    browser_agent_module.ALLOW_WALL_ACTIONS = not args.no_walls
    install_safe_wall_patches()
    print(f"[System] Выбрана модель: {model_path}")
    print("[System] Safe wall mode: tray drag targets, no strategic guards")
    if args.no_walls:
        print("[System] Wall-actions отключены с запуска (--no-walls)")
    browser_agent_module.BrowserAgent().run()


if __name__ == "__main__":
    main()
