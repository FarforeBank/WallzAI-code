import argparse
from pathlib import Path

import browser_agent as browser_agent_module


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_ALIASES = {
    "best": ROOT_DIR / "models" / "best_model" / "best_model.zip",
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
        description="Run Wallz browser agent with a selected MaskablePPO model."
    )
    parser.add_argument(
        "--model",
        default="best",
        help=(
            "Model alias or path. Aliases: best=models/best_model/best_model.zip, "
            "empty=models/empty_model/best_model.zip. Relative paths are resolved from repo root."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model)
    browser_agent_module.MODEL_PATH = model_path
    print(f"[System] Выбрана модель: {model_path}")
    browser_agent_module.BrowserAgent().run()


if __name__ == "__main__":
    main()
