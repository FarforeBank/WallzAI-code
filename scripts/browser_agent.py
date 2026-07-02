"""Stable Wallz browser-agent entrypoint.

This file intentionally delegates to the clean browser runtime.  The previous
experimental implementation mixed SVG parsing and wall dragging in one large file
and could fail before runtime.  The default mode here is move-only because wall
placement on wallz.gg is a pointer-drag interaction and is still less reliable
than pawn moves in Playwright.

Use --allow-walls only when actively testing the experimental wall drag layer.
"""

import argparse

from browser_agent_clean import CleanWallzAgent, WALLZ_URL, resolve_model_path


def parse_args():
    parser = argparse.ArgumentParser(description="Stable Wallz browser agent")
    parser.add_argument(
        "--model",
        default="stage8",
        help="Model alias or path: best, stage8, empty, or file path",
    )
    parser.add_argument("--url", default=WALLZ_URL)
    parser.add_argument(
        "--allow-walls",
        action="store_true",
        help="Enable experimental wall dragging. Disabled by default for stability.",
    )
    parser.add_argument(
        "--wall-fail-limit",
        type=int,
        default=2,
        help="Disable walls after this many failed wall drops when --allow-walls is enabled.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    allow_walls = bool(args.allow_walls)
    agent = CleanWallzAgent(
        model_path=resolve_model_path(args.model),
        allow_walls=allow_walls,
        wall_fail_limit=args.wall_fail_limit if allow_walls else 0,
    )
    mode = "experimental walls enabled" if allow_walls else "move-only stable mode"
    print(f"[System] Browser agent mode: {mode}")
    agent.run(args.url)


if __name__ == "__main__":
    main()
