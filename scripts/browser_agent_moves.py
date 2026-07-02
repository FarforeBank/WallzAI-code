import argparse

from browser_agent_clean import CleanWallzAgent, WALLZ_URL, resolve_model_path


def parse_args():
    parser = argparse.ArgumentParser(description="Stable move-only Wallz browser agent")
    parser.add_argument("--model", default="stage8", help="Model alias or path: best, stage8, empty, or file path")
    parser.add_argument("--url", default=WALLZ_URL)
    return parser.parse_args()


def main():
    args = parse_args()
    agent = CleanWallzAgent(
        model_path=resolve_model_path(args.model),
        allow_walls=False,
        wall_fail_limit=0,
    )
    print("[System] Move-only runtime: wall actions are disabled in the browser mask.")
    agent.run(args.url)


if __name__ == "__main__":
    main()
