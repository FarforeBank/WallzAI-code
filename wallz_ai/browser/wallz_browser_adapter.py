from __future__ import annotations

import argparse
import asyncio

from wallz_ai.agents.ppo import sample_action
from wallz_ai.env.wallz_env import WallzEnv
from wallz_ai.training.evaluate import load_checkpoint
from .action_executor import WallzActionExecutor
from .state_reader import WallzStateReader


async def run(url: str, checkpoint: str, dry_run: bool, allow_live: bool) -> None:
    if not dry_run and not allow_live:
        raise SystemExit("Refusing live browser control without --allow-live. Default mode is dry-run only.")
    from playwright.async_api import async_playwright

    model, _ = load_checkpoint(checkpoint)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto(url)
        state = await WallzStateReader(page).read_state()
        env = WallzEnv(max_moves=state.max_moves)
        env.state = state
        action, _, _, _ = sample_action(model, env.observation(), env.legal_action_mask(), next(model.parameters()).device, deterministic=True)
        await WallzActionExecutor(page, dry_run=dry_run).execute(state, action)
        await browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Wallz.gg browser dry-run adapter.")
    parser.add_argument("--url", default="https://www.wallz.gg/")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--allow-live", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.url, args.checkpoint, args.dry_run, args.allow_live))


if __name__ == "__main__":
    main()
