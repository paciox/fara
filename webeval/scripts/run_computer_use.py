"""Command-line runner for Computer Use System."""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from webeval.systems.computer_use import ComputerUseSystem


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


async def run_task(
    task: str,
    executor_config: dict,
    output_dir: str,
    max_rounds: int = 20,
    save_screenshots: bool = True,
):
    """Run a single computer use task."""
    system = ComputerUseSystem(
        system_name="computer_use_cli",
        executor_client_cfg=executor_config,
        max_rounds=max_rounds,
        save_screenshots=save_screenshots,
        downloads_folder=output_dir,
        use_multi_agent=False,  # Single agent for now
    )
    
    example_data = {"task": task}
    
    trajectory = system.get_answer(
        question_id="cli_task",
        example_data=example_data,
        output_dir=output_dir,
    )
    
    if trajectory:
        print(f"\n✅ Task completed!")
        print(f"📝 Final answer: {trajectory.final_answer.answer}")
        print(f"📊 Actions taken: {len(trajectory.action)}")
        print(f"💾 Outputs saved to: {output_dir}")
    else:
        print("\n❌ Task failed!")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run computer use tasks with multi-agent orchestration"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task description to execute"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./computer_use_outputs",
        help="Directory to save outputs and screenshots"
    )
    
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:1234/v1",
        help="Base URL for the executor model endpoint"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/Fara-7B",
        help="Model name for the executor"
    )
    
    parser.add_argument(
        "--api_key",
        type=str,
        default="lm-studio",
        help="API key for the model endpoint"
    )
    
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=20,
        help="Maximum number of action rounds"
    )
    
    parser.add_argument(
        "--no_screenshots",
        action="store_true",
        help="Disable screenshot saving"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON config file for executor"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Build executor config
    if args.config:
        with open(args.config) as f:
            executor_config = json.load(f)
    else:
        executor_config = {
            "model": args.model,
            "base_url": args.base_url,
            "api_key": args.api_key,
        }
    
    # Run the task
    asyncio.run(run_task(
        task=args.task,
        executor_config=executor_config,
        output_dir=args.output_dir,
        max_rounds=args.max_rounds,
        save_screenshots=not args.no_screenshots,
    ))


if __name__ == "__main__":
    main()
