"""Entry point for `python -m openmind`."""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="openMind - Continuous Learning System for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python -m openmind                        # Claude API backend (default)
  python -m openmind --model mistral/7B     # Local HuggingFace model
  python -m openmind --config my_config.yaml
  python -m openmind --data-dir ./my_data
  python -m openmind --no-explore           # Disable autonomous exploration
""",
    )
    parser.add_argument(
        "--model", "-m",
        help="HuggingFace model path (switches to local backend)",
        default=None,
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to YAML config file",
        default=None,
    )
    parser.add_argument(
        "--data-dir", "-d",
        help="Directory for persistent data (default: ./openmind_data)",
        default="./openmind_data",
    )
    parser.add_argument(
        "--no-explore",
        action="store_true",
        help="Disable autonomous exploration (interactive only)",
    )

    args = parser.parse_args()

    from openmind.runner import run

    run(
        base_model=args.model,
        config_path=args.config,
        data_dir=args.data_dir,
        idle_think_enabled=not args.no_explore,
    )


if __name__ == "__main__":
    main()
