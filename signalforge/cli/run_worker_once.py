import argparse
import logging
import os

from dotenv import load_dotenv

from signalforge.worker import WorkerConfig, run_worker_cycle


def main() -> None:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    config = WorkerConfig.from_environment()
    result = run_worker_cycle(config)

    print(
        "Worker cycle completed: "
        f"{len(result.ingestion_results)} sources processed, "
        f"{result.vectorization_result.indexed_count} chunks indexed"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one approved-source ingestion and vectorization cycle."
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("SIGNALFORGE_LOG_LEVEL", "INFO"),
        help="Python logging level for the one-shot worker cycle.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
