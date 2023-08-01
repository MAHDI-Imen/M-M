import os
from glob import glob
from time import time, strftime, gmtime


def main():
    config_files_paths = list(glob("scripts/config/*.py"))
    config_files = [
        os.path.splitext(os.path.basename(path))[0] for path in config_files_paths
    ]

    if not os.environ.get("VIRTUAL_ENV"):
        os.system("source .venv/bin/activate")

    time_format = "%H:%M:%S"
    experiment_start_time = time()

    for config_file in config_files:
        run_start_time = time()
        print(f"Running pipeline for {config_file}...")
        os.system(f"python scripts/pipeline.py -c config.{config_file}")
        run_end_time = time()
        elapsed_time = strftime(time_format, gmtime(run_end_time - run_start_time))
        print(f"Finished {config_file} in {elapsed_time}.")

    experiment_end_time = time()
    elapsed_time = strftime(
        time_format, gmtime(experiment_end_time - experiment_start_time)
    )
    print(f"Finished all experiments in {elapsed_time}.")


if __name__ == "__main__":
    main()
