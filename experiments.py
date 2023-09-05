import os
from scripts.utils import Timer, get_file_basenames_with_path_format, get_venv_status

TIME_FORMAT = "%H:%M:%S"
VENV_ACTIVATED = True
CONFIG_FILE_PATH_FORMAT = "scripts/config/*.py"


@Timer(time_format=TIME_FORMAT, function_name="all experiments")
def main():
    config_file_names = get_file_basenames_with_path_format(CONFIG_FILE_PATH_FORMAT)

    VENV_STATUS = get_venv_status()
    if VENV_STATUS != VENV_ACTIVATED:
        os.system("source .venv/bin/activate")

    for config_file_name in config_file_names:
        run_pipeline_with_config(config_file_name)


@Timer(time_format=TIME_FORMAT, function_name="pipeline")
def run_pipeline_with_config(config_file_name):
    print(f"Running pipeline for {config_file_name}...")
    os.system(f"python scripts/pipeline.py -c config.{config_file_name}")


if __name__ == "__main__":
    main()
