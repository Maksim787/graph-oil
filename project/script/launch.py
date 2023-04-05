import sys
import os
import yaml
import subprocess
from pathlib import Path


def execute_command(cmd: str, working_directory: Path):
    print(f'Execute {cmd} in {working_directory}')
    result = subprocess.run(cmd, cwd=str(working_directory), shell=True, stdout=sys.stdout, stderr=sys.stderr)
    print(f'Return code: {result.returncode}')


def main():
    # Read config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    binary_path = config['binary_path']
    run_file = config['run_file']

    # Check paths
    assert Path(binary_path).exists()  # binary exists
    assert Path.cwd().name == 'script'  # current folder is scripts

    runs_folder = Path.cwd().parent / 'runs'
    run_path = (runs_folder / run_file).absolute()
    assert run_path.exists()  # run_file exists in runs folder

    output_directory = (Path.cwd().parent / 'output').absolute()
    output_directory.mkdir(exist_ok=True)
    assert output_directory.exists()
    # delete output/*
    for file in output_directory.iterdir():
        file.unlink()
    # delete runs/* which does not end with .run and .pvsm
    for file in runs_folder.iterdir():
        if not file.name.endswith('.run') and not file.name.endswith('.pvsm'):
            file.unlink()
    execute_command(f'"{binary_path}" "{run_path}"', working_directory=output_directory)

    for file in runs_folder.iterdir():
        if 'CSV' in file.name:
            file.rename(output_directory / file.name)


if __name__ == '__main__':
    main()
