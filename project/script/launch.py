import sys
import yaml
import subprocess
import shutil
from pathlib import Path


def execute_command(cmd: str, working_directory: Path):
    print(f'Execute {cmd}')
    result = subprocess.run(cmd, cwd=working_directory, shell=True, stdout=sys.stdout, stderr=sys.stderr)
    print(f'Return code: {result.returncode}')


def main():
    # Read config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    project_path = Path(config['project_path'])
    run_file = config['run_file']
    tmp_suffixes = config['tmp_suffixes']

    binary_path = project_path / 'bin/H64.EXE'

    # Check paths
    assert project_path.exists()  # binary exists
    assert Path.cwd().name == 'script'  # current folder is scripts

    runs_folder = Path.cwd().parent / 'runs'
    run_path = (runs_folder / run_file).absolute()
    assert run_path.exists()  # run_file exists in runs folder

    styles_folder = Path.cwd().parent / 'styles'
    assert styles_folder.exists()

    styles_file = f'{run_file}.pvsm'
    style_path = styles_folder / styles_file
    assert style_path.exists()

    output_directory = (Path.cwd().parent / 'output').absolute()
    output_directory.mkdir(exist_ok=True)
    assert output_directory.exists()

    # Delete output/*
    for file in output_directory.iterdir():
        file.unlink()

    # Delete runs/* which do contain temp tmp_suffixes
    for file in runs_folder.iterdir():
        for suffix in tmp_suffixes:
            if file.name.endswith(suffix):
                file.unlink()

    # Replace .pvsm file paths
    with open(style_path, 'r') as f:
        content = f.read()
    content = content.replace(rf'F:\SIMULATIONS\TESTING\{run_file}', str(runs_folder))
    with open(runs_folder / styles_file, 'w') as f:
        f.write(content)

    # Execute simulation
    execute_command(f'"{binary_path}" "{run_path}"', working_directory=runs_folder)

    # Move csv files to output directory
    for file in runs_folder.iterdir():
        if 'CSV' in file.name:
            shutil.copy(file, output_directory / file.name)


if __name__ == '__main__':
    main()
