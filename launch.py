import sys
import yaml
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    project_path: Path

    binary_path: Path

    sources_folder: Path
    output_folder: Path
    run_folder: Path

    run_name: str
    styles_name: str
    include_names: list[str]

    def __init__(self, config: dict):
        self.project_path = Path(config['project_path'])

        self.binary_path = self.project_path / config['binary_path']

        self.sources_folder = self.project_path / config['sources_folder']
        self.output_folder = self.project_path / config['output_folder']
        self.run_folder = self.project_path / config['run_folder']

        self.run_name = config['run_file']
        self.styles_name = self.run_name + '.pvsm'
        self.include_names = config['include_files']

        self.run_folder.mkdir(exist_ok=True)

        for path in [self.project_path, self.binary_path, self.sources_folder, self.run_folder]:
            assert path.exists(), path
        for path in [self.run_name, self.styles_name] + self.include_names:
            assert (self.sources_folder / path).exists()


def read_config(path: Path):
    print('Read config')
    assert path.exists(), path
    with open(path, encoding='utf-8') as f:
        return Config(yaml.safe_load(f))


def clear_folders(config: Config):
    print('Clear run and output folders')
    shutil.rmtree(config.run_folder)
    shutil.rmtree(config.output_folder)
    config.run_folder.mkdir()
    config.output_folder.mkdir()


def correct_styles(config: Config):
    print(f'Correct {config.styles_name}')
    with open(config.run_folder / config.styles_name, encoding='utf-8') as f:
        content = f.read()
    content = content.replace(rf'F:\SIMULATIONS\TESTING\{config.run_name}', str(config.run_folder))
    with open(config.run_folder / config.styles_name, 'w', encoding='utf-8') as f:
        f.write(content)


def copy_sources(config: Config):
    print('Copy sources:')
    for file in config.sources_folder.iterdir():
        if config.run_name in file.name:
            print(f'Copy {file.name}')
            shutil.copy(file, config.run_folder / file.name)
    for file in config.include_names:
        print(f'Copy {file}')
        shutil.copy(config.sources_folder / file, config.run_folder / file)


def copy_csv(config: Config):
    for file in config.run_folder.iterdir():
        if 'csv' in file.name.lower():
            shutil.copy(file, config.output_folder / file.name)


def execute_command(cmd: str, working_directory: Path):
    print(f'Execute {cmd}')
    result = subprocess.run(cmd, cwd=working_directory, shell=True, stdout=sys.stdout, stderr=sys.stderr)
    print(f'Return code: {result.returncode}')


def main():
    config = read_config(Path('config.yaml'))

    # Clear output/, run/
    clear_folders(config)

    # Copy run files to run/
    copy_sources(config)

    # Correct run/style.pvsm
    correct_styles(config)

    # 10-wells simulation specific correction
    # TODO: edit schedule

    # Execute simulation
    execute_command(f'"{config.binary_path}" "{config.run_name}"', working_directory=config.run_folder)

    # Move csv files to output directory
    copy_csv(config)


if __name__ == '__main__':
    main()
