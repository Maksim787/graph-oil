import sys
import time

import yaml
import subprocess
import shutil
import random
import pandas as pd
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    project_path: Path

    binary_path: Path

    sources_folder: Path
    output_folder: Path
    run_folder: Path
    data_folder: Path

    run_name: str
    styles_name: str
    include_names: list[str]

    def __init__(self, config: dict):
        self.project_path = Path(config['project_path'])

        self.binary_path = self.project_path / config['binary_path']

        self.sources_folder = self.project_path / config['sources_folder']
        self.output_folder = self.project_path / config['output_folder']
        self.run_folder = self.project_path / config['run_folder']
        self.data_folder = self.project_path / config['data_folder']

        self.run_name = config['run_file']
        self.styles_name = self.run_name + '.pvsm'
        self.include_names = config['include_files']

        self.run_folder.mkdir(exist_ok=True)
        self.data_folder.mkdir(exist_ok=True)

        for path in [self.project_path, self.binary_path, self.sources_folder, self.run_folder, self.data_folder]:
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


def copy_sources(config: Config):
    print('Copy sources:')
    for file in config.sources_folder.iterdir():
        if config.run_name in file.name:
            print(f'Copy {file.name}')
            shutil.copy(file, config.run_folder / file.name)
    for file in config.include_names:
        print(f'Copy {file}')
        shutil.copy(config.sources_folder / file, config.run_folder / file)


def correct_styles(config: Config):
    print(f'Correct {config.styles_name}')
    styles_file = config.run_folder / config.styles_name
    with open(styles_file, encoding='utf-8') as f:
        content = f.read()
    content = content.replace(rf'F:\SIMULATIONS\TESTING\{config.run_name}', str(config.run_folder))
    with open(styles_file, 'w', encoding='utf-8') as f:
        f.write(content)


def parse_fields(content: str) -> list[dict]:
    # Parse parts
    parts = []
    start = 0
    finish = 0
    while finish != len(content):
        finish = content.find('DATES', start + 1)
        if finish == -1:
            finish = len(content)
        parts.append(content[start:finish])
        start = finish
    # Parse fields
    parts_fields = []
    for part in parts:
        part_fields = {}
        assert part.startswith('DATES')
        date_start = part.find('1')
        date_finish = part.find('/', date_start)
        day, month, year = part[date_start:date_finish].split()
        assert day == '1'
        part_fields['date'] = {'day': int(day), 'month': month, 'year': year}
        start = date_finish
        while True:
            try:
                start = part.find(next(filter(str.isalpha, part[start:])), start)
            except StopIteration:
                break
            finish = part.find('\n', start)
            field = part[start:finish]
            start = finish

            finish = part.find('\n/\n', start)
            data = part[start:finish].strip().split('\n')
            start = finish

            data = [x.strip().removesuffix('/').split() for x in data]
            part_fields[field] = data
        parts_fields.append(part_fields)
    return parts_fields


def is_unique(data: list[list[str]]) -> bool:
    wells = set()
    for row in data:
        well = row[0]
        if well in wells:
            return False
        wells.add(well)
    return True


def group_by_well(data: list[list[str]]) -> dict[str, list[str]]:
    result = {}
    for row in data:
        well = row[0]
        assert well not in result
        result[well] = row[1:]
    return result


def dict_to_data(data: dict[str, list[str]]) -> list[list[str]]:
    result = []
    for key, value in data.items():
        result.append([key] + value)
    return result


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def interpolate_values(first: float, second: float, i: int, n_periods: int) -> float:
    return first + i / n_periods * (second - first)


def create_random_values(first: float, second: float, i: int, n_periods: int) -> float:
    return first + i / n_periods * (second - first) + random.normalvariate(0, second - first)


def interpolate_field(field: str, data_current: list[list[str]], data_next: list[list[str]] | None, n_periods: int, i: int, date: dict[str, str]) -> list[list[str]]:
    assert 0 <= i <= n_periods - 1
    if data_next is None or field == 'WELTARG':
        return data_current
    if not is_unique(data_current) or not is_unique(data_next):
        return data_current
    if field == 'WPIMULT':
        new_rows = []
        for row in data_current:
            new_row = row.copy()
            new_row[1] = str(float(new_row[1]) ** (1 / n_periods))
            new_rows.append(new_row)
        return new_rows
    data_current = group_by_well(data_current)
    data_next = group_by_well(data_next)
    if field in ['WELSPECS', 'WEFAC', 'WCONHIST', 'COMPDAT']:
        data_result = {}
        for well, row in data_current.items():
            if well in data_next:
                new_row = []
                for j, (value_current, value_next) in enumerate(zip(row, data_next[well])):
                    if is_float(value_current):
                        assert is_float(value_next)
                        value_current = float(value_current)
                        value_next = float(value_next)
                        if field == 'WCONHIST' and 2 <= j <= 4:
                            value_new = create_random_values(value_current, value_next, i, n_periods)
                        else:
                            value_new = interpolate_values(value_current, value_next, i, n_periods)
                        print(f'Interpolate field: {date}: {field}; well: {well}; prev={row}, new={data_next[well]} {value_current}->{value_next}: {value_new}')
                    else:
                        value_new = value_current
                    new_row.append(str(value_new))
                data_result[well] = new_row
            else:
                data_result[well] = row
        return dict_to_data(data_result)
    assert False, 'Unreachable'


def interpolate_fields(parts_fields: list[dict], n_periods: int) -> list[dict]:
    new_parts_fields = []
    n = len(parts_fields)
    days_in_period = round(30 / n_periods)
    days = [1 + days_in_period * i for i in range(n_periods)]
    print(f'Days: {days} ({n_periods=}; diffs={[days[i] - days[i - 1] for i in range(1, len(days))] + [31 - days[-1]]})')
    for i in range(n - 1):
        current_part = parts_fields[i]
        next_part = parts_fields[i + 1]
        for i, day in enumerate(days):
            new_part_fields = {'date': {'day': day, 'month': current_part['date']['month'], 'year': current_part['date']['year']}}
            for field in current_part.keys():
                if field == 'date':
                    continue
                new_part_fields[field] = interpolate_field(field, current_part[field], next_part.get(field), n_periods, i, new_part_fields['date'])
            new_parts_fields.append(new_part_fields)
    new_parts_fields.append(parts_fields[-1])
    return new_parts_fields


def merge_fields(parts_fields: list[dict]):
    content = ''
    for part_fields in parts_fields:
        date = part_fields['date']
        content += f'DATES\n{date["day"]} {date["month"]} {date["year"]} /\n/\n\n'
        for field, data in part_fields.items():
            if field == 'date':
                continue
            content += f'{field}\n'
            for row in data:
                content += ' '.join(row) + ' /\n'
            content += '/\n\n'
    return content


def correct_schedule(config: Config, n_periods: int):
    schedule_name = f'{config.run_name}.SCHEDULE'
    print(f'Correct {config.run_name}')
    assert config.run_name == 'OW2P-RT', 'Comment this function if you use other simulation'
    schedule_path = config.run_folder / schedule_name
    assert schedule_path.exists()
    with open(schedule_path, encoding='utf-8') as f:
        content = f.read()

    parts_fields = parse_fields(content)
    parts_fields = interpolate_fields(parts_fields, n_periods=n_periods)
    columns = ['date', 'well', 'oilrate', 'watrate', 'gasrate']
    features = []
    for part_fields in parts_fields:
        date = part_fields['date']
        if 'WCONHIST' not in part_fields:
            continue
        data = part_fields['WCONHIST']
        for row in data:
            features.append([
                f'{date["day"]} {date["month"]} {date["year"]}',
                row[0], row[3], row[4], row[5]
            ])
    df = pd.DataFrame(features, columns=columns)
    df.to_csv(config.output_folder / 'features.csv', index=False)
    print('Features:')
    print(df.to_string())
    new_content = merge_fields(parts_fields)

    with open(schedule_path, 'w', encoding='utf-8') as f:
        print(''.join(new_content), file=f, end='')


def copy_csv(config: Config):
    for file in config.run_folder.iterdir():
        if 'csv' in file.name.lower():
            shutil.copy(file, config.output_folder / file.name)


def copy_to_data_folder(config: Config, iteration: int):
    iteration_folder = config.data_folder / f'iteration_{iteration}'
    iteration_folder.mkdir()
    for file in config.output_folder.iterdir():
        shutil.copy(file, iteration_folder / file.name)


def execute_command(cmd: str, working_directory: Path):
    print(f'Execute {cmd}')
    result = subprocess.run(cmd, cwd=working_directory, shell=True, stdout=sys.stdout, stderr=sys.stderr)
    print(f'Return code: {result.returncode}')


def run_simulation(n_periods: int, iteration: int):
    print(f'run_simulation({iteration=}) with {n_periods=}')
    config = read_config(Path('config.yaml'))

    # Clear output/, run/
    clear_folders(config)

    # Copy run files to run/
    copy_sources(config)

    # Correct run/style.pvsm
    correct_styles(config)

    # 10-wells simulation specific correction
    correct_schedule(config, n_periods=n_periods)

    # Execute simulation
    execute_command(f'"{config.binary_path}" "{config.run_name}"', working_directory=config.run_folder)

    # Move csv files to output directory
    copy_csv(config)

    # Move results to data directory
    copy_to_data_folder(config, iteration)


def main(start_iteration, n_iterations, n_periods):
    for i in range(start_iteration, start_iteration + n_iterations):
        start = time.time()
        run_simulation(n_periods=n_periods, iteration=i)
        print(f'Time: {time.time() - start} seconds')


if __name__ == '__main__':
    main(start_iteration=869, n_iterations=100000, n_periods=6)
