import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def create_dataset(data_folder: Path):
    assert data_folder.exists(), data_folder
    wells = [_well_name(n) for n in range(10)]
    features_columns = ['oilrate', 'watrate', 'gasrate']

    features_by_iteration = []
    targets_by_iteration = []

    dates = None  # only for check

    iterations = filter(lambda x: 'iteration_' in x.name, data_folder.iterdir())
    iterations_sorted = sorted(iterations, key=lambda x: int(x.name.split('_')[-1]))
    for iteration_folder in tqdm(iterations_sorted):
        # read features
        features_df = pd.read_csv(iteration_folder / 'features.csv')
        features_df['date'] = pd.to_datetime(features_df['date'], format="%d '%b' %Y")

        if dates is None:
            dates = features_df['date']
        else:
            assert np.all(dates == features_df['date'])

        features_for_iteration = []
        for date, group_ind in features_df.groupby('date').groups.items():
            group_df = features_df.loc[group_ind]
            features_for_date = []
            for well in wells:
                well_features = group_df[group_df['well'] == f"'{well}'"]
                if len(well_features) > 0:
                    assert len(well_features) == 1
                    features_for_date.append(well_features.iloc[0][features_columns].values.squeeze().astype(float))
                else:
                    features_for_date.append(np.full(len(features_columns), np.nan, dtype=float))
            features_for_iteration.append(features_for_date)

        # read targets
        targets_for_iteration = []
        for well in wells:
            well_df = pd.read_csv(iteration_folder / f'OW2P-RT.WELL.{well}.CSV')
            well_df.columns = [col.strip() for col in well_df.columns]
            well_df = well_df.iloc[1:-1]
            target = well_df['WOPR (M3/DAY)']
            assert len(target) == len(features_for_iteration)
            targets_for_iteration.append(target)
        features_by_iteration.append(features_for_iteration)
        targets_by_iteration.append(targets_for_iteration)

    assert len(features_by_iteration) == len(targets_by_iteration)
    features_by_iteration = np.array(features_by_iteration, dtype=float)
    features_by_iteration = features_by_iteration.transpose((0, 2, 1, 3))
    targets_by_iteration = np.array(targets_by_iteration, dtype=float)

    n_iterations = len(features_by_iteration)
    n_wells = len(wells)
    n_dates = len(targets_by_iteration[0][0])
    n_features = len(features_columns)

    print(f'Found {n_iterations} iterations')
    assert features_by_iteration.shape == (n_iterations, n_wells, n_dates, n_features)
    assert targets_by_iteration.shape == (n_iterations, n_wells, n_dates)
    np.save(str(data_folder / 'features_(n_iterations, n_wells, n_dates, n_features).npy'), features_by_iteration)
    np.save(str(data_folder / 'targets_(n_iterations, n_wells, n_dates).npy'), targets_by_iteration)


def _well_name(n: int) -> str:
    if n == 0:
        return 'P3'
    if n == 9:
        return 'WELL9G'
    return f'WELL{n}'


if __name__ == '__main__':
    create_dataset(Path('data/'))
