import glob
import pandas as pd
from pathlib import Path


def import_bearing_data_to_pickle(data_directory_path):
    """
    Imports and converts the original bearing accelerometer CSV file from the given directory into pandas dataframe!!
    Then it stores as a pickle file!

    :param data_directory_path: The path to the file's directory!
    """
    data_dir = Path(data_directory_path)
    pickle_dir = data_dir / "pickles"
    pickle_dir.mkdir(exist_ok=True)  # create the pickles directory if it doesn't exist

    columns = ['Hour', 'Minute', 'Second', 'micro-second', 'Horiz. accel.', 'Vert. accel.']
    print('Importing Started!')

    try:
        subdirs = [subdir for subdir in data_dir.iterdir() if subdir.is_dir()]

        if not subdirs:
            raise FileNotFoundError(f"No subdirectories found in folder: {data_dir}")

        for subdir in subdirs:
            print(f"Importing: {subdir}")
            acm_dfs = []

            for file in subdir.glob("acc*.csv"):  # Match only files starting with 'acc' and ending with .csv
                try:
                    data = pd.read_csv(file, header=None)

                    # Check if values are semicolon-separated in the first value
                    if data.shape[1] == 1 and ';' in str(data.iloc[0, 0]):
                        data = pd.read_csv(file, delimiter=';', header=None)
                    acm_dfs.append(data)

                except pd.errors.EmptyDataError:
                    print(f"Skipped empty file: {file}")

                except Exception as file_error:
                    print(f"Failed to read {file}: {file_error}")

            if acm_dfs:
                acm_df = pd.concat(acm_dfs, ignore_index=True)
                acm_df.columns = columns

                # Combine the time information into a single column!!!
                time = pd.to_datetime(acm_df['Hour'].astype(str) + ':' + acm_df['Minute'].astype(str) + ':' +
                                      acm_df['Second'].astype(str) + '.' + acm_df['micro-second'].astype(int).astype(
                    str),
                                      format='%H:%M:%S.%f')
                acm_df.insert(0, "Time", time)
                acm_df_new = acm_df.drop(['Hour', 'Minute', 'Second', 'micro-second'], axis=1)
                acm_df_new = acm_df_new.reset_index(drop=True)

                # Save to pickle inside the 'pickles' directory
                output_pickle = pickle_dir / f"{subdir.name}_DF.pkl"
                acm_df_new.to_pickle(output_pickle)

            print('Imported to DF:', subdir)
    except FileNotFoundError as e:
        print(e)
    return