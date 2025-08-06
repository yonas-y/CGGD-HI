import glob
import pandas as pd
from pathlib import Path
import re
import pyarrow.dataset as ds
import shutil

# Find CSV files and sort by numeric value extracted from filename
def extract_number(f):
    # extract number before .csv (works for filenames like '1.csv', 'file_1.csv', etc.)
    match = re.search(r'(\d+)', f.stem)
    return int(match.group(1)) if match else float('inf')  # put non-matching files at the end

def import_pronostia_data_to_pickle(data_directory_path, pickle_directory_path):
    """
    Imports and converts the original bearing accelerometer CSV file from the given directory into pandas dataframe!!
    Then it stores as a pickle file in a directory!

    :param data_directory_path: The path to the file's directory!
    :param pickle_directory_path: The path to store the converted pickle directory!'
    """
    data_dir = Path(data_directory_path)
    pickle_dir = Path(pickle_directory_path)
    pickle_dir.mkdir(exist_ok=True)  # create the pickles directory if it doesn't exist

    columns = ['Hour', 'Minute', 'Second', 'micro-second', 'Horiz. accel.', 'Vert. accel.']
    print('Importing Started!')

    try:
        subdirs = [subdir for subdir in data_dir.iterdir() if subdir.is_dir()]

        if not subdirs:
            raise FileNotFoundError(f"No subdirectories found in folder: {data_dir}")

        for subdir in subdirs:
            output_pickle = pickle_dir / f"{subdir.name}_DF.pkl"

            if output_pickle.exists():
                print(f"[✓] Skipping {subdir.name} — already imported!")
                continue

            print(f"[→] Importing {subdir.name}")
            acm_dfs = []
            batch_size = 100

            csv_files = sorted(subdir.glob("acc*.csv"), key=extract_number) # find all the CSV files in sorted order.
            total_files_loaded = 0  # counter

            for i in range(0, len(csv_files), batch_size):
                batch_files = csv_files[i:i + batch_size]
                batch_dfs = []

                for file in batch_files:
                    try:
                        data = pd.read_csv(file, header=None)
                        # Check if values are semicolon-separated in the first value
                        if data.shape[1] == 1 and ';' in str(data.iloc[0, 0]):
                            data = pd.read_csv(file, delimiter=';', header=None)
                        batch_dfs.append(data)
                        total_files_loaded += 1
                    except pd.errors.EmptyDataError:
                        print(f"Skipped empty file: {file}")

                    except Exception as file_error:
                        print(f"Failed to read {file}: {file_error}")

                if batch_dfs:
                    batch_df = pd.concat(batch_dfs, ignore_index=True)
                    acm_dfs.append(batch_df)

            # Final full DataFrame
            if acm_dfs:
                acm_df = pd.concat(acm_dfs, ignore_index=True)
                acm_df.columns = columns

                print(f"✅ Final DataFrame shape: {acm_df.shape}")
                print(f"📦 Total files successfully loaded: {total_files_loaded}")

                # Combine the time information into a single column!!!
                time = pd.to_datetime(acm_df['Hour'].astype(str) + ':' + acm_df['Minute'].astype(str) + ':' +
                                      acm_df['Second'].astype(str) + '.' + acm_df['micro-second'].astype(int).astype(
                    str),
                                      format='%H:%M:%S.%f')
                acm_df.insert(0, "Time", time)
                acm_df_new = acm_df.drop(['Hour', 'Minute', 'Second', 'micro-second'], axis=1)
                acm_df_new = acm_df_new.reset_index(drop=True)

            else:
                acm_df_new = pd.DataFrame()
                print("⚠️ No data loaded.")

            # Save to pickle inside the output 'pickles' directory
            acm_df_new.to_pickle(output_pickle)

            print('Imported to DF:', subdir)

    except FileNotFoundError as e:
        print(e)
    return

def import_XJTU_SY_data_to_pickle(data_directory_path, pickle_directory_path):
    """
    Imports and converts the original bearing accelerometer CSV file from the given directory into pandas dataframe!!
    Then it stores as a pickle file in a directory!

    :param data_directory_path: The path to the file's directory!
    :param pickle_directory_path: The path to store the converted pickle directory!'
    """
    data_dir = Path(data_directory_path)
    pickle_dir = Path(pickle_directory_path)
    pickle_dir.mkdir(exist_ok=True)  # create the pickles directory if it doesn't exist

    columns = ['Horiz. accel.', 'Vert. accel.']
    print('Importing Started!')
    try:
        subdirs = [subdir for subdir in data_dir.iterdir() if subdir.is_dir()]
        if not subdirs:
            raise FileNotFoundError(f"No subdirectories found in folder: {data_dir}")

        for subdir in subdirs:
            output_pickle = pickle_dir / f"{subdir.name}_DF.pkl"

            if output_pickle.exists():
                print(f"[✓] Skipping {subdir.name} — already imported!")
                continue

            print(f"[→] Importing {subdir.name}")
            acm_dfs = []
            batch_size = 25

            csv_files = sorted(subdir.glob("*.csv"), key=extract_number)  # find all the CSV files in sorted order.
            total_files_loaded = 0  # counter

            # Create an output folder for Parquet batches
            parquet_dir = Path("data/raw_pickles/XJTU_SY/parquet_batches")

            # Delete folder if it exists
            if parquet_dir.exists():
                shutil.rmtree(parquet_dir)

            # Recreate empty folder
            parquet_dir.mkdir(parents=True, exist_ok=True)

            for i in range(0, len(csv_files), batch_size):
                batch_files = csv_files[i:i + batch_size]

                batch_dfs = []

                for file in batch_files:
                    try:
                        data = pd.read_csv(file, header=None)
                        batch_dfs.append(data.iloc[1:])  # Drop the first row which is a column title.
                        total_files_loaded += 1

                    except pd.errors.EmptyDataError:
                        print(f"Skipped empty file: {file}")

                    except Exception as file_error:
                        print(f"Failed to read {file}: {file_error}")

                if batch_dfs:
                    batch_df = pd.concat(batch_dfs, ignore_index=True)
                    parquet_file = parquet_dir / f"batch_{i // batch_size + 1}.parquet"
                    batch_df.to_parquet(parquet_file)
                else:
                    print(f"No data in batch {i // batch_size + 1}")

            # Find all batch parquet files! Parquet is faster!
            parquet_files = sorted(parquet_dir.glob('batch_*.parquet'))
            if parquet_files:
                print(f"📦 Total Parquet batches: {len(parquet_files)}")

                dataset = ds.dataset(parquet_files, format="parquet")
                final_df = dataset.to_table().to_pandas()
                print(f"✅ Final concatenated DataFrame shape: {final_df.shape}")
                print(f"📦 Total files successfully loaded: {total_files_loaded}")
            else:
                final_df = pd.DataFrame()
                print("⚠️ No data loaded.")

            final_df.columns = columns

            # Save the parquet or pickle inside the output 'pickles' directory
            final_df.to_pickle(output_pickle)
            # final_df.to_parquet(output_parquet, index=False)
            print(f"Final dataset saved to {output_pickle}")

    except FileNotFoundError as e:
        print(e)
    return