from pathlib import Path
import pandas as pd
import numpy as np

def get_result_df_list(experiment_dir: Path, uuid_list: list[str]):
    result_list = []
    failed_uuid_list = []
    missing_uuid_list = []
    for uuid in uuid_list:
        if (experiment_dir / f"{uuid}.csv").exists():
            try:
                result_list.append(pd.read_csv(
                    experiment_dir / f"{uuid}.csv", index_col=["uuid", "replication"]
                ))
            except pd.errors.ParserError:
                failed_uuid_list.append(uuid)
        else:
            missing_uuid_list.append(uuid)

    print("The following UUIDs did not have a CSV file:")
    print(missing_uuid_list)
    print()
    print("The following UUIDs failed due to a pandas.errors.ParserError:")
    print(failed_uuid_list)
    return result_list

def convert_str_to_float_list(str_list: str, list_len: int) -> list[float]:
    if str_list == "[]":
        return [np.nan for _ in range(list_len)]
    else:
        return [float(x) for x in str_list.strip('[]').split(',')]