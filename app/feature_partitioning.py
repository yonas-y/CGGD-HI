from typing import List
import numpy as np


def create_feature_portions(input_data: List[List[float]], percentages: List[float]) -> List[List[List[float]]]:
    """
    Divide each list of features in input_data into segments according to given percentages.

    Args:
        input_data: List of feature sequences (each is a list).
        percentages: List of percentages (floats between 0 and 1) that sum to 1,
                     defining how to split each sequence.

    Returns:
        all_features_percentage_list: A list where each element corresponds to one feature sequence
                                      and contains its segments as sublists.
    """
    all_features_percentage_list = []

    # From which run the data came from!!
    run_train = [np.array([i + 1] * len(input_data[1][i])).reshape(-1, 1) for i in range(len(input_data[1]))]

    input_data.append(run_train)
    for feature in input_data:
        # Initialize a list of lists to store the intermediate values
        extracted_portions = [[] for _ in range(len(percentages))]

        # Iterate over each array and extract portions based on the percentages
        for data in feature:
            data_length = len(data)
            start_index = 0
            # Iterate over each percentage and extract the corresponding portion
            for i, percentage in enumerate(percentages):
                portion_length = int(data_length * percentage)
                extracted_portion = data[start_index:start_index + portion_length, :]
                if i == len(percentages) - 1:
                    extracted_portion = data[start_index:, :]
                extracted_portions[i].append(extracted_portion)
                start_index += portion_length

        # Concatenate the extracted portions to create n groups!
        feature_groups = [np.concatenate(extracted_portion_list) for extracted_portion_list in extracted_portions]
        all_features_percentage_list.append(feature_groups)

    return all_features_percentage_list