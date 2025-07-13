# import tensorflow as tf
from typing import List, Tuple
import numpy as np
import random
import time

def create_feature_portions(train_mel_feature: List[np.ndarray],
                            ene_rul_feat: List[np.ndarray],
                            percentages: List[float]) -> List[List[np.ndarray]]:
    """
    Divide each list of features in input_data into segments according to given percentages.

    Args:
        train_mel_feature: Training data mel feature sequences.
        ene_rul_feat: Training data energy, rul and order feature list.
        percentages: List of percentages (floats between 0 and 1) that sum to 1,
                     defining how to split each sequence.

    Returns:
        all_features_percentage_list: A list where each element corresponds to one feature sequence
                                      and contains its segments as sublists.
    """
    all_features_percentage_list = []
    input_data = [train_mel_feature, ene_rul_feat]

    # From which run the data came from!!
    run_train = [np.array([i + 1] * len(ene_rul_feat[i])).reshape(-1, 1) for i in range(len(ene_rul_feat))]

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


def shuffle_batched_interleaved(percentage_partitioned_data: List[List[np.ndarray]],
                                batch_percentages: List[float],
                                val_split: float = 0.2,
                                batch_size: int = 16) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
    """
    A function that Shuffles and creates a training and validation sets bt taking data from the different segments pf the
    run by partitioning based on the batch percentage value. More data compared to its size is included from the
    start and end segments of the dataset.

    :param percentage_partitioned_data: The different features we want to create the portions (list of lists).
    :param batch_percentages: The percentage of data included in the batches from different sections.
    :param val_split: The percentage of data reserved for the validation set.
    :param batch_size: The size of the batches.

    :return:
        final_batched_datasets: The batches as a list of tensors.
    """

    # Shuffle and batch the datasets from each group
    final_batched_datasets = []

    total_nu_samples = sum(len(part) for part in percentage_partitioned_data[0])
    n_batches = 3 * (total_nu_samples // batch_size)

    for i in range(n_batches):
        random.seed(time.time())
        seed = random.randint(0, 1000)  # Adjust the seed range as needed!
        extracted_elements_to_batch_list = []
        for j in range(len(percentage_partitioned_data[0])):
            features_portion = [sublist[j] for sublist in percentage_partitioned_data]

            # Shuffle each proportion of feature!
            features_portion_shuffled_list = []
            for sel_feature_to_shuffle in features_portion:
                np.random.seed(seed)  # Set the seed for shuffling
                features_portion_shuffled = np.random.permutation(sel_feature_to_shuffle)
                features_portion_shuffled_list.append(features_portion_shuffled)

            # Calculate the number of samples from this group in each batch!
            num_samples_batch = int(batch_size * batch_percentages[j])

            # Extract data from the portions based on the number of samples!
            start = random.randint(0, len(features_portion_shuffled_list[0]) - num_samples_batch)
            extracted_elements_to_batch = [sublist[start:start + num_samples_batch] for sublist in
                                           features_portion_shuffled_list]
            extracted_elements_to_batch_list.append(extracted_elements_to_batch)

        # Now concatenate the similar feature portions together to have the full batch size features!
        full_batch_size_element_list = []

        for k in range(len(extracted_elements_to_batch_list[0])):
            common_feat_element_list = []
            for j in range(len(extracted_elements_to_batch_list)):
                common_feat_element_list.extend(extracted_elements_to_batch_list[j][k])
            full_batch_size_element_list.append(np.array(common_feat_element_list))

        # Shuffle the new full batch size features!
        full_batch_size_element_shuffled_list = []
        for sel_full_batch_size_element in full_batch_size_element_list:
            np.random.seed(seed)  # Set the seed for shuffling
            full_batch_size_element_shuffled = np.random.permutation(sel_full_batch_size_element)
            # full_batch_size_element_shuffled = tf.cast(full_batch_size_element_shuffled, dtype=tf.float32)
            full_batch_size_element_shuffled_list.append(full_batch_size_element_shuffled)

        final_batched_datasets.append(full_batch_size_element_shuffled_list)

    ########## =================== Split the data into training/validation! ============ ############
    random.seed(seed)
    # Shuffle the combined data
    random.shuffle(final_batched_datasets)
    # Calculate the split index
    split_index = int(len(final_batched_datasets) * (1 - val_split))
    # Split the combined data into training and validation sets
    train_data, val_data = final_batched_datasets[:split_index], final_batched_datasets[split_index:]

    return train_data, val_data
