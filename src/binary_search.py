from typing import List, Literal, Tuple, Optional

def num_partitions(token_counts: List[int], threshold: int, prompt_token_count: int):
        """
        Helper function for binary search which determines the optimal number of tokens
        for the query for both documents_to_be_updated and documents_to_be_checked groups.
        """
        partitions = 1
        current_sum = 0
        for tokens in token_counts:
            if current_sum + tokens > threshold:
                partitions += 1
                current_sum = tokens + prompt_token_count
            else:
                current_sum += tokens
        return partitions

def min_threshold_for_group(token_counts: List[int], max_partitions: int, prompt_token_count: int):
        """
        Find the smallest threshold T such that the documents (with known token counts)
        can be partitioned into <= max_partitions subgroups.
        """
        lo = max(token_counts)  # At minimum, a threshold must fit the largest document
        hi = sum(token_counts)  # At most, all documents fit in one partition
        best = hi
        while lo <= hi:
            mid = (lo + hi) // 2
            parts = num_partitions(token_counts, mid, prompt_token_count)
            if parts <= max_partitions:
                best = mid
                hi = mid - 1  # Try to minimize T further
            else:
                lo = mid + 1
        return best

def partition_documents(token_counts: List[int], threshold: int, prompt_token_count: int) -> List[Tuple[int, int]]:
    """
    Function partiotioning the documents given token count threshold.
    """
    partitions = []
    current_partition = []
    current_sum = 0
    start = 0
    for i, tokens in enumerate(token_counts):
        if current_sum + tokens > threshold:
            partitions.append((start, i))
            start = i
            current_partition = [tokens + prompt_token_count]
            current_sum = tokens
        else:
            current_partition.append(tokens)
            current_sum += tokens
    if current_partition:
        partitions.append((start, len(token_counts)))
    return partitions

def find_partitions(
    max_partitions: int,
    tokens_updated_group: Optional[List[int]],
    tokens_checked_group: Optional[List[int]],
    prompt_updated_token_count: Optional[int],
    prompt_checked_token_count: Optional[int],
    find_best_thresholds: Literal["minsum", "minmax", "minvariance"] = "minmax",
    max_token_count_per_call: int = 50000,
    ) -> Tuple[Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]]:
    """
    The function finds partitions for both checked and updated document groups.
    """
    if tokens_updated_group is not None and tokens_checked_group is not None:
        group_thresholds = []
        for partitions_count in range(1, max_partitions):
            token_threshold_updated_group = min_threshold_for_group(tokens_updated_group,
                                                                    partitions_count,
                                                                    prompt_token_count=prompt_updated_token_count)
            token_threshold_checked_group = min_threshold_for_group(tokens_checked_group,
                                                                    max_partitions - partitions_count,
                                                                    prompt_token_count=prompt_checked_token_count)

            group_thresholds.append((
                token_threshold_updated_group,
                token_threshold_checked_group
            ))
        
        filtered_thresholds = list(filter(
            lambda x: x[0] < max_token_count_per_call and x[1] < max_token_count_per_call,
            group_thresholds
            ))
        
        if find_best_thresholds == "minsum":
            optimal_thresholds = min(filtered_thresholds, lambda x: x[0] + x[1])
        elif find_best_thresholds == "minmax":
            optimal_thresholds = min(filtered_thresholds, key=lambda x: max(x[0], x[1]))
        elif find_best_thresholds == "minvariance":
            optimal_thresholds = min(filtered_thresholds, key=lambda x: (x[0] - x[1]) ** 2)
        else:
            raise ValueError("Unknown thresholds aggregation method.")

        threshold_updated_group, threshold_checked_group = optimal_thresholds
        partition_updated_group = partition_documents(
            tokens_updated_group,
            threshold_updated_group,
            prompt_updated_token_count
        )
        partition_checked_group = partition_documents(
            tokens_checked_group,
            threshold_checked_group,
            prompt_checked_token_count
        )

    
    elif tokens_checked_group is None:
        threshold_updated_group = min_threshold_for_group(tokens_updated_group,
                                                        max_partitions,
                                                        prompt_token_count=prompt_updated_token_count)
        partition_updated_group = partition_documents(
            tokens_updated_group,
            threshold_updated_group,
            prompt_updated_token_count
        )

        partition_checked_group = None
    
    elif tokens_updated_group is None:
        threshold_checked_group = min_threshold_for_group(tokens_checked_group,
                                                        max_partitions,
                                                        prompt_token_count=prompt_checked_token_count)
        partition_checked_group = partition_documents(
            tokens_checked_group,
            threshold_checked_group,
            prompt_checked_token_count
        )

        partition_updated_group = None
    
    return partition_checked_group, partition_updated_group

