import gc
import pandas as pd

from itertools import combinations

def get_data_task1(
    data: pd.DataFrame,
    sample_size: int=10,
    ) -> pd.DataFrame:
    
    """Make the combinations of the text in the dataframe for task1
    Args:
        data (DataFrame): DataFrame having column alg
        sample_size (int): Sample size by algorithm
    """
    
    if 'alg' not in data.columns:
        raise ValueError("Check your dataframe, alg column missing")
    
    # Get sample size index of every alg
    grouped_data = data[["alg"]].groupby('alg').apply(
        lambda s: s.sample(sample_size)
    ).droplevel(level=0)
    
    # freeing up the memory
    del data
    gc.collect()
    data = None
    
    # making every possible combination of length 2
    iterable = combinations(grouped_data.index, 2)
    index_labels_map = grouped_data["alg"].to_dict() 
    
    #TODO: Introduce batching otherwise it may crash on large data
    data_combination_df = pd.DataFrame(
        [indexes for indexes in iterable], columns=["first_idx", "second_idx"]
    )
    
    # True labels/algorithm
    data_combination_df["true_labels"] = (
        data_combination_df[["first_idx", "second_idx"]].apply(
            lambda x: (index_labels_map[x[0]], index_labels_map[x[1]]), axis=1
        )
    )
    
    # Ground truth for task 1
    data_combination_df["ground_truth"] = (
        data_combination_df["true_labels"].apply(lambda x: x[0] == x[1])
    )

    return data_combination_df
    