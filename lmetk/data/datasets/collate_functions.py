import torch
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional
from types import SimpleNamespace

# Define a type hint for a single sample from your dataset for clarity.
# It's a dictionary where keys are strings and values are tuples.
SampleType = Dict[str, Tuple[Optional[torch.Tensor], ...]]


def forecasting_collate_fn(batch: List[SampleType]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to process a batch of data samples.

    This function takes a list of dictionaries (the batch), where each dictionary
    represents a sample from the dataset. It unnests the data, creating a single
    flat dictionary where tensors from across the batch are concatenated.

    Args:
        batch: A list of samples. Each sample is a dictionary like:
               {'features': (tensor_x, tensor_y, None, ...), ...}

               if {'features': tensor_x, ...} then it is treated as a static feature.

    Returns:
        A single dictionary with flattened keys and concatenated tensors.
        Example: {'features_seq_x': batched_tensor_x, 'features_seq_y': batched_tensor_y}
        Static features remain as 'features': batched_tensor.
    """
    # Use defaultdict(list) to easily append tensors for the same key.
    # If a key is accessed for the first time, it's initialized with an empty list.
    collated_lists = defaultdict(list)

    # Define the suffixes for your new keys. This makes it easy to manage naming.
    # The first tensor in a tuple gets '_seq_x', the second gets '_seq_y', etc.
    suffixes = ["_seq_x", "_seq_y"]  # You can extend this list if needed

    # 1. Iterate through each sample dictionary in the batch.
    for sample in batch:
        # 2. Iterate through the items (e.g., 'features': (tensor, ...)) in the sample.
        for key, value in sample.items():

            if type(value) is tuple:
                # 3. Iterate through each element within the tuple.
                for idx, item in enumerate(value):
                    # 4. We only process the item if it's actually a tensor (ignoring Nones).
                    if torch.is_tensor(item):
                        # 5. Create the new, flattened key (e.g., 'features' + '_seq_x').
                        # This logic handles cases where you might have more tensors than defined suffixes.
                        suffix = suffixes[idx] if idx < len(suffixes) else f"_{idx}"
                        new_key = f"{key}{suffix}"

                        # 6. Append the tensor to the list for that new key.
                        collated_lists[new_key].append(item)

            # Case 2: The value is a single tensor (like a static feature)
            elif torch.is_tensor(value):
                # Use the original key directly, without adding a suffix
                collated_lists[key].append(value)
            else:
                raise ValueError(f"Unsupported value type in sample: {type(value)}")

    # 7. Now, concatenate the lists of tensors into final batched tensors.
    final_batch = {}
    for key, tensor_list in collated_lists.items():
        if tensor_list:
            # Use torch.cat to stack the tensors along the batch dimension (dim=0).
            final_batch[key] = torch.cat(tensor_list, dim=0)

    return final_batch


def rul_context_collate_fn(batch: List[Any]) -> Dict[str, Dict[str, torch.Tensor]]:
    """


    - Input sample: batch {'batch_idx': int, 'features': tensor, 'rul': tensor,...}
     ll tensors are static features can be procesed with forecasting_collate_fn, case 2.
    - Operation: Applies `forecasting_collate_fn` to batch
    - Output: {
                'batch_idx': torch.Tensor,
                'rul': torch.Tensor,
                'features': torch.Tensor
                ...
              }
    """

    batch_idxs = torch.tensor([sample.pop("batch_idx") for sample in batch])
    out = forecasting_collate_fn(batch)
    out["batch_idx"] = batch_idxs
    return out


def context_collate_fn(batch: List[Any]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Collate function for context/target tasks where each sample is a nested object.
    This version creates a nested dictionary as output by processing the 'target'
    and 'context' attributes of each sample separately.

    - Input sample: An object `item` where `item.target` is a dictionary like
      {'features': (tensor, ...)} that can be processed by `forecasting_collate_fn`.
    - Operation: Applies `forecasting_collate_fn` to the target and context data.
    - Output: {
                'target': {'features_seq_x': ..., 'features_seq_y': ...},
                'context': {'features_seq_x': ..., 'features_seq_y': ...}
              }
    """
    # 1. Extract the list of 'target' and 'context' dictionaries from the batch.
    # Each of these lists is now a valid batch for `forecasting_collate_fn`.
    target_batch = [sample.target for sample in batch]
    context_batch = [sample.context for sample in batch]

    # 2. Collate each sub-batch independently using the forecasting function.
    collated_target = forecasting_collate_fn(target_batch)
    collated_context = forecasting_collate_fn(context_batch)

    # 3. Assemble the final nested dictionary.
    return {
        "target": collated_target,
        "context": collated_context,
    }


def collate_identity(batch):
    """
    Collate function that returns the batch as is.
    Useful when each item in the batch is already a complete batch.
    E.g., when using a BatchSampler that groups indices.
    """
    # batch is a list of batch-returns
    # each element is already the full batch from dataset
    # just return the first (since BatchSampler groups them)
    return batch[0]


# def forecasting_collate_fn(batch):
#     batch_x = torch.stack([item[0] for item in batch], dim=0)
#     batch_y = torch.stack([item[1] for item in batch], dim=0)
#     batch_x_mark = torch.stack([item[2] for item in batch], dim=0)
#     batch_y_mark = torch.stack([item[3] for item in batch], dim=0)

#     return {
#         "batch_x": batch_x,  # (B, L, D)
#         "batch_y": batch_y,  # (B, L, D)
#         "batch_x_mark": batch_x_mark,  # (B, L, D)
#         "batch_y_mark": batch_y_mark,  # (B, L, D)
#     }


# def context_collate_fn(batch):
#     target_x = torch.stack([item.target.x for item in batch], dim=0)  # (B, L, D)
#     target_y = torch.stack([item.target.y for item in batch], dim=0)
#     target_x_mark = torch.stack(
#         [item.target.x_mark for item in batch], dim=0
#     )  # (B, L, D)
#     target_y_mark = torch.stack([item.target.y_mark for item in batch], dim=0)
#     target = {
#         "x": target_x,
#         "y": target_y,
#         "x_mark": target_x_mark,
#         "y_mark": target_y_mark,
#     }

#     context_x = torch.stack([item.context.x for item in batch], dim=0)  # (B, L, D)
#     context_y = torch.stack([item.context.y for item in batch], dim=0)
#     context_x_mark = torch.stack(
#         [item.context.x_mark for item in batch], dim=0
#     )  # (B, L, D)
#     context_y_mark = torch.stack([item.context.y_mark for item in batch], dim=0)
#     context = {
#         "x": context_x,
#         "y": context_y,
#         "x_mark": context_x_mark,
#         "y_mark": context_y_mark,
#     }

#     return {"target": target, "context": context}

# TODO: for the feature test.
if __name__ == "__main__":
    BATCH_SIZE = 4

    # --- 1. Test `forecasting_collate_fn` ---
    print("--- 1. Testing `forecasting_collate_fn` (with torch.cat) ---")
    forecasting_batch_data = [
        {"features": (torch.randn(10, 1, 3), torch.randn(1, 1, 3), None)},
        {"features": (torch.randn(12, 1, 3), torch.randn(1, 1, 3), None)},
        {"features": (torch.randn(8, 1, 3), torch.randn(1, 1, 3), None)},
        {"features": (torch.randn(15, 1, 3), torch.randn(1, 1, 3), None)},
    ]
    collated_output = forecasting_collate_fn(forecasting_batch_data)
    print("Output shapes:")
    for key, tensor in collated_output.items():
        print(f"  '{key}': {tensor.shape}")
    print("-" * 40 + "\n")

    # --- 2. Test `context_collate_fn` ---
    print("--- 2. Testing `context_collate_fn` (with nested output) ---")

    # Helper to create a mock data object for the context example.
    # The `target` and `context` attributes must be dictionaries that
    # `forecasting_collate_fn` can process.
    def create_mock_context_item():
        target_data = {"features": (torch.randn(10, 1, 3), torch.randn(1, 1, 3), None)}
        context_data = {
            "features": (torch.randn(12, 1, 3), torch.randn(1, 1, 3), None),
            "time_features": (torch.randn(12, 4), torch.randn(1, 4)),
        }
        return SimpleNamespace(target=target_data, context=context_data)

    context_batch_data = [create_mock_context_item() for _ in range(BATCH_SIZE)]
    collated_output = context_collate_fn(context_batch_data)
    print("Output shapes (nested dictionary):")
    for main_key, sub_dict in collated_output.items():
        print(f"  '{main_key}':")
        for sub_key, tensor in sub_dict.items():
            print(f"    '{sub_key}': {tensor.shape}")
