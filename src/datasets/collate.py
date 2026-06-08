import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {
        "lensless": torch.stack([x["lensless"] for x in dataset_items]),
        "psf": torch.stack([x["psf"] for x in dataset_items]),
        "lensed": torch.stack([x["lensed"] for x in dataset_items]),
        "id": [x["id"] for x in dataset_items],
    }

    return result_batch
