import torch 

def collate_fn(batch):
    batch_x = torch.stack([item[0] for item in batch], dim=0)
    batch_y = torch.stack([item[1] for item in batch], dim=0)
    batch_x_mark = torch.stack([item[2] for item in batch], dim=0)
    batch_y_mark = torch.stack([item[3] for item in batch], dim=0)

    return {
        'batch_x': batch_x,           # (B, L, D)
        'batch_y': batch_y,           # (B, L, D)
        'batch_x_mark': batch_x_mark, # (B, L, D)
        'batch_y_mark': batch_y_mark  # (B, L, D)
    }