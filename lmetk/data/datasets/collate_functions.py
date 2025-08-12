import torch 

def forecasting_collate_fn(batch):
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

def context_collate_fn(batch):
    target_x = torch.stack([item.target.x for item in batch], dim=0) # (B, L, D)
    target_y = torch.stack([item.target.y for item in batch], dim=0)
    target_x_mark = torch.stack([item.target.x_mark for item in batch], dim=0) # (B, L, D)
    target_y_mark = torch.stack([item.target.y_mark for item in batch], dim=0)
    target = {'x': target_x, 'y': target_y, 'x_mark': target_x_mark, 'y_mark': target_y_mark}

    context_x = torch.stack([item.context.x for item in batch], dim=0) # (B, L, D)
    context_y = torch.stack([item.context.y for item in batch], dim=0)
    context_x_mark = torch.stack([item.context.x_mark for item in batch], dim=0) # (B, L, D)
    context_y_mark = torch.stack([item.context.y_mark for item in batch], dim=0)
    context = {'x': context_x, 'y': context_y, 'x_mark': context_x_mark, 'y_mark': context_y_mark}

    return {'target': target, 'context': context} 
