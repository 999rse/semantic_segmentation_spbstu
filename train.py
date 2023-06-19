from imports import *

def train(dataloader, val_dataloader, model, loss_fn, optimizer, epochn, device, optimSch=False):
    '''
    The function for train model/
    
    Input: 
    + dataloader - dataloader of train dataset
    + val_dataloader - dataloader of validation dataset
    + model_unet - Instance of model
    + loff_fn - loss function
    + optimizer - optimizer
    + epochn - count of epoch's
    + device - device CPU or GPU
    + optimSch - Schedluer for optimizer (by default: False)
        
    Return: None
    '''
    
    # Schedluer for optimizer
    if optimSch:
        optimSchstep = ReduceLROnPlateau(optimizer, mode='min', factor=0.1)

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        
        #train_loss_history.append(loss)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Epoch: {epochn} | Train loss: {loss:.7f} | Batch: [{current:>5d}/{size:>5d}]")
            
    
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    
    size_v = len(dataloader.dataset)
    model.eval()

    with torch.no_grad():
        for X_v, y_v in val_dataloader:
            X_v = X_v.to(device)
            y_v = y_v.to(device)
            preds = torch.sigmoid(model(X_v))
            
            loss_v = loss_fn(preds, y_v)

            if optimSch:
                optimSchstep.step(loss_v)
            
            preds = (preds > 0.5).float()
            num_correct += (preds == y_v).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y_v).sum()) / (
                (preds + y_v).sum() + 1e-8
            )
    
    
    print(f"Epoch: {epochn} | Score_valid: {dice_score/len(val_dataloader)}")


