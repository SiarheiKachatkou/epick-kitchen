import numpy as np
import pandas as pd
import torch

def init_metrics(epochs):
    nan_data = np.full((epochs + 1, 6), np.nan)
    metrics = pd.DataFrame(data=nan_data,
                           columns=['train_acc', 'val_acc', 'test_acc', 'train_loss', 'val_loss', 'test_loss'],
                           copy=True)
    
    return metrics


def evaluate(metrics,model,criterion,train_loader, val_loader, test_loader, epoch):

    model.eval()

    for column,loader in zip(['train','val','test'],[train_loader,val_loader,test_loader]):
        losses=[]
        accs=[]
        for snippets,targets in loader:
            logits=model(snippets)
            loss=criterion(logits, targets)
            losses.append(loss.detach().cpu().numpy())
            preds=torch.argmax(logits,dim=1)
            acc=torch.eq(targets,preds).float().mean()
            accs.append(acc.detach().cpu().numpy())
        metrics[column+'_acc'][epoch]=np.mean(accs)
        metrics[column + '_loss'][epoch] = np.mean(losses)

    return metrics



