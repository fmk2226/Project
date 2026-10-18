import torch

def accuracy(output,target):
    """ Evaluate accuracy of one epoch
    
    Args:
        output: Class logits with shape [batch_size,num_classes].
        target: Integar Class labels with shape [batch_size].

    Returns:
        acc: Accuracy
    """
    with torch.no_grad():
        pred=torch.argmax(output,dim=1)
        assert pred.shape[0]==len(target)
        correct,acc=0,0.0
        correct+=torch.sum(pred==target).item()
        acc=correct/len(target)
        return acc

def top_k_acc(output,target,k=3):
    """ Evaluate top-k accuracy for a batch
    
    Args:
        output: Class logits with shape [batch_size,num_classes].
        target: Integar Class labels with shape [batch_size].
        k: Number of highest-scoring classes to consider. Defaults to 3.

    Returns:
        Top-k accuracy
    """
    with torch.no_grad():
        pred=torch.topk(output,k,dim=1)[1]
        assert pred.shape[0]==len(target)
        correct=0
        for i in range(k):
            correct+=torch.sum(pred[:,i]==target).item
        acc=correct/len(target)
        return acc