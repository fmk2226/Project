import torch
import pandas as pd
from pathlib import Path

from data_loader.data_loaders import LeafDataLoader
from model.model import LeafNet

ROOT=Path(__file__).resolve().parent
DATA_DIR=ROOT.parents[1]/'data'/'classify_leaves'
MODEL_DIR=ROOT/'saved'/'model_weights'
OUTPUT_DIR=ROOT/'saved'/'submissions'

MODEL_NAME='resnet50d'
FOLDS=[0]

config={
    'check_points':[str(MODEL_DIR/MODEL_NAME/f'fold{f}'/'model_best.pth') for f in FOLDS],
    'data_loader':{'csv_file':str(DATA_DIR/'test.csv'),'data_dir':str(DATA_DIR),
                   'batch_size':512,'num_workers':8,'Training':False},
    'submission_path':str(OUTPUT_DIR/'submission.csv')
}

def load_model(checkpoint_path,device):
    """Load the model from a training checkpoint.
    
    Args:
        checkpoint_path: Path to a checkpoint saved by Trainer._save_checkpoint.
        device: Device on which to load the model.

    Returns:
        model: Loaded model in evaluation mode.
        label_to_id: Mapping from class labels to IDs.
    """
    checkpoint=torch.load(checkpoint_path,map_location=device)
    # Load the model architecture from training, while skip downloading ImageNet weights
    model=LeafNet(**{**checkpoint['config']['arch'],'pretrained':False})
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model,checkpoint['label_to_id']


def main(config):
    device=torch.device('cuda')
    models,label_to_id=[],None
    for path in config['check_points']:
        model,mapping=load_model(path,device)
        models.append(model)
        label_to_id=mapping

    id_to_label={idx:label for label,idx in label_to_id.items()}
    data_loader=LeafDataLoader(**config['data_loader'])

    preds=[]
    for X in data_loader:
        with torch.no_grad():
            X=X.to(device)
            y=[model(X) for model in models]
            # calculate the average probability from logits
            probs=sum(torch.softmax(logit,dim=1) for logit in y)/len(models)
            preds.extend(probs.argmax(dim=1).tolist())

    submission=pd.read_csv(config['data_loader']['csv_file'])
    submission['label']= [id_to_label[pred] for pred in preds]
    submission_path=Path(config['submission_path'])
    submission_path.parent.mkdir(parents=True,exist_ok=True)
    submission.to_csv(submission_path,index=False)
    print(f"Saved {len(submission)} predictions to {submission_path}")

if __name__=='__main__':
    main(config)