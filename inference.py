import torch
from transformers import AutoTokenizer

from torch.utils.data import DataLoader

import os
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F

from tqdm.auto import tqdm
from functools import partial

from dataset import NLI_Dataset, NLI_Explain_Dataset, num_to_label
from utils.random_seed import seed_everything
from utils.collate_functions import collate_to_max_length
from utils.mk_data import get_dataset, basic_preprocess
from model import ExplainableModel

def inference_baseline(dataset, args):

    # Download Model & Tokenizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_name_or_path = "klue/roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = torch.load(args.model_path).to(device)

    # Dataset
    test_data = NLI_Dataset(dataset, tokenizer, args.max_length)

    # DataLoader
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Predict
    output_pred = []
    output_prob = []
    model.eval()

    for _, data in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids = data['input_ids'].to(device),
                attention_mask = data['attention_mask'].to(device)
            )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)
    
    pred_answer, output_prob = np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()
    return pred_answer, output_prob

def inference_explain(dataset, args):
    
    # Download Model & Tokenizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_name_or_path = "klue/roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = ExplainableModel(model_name_or_path)
    model = torch.load(args.model_path).to(device)

    # Dataset
    test_data = NLI_Explain_Dataset(dataset, tokenizer, args.max_length)

    # DataLoader
    dataloader = DataLoader(
        test_data,
        batch_size=24,
        shuffle=False,
        collate_fn=partial(collate_to_max_length, fill_values=[1,0,0]),
        drop_last=False
    )

    # Predict
    output_pred = []
    output_prob = []
    model.eval()

    for _, data in enumerate(tqdm(dataloader)):
        input_ids, labels, length, start_indexs, end_indexs, span_masks = data

        input_ids = input_ids.to(device)
        start_indexs = start_indexs.to(device)
        end_indexs = end_indexs.to(device)
        span_masks = span_masks.to(device)

        with torch.no_grad():
            y_hat, a_ij = model(input_ids, start_indexs, end_indexs, span_masks)

        logits = y_hat.detach().cpu().numpy()
        prob = F.softmax(y_hat, dim=1).detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)
    
    pred_answer, output_prob = np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()
    return pred_answer, output_prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # -- choose model
    parser.add_argument('--mode', type=str, default='baseline', help="select one between baseline and explain (default: baseline)")
    
    # -- model path
    parser.add_argument('--model_path', type=str, default='./save_model/epoch_10.pt', help="trained model path")
    
    # -- inference arguments
    parser.add_argument('--max_length', type=int, default=128, help="max length of tensor (default: 128)")
    parser.add_argument('--batch_size', type=int, default=24, help="batch size (default: 24)")
    
    # -- save predict
    parser.add_argument('--save_path', type=str, default='./submission/', help="save path for submission")
    
    args = parser.parse_args()
    seed_everything(42)

    # Call Dataset
    test = get_dataset('./data/open/test_data.csv')

    # Preprocessing
    test['premise'] = basic_preprocess(test['premise'])
    test['hypothesis'] = basic_preprocess(test['hypothesis'])

    # Predict
    if args.mode == 'baseline':
        pred_answer, output_prob = inference_baseline(test, args)
    elif args.mode == 'explain':
        pred_answer, output_prob = inference_explain(test, args)

    # Save
    answer = num_to_label(pred_answer)

    dataframe = pd.DataFrame(answer, columns=['index', 'label'])
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)
    dataframe.to_csv(os.path.join(args.save_path, "submission.csv"), index=False)  
        