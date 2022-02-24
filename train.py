from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm
from functools import partial

import os
import torch
import torch.nn as nn
import pandas as pd
import argparse

from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import NLI_Dataset, NLI_Explain_Dataset
from utils.random_seed import seed_everything
from utils.loss import FocalLoss
from utils.collate_functions import collate_to_max_length
from utils.mk_data import get_dataset, basic_preprocess
from model import ExplainableModel



def train_baseline(dataset, args):
    # Dataset Split
    train_df, valid_df = train_test_split(
                            dataset,
                            test_size=args.split_ratio,
                            shuffle=True,
                            stratify=dataset['label']
                        )
    train_df.reset_index(inplace = True)
    valid_df.reset_index(inplace = True)   

    # Download Model & Tokenizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_name_or_path = "klue/roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
    model = model.to(device)

    # Dataset
    train_data = NLI_Dataset(train_df, tokenizer, args.max_length)
    valid_data = NLI_Dataset(valid_df, tokenizer, args.max_length)

    # DataLoader
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )

    if args.loss == 'cross':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'fl':
        criterion = FocalLoss(gamma=args.fl_gamma)
    
    for epoch in tqdm(range(args.epochs)):
        train_loss, train_acc = 0, 0
        valid_loss, valid_acc = 0, 0

        total_step=0
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()

            label = batch['label'].to(device)
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            
            pred_logits = model(**inputs)
            preds = torch.argmax(pred_logits[0], dim=-1)

            loss = criterion(pred_logits[0], label)
            acc = torch.sum(preds == label.data)

            train_loss += loss.item()
            train_acc += acc
            total_step += 1

            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"Epoch {epoch+1} - Train Loss: {train_loss/total_step:.4f} | Train Acc: {(train_acc/total_step)/32:.4f}")

        total_step = 0
        model.eval()
        for _, batch in enumerate(tqdm(valid_dataloader)):
            label = batch['label'].to(device)
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
            }

            with torch.no_grad():
                pred_logits = model(**inputs)
                preds = torch.argmax(pred_logits[0], dim=-1)

                loss = criterion(pred_logits[0], label)
                acc = torch.sum(preds == label.data)
            total_step += 1

            valid_loss += loss.item()
            valid_acc += acc
        print(f"Epoch {epoch+1} - Valid Loss: {valid_loss/total_step:.4f} | Valid Acc: {(valid_acc/total_step)/32:.4f}")

        if epoch % args.save_steps == 0:
            torch.save(model, f"{args.save_path}epoch_{epoch+1}.pt")
        
def train_explain(dataset, args):
    # Dataset Split
    train_df, valid_df = train_test_split(
                            dataset,
                            test_size=args.split_ratio,
                            shuffle=True,
                            stratify=dataset['label']
                        )
    train_df.reset_index(inplace = True)
    valid_df.reset_index(inplace = True)   

    # Download Model & Tokenizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_name_or_path = "klue/roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = ExplainableModel(model_name_or_path)
    model = model.to(device)

    # Dataset
    train_data = NLI_Explain_Dataset(train_df, tokenizer, args.max_length)
    valid_data = NLI_Explain_Dataset(valid_df, tokenizer, args.max_length)

    # DataLoader
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_to_max_length, fill_values=[1,0,0]),
        drop_last=False
    )
    valid_dataloader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_to_max_length, fill_values=[1,0,0]),
        drop_last=False
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps
    )

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )

    if args.loss == 'cross':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'fl':
        criterion = FocalLoss(gamma=args.fl_gamma)
    
    for epoch in tqdm(range(args.epochs)):
        train_loss, train_acc = 0, 0
        valid_loss, valid_acc = 0, 0

        total_step=0
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()

            input_ids, labels, length, start_indexs, end_indexs, span_masks = batch
            y = labels.view(-1).to(device)

            input_ids = input_ids.to(device)
            start_indexs = start_indexs.to(device)
            end_indexs = end_indexs.to(device)
            span_masks = span_masks.to(device)

            y_hat, a_ij = model(input_ids, start_indexs, end_indexs, span_masks)

            predict_scores = F.softmax(y_hat, dim=1)
            
            # compute loss
            ce_loss = criterion(predict_scores, y)
            reg_loss = 1.0 * a_ij.pow(2).sum(dim=1).mean()
            loss = ce_loss + reg_loss

            # compute acc
            preds = torch.argmax(predict_scores, dim=-1)
            acc = torch.sum(preds == y)

            train_loss += loss.item()
            train_acc += acc
            total_step += 1

            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"Epoch {epoch+1} - Train Loss: {train_loss/total_step:.4f} | Train Acc: {(train_acc/total_step)/32:.4f}")

        total_step = 0
        model.eval()
        for _, batch in enumerate(tqdm(valid_dataloader)):
            input_ids, length, labels, start_indexs, end_indexs, span_masks = batch
            y = labels.view(-1).to(device)

            input_ids = input_ids.to(device)
            start_indexs = start_indexs.to(device)
            end_indexs = end_indexs.to(device)
            span_masks = span_masks.to(device)

            with torch.no_grad():
                y_hat, a_ij = model(input_ids, start_indexs, end_indexs, span_masks)
                predict_scores = F.softmax(y_hat, dim=1)

                # compute loss
                ce_loss = criterion(predict_scores, y)
                reg_loss = 1.0 * a_ij.pow(2).sum(dim=1).mean()
                loss = ce_loss + reg_loss

                # compute acc
                preds = torch.argmax(predict_scores, dim=-1)
                acc = torch.sum(preds == y)

            total_step += 1

            valid_loss += loss.item()
            valid_acc += acc
        print(f"Epoch {epoch+1} - Valid Loss: {valid_loss/total_step:.4f} | Valid Acc: {(valid_acc/total_step)/32:.4f}")

        if epoch % args.save_steps == 0:
            if not os.path.isdir(args.save_path):
                os.mkdir(args.save_path)
            torch.save(model, f"{args.save_path}epoch_{epoch+1}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # -- choose model
    parser.add_argument('--mode', type=str, default='baseline', help="select one between baseline and explain (default: baseline)")

    # -- dataset
    parser.add_argument('--split_ratio', type=float, default=.2, help="train / valid split ratio (default: 0.2)")
    
    # -- training arguments
    parser.add_argument('--epochs', type=int, default=10, help="number of epochs to train (default: 10)")
    parser.add_argument('--batch_size', type=int, default=24, help="batch size (default: 24)")
    parser.add_argument('--lr', type=float, default=2e-5, help="learning rate (default: 2e-5)")
    parser.add_argument('--beta1', type=float, default=0.9, help="adamw's beta1 parameter (default: 0.9)")
    parser.add_argument('--bata2', type=float, default=0.999, help="adamw's beta2 parameter (default: 0.999)")
    parser.add_argument('--eps', type=float, default=1e-9, help="adamw's epsilon parameter (default: 1e-9)")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="strength of weight decay (default: 0.01)")
    parser.add_argument('--warmup_steps', type=int, default=500, help="number of warmup steps for learning rate scheduler (default: 500)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="gradient accumulation steps (default: 1)")
    parser.add_argument('--loss', type=str, default='cross', help="training objective (default: cross entropy loss)")
    parser.add_argument('--fl_gamma', type=float, default=1.0, help="amma value in Focal Loss (default: 1.0)")
    parser.add_argument('--max_length', type=int, default=128, help="max length of tensor (default: 128)")
    
    # -- utils
    parser.add_argument('--save_path', type=str, default='./save_model/', help="path about save model (default: ./save_model/")
    parser.add_argument('--save_steps', type=int, default=1, help="save model per save_steps (default: 1)")
    parser.add_argument('--seed', type=int, default=42, help="random seed (default: 42)")
    
    args = parser.parse_args()
    seed_everything(42)

    # Call Dataset
    original_data = get_dataset('./data/open/train_data.csv')
    klue_dev = get_dataset('./data/klue_dev.csv')
    kor_nli = get_dataset('./data/kor_nli_valid.csv')
    train = pd.concat([original_data, klue_dev, kor_nli])

    # Preprocessing
    train['premise'] = basic_preprocess(train['premise'])
    train['hypothesis'] = basic_preprocess(train['hypothesis'])

    if args.mode == 'baseline':
        train_baseline(train, args)
    elif args.mode == 'explain':
        train_explain(train, args)
    