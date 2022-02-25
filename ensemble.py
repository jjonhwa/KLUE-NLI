import pandas as pd
import numpy as np

import argparse
import os

from dataset import num_to_label

def ensemble(args):
    files = os.listdir(args.prob_dir)

    datas = []
    for f in files:
        if f.endswith(".csv"):
            file_path = os.path.join(args.prob_dir, f)
            dataframe = pd.read_csv(file_path)
            datas.append(dataframe)

    ensemble_prob = []
    for data in datas:
        prob = []
        try: # Data Probability가 list형태로 올바르게 삽입되어있을 경우
            check = eval(data['probability'][0]) # check
            for i in range(len(data)):
                prob.append(eval(data['probability'][i]))
        except: # comma가 아닌 space로 나뉘어져 있을 경우
            for i in range(len(data)):
                probability = data['probability'][i][1:-1]
                
                probability_list = []
                for p in probability.split():
                    probability_list.append(float(p))
                
                prob.append(probability_list)
        
        ensemble_prob.append(prob)
    
    oof_pred_mean = np.mean(ensemble_prob, axis=0)
    single_label = np.argmax(oof_pred_mean, axis=-1)
    answer = num_to_label(list(single_label))

    all_prob = []
    for pred in oof_pred_mean:
        all_prob.append(pred)
    
    return answer, all_prob
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # -- dir
    parser.add_argument("--prob_dir", type=str, default="./result/")
    parser.add_argument('--save_path', type=str, default='./ensemble/', help="save path for output_prob")
    parser.add_argument('--submission_path', type=str, default='./submission/', help="save path for submission")
    
    args = args = parser.parse_args()

    answer, all_prob = ensemble(args)
    dataframe = pd.DataFrame(answer, columns=['index', 'label'])
    dataframe['probability'] = all_prob
    submission = dataframe[['index', 'label']]

    if not os.path.isdir(args.save_path):
        os.mkdir(os.path.join('./', args.save_path))
    if not os.path.isdir(args.submission_path):
        os.mkdir(os.path.join('./', args.submission_path))
    
    submission.to_csv(os.path.join(args.submission_path, "submission.csv"), index=False)  
    dataframe.to_csv(os.path.join(args.save_path, "prob.csv"), index=False)