import json
import pandas as pd


def read_json(file_path):
    with open(file_path) as f:
        return json.load(f)

def create_pandas(data: dict) -> pd.DataFrame:
    guid = []
    source = []
    premise = []
    hypothesis = []
    label = []
    for entry in data:
        guid.append(entry["guid"])
        if "genre" in entry.keys():
            source.append(entry["genre"])
        else:
            source.append(entry["source"])
        premise.append(entry["premise"])
        hypothesis.append(entry["hypothesis"])
        if entry["gold_label"] == "entailment":
            label.append('entailment')
        elif entry["gold_label"] == "contradiction":
            label.append('contradiction')
        elif entry["gold_label"] == "neutral":
            label.append('neutral')

    df = pd.DataFrame(
        data={
            "guid": guid,
            "source": source,
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label,
        }
    )
    return df

def get_dataset(path):
    dataset = pd.read_csv(path, encoding='utf-8')
    return dataset

def basic_preprocess(series: pd.Series):     
    series = series.str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣 0-9]', '')
    return series