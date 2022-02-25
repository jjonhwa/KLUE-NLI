# KLUE-NLI

## 대회 설명
- Premise문장을 참고하여 hypothesis 문장의 참, 거짓, 중립을 판별해야한다.

```
premise: 씨름은 상고시대로부터 전해져 내려오는 남자들의 대표적인 놀이로서, 소년이나 장정들이 넓고 평평한 백사장이나 마당에서 모여 서로 힘과 슬기를 겨루는 것이다.
hypothesis: 씨름의 여자들의 놀이이다.

label: contradiction
```

### 평가방법
- **Accuracy**
- Public: Test Data 중 Random sampling한 60%
- Private: 전체 Test Data

### Dataset
- Train: 24998
- Test: 1666

### Hardware
- `GPU: Colab Pro P100`

## 실행

### Install Requirements and Data Unzip
```python
pip install -r requirements.txt
unzip -q './data/open.zip' -d './data'
```

### Run
```python
# Train
python train.py --explain

# Inference
python inference.py --explain
```

## Code

```
+- data
|   +- klue-nli-v1.1.tar.gz (klue_dev 제작에 사용)
|   +- klue_dev.csv (KLUE OFFICAL dev dataset)
|   +- kor_nli_valid.csv (kakaobrain - kornli dataset) 
|   +- open.zip (Original Dataset)
+- EDA
|   +- EDA.ipynb (Dataset EDA)
|   +- aug_dataset.ipynb (Dataset augmentation => klue_dev.csv & kor_nli_valid.csv)
+- utils
|   +- collate_functions.py
|   +- loss.py
|   +- mk_data.py
|   +- nlpdata_eda.py
|   +- random_seed.py
+- requirements.txt
+- dataset.py
+- model.py
+- train.py
+- train_kfold.py
+- inference.py
```

## Core Strategy
- **KLUE/RoBERTa-large + Classifier Head with Hyperparmeter Tuning** (Baseline으로 지정) 
  - KLUE/RoBERTa-large를 backbone으로 활용한 NLI 모델 적용
  - 다양한 HyperParameter Tuning 실험을 통한 성능 향상
- **Self-Explaining Structures Improve NLP Models** [(Paper Review 참고)](https://jjonhwa.github.io/booststudy/2022/02/21/booststudy-paper-Self_Explaining_Structures_Improve_NLP_Models/)
  -  **KLUE/RoBERTa-large를 backbone**으로 활용 (intermediate layer)
  -  **SIC layer추가 (backbone model에서의 output layer들 사이의 조합 생성) => span 정보 전달**
  -  **Interpreatation layer를 추가 => span에서의 가중치 추출**
  -  **추출된 가중치와 span 정보를 weighted sum하여 최종 output 출력**
-  **외부 Dataset 정제 및 활용**
  - KLUE OFFICIAL Dev Dataset 활용
  - KakaoBrain KorNLI Dataset 중 Human Trnaslated Data만 활용 (Original Dataset과 유사한 Data 추가)
-  **Out of Fold Ensemble**
  - Stratified KFold를 Ensemble 진행
  - Baseline + Explaining Model Ensemble

## 결과
||Single Baseline(train:valid=8:2)|Single Self-Explaining(train:valid=8:2)|
|---|---|---|
|Accuracy|0.872|0.864|

||Baseline KFold|Self-Explaining KFold|Baseline + Self-Explaining|
|---|---|---|---|
|Accuracy|0.888|||

## 과제
- Self-Explaining에 대한 Error Analysis
  - 어떤 부분에 weight를 주어 예측을 진행했는데 출력해보기
