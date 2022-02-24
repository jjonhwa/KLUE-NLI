######################## implement in termial ##############################
################ You have to specify the file path #########################
# curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1zib1GI8Q5wV08TgYBa2GagqNh4jyfXZz" > /dev/null
# curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1zib1GI8Q5wV08TgYBa2GagqNh4jyfXZz" -o my_data/wiki_20190620_small.txt
# pip install konlpy
# bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
# pip install transformers
############################################################################

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def corpus_statistic_with_graph(texts: list, tokenizer_type: str, tokenizer = None) -> dict :
    '''
    입력 text(corpus)에 대한 다음의 통계값을 dict형태로 출력하고 그래프로 표현한다.
    - 전체 문장 개수
    - 최소, 최대 문장 길이
    - 문장 길이의 평균값, 중앙값
    - unique한 전체 토큰 개수
    - 문장 내의 최소 및 최대 토큰 개수
    - 문장별 토큰 개수의 평균, 중앙값
    - 코퍼스에서 가장 많이 나온 상위 10개 단어(= 토큰)

    - 단어들을 빈도순으로 정렬하여 등장 횟수를 그린 그래프
    - 문장 내 단어 개수 분포에 대한 히스토그램

    tokenizer -> if tokenizer_type == 'morph' : Kkma() or Mecab() etc
              -> if tokenizer_type == 'wordpiece' : trained_tokenizer (like trained BertWordPieceTokenizer)
    if you want to know how to train the wordpiece model, Go to this path (/opt/ml/mine_uk/전처리 연습/wp_tokenizer.ipynb)
    '''
    assert tokenizer_type in ['word', 'morph', 'syllable', 'wordpiece'], '정의되지 않은 tokenizer_type입니다.'
    if tokenizer :
        tokenizer = tokenizer

    texts_lens = []
    word_list = []
    word_lens_per_sent = []

    for text in texts :
        texts_lens.append(len(text))

        if tokenizer_type == 'word' : # 어절 단위
            words = text.split(' ')
        elif tokenizer_type == 'syllable' : # 음절 단위
            words = list(text)
        elif tokenizer_type == 'morph' : # 형태소 단위
            words = tokenizer.morphs(text)
        elif tokenizer_type == 'wordpiece' : # Wordpiece 단위
            words = tokenizer.tokenize(text) # BertWordPieceTokenizer일 경우 이므로 다른 wordpiece를 사용할 경우 수정.
        word_list.extend(words)
        word_lens_per_sent.append(len(words))

    counter = Counter(word_list)
    word_list = counter.most_common(n = 10) # 빈도수 기준 상위 10개 단어
    word_list = [word[0] for word in word_list] # 단어만 순서대로 저장

    # 등장 빈도순으로 단어를 시각화
    sorted_words = sorted(counter.items(), key = lambda item: (-item[1], item[0]))
    sorted_frequency_logscale = [np.log10(el[1]) for el in sorted_words]
    indices = np.arange(len(sorted_frequency_logscale))
    plt.plot(indices, sorted_frequency_logscale)
    plt.ylabel('log10(frequency)', fontsize = 16)
    plt.show()
    plt.cla()

    # 문장 단어 개수에 대한 히스토그램 시각화
    plt.hist(word_lens_per_sent, bins = 20)
    plt.xticks(np.arange(0, 250, 50))
    plt.xlabel('# of words/sent.')
    plt.show()


    return {'texts' : len(texts),
            'num_unique_words' : len(counter),
            'maxinum' : np.max(texts_lens),
            'minimum' : np.min(texts_lens),
            'mean' : np.mean(texts_lens),
            'median' : np.median(texts_lens),
            'word_maximum' : np.max(word_lens_per_sent),
            'word_minimum' : np.min(word_lens_per_sent),
            'word_mean' : np.mean(word_lens_per_sent),
            'word_median' : np.median(word_lens_per_sent),
            'TOP10_word' : word_list[:10]}

