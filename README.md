# Advanced-Text-Tutorial-with-BERT
## BERT
BERT 및 Transformer 인코더 아키텍처는 NLP의 다양한 작업에서 큰 성공을 거두었습니다. 딥러닝 모델에 사용하기에 적합한 자연언어의 벡터 공간 표현을 계산한다. BERT 모델 제품군은 Transformer 인코더 아키텍처를 사용하여 전후의 모든 토큰의 전체 컨텍스트에서 입력 텍스트의 각 토큰을 처리하므로 : bidirectional encoder representations from Transformers

# 실습해볼 내용
## 1. Classify text with BERT
 : TF-hub의 pretrained-BERT 모델을 활용하여 IMDB의 영화리뷰 데이터를 fine-tuning하여 evaluate 해보는 과정을 실습해볼 것이다.
 - 사용데이터
 ```python
 url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url,
                                  untar = True, cache_dir='.',
                                  cache_subdir='')
 ```
- 활용한 TF-hub 모델
```python
# 전처리 모델
bert_preprocess_model = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

# BERT 모델
bert_model = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2",
    trainable=True)
```
- 튜토리얼에서 사용할 small-bert 아키텍처<br>
 <img src='./Classify text with BERT/img1.png' width=250><br>

- 결과<br>
 <img src='./Classify text with BERT/img2.png' width=650><br>

<br>

### 스터디를 위해 예제의 일부를 변형하거나 추가적인 코드를 사용될 수 있습니다.
이 레파지토리에서 실습해볼 예제의 출처는 아래와 같습니다 <br> https://www.tensorflow.org/text/tutorials/classify_text_with_bert?hl=en
