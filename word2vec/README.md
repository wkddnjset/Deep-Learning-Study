# Word2Vec

    - 단어를 벡터화 함으로써 각 단어들 사이에 유사도를 측정 할 수있다.
    - 각 단어들의 벡터 연산을 통해 추론을 내릴 수있다.
    
## 목표
- Word2Vec에 대한 이해
- skip gram neural network에 대한 이해

## 정리

### Word2Vec 활용 방법

- CBOW(Continuous Bag-Of-Words) model – 소스 컨텍스트에서 타겟 단어를 예측한다.
    - 예를 들어, ‘the cat sits on the’라는 소스 컨텍스트로부터 ‘mat’이라는 타겟 단어를 예측한다. CBOW는 smaller 데이터셋에 적합하다.

 

- Skip-Gram model – 타겟 단어로부터 소스 컨텍스트를 예측한다.
    - 예를 들어, ‘mat’이라는 타겟 단어로부터 ‘the cat sits on the’라는 소스 컨텍스트를 예측한다. Skip-Gram model은 larger 데이터셋에 적합하다. 따라서, 앞으로 Ski-Gram model에 초점을 맞춰서 설명을 진행할 것이다.

## 참고

[Word2Vec Tutorial](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) (Word2Vec에 대한 자세한 설명)

[영화추천](http://yujuwon.tistory.com/entry/word2vec%EC%9C%BC%EB%A1%9C-%EC%98%81%ED%99%94-%EC%B6%94%EC%B2%9C-%ED%95%98%EA%B8%B0) (Word2Vec을 사용한 영화추천)

[솔라리스 인공지능 연구실](http://solarisailab.com/archives/374) (텐서플로우를 이용한 자연어처리)
