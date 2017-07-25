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

### Skip-Gram model

 예를 들어, 아래와 같은 데이터셋이 주어졌다고 가정해보자.

         the quick brown fox jumped over the lazy dog
        
먼저 콘텍스트를 정의해야한다. 콘텍스트는 어떤 형태로든 정의할 수 있지만, 사람들은 보통 구문론적 콘텍스트를 정의한다. 이번 구현에서는 간단하게, 콘텍스트를 타겟 단어의 왼쪽과 오른쪽 단어들의 윈도우로 정의한다. 윈도우 사이즈를 1로하면 (context, target) 쌍으로 구성된 아래와 같은 데이터셋을 얻을 수 있다.

        ([the, brown], quick), ([quick fox], brown), ([brown, jumped], fox), …
        
skip-gram 모델은 타겟 단어로부터 콘텍스트를 예측한다는 점을 상기하라. 따라서 우리가 해야할 일은 ‘quick’이라는 타겟단어로부터 콘텍스트 ‘the’와 ‘brown’을 예측하는 것이다. 따라서 우리 데이터셋은 아래와 같은 (input, output) 쌍으로 표현할 수 있다.

        (quick, the), (quick, brown), (brown, quick), (brown, fox), …
        
목적 함수는 전체 데이터셋에 대해 정의될 수 있다. 하지만, 우리는 보통 이를 stochastic gradient descent(SGD) 방식으로 한번에 하나의 예제에 대해서 최적화한다. 또는 ‘minibatch’라고 하는 일정 개수의 배치로 묶어서 최적화한다. 일반적으로 배치 사이즈는 16 <= batch_size <= 512)이다.

## 참고

[Word2Vec Tutorial](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) (Word2Vec에 대한 자세한 설명)

[영화추천](http://yujuwon.tistory.com/entry/word2vec%EC%9C%BC%EB%A1%9C-%EC%98%81%ED%99%94-%EC%B6%94%EC%B2%9C-%ED%95%98%EA%B8%B0) (Word2Vec을 사용한 영화추천)

[솔라리스 인공지능 연구실](http://solarisailab.com/archives/374) (텐서플로우를 이용한 자연어처리)
