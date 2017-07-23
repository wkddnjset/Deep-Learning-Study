# Word2Vec

    - 단어를 벡터화 함으로써 각 단어들 사이에 유사도를 측정 할 수있다.
    - 각 단어들의 벡터 연산을 통해 추론을 내릴 수있다.
    
## 목표
- Word2Vec에 대한 이해
- skip gram neural network에 대한 이해

## 정리

### 모델

- 우선 우리가 무엇을 해야하는지에 대한 이해부터 시작해 봅시다. Word2vec은 여러분들이 다른 머신러닝에서 보았을 수도 있는 트릭을 사용합니다. 우리는 하나의 히든 레이어를 가진 단순한 뉴럴넷을 통해 숨겨진 신경 네트워크를 통해 특정한 임무를 수행할 수 있는 단순한 신경 회로를 훈련시킬 것입니다. 하지만 우리가 훈련한 임무를 위해 신경 회로망을 사용하지는 않습니다. 대신에, 그 목표는 실제로 숨겨져 있는 것들의 무게를 배우는 것입니다. 우리가 배우려고 하는 "단어 벡터"라는 것이 사실입니다.

## 참고

[Word2Vec Tutorial](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) (Word2Vec에 대한 자세한 설명)

[영화추천](http://yujuwon.tistory.com/entry/word2vec%EC%9C%BC%EB%A1%9C-%EC%98%81%ED%99%94-%EC%B6%94%EC%B2%9C-%ED%95%98%EA%B8%B0) (Word2Vec을 사용한 영화추천)
