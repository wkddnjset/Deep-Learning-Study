# Word2Vec 모델을 간단하게 구현해봅니다.
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# matplot 에서 한글을 표시하기 위한 설정
font_name = matplotlib.font_manager.FontProperties(
                fname="/Windows/Fonts/NanumGothic.ttf"  # 한글 폰트 위치를 넣어주세요
            ).get_name()
matplotlib.rc('font', family=font_name)

# 단어 벡터를 분석해볼 임의의 문장들
sentences = ["나 고양이 좋다",
             "나 강아지 좋다",
             "나 동물 좋다",
             "강아지 고양이 동물",
             "여자친구 고양이 강아지 좋다",
             "고양이 생선 우유 좋다",
             "강아지 생선 싫다 우유 좋다",
             "강아지 고양이 눈 좋다",
             "나 여자친구 좋다",
             "여자친구 나 싫다",
             "여자친구 나 영화 책 음악 좋다",
             "나 게임 만화 애니 좋다",
             "고양이 강아지 싫다",
             "강아지 고양이 좋다"]

# 문장을 전부 합친 후 공백으로 단어들을 나누고 고유한 단어들로 리스트를 만듭니다.
word_list = " ".join(sentences).split()
word_list = list(set(word_list))

'''
#set과 list의 차이점,
set은 집합 자료형으로 중복을 허용하지 않으며, 순서가 없다.
그래서 인덱싱이 불가능하다.

하지만 집합연산에 자주 사용된다. 참고 - (https://wikidocs.net/1015)

word_list = " ".join(sentences).split()
word_list_1 = set(word_list)           #set
word_list_2 = list(set(word_list))     #list
----------------------출력-------------------------
word_list => ['나', '고양이', '좋다', '나', '강아지', '좋다', '나', '동물', '좋다', '강아지', '고양이', '동물', '여자친구', '고양이', '강아지', '좋다', '고양이', '생선', '우유', '좋다', '강아지', '생선', '싫다', '우유', '좋다', '강아지', '고양이', '눈', '좋다', '나', '여자친구', '좋다', '여자친구', '나', '싫다', '여자친구', '나', '영화', '책', '음악', '좋다', '나', '게임', '만화', '애니', '좋다', '고양이', '강아지', '싫다', '강아지', '고양이', '좋다']
word_list_1 => {'만화', '좋다', '동물', '고양이', '책', '애니', '여자친구', '우유', '눈', '강아지', '음악', '게임', '나', '싫다', '생선', '영화'}
word_list_2 => ['만화', '좋다', '동물', '고양이', '책', '애니', '여자친구', '우유', '눈', '강아지', '음악', '게임', '나', '싫다', '생선', '영화']
'''

# 문자열로 분석하는 것 보다, 숫자로 분석하는 것이 훨씬 용이하므로
# 리스트에서 문자들의 인덱스를 뽑아서 사용하기 위해,
# 이를 표현하기 위한 연관 배열과, 단어 리스트에서 단어를 참조 할 수 있는 인덱스 배열을 만듭합니다
word_dict = {w: i for i, w in enumerate(word_list)}
word_index = [word_dict[word] for word in word_list]

'''
-----------------------출력----------------------------
word_dict => {'싫다': 0, '책': 1, '우유': 3, '여자친구': 4, '게임': 6, '만화': 13, '강아지': 7, '애니': 8, '고양이': 9, '생선': 11, '좋다': 2, '눈': 12, '영화': 5, '음악': 15, '나': 14, '동물': 10}
word_index => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
'''

# 윈도우 사이즈를 1 로 하는 skip-gram 모델을 만듭니다.
# 예) 나 게임 만화 애니 좋다
#   -> ([나, 만화], 게임), ([게임, 애니], 만화), ([만화, 좋다], 애니)
#   -> (게임, 나), (게임, 만화), (만화, 게임), (만화, 애니), (애니, 만화), (애니, 좋다)
skip_grams = []

for i in range(1, len(word_index) - 1):
    # (context, target) : ([target index - 1, target index + 1], target)
    target = word_index[i]
    context = [word_index[i - 1], word_index[i + 1]]
    # (target, context[0]), (target, context[1])..
    for w in context:
        skip_grams.append([target, w])

'''
Word2Vec에는 2가지 활용 방법이 있고, skip_grams는 그 중 하나이다.
skip gram은 traget단어로부터 context단어를 예측하는 모델이다.
window size는 단어 앞, 뒤에 단어들의 갯수를 의마한다.
예)window size 5는 target 단어 앞에 5개, 뒤에 5개 단어 총 10개의 단어를 의미한다.
---------------------출력--------------------
target : 1
context :  [0, 2]
skip_grams :  [[1, 0], [1, 2]]
target :  2
context :  [1, 3]
skip_grams :  [[1, 0], [1, 2], [2, 1], [2, 3]]
target :  3
context :  [2, 4]
skip_grams :  [[1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4]]
target :  4
context :  [3, 5]
skip_grams :  [[1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5]]
target :  5
context :  [4, 6]
skip_grams :  [[1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [5, 4], [5, 6]]
target :  6
context :  [5, 7]
skip_grams :  [[1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [5, 4], [5, 6], [6, 5], [6, 7]]
target :  7
context :  [6, 8]
skip_grams :  [[1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8]]
target :  8
context :  [7, 9]
skip_grams :  [[1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9]]
target :  9
context :  [8, 10]
skip_grams :  [[1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [9, 8], [9, 10]]
target :  10
context :  [9, 11]
skip_grams :  [[1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [9, 8], [9, 10], [10, 9], [10, 11]]
target :  11
context :  [10, 12]
skip_grams :  [[1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12]]
target :  12
context :  [11, 13]
skip_grams :  [[1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13]]
target :  13
context :  [12, 14]
skip_grams :  [[1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [13, 12], [13, 14]]
target :  14
context :  [13, 15]
skip_grams :  [[1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [13, 12], [13, 14], [14, 13], [14, 15]]
'''


# skip-gram 데이터에서 무작위로 데이터를 뽑아 입력값과 출력값의 배치 데이터를 생성하는 함수
def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(data[i][0])  # target
        random_labels.append([data[i][1]])  # context word

    return random_inputs, random_labels

'''
random_inputs = []
random_labels = []
random_index = np.random.choice(range(len(skip_grams)), 20, replace=False)

for i in random_index:
    random_inputs.append(skip_grams[i][0])  # target
    random_labels.append([skip_grams[i][1]])  # context word
    
    print(random_inputs)
    print(random_labels)
    
-----------------------출력-----------------------------
[7]
[[8]]
[7, 9]
[[8], [10]]
[7, 9, 1]
[[8], [10], [0]]
[7, 9, 1, 3]
[[8], [10], [0], [2]]
[7, 9, 1, 3, 10]
[[8], [10], [0], [2], [9]]
[7, 9, 1, 3, 10, 11]
[[8], [10], [0], [2], [9], [10]]
[7, 9, 1, 3, 10, 11, 14]
[[8], [10], [0], [2], [9], [10], [15]]
[7, 9, 1, 3, 10, 11, 14, 6]
[[8], [10], [0], [2], [9], [10], [15], [7]]
[7, 9, 1, 3, 10, 11, 14, 6, 2]
[[8], [10], [0], [2], [9], [10], [15], [7], [1]]
[7, 9, 1, 3, 10, 11, 14, 6, 2, 2]
[[8], [10], [0], [2], [9], [10], [15], [7], [1], [3]]
[7, 9, 1, 3, 10, 11, 14, 6, 2, 2, 12]
[[8], [10], [0], [2], [9], [10], [15], [7], [1], [3], [11]]
[7, 9, 1, 3, 10, 11, 14, 6, 2, 2, 12, 10]
[[8], [10], [0], [2], [9], [10], [15], [7], [1], [3], [11], [11]]
[7, 9, 1, 3, 10, 11, 14, 6, 2, 2, 12, 10, 9]
[[8], [10], [0], [2], [9], [10], [15], [7], [1], [3], [11], [11], [8]]
[7, 9, 1, 3, 10, 11, 14, 6, 2, 2, 12, 10, 9, 13]
[[8], [10], [0], [2], [9], [10], [15], [7], [1], [3], [11], [11], [8], [14]]
[7, 9, 1, 3, 10, 11, 14, 6, 2, 2, 12, 10, 9, 13, 1]
[[8], [10], [0], [2], [9], [10], [15], [7], [1], [3], [11], [11], [8], [14], [2]]
[7, 9, 1, 3, 10, 11, 14, 6, 2, 2, 12, 10, 9, 13, 1, 7]
[[8], [10], [0], [2], [9], [10], [15], [7], [1], [3], [11], [11], [8], [14], [2], [6]]
[7, 9, 1, 3, 10, 11, 14, 6, 2, 2, 12, 10, 9, 13, 1, 7, 11]
[[8], [10], [0], [2], [9], [10], [15], [7], [1], [3], [11], [11], [8], [14], [2], [6], [12]]
[7, 9, 1, 3, 10, 11, 14, 6, 2, 2, 12, 10, 9, 13, 1, 7, 11, 3]
[[8], [10], [0], [2], [9], [10], [15], [7], [1], [3], [11], [11], [8], [14], [2], [6], [12], [4]]
[7, 9, 1, 3, 10, 11, 14, 6, 2, 2, 12, 10, 9, 13, 1, 7, 11, 3, 8]
[[8], [10], [0], [2], [9], [10], [15], [7], [1], [3], [11], [11], [8], [14], [2], [6], [12], [4], [7]]
[7, 9, 1, 3, 10, 11, 14, 6, 2, 2, 12, 10, 9, 13, 1, 7, 11, 3, 8, 14]
[[8], [10], [0], [2], [9], [10], [15], [7], [1], [3], [11], [11], [8], [14], [2], [6], [12], [4], [7], [13]]
'''

#########
# 옵션 설정
######
# 학습을 반복할 횟수
training_epoch = 300
# 학습률
learning_rate = 0.1
# 한 번에 학습할 데이터의 크기
batch_size = 20
# 단어 벡터를 구성할 임베딩 차원의 크기
# 이 예제에서는 x, y 그래프로 표현하기 쉽게 2 개의 값만 출력하도록 합니다.
embedding_size = 2
# word2vec 모델을 학습시키기 위한 nce_loss 함수에서 사용하기 위한 샘플링 크기
# batch_size 보다 작아야 합니다.
num_sampled = 15
# 총 단어 갯수
voc_size = len(word_list)

#########
# 신경망 모델 구성
######
inputs = tf.placeholder(tf.int32, shape=[batch_size])
# tf.nn.nce_loss 를 사용하려면 출력값을 이렇게 [batch_size, 1] 구성해야합니다.
labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
# word2vec 모델의 결과 값인 임베딩 벡터를 저장할 변수입니다.
# 총 단어 갯수와 임베딩 갯수를 크기로 하는 두 개의 차원을 갖습니다.
embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
# 임베딩 벡터의 차원에서 학습할 입력값에 대한 행들을 뽑아옵니다.
# 예) embeddings     inputs    selected
#    [[1, 2, 3]  -> [2, 3] -> [[2, 3, 4]
#     [2, 3, 4]                [3, 4, 5]]
#     [3, 4, 5]
#     [4, 5, 6]]
selected_embed = tf.nn.embedding_lookup(embeddings, inputs)

'''
inputs = tf.placeholder(tf.int32, shape=[batch_size])
labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
selected_embed = tf.nn.embedding_lookup(embeddings, inputs)

print(inputs)
print(labels)
print(embeddings)
print(selected_embed)
------------------출력------------------
Tensor("Placeholder:0", shape=(20,), dtype=int32)
Tensor("Placeholder_1:0", shape=(20, 1), dtype=int32)
<tf.Variable 'Variable:0' shape=(16, 2) dtype=float32_ref>
Tensor("embedding_lookup:0", shape=(20, 2), dtype=float32)

----------------------------------------------
*******************************************
#tf.nn.embedding_lookup
하나의 단어를 벡터 공간상의 하나의 점으로 맵핑해주는 명령어 인듯..
embedding_lookup(
    params,
    ids,
    partition_strategy='mod',
    name=None,
    validate_indices=True,
    max_norm=None
******************************************
'''

# nce_loss 함수에서 사용할 변수들을 정의합니다.
nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))

# nce_loss 함수를 직접 구현하려면 매우 복잡하지만,
# nce_loss??????
# 함수를 텐서플로우가 제공하므로 그냥 tf.nn.nce_loss 함수를 사용하기만 하면 됩니다.
loss = tf.reduce_mean(
            tf.nn.nce_loss(nce_weights, nce_biases, labels, selected_embed, num_sampled, voc_size))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


'''
#########
# 신경망 모델 학습
######
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(1, training_epoch + 1):
        batch_inputs, batch_labels = random_batch(skip_grams, batch_size)

        _, loss_val = sess.run([train_op, loss],
                               feed_dict={inputs: batch_inputs,
                                          labels: batch_labels})

        if step % 10 == 0:
            print("loss at step ", step, ": ", loss_val)

    # matplot 으로 출력하여 시각적으로 확인해보기 위해
    # 임베딩 벡터의 결과 값을 계산하여 저장합니다.
    # with 구문 안에서는 sess.run 대신 간단히 eval() 함수를 사용할 수 있습니다.
    trained_embeddings = embeddings.eval()


#########
# 임베딩된 Word2Vec 결과 확인
# 결과는 해당 단어들이 얼마나 다른 단어와 인접해 있는지를 보여줍니다.
######
for i, label in enumerate(word_list):
    x, y = trained_embeddings[i]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom')

plt.show()
'''