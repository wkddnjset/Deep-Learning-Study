import tensorflow as tf
from PIL import Image
import urllib.request
import urllib
import io
import numpy as np
from resizeimage import resizeimage
import matplotlib.pyplot as plt

#ImageNet에서 Cat 사진 URL 파싱하기
all_data_url = urllib.request.Request("http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02121620")
all_data = urllib.request.urlopen(all_data_url).read()

#\r\n으로  텍스트 분리
all_data = all_data.decode().split("\r\n")



# 함수정의
def parsing(data):
    cat_test = []
    # i를 0부터 all_data길이 만큼 반복해서 index 값을 imgUrl에 넣는다
    #for i in range(0, len(all_data)):
    for i in range(0, len(data)):
        imgUrl = (all_data[i])
        try:
            with urllib.request.urlopen(imgUrl) as url:
                f = io.BytesIO(url.read())
                img = Image.open(f)
                a = np.array(img)
                img = img.resize((400, 400), Image.ANTIALIAS)
                a = np.array(img)
                b = a.ndim

                # 3차원인 배열만 저장하기
                if b == 3:
                    # cat_test 배열에 순서대로 데이터 쌓기
                    cat_test.append(a)
                    b = np.array(cat_test)

                #print(cover)
                #b = np.array(cover)
                #print(a.shape)
                #print(imgUrl)
        except:
            continue
    return b

a = parsing([0,1,2])
a = a.astype('float32')
print(a.shape)
a.tofile('cat_all_test.dat')
