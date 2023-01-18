# 딥러닝 기반 흉부 X-Ray 폐질환 분류 (Capstone Design 2022-1)
* 소프트웨어융합학과 이인석

## 1. 과제 개요

### 가. 과제 선정 배경 및 필요성
결핵 발생과 사망률을 줄이기 위해서는 결핵 환자를 조기에 발견하는 것이 무엇보다 중요하다. 하지만1차 검진 수단인 흉부 x선 영상으로 결핵을 판독하는 방사선과 전문의는 전체 의사의 5%도 안되고, 전체 병의원의 70% 이상이 방사선과 전문의 없이 운영되고 있다. 병의원의 의뢰를 받아 영상판독센터가 영상을 대신 분석해주고 있지만, 전국에 몇곳이 안되어서 이메일로 판독을 의뢰하면 일러야 하루, 늦으면 1주일이 소요된다. 객담이나 혈액 분석을 통한 2차 검진으로 결핵을 확진하기까지는 빠르면 1주일 길게는 8주까지도 소요된다. 폐 영상을 찍은 후 곧바로 결과가 나오지 않는 현재의 진단 기술로는 2차감염을 차단할 골든타임을 놓치기 쉽다. 따라서 X-Ray를 통해 병명을 조기 진단하는 연구가 필요하다.

### 나. 과제 주요내용
- 폐질환 데이터 수집
- 데이터 전처리 및 증강
- 5-Layer CNN, ResNet-18 모델
- 모델의 성능 측정 및 평가

## 2. 과제의 목표

### 가. 최종결과물의 목표 (정량적/정성적 목표 설정)

1) 폐질환 질병명을 분류하는 딥러닝 모델의 성능의 정확도를 70% 이상을 목표로 한다.

2) Data Augmentation 추가하여 성능을 비교한다.

3) 학습에 사용한 각 모델별 성능을 비교한다.

### 나. 최종결과물의 세부내용 및 구성


1) 데이터 수집 및 전처리

2) Data Augmentation

3) CNN Architecture (5-layer CNN, ResNet)

4) 모델 성능 평가 지표

5) 모델별 성능 비교


## 3. 기대효과 및 활용방안

결핵 여부에 대한 최종 판단은 객담이나 혈액 분석을 거쳐 2명 이상의 의사가 확진을 내려야만 가능하다. AI가 의사를 완전히 대체하지는 못하겠지만, 2차 검진이 필요한 대상자를 추려내 주는 역할을 하면서 하루 수백장의 흉부 영상을 살펴봐야 하는 의사의 작업 피로도를 덜어줄 뿐더러 하루에 수십만명의 코로나 환자가 속출하는 상황에서 보조적인 개념으로는 충분한 가치가 있다. 이러한 모델을 활용한 병원 측에서는 결핵, 폐렴, 코로나19 환자를 조기에 발견하여 치료할 수 있을 것을 기대해 볼 수 있다.

## 4. 수행방법

### 가. 과제수행을 위한 도구적 방법

Python, Colab, Pytorch

### 나. 과제수행 계획

이미지 데이터 수집 → 훈련데이터와 테스트데이터 분할 → CNN 모델로 학습 → 모델 성능 측정 → 성능 향상을 위해 Data Augmentation 기법 적용 → 모델 학습 → 모델 성능 측정 → 모델별 성능 비교

### 다. 과제 수행 방법 
가)	Chest X-Ray(흉부 엑스레이) 데이터셋
![그림1](https://user-images.githubusercontent.com/92963189/213145217-100ed0d6-d928-48c5-bbdb-4603b9397d68.png)

1)	데이터 수집  
Kaggle에서 Chest X-Ray 데이터셋을 수집했다. 이 데이터 셋은 Pneumonia(폐렴), Covid-19(코로나19), Tuberculosis(결핵), Normal(정상) 4가지 질병명으로 진단된 흉부 X-Ray 이미지이다. 3개의 폴더(train, test, validationl)로 구성되며 각 이미지 레이블(Normal/Pneumonia/Covid-19/Tuberculosis)에 대한 하위 폴더를 포함한다. 총 7,135장의 이미지가 있으며 데이터를 살펴보면 코로나19, 폐결핵 데이터에 비해 폐렴 데이터가 압도적으로 많은것을 알 수 있습니다.

![데이터수](https://user-images.githubusercontent.com/92963189/213143608-89e4ed34-1487-4149-9098-370eaee7f591.png)


2)	데이터 전처리  
프로젝트에서는 validation데이터를 train데이터로 옮겨주고 8:2의 train, test 데이터셋으로 나누어 주었다. 대부분 고해상도 크기가 서로 다른 이미지를 224x224크기로 Resize하고, 0-255범위의 픽셀 값을 정규화를 통해 0-1 범위로 만들어 주었다.  

3)	데이터 증강  
모델 성능 향상을 위해 RandomHorizontalFlip, RandomRotation, RandomCrop기법을 추가하여 데이터 증강을 한 후 학습을 진행해보았다. 왼쪽은 augmentation 기법을 적용하기전, 오른쪽은 적용한 후의 이미지이다.  
![그림2](https://user-images.githubusercontent.com/92963189/213145319-262a25b5-4cf5-4ff6-8b3b-850442968715.png)
 
나)	Model Architecture
1)	5-Layer CNN  
합성곱(nn.Conv2d) + 활성화 함수(nn.ReLU) + 맥스풀링(nn.MaxPoold2d)을 하나의 합성곱 층으로 보고 이렇게 이루어진 3개의 convolutional layer와 2개의 Fully-connected layer로 이루어져 있다. 학습파라미터는 다음과 같다.(learning_rate = 0.001, training_epochs =30, batch_size = 32)

2)	ResNet-18  
 ![그림3](https://user-images.githubusercontent.com/92963189/213145391-1b1b89e3-52cc-41cf-8c37-fbb339f12315.png)  
ResNet-18은 18개의 층으로 이루어진 ResNet이다. 이 ResNet은 Residual Block 단위로 이루어져 있고 18개의 층은 크게 5개의 Block과 fully connected layer로 나뉜다. 이 블럭의 개수에 따라 Resnet-18, resnet-34, resnet-50, resnet-101등이 존재한다. ResNet 연구팀은 층이 깊어질수록 성능이 좋아진다는 사실을 심층적으로 연구한 결과 layer가 너무 깊어져도 성능이 떨어지는 현상을 발견했다. 아래는 층 깊이에 따른 일반 모델과 ResNet 모델의 성능을 비교한 그래프이다. 왼쪽 그래프를 보면 34-layer의 plain 모델보다 18-layer의 모델이 더 성능이 좋다는 사실을 확인할 수 있다.  
![그림4](https://user-images.githubusercontent.com/92963189/213145560-9a7d9db4-3dbd-4092-a321-4ced5f6f7569.png)  
그 이유는 Layer가 깊어질수록 미분을 점점 많이 하게 되고, 미분 값이 작아져 weight의 영향이 미비해지는 Vanishing Gradient 이 발생하여 training data로 학습이 되지 않는 문제가 발생한다고 한다. 이를 해결한 방법이 기울기가 잘 전파될 수 있도록 일종의 숏 컷(skip connection)을 만들어 주는 것이다. 일반적인 구조와는 다르게 Residual block은 아이덴티티 매핑을 통해 입력 x가 어떤 함수를 통과하더라도 다시 x 형태로 출력되도록 해준다. 이렇게 전방의 인풋 값을 출력층까지 가져가기 때문에 층이 깊어져도 Vanishing gradient 문제를 해결할 수 있다. 따라서 ResNet 모델은 신경망의 깊이가 깊어질수록 성능이 좋다는 것을 알 수 있다. 하지만 학습하고자 하는 폐질환 데이터 셋의 크기, 오버 피팅 문제를 고려하여 가장 작은 크기의 네트워크인 ResNet18모델을 사용하기로 결정하였다.  

다. 모델 성능 평가 지표
1) Confusion Matrix: 예측 값이 실제 값을 얼마나 정확히 예측했는지 보여주는 행렬
2) Accuracy: 전체 중에 정답을 맞춘 비율
3) Precision: Positive 로 예측한 것 중에서 실제 Positive 의 비율
4) Recall: 실제 Positive 인 것 중에서 Positive 로 예측한 비율
5) F-1 Score: Precision 과 Recall 의 조화 평균 (* Imbalanced Data 인 경우)  


## Schedule
| Contents | March | April |  May  | June  |   Progress   |
|----------|-------|-------|-------|-------|--------------|
|  데이터 수집  |   O   |       |       |       |     Link1    |
|  관련 기법 학습  |   O   |   O    |       |       |     Link2    |
|  모델 구축 및 수정  |       |      |   O   |   O    |     Link3    |
|  성능 비교  |       |       |      |   O    |     Link4    |
|  보고서 작성  |       |       |       |   O   |     Link5    |


## Results
### 모델별 F1-Score
|     Model   | No Aug | Aug    |
|-------------|--------|--------|
|    CNN      |0.7548| None |
|    ResNet   |0.3537|0.7898|


## Conclusion

* 활용한 데이터셋을 살펴보면 폐렴 데이터에 비해 코로나19데이터와 폐결핵 데이터는 매우 적고 데이터가 불균형하다. 하지만 실제로 4가지 병명 중에는 폐렴이 가장 흔하고 모집단도 크게 다르지 않다고 생각한다. 따라서 추가적인 데이터 없이 불균형한 데이터 셋에 맞게 최선의 성능을 내는 모델을 학습하고 평가하고자 했다. 

* 데이터 증강을 적용하지 않은 5계층 CNN모델의 정확도가 73% 나온 반면 ResNet18 모델의 정확도가 52%가 나왔다. 또한 Flip, rotation, crop등의 데이터 증강 기법을 적용한 데이터셋을 사용한 ResNet18 모델은 80%의 정확도가 나왔다. 데이터 증강 기법이 성능 향상에 크게 기여를 했음을 알 수 있다. 또한 층이 깊고 복잡한 모델일수록 꼭 성능이 좋은 것은 아니라는 것을 직접 확인할 수 있었다. 

* 연구 과제에서 ResNet으로 학습한 모델은 실제 폐 질환 데이터 셋이기 때문에 영상 의학에서 활용이 가능하지만 아직은 성능 향상을 위해 최적화가 필요해 보인다. 이번에 총 7135장의 이미지 데이터로 3-D 5계층CNN 과 ResNet18 모델을 학습시키고 성능 평가를 진행하는 연구를 진행했다면, 다음은 더 큰 데이터셋을 확보하고 큰 네트워크를 가진 모델을 사용해보고 좀 더 다양한 하이퍼파라미터 조정을 통해 최적화를 하는 연구를 진행해보고자 한다. 데이터셋의 크기에 따라 어떤 크기의 딥러닝 네트워크를 학습시킬지 어떤 최적화 기법을 사용할지 인사이트를 길러야 겠다는 확신이 든 이번 연구 과제였다. 

## Reports
* Upload or link (e.g. Google Drive files with share setting)
* Midterm: [Report](Reports/Midterm.pdf)
* Final: [Report](Reports/Final.pdf), [Demo video](Reports/Demo.mp4)
