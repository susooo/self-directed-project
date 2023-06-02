# self-directed-project
### technical stack : docker &amp; k8s &amp; mlflow <br>
<br>
 public 클라우드를 사용하지 않고 private 클라우드를 구성하여 기계학습을 진행할 수 있도록 On-Premise Kubernetes 환경을 구축하고, 오픈 소스 MLOps 도구인 mlflow를 이용하여 MLOps를 구현한다. 

 MLOps 개발의 2차적인 효과로 GPU에 대한 효율성을 향상시킨다. GPU를 사용하는 기계학습에서 학습과정에서뿐만 아니라 
코딩 과정에서도 GPU가 할당된 채 사용되지 않는 경우가 있다. 
기계학습 모델 개발을 위한 코드와 모델 실행을 위한 코드를 분리하여 학습 과정에서만 GPU를 할당함으로써 다수의 사용자가 GPU 자원을 활용할 수 있도록 하는 방안을 구현한다. 

<br>

## [시스템 구조]
![image](https://github.com/susooo/self-directed-project/assets/92291198/2d5287c5-8deb-4015-9c4b-4f57b870e53e)

<br>

## [사용자 사용방법]
python execution.py를 실행하여 실행할 기계학습 파일과 데이터셋을 전달한다.
※기계학습 파일은 /user 디렉토리에 있어야하며, 데이터셋은 /user/data 디렉토리에 존재해야 한다.

<br>

## [실행결과]
- 날짜별로 실행결과 저장

![image](https://github.com/susooo/self-directed-project/assets/92291198/6171900b-24b3-44e0-bcd4-e7cf3fa830da)

- 실행에 대한 다양한 정보 확인가능

![mlflow image](https://github.com/susooo/self-directed-project/assets/92291198/e6dc0f97-e139-4829-9e08-f85b86165378)
