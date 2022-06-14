<h3>pyhon을 통한 가상화폐 가격 예측 시스템 구현</h3>

기간 : 2021.3.30 ~ 2021.5.17

사용 툴 : pycharm + python

시스템 설계도 

![image](https://user-images.githubusercontent.com/100816231/170006577-83c1b9d4-c535-43ee-986c-18e71c277ac3.png)

시스템 흐름도

![image](https://user-images.githubusercontent.com/100816231/170006738-8c98af29-3b4a-4d17-968b-3cdcb3c4bdcf.png)

사용 방법 

1. code/데이터전처리/main.py 를 통해 원하는 가상화폐의 가격 데이터를 수집한다.
defaut 값은 coin_list = ['BTC'] // time_units = ['days', 'weeks'] //minutes_units = [60] 으로 설정되어 있는데,
각각 가상화폐 종류, 주 단위, 일단위, 수집 간격 이것을 조정하여 가격 데이터 수집

2. 수집한 가격 데이터를 보조지표.py를 통해 지표 생성

3. 수집한 데이터를 가지고 code/비트코인 가격예측에 있는 파일을 활용하여 모델을 학습하고 가격을 예측한다.
