age = int(input('당신의 나이는?: '))
flag = 0
flag = int(input('생일이 지났으면 1, 그렇지 않으면 0을 입력하세요: '))

if flag == 1 :
	age -= 1
 	print('당신의 미국 나이는{}살 입니다.'.format(age))

else :
	age -= 2
 	print('당신의 미국 나이는{}살 입니다.'.format(age))