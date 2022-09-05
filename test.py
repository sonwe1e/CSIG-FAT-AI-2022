import cv2
import random
import time
import os
from pyfat_implement import PyFAT

app = PyFAT(N=10)
app.load('./assets')

mode = 'rank1'

for iddir in range(0, 100):
    if mode == 'rank1':
        im_a = cv2.imread(f'./test_dataset_A/{iddir+1:03d}/0.png')
        im_v = cv2.imread(f'./test_dataset_A/{iddir+1:03d}/1.png')
    elif mode == 'rank2':
        p1 = random.randint(2, 19)
        p2 = random.randint(2, 19)
        n1 = random.randint(1, 89)
        n2 = random.randint(1, 89)
        if p1 == p2:
            p2 += 1
        print(f'{p1:03d}/{n1:03d}'+' and '+ f'{p2:03d}/{n2:03d}')
        im_a = cv2.imread(f'./test_dataset_B1/{p1:03d}/{n1:03d}.jpg')
        im_v = cv2.imread(f'./test_dataset_B1/{p2:03d}/{n2:03d}.jpg')
    elif mode =='rank3':
        im_a = cv2.imread(f'./test_dataset_B2/data05/00018.jpg')
        im_v = cv2.imread(f'./test_dataset_B2/data05/00019.jpg')
    else:
        path = './test_dataset_B2/'
        out_list = os.listdir('./test_dataset_B2')
        for group in out_list:
            in_list = os.listdir(path+group)
            for pic in range(len(in_list)-1):
                print(group+'/'+in_list[pic])
                im_a = cv2.imread(path+group+'/'+in_list[pic])
                im_v = cv2.imread(path+group+'/'+in_list[pic+1])
                for i in range(app.size()):
                    t1 = time.time()
                    im_av = app.generate(im_a, im_v, i)
                    print(cv2.imwrite(f'./{i:02d}.png', im_av))
                    print(time.time()-t1)
    for i in range(app.size()):
        # im_a = cv2.imread(f'./test_output/{i:02d}.png')
        t1 = time.time()
        im_av = app.generate(im_a, im_v, i)
        print(cv2.imwrite(f'./test_output/{i:02d}.png', im_av))
        print(time.time() - t1)

    break
