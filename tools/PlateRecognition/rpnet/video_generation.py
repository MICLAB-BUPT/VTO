import os
import cv2
import numpy as np

path = 'E:\Dataset\CCPD2019\ccpd_base/'
filelist = os.listdir(path)


fps = 15
# 这里的size是反的
size = (720, 1160)

video_list = 'Beijing,Shanghai,Guangzhou,Shenzhen,Tianjin,Chongqing,Hangzhou,Chengdu,Xian,Nanjing,Wuhan,Shenyang,' \
             'Suzhou,Harbin,Guilin,Lhasa,Lanzhou,Kunming,Xiamen,Dalian,Qingdao,Changsha,Zhengzhou,Jinan,Ningbo,' \
             'Changchun,Fuzhou,Nanchang,Urumqi,Hefei,Taiyuan,Xianyang,Wenzhou,Foshan,Dongguan,Jinjiang,Wuxi,' \
             'Taizhou,Shijiazhuang,Changzhou,Nanning,Ürümqi,Guiyang,Yantai,Xining,Nantong,Huizhou,Baotou,Anshan,' \
             'Tangshan,Weifang,Zhuhai,Shaoxing,Yangzhou,Yancheng,Zibo,Huaian,Jinhua,Nanchong,Xuzhou,Fushun,' \
             'Wuhu,Datong,Quanzhou,Yinchuan,Jilin,Zhangzhou,Shantou,Luoyang,Zhanjiang'
video_list = video_list.split(',')
provincelist = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫",
                "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "西", "陕", "甘", "青", "宁", "新"]
wordlist = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
            "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
frame_index = 0
video_index = 0
video = cv2.VideoWriter("dataset/videos/"+video_list[video_index]+".mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                fps, size)
file = open('dataset/query/{}.txt'.format(video_list[video_index]), 'w')

for item in filelist:
    # 每个视频的长度为5s，每75帧就新建一个视频和对应的.txt文件
    print(frame_index)
    print(video_list[video_index])
    if frame_index >= 75:
        frame_index = 0
        video_index += 1
        video.release()
        file.close()
        cv2.destroyAllWindows()
        video = cv2.VideoWriter("dataset/videos/"+video_list[video_index]+".mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                fps, size)
        file = open('dataset/query/{}.txt'.format(video_list[video_index]), 'w')

    if item.endswith('.jpg'):
        frame_index += 1
        item = path + item
        img = cv2.imread(item)
        lpn = item.split('-')[4].split('_')
        lpn = [int(i) for i in lpn]
        lpn = provincelist[lpn[0]] + wordlist[lpn[1]] + wordlist[lpn[2]] + wordlist[lpn[3]] + wordlist[lpn[4]] + \
        wordlist[lpn[5]] + wordlist[lpn[6]]
        file.write(lpn+'\n')
        video.write(img)

