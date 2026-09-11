import os
import argparse
from shutil import copyfile
from IPython import embed

video_list = 'Beijing,Shanghai,Guangzhou,Shenzhen,Tianjin,Chongqing,Hangzhou,Chengdu,Xian,Nanjing,Wuhan,Shenyang,Suzhou,Harbin,Guilin,Lhasa,Lanzhou,Kunming,Xiamen,Dalian,Qingdao,Changsha,Zhengzhou,Jinan,Ningbo,Changchun,Fuzhou,Nanchang,Urumqi,Hefei,Taiyuan,Xianyang,Wenzhou,Foshan,Dongguan,Jinjiang,Wuxi,Taizhou,Shijiazhuang,Changzhou,Nanning,Ürümqi,Guiyang,Yantai,Xining,Nantong,Huizhou,Baotou,Anshan,Tangshan,Weifang,Zhuhai,Shaoxing,Yangzhou,Yancheng,Zibo,Huaian,Jinhua,Nanchong,Xuzhou,Fushun,Wuhu,Datong,Quanzhou,Yinchuan,Jilin,Zhangzhou,Shantou,Luoyang,Zhanjiang'
video_list = video_list.split(',')[40:]
new_video_list = 'Hefei,Wuhu,Bengbu,Huainan,Maanshan,Anqing,Suzhou,Fuyang,Bozhou,Huangshan,Chuzhou,Xuancheng,Luan,Chizhou,Tongling'
new_video_list = new_video_list.split(',')
os.makedirs('/home/zhongyaoyao/Data/plate/new_gallery/', exist_ok=True)

for i in range(len(new_video_list)):
    print(i)
    new_name = new_video_list[i]
    past_name = video_list[i]
    past_file = "/home/zhongyaoyao/Data/plate/query/{}.txt".format(past_name)
    new_file = "/home/zhongyaoyao/Data/plate/query/{}.txt".format(new_name)
    copyfile(past_file, new_file)
    past_video = "/home/zhongyaoyao/Data/plate/videos/{}.mp4".format(past_name)
    new_video = "/home/zhongyaoyao/Data/plate/new_gallery/{}.mp4".format(new_name)
    copyfile(past_video, new_video)