# from PIL import Image
# from torchvision import transforms

# img = Image.open('test.jpg')
# w, h = img.size
# resize = transforms.Resize([224,244])
# img = resize(img)
# img.save('2.jpg')
# resize2 = transforms.Resize([h, w])
# img = resize2(img)
# img.save('3.jpg')


import pandas as pd


dict_a = {'a': 'rrts', 'k':'zjhjg', 'c':'rrts'}
# df = pd.DataFrame(dict_a)
df = pd.DataFrame(list(dict_a.items()))
df.to_csv('test.csv', header=None, index=None)