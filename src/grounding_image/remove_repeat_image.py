import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool

class XHash:
    def __init__(self, image_path, hash_type):
        self.image_path = image_path
        self.hash_size = 8
        self.type = hash_type
        if self.type == 'aHash':
            self.hash = self.__aHash()
        elif self.type == 'dHash':
            self.hash = self.__dHash()

    def __get_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def __difference(self):
        img = cv2.imdecode(np.fromfile(self.image_path,dtype=np.uint8),cv2.IMREAD_COLOR)
        resize_img = cv2.resize(img, (self.hash_size+1, self.hash_size))
        gray = self.__get_gray(resize_img)

        differences = []
        for t in range(resize_img.shape[1] - 1):
            differences.append(gray[:, t] > gray[:, t + 1])
        return np.stack(differences).T

    def __average(self):
        img = cv2.imdecode(np.fromfile(self.image_path,dtype=np.uint8),cv2.IMREAD_COLOR)
        resize_img = cv2.resize(img, (self.hash_size, self.hash_size))
        gray = self.__get_gray(resize_img)
        return gray > gray.mean()

    def __binarization(self, hash_image):
        return ''.join(hash_image.astype('B').flatten().astype('U').tolist())

    def __seg(self, hash_image):
        img_bi = self.__binarization(hash_image)
        return list(
            map(lambda x: '%x' % int(img_bi[x:x + 4], 2), range(0, 64, 4)))

    def __aHash(self):
        return self.__seg(self.__average())

    def __dHash(self):
        return self.__seg(self.__difference())


class Pairs:
    def __init__(self, root):
        self.root = root
        self.__del_all_null_images()
        self.__hashs = np.array([
            XHash(self.names[name], 'dHash').hash
            for name in self.names.keys()
        ])
        self.__cal_haming_distance(self.__hashs)

    def __del_null_image(self, image_path):
        os.remove(image_path)
        print('remove', image_path)

    def __del_all_null_images(self):
        self.names = {}
        for j, name in enumerate(os.listdir(self.root)):
            if not name.endswith('txt' or '.py' or '.ipynb'):
                image_path = os.path.join(self.root, name)
                try:
                    img = cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),cv2.IMREAD_COLOR)
                    if img is not None:
                        self.names[j] = image_path
                except:
                    pass

    def __cal_haming_distance(self, hashs):
        j = 0
        pairs = {}
        while j < hashs.shape[0]:
            for i in range(j + 1, hashs.shape[0]):  # 图片对，过滤到已经计算过的 pairs
                pairs[j] = pairs.get(j, []) + \
                    [np.array(hashs[i] != hashs[j]).sum()]
                continue
            j += 1
        self.pairs = pairs

    def get_names(self):
        n = len(self.pairs)
        temp = {}
        while n > 0:
            n -= 1
            for i, d in enumerate(self.pairs[n]):
                if d == 0:
                    temp[n] = temp.get(n, []) + [i + n + 1]
                    continue
        return temp

    def del_repeat(self):
        P = self.get_names()
        for j in P:
            for i in P[j]:
                try:
                    os.remove(self.names[i])
                    print('remove', self.names[i])
                except FileNotFoundError:
                    print(f'removed {self.names[i]}cannot remove again')
        print('done')


if __name__ == "__main__":
    def func(path):
        if os.path.isdir(path):
            xhash = Pairs(path)
            xhash.del_repeat()

    root = "datas/commongen/bing_image_for_commongen"
    paths = os.listdir(root)
    paths = [os.path.join(root, path) for path in paths]
    with Pool(32) as p:
        tqdm(p.imap(func, paths), total=len(paths))
