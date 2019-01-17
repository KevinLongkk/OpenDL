import os
import cv2
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import random


class Data(object):

    def __init__(self, data_path, batch_size, image_size, istrain=True, shuffle=True):
        self.data_path = data_path
        self.batch_size = batch_size
        self.istrain = istrain
        self.index = 0
        self.image_size = image_size
        self.shuffle = shuffle
        self.epoch = 0
        self.img_path = os.path.join(self.data_path, 'JPEGImages')
        self.label_path = os.path.join(self.data_path, 'Annotations')
        self.img_list = os.listdir(self.img_path)
        if data_path.find('VOC') >= 0 and data_path.find('coco') < 0 and data_path.find('COCO') < 0:
            with open(os.path.join(data_path, 'ImageSets', 'Main', 'aeroplane_trainval.txt')) as txt:
                lines = txt.readlines()
                image_list = []
                for line in lines:
                    image_list.append(line.split(' ')[0] + '.jpg')
            self.img_list = image_list
        self.img_list.sort()
        # random.shuffle(self.img_list)


    """
        check the data_path whether include "Annotations","ImageSets","JPEGImages"
    """
    def check(self):
        assert "Annotations" in os.listdir(self.data_path) and "JPEGImages" in os.listdir(self.data_path)

    """
        load image
    """
    def load_image(self, image_path, flag=None, zoom=None):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if flag == 0 and random.random() < 0.2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            a, b, c = (random.random() * 0.3 + 0.7), ((random.random() * 2 - 1.0) * 0.4 + 0.7), (
                        (random.random() * 2 - 1.0) * 0.4 + 0.7)
            image[..., 0] = image[..., 0] * a
            image[..., 1] = image[..., 1] * b
            image[..., 2] = image[..., 2] * c
            image[image > 255] = 255
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        # elif flag == 0 and random.random() < 0.6:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if zoom is not None:
            # print('zoom')
            width, height = image.shape[1], image.shape[0]
            top, buttom = int(zoom[0] * height), int(zoom[1] * height)
            left, right = int(zoom[2] * width), int(zoom[3] * width)
            image = image[top:(height-buttom), left:(width-right)]
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image / 255.0
        if flag == 0:
            image = cv2.flip(image, 1)
        elif flag == 1:
            image = cv2.flip(image, 0)
        elif flag == 2:
            image = cv2.transpose(image)
            image = cv2.flip(image, 1)
        return image

    def load_label(self, label_path, flag=None, zoom=None):
        label = []
        tree = ET.parse(label_path)
        objs = tree.findall('object')
        size = tree.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        for obj in objs:
            obj_list = []
            obj_name = obj.find('name').text
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text) / width
            y1 = float(bbox.find('ymin').text) / height
            x2 = float(bbox.find('xmax').text) / width
            y2 = float(bbox.find('ymax').text) / height
            w = x2 - x1
            h = y2 - y1

            if zoom is not None:
                top, buttom = zoom[0], zoom[1]
                left, right = zoom[2], zoom[3]
                x1 -= left
                x2 -= left
                y1 -= top
                y2 -= top
                x1 = 0. if x1 < 0 else x1
                y1 = 0. if y1 < 0 else y1
                if x2 < 0 or y2 < 0:
                    continue
                x1 /= 1-(left+right)
                x2 /= 1-(left+right)
                y1 /= 1-(top+buttom)
                y2 /= 1-(top+buttom)
                if x1 > 1. or y1 > 1:
                    continue
                x2 = 1. if x2 > 1 else x2
                y2 = 1. if y2 > 1 else y2
                if x1 == 0. or y1 == 0. or x2 == 1. or y2 == 1.:
                    if (x2 - x1) / w < 0.5 or (y2 - y1) / h < 0.5:
                        # print('too small')
                        continue

            if flag == 0:
                tmp = 1 - x1
                x1 = 1 - x2
                x2 = tmp
            elif flag == 1:
                tmp = y1
                y1 = 1 - y2
                y2 = 1 - tmp
            elif flag == 2:
                tmp_x1, tmp_y1, tmp_x2 = x1, y1, x2
                x1 = 1 - y2
                y1 = tmp_x1
                x2 = 1 - tmp_y1
                y2 = tmp_x2
            obj_list.append(obj_name)
            obj_list.append(x1)
            obj_list.append(y1)
            obj_list.append(x2)
            obj_list.append(y2)
            label.append(obj_list)
        return label

    """
        @:param data_augmentation
                when data_augmentation is equal 0, image will horizontal flip randomly
                when data_augmentation is equal 1, image will vertical and horizontal flip randomly
    """
    def load_data(self, data_augmentation=None, is_print=False, batch_size=None, shuffle=True):
        self.check()
        imgs, labels = [], []
        num_images = len(self.img_list)
        if self.index % num_images < self.batch_size and shuffle:
            random.shuffle(self.img_list)
            self.epoch += 1
            print('index:%d, shuffling dataset...' % self.index)
        batch_size = self.batch_size if batch_size is None else batch_size
        # for i in range(batch_size):
        i = 0
        while i < batch_size:
            if random.random() < 0.5:
                flag = 0
            else:
                flag = None
            if data_augmentation == 0 and random.random() < 0.95:
                # zoom = None
                zoom = [random.random() * 0.03, random.random() * 0.03, random.random() * 0.03, random.random() * 0.03]
            elif data_augmentation == 1:
                zoom = None
                if random.random() < 0.5:
                    flag = 0
                elif 1/2 < random.random() < 5/8:
                    flag = 1
                else:
                    flag = None
            elif data_augmentation == 2:
                zoom = None
                if random.random() < 0.4:
                    flag = 0
                elif 0.4 < random.random() < 0.6:
                    flag = 2
                else:
                    flag = None
            else:
                flag = None
                zoom = None
            self.index %= num_images
            if is_print:
                print(self.img_list[self.index])
            img = self.load_image(os.path.join(self.img_path, self.img_list[self.index]), flag, zoom)
            label = self.load_label(os.path.join(self.label_path, self.img_list[self.index][:-4]+'.xml'), flag, zoom)
            self.index += 1
            if len(label) != 0:
                i += 1
            else:
                continue
            imgs.append(img)
            labels.append(label)
        return imgs, labels

    def load_test_data(self, index=0):
        self.check()
        imgs, labels = [], []
        img_path = os.path.join(self.data_path, 'JPEGImages')
        label_path = os.path.join(self.data_path, 'Annotations')
        img_list = os.listdir(img_path)
        img_list.sort()
        # num_images = len(img_list)
        # print('loading test image: '+img_list[index])
        # print(img_list[index])
        for i in range(self.batch_size):
            # self.index %= num_images
            img = self.load_image(os.path.join(img_path, img_list[index]))
            label = self.load_label(os.path.join(label_path, img_list[index][:-4]+'.xml'))
            # self.index += 1
            imgs.append(img)
            labels.append(label)
        return imgs, labels


# def test_data_process():
#     data_path = '/home/kevin/DataSet/COCO/VOC_COCO_with_cls/2017'
#     data = Data(data_path, 1, 300)
#     while True:
#         image, label = data.load_data(0)
#         image = image[0]
#         width, height = image.shape[1], image.shape[0]
#         for l in label[0]:
#             if int(l[1] * width) < 0:
#                 print('top_left out of bound')
#             if int(l[2]*height) < 0:
#                 print('top_left out of bound')
#             if int(l[3]*width) > width:
#                 print('top_left out of bound')
#             if int(l[4]*height) > height:
#                 print('top_left out of bound')
#             cv2.rectangle(image, (int(l[1]*width), int(l[2]*height)), (int(l[3]*width), int(l[4]*height)), (0, 255, 0), 1)
#             cv2.putText(image, l[0], (int(l[1]*width), int(l[2]*height)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
#         cv2.imshow('', image)
#         cv2.waitKey(0)
# test_data_process()




