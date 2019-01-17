import os
import json
from xml.dom.minidom import Document
from xml.etree.ElementTree import parse, Element
import cv2
import random

"""
COCO to VOC

@:param str annotaions_path, example:./instances_train2014.json
@:param str output_path, the path of xmls' output

'add_ele' should be called before 'creat_xml'
"""
class coco2voc(object):

    def __init__(self, annotaions_path, output_path, name_pattern):
        self.annotations_path = annotaions_path
        self.output_path = output_path
        self.file = json.load(open(self.annotations_path))
        self.annotations = self.file['annotations']
        self.categories = self.file['categories']
        self.images = self.file['images']
        self.name_pattern = name_pattern

        self.VOC_CLASS = {
            'aeroplane': 5,
            'bicycle': 2,
            'bird': 16,
            'boat': 9,
            'bottle': 44,
            'bus': 6,
            'car': 3,
            'cat': 17,
            'chair': 62,
            'cow': 21,
            'diningtable': 67,
            'dog': 18,
            'horse': 19,
            'motorbike': 4,
            'person': 1,
            'pottedplant': 64,
            'sheep': 20,
            'sofa': 63,
            'train': 7,
            'tvmonitor': 72,
        }
        self.VOC_CLASS_ = {
            5: 'aeroplane',
            2: 'bicycle',
            16: 'bird',
            9: 'boat',
            44: 'bottle',
            6: 'bus',
            3: 'car',
            17: 'cat',
            62: 'chair',
            21: 'cow',
            67: 'diningtable',
            18: 'dog',
            19: 'horse',
            4: 'motorbike',
            1: 'person',
            64: 'pottedplant',
            20: 'sheep',
            63: 'sofa',
            7: 'train',
            72: 'tvmonitor',
        }


    def creat_xml(self):
        index = 0
        for image in self.images:
            id = image['id']
            file_name = image['file_name']
            width = image['width']
            height = image['height']

            doc = Document()
            ann = doc.createElement('annotation')
            doc.appendChild(ann)
            name = doc.createElement('file_name')
            name_text = doc.createTextNode(file_name)
            name.appendChild(name_text)
            ann.appendChild(name)
            size = doc.createElement('size')
            w = doc.createElement('width')
            h = doc.createElement('height')
            d = doc.createElement('depth')
            w_text = doc.createTextNode(str(width))
            h_text = doc.createTextNode(str(height))
            d_text = doc.createTextNode('3')
            w.appendChild(w_text)
            h.appendChild(h_text)
            d.appendChild(d_text)
            size.appendChild(w)
            size.appendChild(h)
            size.appendChild(d)
            ann.appendChild(size)
            xml_name = os.path.join(self.output_path, file_name[:-3]+'xml')

            f = open(xml_name, 'w')
            f.write(doc.toprettyxml(indent="  "))
            f.close()
            print('creating...  '+str(index))
            index += 1

    def add_ele(self):
        index = 0
        cats = {}
        for cat in self.categories:
            cats[cat['id']] = cat['name']
        for ann in self.annotations:
            id = ann['category_id']
            if id in self.VOC_CLASS_.keys():
                image_id = str(ann['image_id'])
                bbox = ann['bbox']
                xml_path = os.path.join(self.output_path, self.name_pattern[:-len(image_id)]+image_id+'.xml')
                doc = parse(xml_path)
                root = doc.getroot()
                obj = Element('object')
                name = Element('name')
                # name.text = cats[id]
                name.text = self.VOC_CLASS_[id]
                bndbox = Element('bndbox')
                xmin, ymin, xmax, ymax = Element('xmin'), Element('ymin'), Element('xmax'), Element('ymax')
                xmin.text, xmax.text = str(int(bbox[0])), str(int(bbox[0] + bbox[2]))
                ymin.text, ymax.text = str(int(bbox[1])), str(int(bbox[1] + bbox[3]))
                bndbox.append(xmin)
                bndbox.append(xmax)
                bndbox.append(ymin)
                bndbox.append(ymax)
                obj.append(name)
                obj.append(bndbox)
                root.append(obj)
                doc.write(xml_path)
                print('adding... '+str(index))
                index += 1

    """
    this is only for test
    """
    def test(self):
        xml_path = '/home/kevin/DataSet/COCO/VOC_COCO_with_cls/2017/Annotations'
        img_path = '/home/kevin/DataSet/COCO/VOC_COCO_with_cls/2017/JPEGImages'
        # xml_path = '/home/kevin/DataSet/VOCdevkit/VOC2007/Annotations'
        # img_path = '/home/kevin/DataSet/VOCdevkit/VOC2007/JPEGImages'
        list_xmls = os.listdir(xml_path)
        list_xmls.sort()
        for xml in list_xmls:
            image = cv2.imread(os.path.join(img_path, xml[:-3]+'jpg'))
            doc = parse(os.path.join(xml_path, xml))
            objs = doc.findall('object')
            for obj in objs:
                bbox = obj.find('bndbox')
                xmin, xmax, ymin, ymax = int(bbox.find('xmin').text), int(bbox.find('xmax').text), \
                                         int(bbox.find('ymin').text), int(bbox.find('ymax').text)
                name = obj.find('name').text
                a, b, c = int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (a, b, c), 2)
                cv2.putText(image, name, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (a, b, c), 1)
            cv2.imshow(' ', image)
            cv2.waitKey()

if __name__ == '__main__':
    pass
    # coco2voc('/home/kevin/DataSet/COCO/annotations_2017/instances_train2017.json',
    #          '/home/kevin/DataSet/COCO/VOC_COCO_with_cls/2017/Annotations', '000000000000').creat_xml()
    # coco2voc('/home/kevin/DataSet/COCO/annotations_2017/instances_train2017.json',
    #          '/home/kevin/DataSet/COCO/VOC_COCO_with_cls/2017/Annotations', '000000000000').add_ele()
    coco2voc('/home/kevin/DataSet/COCO/annotations_2017/instances_train2017.json',
             '/home/kevin/DataSet/COCO/VOC_COCO_with_cls/2017/Annotations', 'COCO_train2014_000000000000').test()
    # coco2voc('/home/kevin/DataSet/COCO/annotations/instances_val2014.json', '/home/kevin/DataSet/COCO/VOC_COCO/Annotations', 'COCO_val2014_000000000000').creat_xml()
    # coco2voc('/home/kevin/DataSet/COCO/annotations/instances_val2014.json', '/home/kevin/DataSet/COCO/VOC_COCO/Annotations', 'COCO_val2014_000000000000').add_ele()
    # coco2voc().test()
