import os
import cv2
from xml.etree.ElementTree import parse, Element
import random

def test(xml_path, img_path):
    # xml_path = '/home/kevin/DataSet/COCO/VOC_COCO_with_cls/Annotations'
    # img_path = '/home/kevin/DataSet/COCO/VOC_COCO_with_cls/JPEGImages'
    list_xmls = os.listdir(xml_path)
    for xml in list_xmls:
        image = cv2.imread(os.path.join(img_path, xml[:-3] + 'jpg'))
        doc = parse(os.path.join(xml_path, xml))
        objs = doc.findall('object')
        print(os.path.join(xml[:-3], 'jpg'))
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
    xml_path = '/home/kevin/DataSet/VOCdevkit/VOC2007/Annotations'
    img_path = '/home/kevin/DataSet/VOCdevkit/VOC2007/JPEGImages'
    test(xml_path, img_path)