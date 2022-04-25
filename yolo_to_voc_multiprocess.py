import os
import argparse
from PIL import Image
import multiprocessing
from xml.etree import ElementTree
from pascal_voc_writer import Writer

config_names = ["thep", ]

def yolo2voc(txt_file):
    w, h = Image.open(os.path.join(config.image_dir, f'{txt_file[:-4]}.jpg')).size
    writer = Writer(f'{txt_file[:-4]}.xml', w, h)
    with open(os.path.join(config.label_dir, txt_file)) as f:
        for line in f.readlines():
            label, x_center, y_center, width, height = line.rstrip().split(' ')
            x_min = int(w * max(float(x_center) - float(width) / 2, 0))
            x_max = int(w * min(float(x_center) + float(width) / 2, 1))
            y_min = int(h * max(float(y_center) - float(height) / 2, 0))
            y_max = int(h * min(float(y_center) + float(height) / 2, 1))
            writer.addObject(config_names[int(label)], x_min, y_min, x_max, y_max)
    writer.save(os.path.join(config.xml_dir, f'{txt_file[:-4]}.xml'))


def voc2yolo(xml_file, output_folder, sub_dir, xml_dir, offset_x, offset_y):
    in_file = open(os.path.join(xml_dir, xml_file))
    tree = ElementTree.parse(in_file)
    size = tree.getroot().find('size')
    height = int(size.find('height').text)
    width = int(size.find('width').text)

    class_exists = False
    for obj in tree.findall('object'):
        name = obj.find('name').text
        if name in config_names:
            class_exists = True

    if class_exists:
        out_file = open(os.path.join(output_folder, "labels", sub_dir + "_" + xml_file[:-4] + ".txt"), 'w')
        # out_file = open(f'{config.label_dir}/{xml_file[:-4]}.txt', 'w')
        for obj in tree.findall('object'):
            difficult = obj.find('difficult').text
            if int(difficult) == 1:
                continue
            xml_box = obj.find('bndbox')
            x_min = float(xml_box.find('xmin').text) + offset_x
            y_min = float(xml_box.find('ymin').text) + offset_y
            x_max = float(xml_box.find('xmax').text) + offset_x
            y_max = float(xml_box.find('ymax').text) + offset_y

            box_x_center = (x_min + x_max) / 2.0 - 1 # according to darknet annotation
            box_y_center = (y_min + y_max) / 2.0 - 1 # according to darknet annotation
            box_w = x_max - x_min
            box_h = y_max - y_min
            box_x = box_x_center * 1. / width
            box_w = box_w * 1. / width
            box_y = box_y_center * 1. / height
            box_h = box_h * 1. / height

            b = [box_x, box_y, box_w, box_h]
            cls_id = config_names.index(obj.find('name').text)
            out_file.write(str(cls_id) + " " + " ".join([str(f'{i:.6f}') for i in b]) + '\n')
        out_file.close()

import shutil
def move_image(image_file, output_folder, sub_dir, image_dir):
    source = os.path.join(image_dir, image_file)
    output = os.path.join(output_folder, "images", sub_dir + "_" + image_file)

    shutil.copy(source, output)


# def voc2yolo_a(xml_file):
#     in_file = open(f'{config.xml_dir}/{xml_file}')
#     tree = ElementTree.parse(in_file)
#
#     class_exists = False
#     for obj in tree.findall('object'):
#         name = obj.find('name').text
#         if name in config_names:
#             class_exists = True
#
#     if class_exists:
#         out_file = open(f'{config.label_dir}/{xml_file[:-4]}.txt', 'w')
#         for obj in tree.findall('object'):
#             difficult = obj.find('difficult').text
#             if int(difficult) == 1:
#                continue
#             xml_box = obj.find('bndbox')
#             x_min = round(float(xml_box.find('xmin').text))
#             y_min = round(float(xml_box.find('ymin').text))
#             x_max = round(float(xml_box.find('xmax').text))
#             y_max = round(float(xml_box.find('ymax').text))
#
#             b = [x_min, y_min, x_max, y_max]
#             cls_id = config_names.index(obj.find('name').text)
#             out_file.write(str(cls_id) + " " + " ".join([str(f'{i}') for i in b]) + '\n')
#         out_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--yolo2voc', action='store_true', help='YOLO to VOC')
    parser.add_argument('--voc2yolo', action='store_true', help='VOC to YOLO')
    # parser.add_argument('--voc2yolo_a', action='store_true', help='VOC to YOLO absolute')
    parser.add_argument(
        "--parent_path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument(
        "--sub_directory", default="", help="path where to save in string with ',' delimiter"
    )
    parser.add_argument(
        "--output_path", default="", help="path where to save in string with ',' delimiter"
    )
    
    args = parser.parse_args()
    try:
      os.makedirs(args.output_path)
    except:
      pass
    try:
      os.makedirs(os.path.join(args.output_path, "images"))
    except:
      pass
    try:
      os.makedirs(os.path.join(args.output_path, "labels"))
    except:
      pass


    if args.yolo2voc:
        print('YOLO to VOC')
        txt_files = [name for name in os.listdir(config.label_dir) if name.endswith('.txt')]

        with multiprocessing.Pool(os.cpu_count()) as pool:
            pool.map(yolo2voc, txt_files)
        pool.join()
    from functools import partial
    
    if args.voc2yolo:
        print('VOC to YOLO')
        for sub_dir in args.sub_directory.split(","):
          xml_dir = os.path.join(args.parent_path, sub_dir, "Annotations")
          image_dir = os.path.join(args.parent_path, sub_dir, "JPEGImages")

          xml_files = [name for name in os.listdir(xml_dir) if name.endswith('.xml')]
          image_files = [name[:-4] + ".jpg" for name in xml_files]

          conv_func=partial(voc2yolo, output_folder=args.output_path, sub_dir=sub_dir, xml_dir=xml_dir, offset_x = 1, offset_y = 2)
          im_cp_func=partial(move_image, output_folder=args.output_path, sub_dir=sub_dir, image_dir=image_dir)

          with multiprocessing.Pool(os.cpu_count()) as pool:
              pool.map(conv_func, xml_files)
              pool.map(im_cp_func, image_files)

          pool.join()

          # with multiprocessing.Pool(os.cpu_count()) as pool:
          #     pool.map(im_cp_func, image_files)
          # pool.join()


    # if args.voc2yolo_a:
    #     xml_files = [name for name in os.listdir(config.xml_dir) if name.endswith('.xml')]
    #     with multiprocessing.Pool(os.cpu_count()) as pool:
    #         pool.map(voc2yolo_a, xml_files)
    #     pool.close()

# python yolo_to_voc_multiprocess.py ^
# --voc2yolo ^
# --parent_path=D:\Projects\VSTech\FieldDataApril\Check\Verified ^
# --sub_directory="29032022,30032022" ^
# --output_path=D:\Projects\VSTech\FieldDataApril\generated_annotations_24_04_yolo\content\data\data_24_04_test