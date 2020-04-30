from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
import json
import torch
import collections
import os 
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from .transforms import build_transforms

from .collate_batch import BatchCollator, BBoxAugCollator
def makeFashionDataLoader(cfg,dataset,test=False):
    # dataset = FashionDataset(cfg,train)
    sampler = torch.utils.data.sampler.RandomSampler(dataset)
    if test:
        image_per_batch =1
    else:
        image_per_batch = 1
    batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, image_per_batch, drop_last=False
        )
    collator= BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
    data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
    return data_loader

class FashionDataset(torch.utils.data.Dataset):
    def __init__(self,cfg,train=False):
        super(FashionDataset,self).__init__()
        self.data=[]
        self.train = train
        self.cfg = cfg
        f=open("G:\OneDrive\code\python\kaggle/fashion/label_descriptions.json",'r')
        self.attributes = json.load(f)['attributes']
        self.id2idx = dict()
        for i,x in enumerate(self.attributes):
            self.id2idx[str(x['id'])]=i
        # 这是源码中定义的数据预处理
        
        self.transforms = None  if not train and cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, train)

        ## 读取我们生成的目录
        # 文件
        if train:
            f=open("G:\OneDrive\code\python\kaggle/fashion/train.json",'r')
            self.data=json.load(f)
        else:
            f=open("G:\OneDrive\code\python\kaggle/fashion/test.json",'r')
            self.data=json.load(f)
        

     
    ## 通过mask计算bbox
    def extract_bboxes(self,mask):   ###函数，计算与掩膜尺寸相同的包围盒
        """Compute bounding boxes from masks.
        mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

        Returns: bbox array [num_instances, (y1, x1, y2, x2)].
        """
        
        boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
        # print('mask.shape[-1]',mask.shape[-1])
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            # print('horizontal_indicies',horizontal_indicies)
            # print('horizontal_indicies_shape',horizontal_indicies.shape)
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i] = np.array([x1, y1, x2, y2])
        return boxes.astype(np.int32)
    
    ## 加载图片的mask，并解码为binaryMask
    def load_mask(self, image_id):
        info = self.data[image_id]
                
        mask = np.zeros((info['height'], info['width'], len(info['annotations'])), dtype=np.uint8)
        labels = []
        
        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):
            sub_mask = np.full(info['height']*info['width'], 0, dtype=np.uint8)
            annotation = [int(x) for x in annotation.split(' ')]
            
            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel+annotation[2*i+1]] = 1

            sub_mask = sub_mask.reshape((info['height'], info['width']), order='F')
            # sub_mask = cv2.resize(sub_mask, (self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
            
            mask[:, :, m] = sub_mask
            labels.append(int(label)+1)
            
        return mask, np.array(labels)

    def __getitem__(self, idx):
        # 图片必须以PIL格式读取（重点）
        image = self.data[idx]

        ## 必须要把RGB转为BGR，因为训练的时候用的是BGR（OPENCV是BGR,PIL和MATPLOTLIB都是RGB) xxxxxx transform时自动使用了toBGR
        img = Image.open(image['path']).convert('RGB')
        # print(np.array(img).shape)
        # img = np.array(img)
        # img = np.array(img)[:, :, [2, 1, 0]]
        # img = Image.fromarray(img)
        ## 训练时生成mask，bbox以及label。并添加到boxlist对象的filed中
        if self.train:
            # if image['height']<800 or image['width']<800:
            #     return self.__getitem__((idx+1)%self.__len__())
            masks,labels = self.load_mask(idx)
            try:
                if 'bbox' not in image:
                    boxes = self.extract_bboxes(masks)
                    self.data[idx]['bbox'] = boxes
                else:
                    boxes = image['bbox']
                boxlist = BoxList(torch.Tensor(boxes), (image['width'],image['height']), mode="xyxy")
                binMasks = SegmentationMask(torch.Tensor(masks).permute(2,0,1),(image['width'],image['height']),mode='mask')
                # for cats in image['cat']:
                    # print(cats)
                cat = [[self.id2idx[x]   for x in cats.split(',')  ]if type(cats)!=float else [] for cats in image['cat']]
                # for cats in image['cat']:
                    # print(f"cats:{cats},{type(cats)}")
                sub_cat = np.zeros((len(cat),self.cfg.MODEL.ROI_BOX_HEAD.NUM_CATEGORIES))
                for i,c in enumerate(cat):
                    
                    sub_cat[i,c]=1
                    # print(f"sub_cat:{sub_cat[i,:]}")
                    # print(sub_cat.shape)
                # print(cat)
                boxlist.add_field('masks',binMasks)
                boxlist.add_field('labels',torch.Tensor(labels))
                boxlist.add_field('categories',torch.Tensor(sub_cat))
            except:
                return self.__getitem__((idx+1)%self.__len__())
        # 测试时由于数据中没有长和宽，手动生成一下
        else:
            self.data[idx]['width']=img.size[0]
            self.data[idx]['height']=img.size[1]
            boxlist = None
        
        # 进行转换
        if self.transforms:
            img, boxlist = self.transforms(img, boxlist)
        
        # 返回图片,（对应的target）和index
        # target是boxlist格式，其中包含多个训练时要用的数据，包括mask，bbox和label等
        # boxlists = list()
        # boxlists.append(boxlist)
        if self.train:
          return img, boxlist, image['image_id']
        
        else: return img,image['image_id']

    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.data)
    

    # 获取图片的长和宽
    def get_img_info(self, idx):
        image = self.data[idx]
        print(image['image_id'])
        if image['height']==None or image['width']==None:
            img = Image.open(image['path'])
            self.data[idx]['width']=img.size[0]
            self.data[idx]['height']=img.size[1]

        return {"height": self.data[idx]['height'], "width": self.data[idx]['width']}
    def get_img_info_by_id(self, image_id):
        for i,image in enumerate(self.data):
            if image_id == image['image_id']:
                # print(image['image_id'])
                if image['height']==None or image['width']==None:
                    img = img = Image.open(image['path'])
                    self.data[i]['width']=img.size[0]
                    self.data[i]['height']=img.size[1]

                return (self.data[i]['height'], self.data[i]['width'])
        return None
