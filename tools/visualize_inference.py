import torch
import cv2
import os
import json
import numpy as np
from maskrcnn_benchmark.utils import cv2_util
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.data import FashionDataset,makeFashionDataLoader
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.transforms import Resize
from PIL import Image
class visualize_tools():
    ## 该变量应该还要修改
   
   

    def __init__(
        self,
        confidence_threshold=0.7,
        masks_per_dim=2,
        min_image_size=224,
    ):
        mask_threshold =  0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.masks_per_dim = masks_per_dim
        f=open("G:\OneDrive\code\python\kaggle/fashion/test.json",'r')
        self.test_json = json.load(f)
        f=open("G:\OneDrive\code\python\kaggle/fashion/label_descriptions.json",'r')
        self.label_desc = json.load(f)

    ## 这几个方法都是参考源码的predictor编写而成的
    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None].long() * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            # print(box)
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            
            image = cv2.rectangle(
                np.ascontiguousarray(image), tuple(top_left), tuple(bottom_right), tuple(color), 1
            )

        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        
        labels = predictions.get_field("labels")
        # print(len(masks),len(labels))
        colors = self.compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            # print(mask)
            thresh = mask[0, :, :].astype(np.uint8)
            # print(thresh)
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # print(contours)
            image = cv2.drawContours(image, contours, -1, color, 3)

        composite = image

        return composite

    # def overlay_keypoints(self, image, predictions):
    #     keypoints = predictions.get_field("keypoints")
    #     kps = keypoints.keypoints
    #     scores = keypoints.get_field("logits")
    #     kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()
    #     for region in kps:
    #         image = vis_keypoints(image, region.transpose((1, 0)))
    #     return image

    def create_mask_montage(self, image, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field("mask")
        # print(masks)
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[: len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros(
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]
        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.label_desc['categories'][i-1]['name']  for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )

        return image
    def overlay_categories(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = np.array(predictions.get_field("cat_scores").tolist())
        # for score in scores:
            # print(score)
        # print(scores)
        #  = predictions.get_field("cat_scores").tolist()
        idx = scores >0.1
        scores = scores[idx]
        rs = []
        # print(idx.shape)
        # for cat in idx:
            # rs.append(self.label_desc['attributes'][cat]['name'] )
        # for cat in idx:
        #     for i in np.argwhere(cat==True):
        #         print(i)
        print(len(self.label_desc['attributes']))
        attributes = np.array(self.label_desc['attributes'])
        # print(attributes.shape)
        cats =[[  attributes[i[0]-2]['name']   for i in np.argwhere(cat==True)]  for cat in idx]
        print(cats.shape)
        # labels = [self.label_desc['categories'][i-1]['name']  for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score,cat in zip(boxes, scores,cats):
            x, y = box[:2]
            s = template.format(" ".join(cats), " ".join(score))
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )

        return image

    ## 这个方法是根据predictor中的run_on_opencv_image修改的
    def run_on_inference_result(self):
       
        # 这是inference的结果，里面保存了{imageId:predictions}的结果
        predictions = torch.load( os.path.join('G:\OneDrive\code\python\kaggle/fashion/inferences/2020-04-23', "predictions.pth"))
        
        # print(predictions)
        for i,key in enumerate(predictions):
            pre = predictions[key]
            pre = self.select_top_predictions(pre)
            # print(key)
            # print(pre.get_field("scores"))
            # print(pre.get_field("labels"))
            # print(pre.get_field("mask"))
            # print(pre.get_field("mask"))

            # 每100张图画一张展示
            if i%1 ==0:
                image = Image.open('G:\OneDrive\code\python\kaggle/fashion/test/'+str(key)+'.jpg')
                # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                # maxLen = max(image.shape[0],image.shape[1])
                # minLen = min(image.shape[0],image.shape[1])
                # sc = 1
                cfg.merge_from_file("G:\OneDrive\code\python\maskrcnn-benchmark\configs\caffe2\e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x_caffe2.yaml")
        # cfg.merge_from_list(args.opts)
                cfg.freeze()
                # 按照配置中对原始图片进行缩放
                resizer = Resize(cfg.INPUT.MIN_SIZE_TEST,cfg.INPUT.MAX_SIZE_TEST)
                image = np.array(resizer(image))
                # if maxLen>cfg.INPUT.MAX_SIZE_TEST:
                #     sc = cfg.INPUT.MAX_SIZE_TEST/maxLen
                # elif minLen<cfg.INPUT.MIN_SIZE_TEST:
                #     sc = cfg.INPUT.MIN_SIZE_TEST/minLen
                # # print(f"maxlen:{maxLen},minlen:{minLen},sc:{sc}")
                # height = int(image.shape[1]*sc)
                # width = int(image.shape[0]*sc)
                # image = cv2.resize(image,(height,width))
                # print(width,height)

                # 将box添加到图片中
                # image = self.overlay_boxes(image, pre)

                # 将mask从二进制转化为bool值
                if pre.has_field("mask"):
                    masks = pre.get_field("mask")
                    masks = self.masker([masks], [pre])[0]
                    pre.add_field("mask", masks)

                ## 将mask添加到图片中    
                image = self.overlay_mask(image, pre)

                # 将类别添加到图片中
                image = self.overlay_class_names(image, pre)

                # image = self.overlay_categories(image, pre)

                # 绘制图片
                image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                cv2.namedWindow(str(key))
                cv2.imshow(str(key),image)
                cv2.waitKey(5000)
    def run_on_train_set(self):
       
        # 这是inference的结果，里面保存了{imageId:predictions}的结果
        cfg.merge_from_file("G:\OneDrive\code\python\maskrcnn-benchmark\configs\caffe2\e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml")
        # cfg.merge_from_list(args.opts)
        cfg.freeze()
        dataset = FashionDataset(cfg,train=True)
        datasets = torch.utils.data.random_split(dataset,[10,len(dataset)-10])
        data_loader = makeFashionDataLoader(cfg,datasets[0])
          
        
        # print(predictions)
        for i,(image,target,idx) in enumerate(data_loader):
            
            # pre = self.select_top_predictions(pre)
            # print(key)
            # print(pre.get_field("scores"))
            # print(pre.get_field("labels"))
            # print(pre.get_field("mask"))
            # print(pre.get_field("mask"))

            # 每100张图画一张展示
            # nparray = np.array(image)
            image = image.tensors.numpy().squeeze(0)
            image = np.transpose(image, (1, 2, 0)) 
            image = image.astype('uint8')  #convert Float to Int
            target = target[0]
        

            # image = cv2.resize(image,(image.shape[1],image.shape[0]))

            # 将box添加到图片中
            # image = self.overlay_boxes(image, target)
        

            # 将mask从二进制转化为bool值
            labels = target.get_field("labels")
            bboxs = target.bbox
            colors = self.compute_colors_for_labels(labels).tolist()
            masks= ([x for x in target.get_field("masks").instances])

            

            labels = target.get_field("labels").tolist()
            
            labels = [self.label_desc['categories'][i-1]['name']  for i in labels]
            boxes = target.bbox
            template = "{}"
            for label,x,color,box in zip(labels,masks,colors,bboxs):
                mask = x>0
                # print(mask.shape)
                thresh = mask.numpy().astype(np.uint8)
                # print(thresh)
                contours, hierarchy = cv2_util.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                image = cv2.drawContours(image, contours, -1, color, 3)
                x, y = box[:2]
                s = template.format(label)
                cv2.putText(
                    image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
                )

                # 将类别添加到图片中
                # image = self.overlay_class_names(image, target)

                # 绘制图片
                cv2.namedWindow('train')
                cv2.imshow('train',image)
                cv2.waitKey(5000)


vt = visualize_tools()
vt.run_on_inference_result()
# vt.run_on_train_set()