import numpy as np

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
class F1_scorer():
    def __init__(self):
        
        self.FN = 0
        self.TP = 0
        self.FP = 0
    def __call__(self,predict,targets):
        for rs,target in zip(predict,targets):
            f1_threshold = 0.5
            gt_labels = target.get_field('labels')
            gt_masks = np.array([(x>0).numpy() for x in target.get_field("masks").instances ])

            # print(gt_labels.shape,gt_masks.shape)
                
            scores = rs.get_field('scores').cpu().numpy()
            labels = rs.get_field('labels').cpu().numpy()
            # print(scores.shape,labels.shape)
            
            mask_threshold =  0.5
            masker = Masker(threshold=mask_threshold, padding=1)
            if rs.has_field("mask"):
                masks = rs.get_field("mask")
                masks = masker([masks], [rs])[0]
            
            idx = scores>0.5
            
            scores = scores[idx]
            labels = labels[idx]
            masks = masks[idx]
            rs = []
            for gt,gt_label in zip(gt_masks,gt_labels):
                maxIou = 0
                for m,score,label in zip(masks,scores,labels):


                    o = np.sum(gt)
                    x = (np.ravel(gt) & np.ravel(m))
                    i = np.sum(x)
                    iou = i/o
                    if iou>maxIou:
                        maxIou = iou
                    if iou >f1_threshold:
                        if label == gt_label and score > f1_threshold:
                            self.TP+=1
                        else:
                            self.FP+=1
                if maxIou <= f1_threshold:
                    self.FN+=1
        # print(f"TP:{self.TP},FP:{self.FP},FN:{self.FN}")
        try:
            P = self.TP/(self.TP+self.FP)
            R = self.TP/(self.TP+self.FN)
            F1 = 2*R*P/(R+P)
        except:
            F1=0
        return F1