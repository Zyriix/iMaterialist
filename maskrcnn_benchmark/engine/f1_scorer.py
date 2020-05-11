import numpy as np

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
class F1_scorer():
    def __init__(self,Attibutes_on):
        self.Attibutes_on = Attibutes_on
        
        self.FN = 0
        self.TP = 0
        self.FP = 0
        self.FP_LABEL=0
        self.FP_ATT=0
    def __call__(self,predict,targets):
        for rs,target in zip(predict,targets):
            f1_threshold = 0.5
            gt_labels = target.get_field('labels')
            gt_masks = np.array([(x>0).numpy() for x in target.get_field("masks").instances ])
            gt_categories = target.get_field("categories").numpy() >0
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
            if  self.Attibutes_on :
                cat_scores = rs.get_field('cat_scores').cpu().numpy()
                cat_scores = cat_scores[idx]
                cat_scores=cat_scores>0.5
            else: 
                cat_scores = [0 for i in range(len(labels))]
            rs = []
            
            for gt,gt_label,gt_catgory in zip(gt_masks,gt_labels,gt_categories):
                # print(gt_catgory.shape,gt_categories.shape)

                maxIou = 0
                tp = 0
                fp = 0
                fn = 0
                # f1 = 0
                for m,score,label,cat_score in zip(masks,scores,labels,cat_scores):


                    o = np.sum(gt)
                    # print(m)
                    x = (np.ravel(gt) & np.ravel(m))
                    i = np.sum(x)
                    iou = i/o
                    if iou>maxIou:
                        maxIou = iou
                    if iou >f1_threshold:
                        # print(gt_catgory,cat_score)
                        if self.Attibutes_on:
                            tp = np.sum(np.ravel(gt_catgory ) & np.ravel(cat_score))
                            fp = np.sum(~np.ravel(gt_catgory ) & np.ravel(cat_score))
                            fn = np.sum(np.ravel(gt_catgory ) & ~np.ravel(cat_score))
                            # print(tp,fp,fn)
                            try:
                                P = tp/(tp+fp)
                                R = tp/(tp+fn)
                                F1 = 2*R*P/(R+P)
                            except:
                                F1=0
                            # print(F1)
                            if label == gt_label and F1 > f1_threshold:
                                

                                self.TP+=1
                            else:
                                if label!=gt_label:
                                    self.FP_LABEL+=1
                                else:
                                    self.FP_ATT+=1
                                self.FP+=1
                        else:
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
        return F1,self.TP,self.FP,self.FN,self.FP_ATT,self.FP_LABEL