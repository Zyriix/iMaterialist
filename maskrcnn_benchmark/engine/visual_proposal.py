
import numpy as np
import cv2
import torch

def visual_proposal(images,proposals):
    image = images.tensors.cpu().numpy().squeeze(0)
    image = np.transpose(image, (1, 2, 0)) 
    image = np.array(image.astype('uint8')) #convert Float to Int
    image=cv2.UMat(image)
    print(image)
    for i,x in enumerate(proposals):
        for j,box in enumerate(x.bbox):
            # print(box)

            # color = ((i+j)*np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])).astype("uint8")
            color = (i+j)%255
            box = box.cpu().to(torch.int64).numpy()
            top_left, bottom_right =(box[0],box[1]),(box[2],box[3])
            # print(type(top_left[0]),type(image),type(color[0]))
            print(color,top_left,bottom_right)

            image = cv2.rectangle(image, top_left, bottom_right, color=color)
            cv2.imshow('train',image)
            cv2.waitKey(10000)