import json
import pandas as pd
from  tqdm import tqdm
import numpy as np
## 添加到字典中
def dealData():
    def add_image(data,image_id,path,labels=None,cat=None,annotations=None,height=None, width=None):
            img = dict()
            img['image_id']=image_id
            img['path']=path
            img['labels']=labels
            img['cat']=cat
            img['annotations']=annotations
            img['height']=height
            img['width']=width
            
            data.append(img)

    ## 读取训练CSV
    trainData = []
    df = pd.read_csv("G:\OneDrive\code\python\kaggle/fashion/train.csv")

    # 将一张图片的不同mask/label/attributes合并到一个数组中
    image_df = df.groupby('ImageId')['EncodedPixels', 'ClassId','AttributesIds'].agg(lambda x: list(x))
    size_df = df.groupby('ImageId')['Height', 'Width'].mean()
    image_df = image_df.join(size_df, on='ImageId')

    ## 将他打包成json，训练时直接读取
    for _, row in tqdm(image_df.iterrows(),desc="processing training data"):
        add_image(trainData,image_id=row.name, 
                    path=str('G:/OneDrive/code/python/kaggle/fashion/train/'+row.name) + '.jpg', 
                    labels=row['ClassId'],
                    cat=row['AttributesIds'],
                    annotations=row['EncodedPixels'], 
                    height=row['Height'], 
                    width=row['Width'])

    trainJson = open('G:\OneDrive\code\python\kaggle/fashion/train.json','w')
    json.dump(trainData,trainJson)


    ## 读取测试数据
    testData = []
    df = pd.read_csv("G:\OneDrive\code\python\kaggle/fashion/test.csv")
    ## 测试数据只需要图片ID和路径
    for _,row in tqdm(df.iterrows(),desc="processing test data"):
        add_image(testData,image_id=row['ImageId'], 
                    path=str('G:/OneDrive/code/python/kaggle/fashion/test/'+row['ImageId']) + '.jpg')
    testJson = open('G:\OneDrive\code\python\kaggle/fashion/test.json','w')
    json.dump(testData,testJson)



def get_pos_weight():
    def check(rows):
        count = np.zeros(294)
        if type(rows)!=float:
          for a_id in rows.split(','):
            count[id2idx[a_id]]+=1
        # print(count)
        return count
    f=open("G:\OneDrive\code\python\kaggle/fashion/label_descriptions.json",'r')
    attributes = json.load(f)['attributes']
    id2idx = dict()
    for i,x in enumerate(attributes):
        id2idx[str(x['id'])]=i
    
    trainData = []
    df = pd.read_csv("G:\OneDrive\code\python\kaggle/fashion/train.csv")
    rs = df['AttributesIds'].apply(check)
    rs = np.sum(rs,axis=0)
    count = df['AttributesIds'].shape[0]
    rs= (2*count-rs)/rs
    print(rs)
    f=open('../configs/pos_weight.json','w')
    json.dump(rs.tolist(),f)
get_pos_weight()