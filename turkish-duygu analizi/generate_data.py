from utils import *
import pandas as pd

    


def callPipeline(filepath):
    pipeline=CleanerPipeline()
    pipeline.readCsv(filepath)
    dataFrame=pipeline.data
    pipeline.data=dataFrame['Yorum']
    pipeline.removePuncuation(remove_stop_words=True)
    pipeline.removeEmoji()
    X=pipeline.getData()
    y=dataFrame['Duygu']
    assert len(X)==(len(y))
    return X,y



def generateData():
    files=[
    '/home/xina/Desktop/Zaidar/Zaid-NLP/turkish-duygu analizi/datasets/olumlu.csv',
    '/home/xina/Desktop/Zaidar/Zaid-NLP/turkish-duygu analizi/datasets/olumsuz.csv',
    '/home/xina/Desktop/Zaidar/Zaid-NLP/turkish-duygu analizi/datasets/train.csv',
    '/home/xina/Desktop/Zaidar/Zaid-NLP/turkish-duygu analizi/datasets/mixed.csv',
    '/home/xina/Desktop/Zaidar/Zaid-NLP/turkish-duygu analizi/datasets/sentiment_analysis.csv'
    ]
    Xs=[]
    Ys=[]
    for file_ in files:
        X,y= callPipeline(file_)
        Xs+= X
        Ys.append(y)
    y=pd.concat(Ys)   
    print('asd=',len(Xs))
    print('asd=',len(y))
    return Xs,y





