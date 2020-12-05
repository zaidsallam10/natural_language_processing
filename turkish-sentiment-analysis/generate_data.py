from utils import *
import pandas as pd

    



def generateData():

    olumlu_cleaner=CleanerPipeline()
    olumlu_cleaner.readCsv('/home/xina/Desktop/Zaidar/Zaid-NLP/turkish-sentiment-analysis/datasets/olumlu.csv')
    dataFrame=olumlu_cleaner.data
    olumlu_cleaner.data=dataFrame['Yorum']
    olumlu_cleaner.removePuncuation(remove_stop_words=True)
    olumlu_cleaner.removeEmoji()
    olumlu_X=olumlu_cleaner.getData()
    olumlu_y=dataFrame['Duygu']
    assert len(olumlu_X)==(len(olumlu_y))


    olumsuz_cleaner=CleanerPipeline()
    olumsuz_cleaner.readCsv('/home/xina/Desktop/Zaidar/Zaid-NLP/turkish-sentiment-analysis/datasets/olumsuz.csv')
    dataFrame=olumsuz_cleaner.data
    olumsuz_cleaner.data=dataFrame['Yorum']
    olumsuz_cleaner.removePuncuation(remove_stop_words=True)
    olumsuz_cleaner.removeEmoji()
    olumsuz_X=olumsuz_cleaner.getData()
    olumsuz_y=dataFrame['Duygu']
    assert len(olumsuz_X)==(len(olumsuz_y))



    cleaner3=CleanerPipeline()
    cleaner3.readCsv('/home/xina/Desktop/Zaidar/Zaid-NLP/turkish-sentiment-analysis/datasets/train.csv')
    dataFrame=cleaner3.data
    cleaner3.data=dataFrame['Yorum']
    cleaner3.removePuncuation(remove_stop_words=True)
    cleaner3.removeEmoji()
    cleaner3_X=cleaner3.getData()
    cleaner3_y=dataFrame['Duygu']
    assert len(cleaner3_X)==(len(cleaner3_y))

    cleaner4=CleanerPipeline()
    cleaner4.readCsv('/home/xina/Desktop/Zaidar/Zaid-NLP/turkish-sentiment-analysis/datasets/mixed.csv')
    dataFrame=cleaner4.data
    cleaner4.data=dataFrame['Yorum']
    cleaner4.removePuncuation(remove_stop_words=True)
    cleaner4.removeEmoji()
    cleaner4_X=cleaner4.getData()
    cleaner4_y=dataFrame['Duygu']
    assert len(cleaner4_X)==(len(cleaner4_y))

    
    X=olumlu_X + olumsuz_X + cleaner3_X+cleaner4_X
    y=pd.concat([olumlu_y, olumsuz_y,cleaner3_y,cleaner4_y])   
    print('asd=',len(X))
    print('asd=',len(y))
    return X,y





