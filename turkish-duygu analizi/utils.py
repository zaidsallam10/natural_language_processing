import re
import pandas as pd




class CleanerPipeline():
    file_name=None
    data=None
    stop_wods_frame=pd.DataFrame(pd.read_csv('/home/xina/Desktop/Zaidar/Zaid-NLP/turkish-duygu analizi/datasets/turkce-stop-words.csv'))
    def __init__(self):
        print('Hello Cleaner world')        
        


    def readFile(self,file_name):
        self.file_name=file_name
        file_= open(self.file_name)
        text=file_.read()
        self.data=text
        return text

    def readCsv(self,file_name):
        self.file_name=file_name
        self.data=pd.DataFrame(pd.read_csv(file_name))

    def removePuncuation(self,remove_stop_words=False):
        if type(self.data) is str:
            data=re.sub(r'[^\w\s]','',self.data)
            self.data=data
            return data
        else:
            data=[]
            for sentence in self.data:
                if remove_stop_words:
                    data.append(self.removeStopWords(re.sub(r'[^\w\s]','',(sentence))))
                else:
                    data.append(re.sub(r'[^\w\s]','',sentence))
            self.data=data
            return data
   
   
    def removeEmoji(self):
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

        if type(self.data) is str:
            data=re.sub(r'[^\w\s]','',self.data)
            self.data=data
            return data
        else:
            data=[]
            for sentence in self.data:
                data.append(emoji_pattern.sub(r'', sentence))
            self.data=data
            return data


    def removeStopWords(self,sentence):
        stopwords = list(self.stop_wods_frame['word'])
        querywords = sentence.split()
        resultwords  = [word.lower() for word in querywords if word.lower() not in stopwords]
        listToStr = ' '.join([str(elem) for elem in resultwords])
        return (listToStr)
        
                     
    def getData(self):
        return(self.data)

