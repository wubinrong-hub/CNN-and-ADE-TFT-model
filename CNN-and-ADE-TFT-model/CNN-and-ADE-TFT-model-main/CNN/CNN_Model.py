import pandas as pd
import numpy as np
#from keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, \
#Dropout, Activation, Input, Flatten, Concatenate
from keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, \
Dropout,Input, Flatten, Concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
#from numpy.random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#import gensim
from nltk.stem.wordnet import WordNetLemmatizer
import string
import matplotlib.pyplot as plt

import copy
from collections import defaultdict
import math
import operator
def feature_select(list_words):

    doc_frequency=defaultdict(int)
    for word_list in list_words:
        for i in word_list:
            doc_frequency[i]+=1
 
    word_tf={}  
    for i in doc_frequency:
        word_tf[i]=doc_frequency[i]/sum(doc_frequency.values())
 

    doc_num=len(list_words)
    word_idf={}
    word_doc=defaultdict(int) 
    for i in doc_frequency:
        for j in list_words:
            if i in j:
                word_doc[i]+=1
    for i in doc_frequency:
        word_idf[i]=math.log(doc_num/(word_doc[i]+1))
 

    word_tf_idf={}
    for i in doc_frequency:
        word_tf_idf[i]=word_tf[i]*word_idf[i]
 
    # 对字典按值由大到小排序
    dict_feature_select=sorted(word_tf_idf.items(),key=operator.itemgetter(1),reverse=True)
    return dict_feature_select

def cnn_model(FILTER_SIZES, \
              # filter sizes as a list
              MAX_NB_WORDS, \
              # total number of words
              MAX_DOC_LEN, \
              # max words in a doc
              EMBEDDING_DIM=100, \
              # word vector dimension
              NUM_FILTERS=64, \
              # number of filters for all size
              DROP_OUT=0.5, \
              # dropout rate 
              NUM_OUTPUT_UNITS=1, \
              # number of output units
              NUM_DENSE_UNITS=100,\
              # number of units in dense layer
              PRETRAINED_WORD_VECTOR=None,\
              # Whether to use pretrained word vectors
              LAM=0.0):            
              # regularization coefficient
    
    main_input = Input(shape=(MAX_DOC_LEN,), \
                       dtype='int32', name='main_input')
    
    if PRETRAINED_WORD_VECTOR is not None:
        embed_1 = Embedding(input_dim=MAX_NB_WORDS+1, \
                        output_dim=EMBEDDING_DIM, \
                        input_length=MAX_DOC_LEN, \
                        # use pretrained word vectors
                        weights=[PRETRAINED_WORD_VECTOR],\
                        # word vectors can be further tuned
                        # set it to False if use static word vectors
                        trainable=True,\
                        name='embedding')(main_input)
    else:
        embed_1 = Embedding(input_dim=MAX_NB_WORDS+1, \
                        output_dim=EMBEDDING_DIM, \
                        input_length=MAX_DOC_LEN, \
                        name='embedding')(main_input)
    # add convolution-pooling-flat block
    conv_blocks = []
    for f in FILTER_SIZES:
        conv = Conv1D(filters=NUM_FILTERS, kernel_size=f, \
                      activation='relu', name='conv_'+str(f))(embed_1)
        conv = MaxPooling1D(MAX_DOC_LEN-f+1, name='max_'+str(f))(conv)
        conv = Flatten(name='flat_'+str(f))(conv)
        conv_blocks.append(conv)
    
    if len(conv_blocks)>1:
        z=Concatenate(name='concate')(conv_blocks)
    else:
        z=conv_blocks[0]
        
    drop=Dropout(rate=DROP_OUT, name='dropout')(z)
    

    dense = Dense(NUM_DENSE_UNITS, activation='relu',\
                    kernel_regularizer=l2(LAM),name='dense')(drop)
    preds = Dense(NUM_OUTPUT_UNITS, activation='sigmoid', name='output')(dense)
    model = Model(inputs=main_input, outputs=preds)
    
    model.compile(loss="binary_crossentropy", \
              optimizer="adam", metrics=["accuracy"]) 
    
    return model

 
if __name__ == "__main__":  
    
    file_path = 'Paris.csv'  
    data=pd.read_csv('Paris.csv', header=0,encoding='gb18030')
   
    #MAX_NB_WORDS=200
    MAX_NB_WORDS= 11000
    MAX_DOC_LEN=4000
    #MAX_NB_WORDS=446
    #MAX_DOC_LEN=179
    EMBEDDING_DIM=100
   

#stopwords=read_csv('stopwords.csv')
    stopwords=pd.read_csv('stopwords.csv',index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='gb18030')
    #stopwords=pd.read_csv('stopwords.csv',index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
    stopwords=stopwords['stopword'].values
    #stopwords=stopwords['stopword'].astype(str)
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    def clean(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stopwords])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized
    doc_clean = [clean(doc).split() for doc in data["text"]]
  
    #tokenizer = Tokenizer(num_words=None)
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
   # tokenizer.fit_on_texts(data_list_new)
    tokenizer.fit_on_texts(doc_clean)
    #tokenizer.fit_on_texts(data["text"])

   # sequences = tokenizer.texts_to_sequences(data_list_new)
    sequences = tokenizer.texts_to_sequences(doc_clean)
    #sequences = tokenizer.texts_to_sequences(data["text"])
    #print(doc_clean)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_DOC_LEN, padding='post', truncating='post')
    #print(padded_sequences)
    mlb = MultiLabelBinarizer()
    Y=mlb.fit_transform(data.label.astype(str).tolist())
    output_units_num=len(mlb.classes_)

    best_predacc=1;
    max_time=100;
    y = np.random.standard_normal(max_time)
    for h in range(max_time): 
        FILTER_SIZES=[3,4,5]    
       # FILTER_SIZES=[2,3,4]
        #FILTER_SIZES=[1,2,3]
        BTACH_SIZE = 55
        NUM_EPOCHES=100
        #num_filters=32
        num_filters=128
        dense_units_num= num_filters*len(FILTER_SIZES)
    
        BEST_MODEL_FILEPATH="best_model"

        X_train, X_test, Y_train, Y_test = train_test_split(\
                    padded_sequences, Y, test_size=0.38, random_state=0)

        model=cnn_model(FILTER_SIZES, MAX_NB_WORDS, MAX_DOC_LEN, EMBEDDING_DIM=EMBEDDING_DIM, NUM_FILTERS=num_filters,
                    NUM_OUTPUT_UNITS=output_units_num, NUM_DENSE_UNITS=dense_units_num)
       # model=cnn_model(FILTER_SIZES, 299, MAX_DOC_LEN, EMBEDDING_DIM=EMBEDDING_DIM, NUM_FILTERS=num_filters,
                    #NUM_OUTPUT_UNITS=output_units_num, NUM_DENSE_UNITS=dense_units_num)
        earlyStopping=EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='min')
        checkpoint = ModelCheckpoint(BEST_MODEL_FILEPATH, monitor='val_loss', \
                                 verbose=2, save_best_only=True, mode='min')

        training=model.fit(X_train, Y_train, \
              batch_size=BTACH_SIZE, epochs=NUM_EPOCHES, \
              callbacks=[earlyStopping, checkpoint],\
              validation_data=(X_test, Y_test), verbose=2)
    
    # load the best model and predict
        model.load_weights("best_model")
       
        pred=model.predict(X_test)
        pred1=copy.deepcopy(pred)
        print(pred)
        pred=np.where(pred>0.5, 1, 0)
        print(pred)
        
        
    # Accuracy
        scores = model.evaluate(X_test, Y_test, verbose=0)
       # print(scores) [0.6921681785583496, 0.505] loss and accuracy
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # Generate performance report
        print(classification_report(Y_test, pred, target_names=mlb.classes_))
        y[h]=scores[1]*100;
        #print(scores[1]*100)
        if scores[1]*100>=best_predacc:
            best_predacc=scores[1]*100
            pred3=copy.deepcopy(pred)
            pred2=pred1
        print(h)
    print(best_predacc)
    print(classification_report(Y_test, pred3, target_names=mlb.classes_))
    print(pred2[:,0])
   # plt.plot(y)
   # plt.show()
    plt.plot(pred2[:,0])
    plt.show()
    
    import xlwt
    book = xlwt.Workbook(encoding='utf-8',style_compression=0)
    sheet = book.add_sheet('mysheet',cell_overwrite_ok=True)
    for i in range(40):
        b=float(pred2[i,0])
        sheet.write(i+1,1,b)
    book.save('C:\\Users\\Administrator\\Desktop\\test.xls')


   
