import tensorflow as tf
import scipy
import pandas as pd
import numpy as np
from periodictable import elements
from sklearn.utils import shuffle
from statistics import mean
from tensorflow.keras import callbacks
from datetime import datetime

def initialize():
    global trimmed, unnested, x_lengths, y_lengths
    trimmed = []
    unnested = []
    x_lengths = []
    y_lengths = []

def trim(x):
    if '>' in x:  
        start = x.find('(') 
        start += 1 
        end = x.find(')')  
        trimmed.append(x[start:end])   
    else:     
        trimmed.append(x)

def unnest(x):
    for i in x:    
        if 'DB' in i:       
            try:  
                j = i.split('; ')  
                for k in j:      
                    unnested.append((k, str(x[x.index(i) + 1])))     
            except: 
                unnested.append((i, str(x[x.index(i) + 1])))

def organize(x):
    for i in x:     
        trim(i)    
    unnest(trimmed)
    return unnested

def subset(x, elements, sub):
    elim = [str(i) for i in elements if str(i) not in sub]
    for i in elim:
        x = x[~x.SMILES.str.contains(i)]
    return x

def reduce(x, x_lengths, y_lengths):
    reduced = []
    for i in x:
        x_lengths.append(len(i.split(',')[0]))
    for i in x:
        y_lengths.append(len(i.split(',')[1])) 
    x_lengths_mean = mean(x_lengths)
    y_lengths_mean = mean(y_lengths)
    x_lengths_SD = np.std(np.array(x_lengths))
    y_lengths_SD = np.std(np.array(y_lengths))
    for i in x:
        if len(i.split(',')[0]) >= (x_lengths_mean - x_lengths_SD) and len(i.split(',')[0]) <= (x_lengths_mean + x_lengths_SD):
            if len(i.split(',')[0]) >= (y_lengths_mean - y_lengths_SD) and len(i.split(',')[1]) <= (y_lengths_mean + y_lengths_SD):
                reduced.append(i)
    return reduced
                
def split(x, axis):
    return [i.split(',')[axis] for i in x]
   
def maxlength(x):       
    return max([len(i) for i in x])

def vectorize(x, char2id, maxlen):
    text_as_int = [char2id[c] for c in x]
    while len(text_as_int) <= maxlen:   
        text_as_int.append(0)   
    return text_as_int

def onehot(x, vocab_size): 
    onehot_encoded = [] 
    for i in x:    
        j = tf.keras.utils.to_categorical(i, vocab_size).tolist()
        onehot_encoded.append(j)
    return np.array(onehot_encoded)

def expand(x, char2id, maxlen, vocab_size):
    data_processed = [vectorize(i, char2id, maxlen) for i in x]
    data_processed = np.array([onehot(i, vocab_size) for i in data_processed])
    return data_processed

def build_model():
    visible = tf.keras.Input(shape = (maxlen_x + 1, vocab_size_x))
    
    conv_2x2_1 = tf.keras.layers.Conv1D(filters = 512, kernel_size = 2, strides = 1, activation = 'relu', padding = 'valid')(visible)
    conv_2x2_2 = tf.keras.layers.Conv1D(filters = 512, kernel_size = 2, strides = 1, activation = 'relu', padding = 'valid')(conv_2x2_1)
    conv_2x2_3 = tf.keras.layers.Conv1D(filters = 512, kernel_size = 2, strides = 1, activation = 'relu', padding = 'valid')(conv_2x2_2)
    flat_2x2 = tf.keras.layers.Flatten()(conv_2x2_3)
    dense_2x2_1 = tf.keras.layers.Dense(256)(flat_2x2)
    drop_2x2 = tf.keras.layers.Dropout(0.2)(dense_2x2_1)
    dense_2x2_2 = tf.keras.layers.Dense(256)(drop_2x2)
    
    conv_3x3_1 = tf.keras.layers.Conv1D(filters = 512, kernel_size = 3, strides = 1, activation = 'relu', padding = 'valid')(visible)
    conv_3x3_2 = tf.keras.layers.Conv1D(filters = 512, kernel_size = 3, strides = 1, activation = 'relu', padding = 'valid')(conv_3x3_1)
    conv_3x3_3 = tf.keras.layers.Conv1D(filters = 512, kernel_size = 3, strides = 1, activation = 'relu', padding = 'valid')(conv_3x3_2)
    flat_3x3 = tf.keras.layers.Flatten()(conv_3x3_3)
    dense_3x3_1 = tf.keras.layers.Dense(256)(flat_3x3)
    drop_3x3 = tf.keras.layers.Dropout(0.2)(dense_3x3_1)
    dense_3x3_2 = tf.keras.layers.Dense(256)(drop_3x3)
    
    conv_2x2_d_1 = tf.keras.layers.Conv1D(filters = 512, kernel_size = 2, strides = 2, activation = 'relu', padding = 'valid')(visible)
    conv_2x2_d_2 = tf.keras.layers.Conv1D(filters = 512, kernel_size = 2, strides = 2, activation = 'relu', padding = 'valid')(conv_2x2_d_1)
    conv_2x2_d_3 = tf.keras.layers.Conv1D(filters = 512, kernel_size = 2, strides = 2, activation = 'relu', padding = 'valid')(conv_2x2_d_2)
    flat_2x2_d = tf.keras.layers.Flatten()(conv_2x2_d_3)
    dense_2x2_d_1 = tf.keras.layers.Dense(256)(flat_2x2_d)
    drop_2x2_d = tf.keras.layers.Dropout(0.2)(dense_2x2_d_1)
    dense_2x2_d_2 = tf.keras.layers.Dense(256)(drop_2x2_d)

    conv_3x3_d_1 = tf.keras.layers.Conv1D(filters = 512, kernel_size = 3, strides = 2, activation = 'relu', padding = 'valid')(visible)
    conv_3x3_d_2 = tf.keras.layers.Conv1D(filters = 512, kernel_size = 3, strides = 2, activation = 'relu', padding = 'valid')(conv_3x3_d_1)
    conv_3x3_d_3 = tf.keras.layers.Conv1D(filters = 512, kernel_size = 3, strides = 2, activation = 'relu', padding = 'valid')(conv_3x3_d_2)
    flat_3x3_d = tf.keras.layers.Flatten()(conv_3x3_d_3)
    dense_3x3_d_1 = tf.keras.layers.Dense(256)(flat_3x3_d)
    drop_3x3_d = tf.keras.layers.Dropout(0.2)(dense_3x3_d_1)
    dense_3x3_d_2 = tf.keras.layers.Dense(256)(drop_3x3_d)
    
    concat = tf.keras.layers.concatenate([dense_2x2_2, dense_3x3_2, dense_2x2_d_2, dense_3x3_d_2])
    dense_1 = tf.keras.layers.Dense(256)(concat)
    drop = tf.keras.layers.Dropout(0.2)(dense_1)
    dense_2 = tf.keras.layers.Dense(256)(drop)
    rv = tf.keras.layers.RepeatVector(maxlen_y + 1)(dense_2)
    lstm = tf.keras.layers.CuDNNLSTM(256, return_sequences = True)(rv)
    
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense((vocab_size_y), activation = 'softmax'))(lstm)
    
    model = tf.keras.Model(inputs = [visible], outputs = [output])
    model.compile(loss = 'categorical_crossentropy', optimizer = tf.train.AdamOptimizer(5e-3), metrics = ['accuracy'])
    return model

def decode(x):
    x = [[np.argmax(i) for i in j] for j in x]
    x = [id_y_2char[i] for i in x]
    return x

if __name__ == '__main__':
    initialize()
    with open('protein.fasta', 'r') as f:    
        protein_data = f.read().replace('\n', '').replace(')', ')\n').replace('>', '\n>')    
        protein_data = protein_data.split('\n')
    organize(protein_data)
    protein_df = pd.DataFrame(unnested)
    protein_df.columns = ['DrugBank ID', 'Sequence']
    compound_df = pd.read_csv('structure links 5.csv')[['DrugBank ID', 'SMILES']]
    data = protein_df.merge(compound_df, on = 'DrugBank ID', how = 'outer').dropna().drop_duplicates()[['Sequence', 'SMILES']]
    sub = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'Se', 'I']
    data = subset(data, elements, sub)
    data['SMILES'] = data['SMILES'].str.replace('Cl', 'L')
    data['SMILES'] = data['SMILES'].str.replace('Br', 'R')
    data['SMILES'] = data['SMILES'].str.replace('Se', 'E')
    data = shuffle(data)
    print(data)
    data.to_csv('data.csv', index = False, header = None)
    with open('data.csv', 'r') as f:  
        data = f.readlines()[:-1]  
        data = reduce(data, x_lengths, y_lengths)
        data_x, data_y  = split(data, 0), split(data, 1)  
        vocab_x, vocab_y  = sorted(set(''.join(data_x))), sorted(set(''.join(data_y)))  
        vocab_size_x, vocab_size_y = len(vocab_x), len(vocab_y)  
        char2id_x, char2id_y  = {u:i for i, u in enumerate(vocab_x)}, {u:i for i, u in enumerate(vocab_y)}
        id_y_2char = np.array(vocab_y)
        maxlen_x, maxlen_y  = maxlength(data_x), maxlength(data_y)
        x_data_train, y_data_train = data_x[:10000], data_y[:10000]
        x_data_test, y_data_test = data_x[10000:], data_y[10000:]
    x_data_train, x_data_test = expand(x_data_train, char2id_x, maxlen_x, vocab_size_x), expand(x_data_test, char2id_x, maxlen_x, vocab_size_x)
    y_data_train, y_data_test = expand(y_data_train, char2id_y, maxlen_y, vocab_size_y), expand(y_data_test, char2id_y, maxlen_y, vocab_size_y)
    model = build_model()
    csv_logger = callbacks.CSVLogger('training_' + datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv')
    model.fit([x_data_train], [y_data_train], epochs = 100, batch_size = 128, validation_split = 0.1, callbacks = [csv_logger])
    model.save('model_' + datetime.now().strftime('%Y%m%d-%H%M%S') + '.h5')
    model.evaluate([x_data_test], [y_data_test], batch_size = 256)
    target = expand(['GTGGKQSSDKSNGKLKVVTTNSILYDMAKNVGGDNVDIHSIVPVGQDPHEYEVKPKDIKKLTDADVILYNGLNLETGNGWFEKALEQAGKSLKDKKVIAVSKDVKPIYLNGEEGNKDKQDPHAWLSLDNGIKYVKTIQQTFIDNDKKHKADYEKQGNKYIAQLEKLNNDSKDKFNDIPKEQRAMITSEGAFKYFSKQYGITPGYIWEINTEKQGTPEQMRQAIEFVKKHKLKHLLVETSVDKKAMESLSEETKKDIFGEVYTDSIGKEGTKGDSYYKMMKSNIETVHGSMK'], char2id_x, maxlen_x, vocab_size_x)
    prediction = model.predict(target)
    prediction = decode(prediction)
    print(prediction)
