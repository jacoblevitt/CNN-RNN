import pandas as pd
import numpy as np
from periodictable import elements
from sklearn.utils import shuffle
from statistics import mean
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

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
    return(unnested)

def subset(x, elements, sub):
    elim = [str(i) for i in elements if str(i) not in sub]
    for i in elim:
        x = x[~x.SMILES.str.contains(i)]
    return(x)

def reduce(x, x_lengths, y_lengths):
    for i in x:
        x_lengths.append(len(i.split(',')[0]))
    for i in x:
        y_lengths.append(len(i.split(',')[1])) 
    x_lengths_mean = mean(x_lengths)
    y_lengths_mean = mean(y_lengths)
    x_lengths_SD = np.std(np.array(x_lengths))
    y_lengths_SD = np.std(np.array(y_lengths))
    return([i for i in x if len(i.split(',')[0]) >= x_lengths_mean - x_lengths_SD and len(i.split(',')[0]) <= x_lengths_mean + x_lengths_SD and len(i.split(',')[0]) >= y_lengths_mean - y_lengths_SD and len(i.split(',')[1]) <= y_lengths_mean + y_lengths_SD])
    
def split(x, axis):
    return([i.split(',')[axis] for i in x])
   
def maxlength(x):
    return(max([len(i) for i in x]))

def vectorize(x, char2id, maxlen):
    text_as_int = [char2id[c] for c in x]
    while len(text_as_int) <= maxlen:   
        text_as_int.append(0)   
    return(text_as_int)

def extract(x, char2id, maxlen, vocab, element, inversion):
    data_global = []
    data_vectorized = [vectorize(i, char2id, maxlen) for i in x]
    for i in data_vectorized:
        data_local = []
        for j in list(i):
            if inversion == 0:
                if j == vocab.index(element):
                    data_local.append(vocab.index(element))
                else:
                    data_local.append(0)
            if inversion == 1:
                if j == vocab.index(element):
                    data_local.append(0)
                else:
                    data_local.append(vocab.index(element))
        data_global.append(data_local)
    return(np.array(data_global))

def onehot(x, vocab_size): 
    onehot_encoded = [] 
    for i in x:    
        j = keras.utils.to_categorical(i, vocab_size).tolist()
        onehot_encoded.append(j)
    return(np.array(onehot_encoded))

def expand(x, char2id, maxlen, vocab_size):
    data_processed = [vectorize(i, char2id, maxlen) for i in x]
    data_processed = np.array([onehot(i, vocab_size) for i in data_processed])
    return(data_processed)

def scale(x, char2id, maxlen):
    global scaler
    scaler = MinMaxScaler()
    data_vectorized = [vectorize(i, char2id, maxlen) for i in x]
    scaler.fit(data_vectorized)
    data_vectorized = scaler.transform(data_vectorized)
    return(np.array(data_vectorized))

def build():
    global model
    visible = keras.Input(shape = (maxlen_x + 1, vocab_size_x))
    conv_2x2_1 = layers.Conv1D(filters = 256, kernel_size = 2, strides = 1, activation = 'relu', padding = 'valid')(visible)
    conv_2x2_2 = layers.Conv1D(filters = 256, kernel_size = 2, strides = 1, activation = 'relu', padding = 'valid')(conv_2x2_1)
    conv_2x2_3 = layers.Conv1D(filters = 256, kernel_size = 2, strides = 1, activation = 'relu', padding = 'valid')(conv_2x2_2)
    flat_2x2 = layers.Flatten()(conv_2x2_3)
    conv_3x3_1 = layers.Conv1D(filters = 256, kernel_size = 3, strides = 1, activation = 'relu', padding = 'valid')(visible)
    conv_3x3_2 = layers.Conv1D(filters = 256, kernel_size = 3, strides = 1, activation = 'relu', padding = 'valid')(conv_3x3_1)
    conv_3x3_3 = layers.Conv1D(filters = 256, kernel_size = 3, strides = 1, activation = 'relu', padding = 'valid')(conv_3x3_2)
    flat_3x3 = layers.Flatten()(conv_3x3_3)
    concat = layers.concatenate([flat_2x2, flat_3x3])
    dense_1 = layers.Dense(128)(concat)
    dense_2 = layers.Dense(128)(dense_1)
    output = layers.Dense((maxlen_y + 1), activation = 'relu')(dense_2)
    model = keras.Model(inputs = visible, outputs = [output])
    print(model.summary())
    return(model)

def fit(model, x_data_train, y_data_train,
        x_data_test, y_data_test,
        filename, optimizer):
    model.compile(loss = 'mse', optimizer = optimizer, metrics = ['accuracy'])
    model.fit(x_data_train, [y_data_train],
              validation_data = (x_data_test, [y_data_test]),
              epochs = 1000, batch_size = 128)
    model.save(filename)

def decode(x, scaler, vocab):
    x = [np.round(scaler.inverse_transform(x)).tolist() for i in x]
    x = [[[vocab[int(k)] for k in j] for j in i] for i in x]
    print(x)

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
    sub = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl']
    data = subset(data, elements, sub)
    data['SMILES'] = data['SMILES'].str.replace('Cl', 'L')
    print(data)
    data = shuffle(data)
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
    x_data_train_processed, x_data_test_processed = expand(x_data_train, char2id_x, maxlen_x, vocab_size_x), expand(x_data_test, char2id_x, maxlen_x, vocab_size_x)
    y_data_train, y_data_test = scale(y_data_train, char2id_y, maxlen_y), scale(y_data_test, char2id_y, maxlen_y)
    model = build()
    fit(model, x_data_train_processed, y_data_train,
        x_data_test_processed, y_data_test,
        'model.h5', keras.optimizers.Adam(1e-4))  
    target = expand(['GTGGKQSSDKSNGKLKVVTTNSILYDMAKNVGGDNVDIHSIVPVGQDPHEYEVKPKDIKKLTDADVILYNGLNLETGNGWFEKALEQAGKSLKDKKVIAVSKDVKPIYLNGEEGNKDKQDPHAWLSLDNGIKYVKTIQQTFIDNDKKHKADYEKQGNKYIAQLEKLNNDSKDKFNDIPKEQRAMITSEGAFKYFSKQYGITPGYIWEINTEKQGTPEQMRQAIEFVKKHKLKHLLVETSVDKKAMESLSEETKKDIFGEVYTDSIGKEGTKGDSYYKMMKSNIETVHGSMK'], char2id_x, maxlen_x, vocab_size_x)
    prediction = model.predict(target)
    decode(prediction, scaler, vocab_y)
