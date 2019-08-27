import tensorflow as tf
import scipy
import pandas as pd
import numpy as np
from periodictable import elements
from sklearn.utils import shuffle
from statistics import mean
# empty lists to hold semi-processed data and calibrate padding
def initialize():
    global trimmed, unnested, x_lengths, y_lengths
    trimmed = []
    unnested = []
    x_lengths = []
    y_lengths = []
# remove parentheses from DrugBank IDs
def trim(x):
    if '>' in x:  
        start = x.find('(') 
        start += 1 
        end = x.find(')')  
        trimmed.append(x[start:end])   
    else:     
        trimmed.append(x)
# match DrugBank IDs to protein sequences
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
# remove drugs with unwanted atoms from the dataset
def subset(x, elements, sub):
    elim = [str(i) for i in elements if str(i) not in sub]
    for i in elim:
        x = x[~x.SMILES.str.contains(i)]
    return x
# remove drugs (and corresponding proteins) and proteins (and corresponding drugs) that are too short or too long
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
# get an x and a y dataset                
def split(x, axis):
    return [i.split(',')[axis] for i in x]
   
def maxlength(x):       
    return max([len(i) for i in x])
# tokenize and pad datasets
def vectorize(x, char2id, maxlen):
    text_as_int = [char2id[c] for c in x]
    while len(text_as_int) <= maxlen:   
        text_as_int.append(0)   
    return text_as_int
# one-hot encode datasets
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
# 2x1x2 dilated convolutions    
    conv_2x2_d2_1 = tf.keras.layers.Conv1D(filters = 128, kernel_size = 2, strides = 2, activation = 'tanh', padding = 'valid')(visible)
    conv_2x2_d2_2 = tf.keras.layers.Conv1D(filters = 128, kernel_size = 2, strides = 2, activation = 'tanh', padding = 'valid')(conv_2x2_d2_1)
    conv_2x2_d2_3 = tf.keras.layers.Conv1D(filters = 128, kernel_size = 2, strides = 2, activation = 'tanh', padding = 'valid')(conv_2x2_d2_2)
    flat_2x2_d2 = tf.keras.layers.Flatten()(conv_2x2_d2_3)
    dense_2x2_d2_1 = tf.keras.layers.Dense(128)(flat_2x2_d2)
    drop_2x2_d2 = tf.keras.layers.Dropout(0.2)(dense_2x2_d2_1)
    dense_2x2_d2_2 = tf.keras.layers.Dense(128)(drop_2x2_d2)
# 2x1x3 dilated convolutions   
    conv_3x3_d2_1 = tf.keras.layers.Conv1D(filters = 128, kernel_size = 3, strides = 2, activation = 'tanh', padding = 'valid')(visible)
    conv_3x3_d2_2 = tf.keras.layers.Conv1D(filters = 128, kernel_size = 3, strides = 2, activation = 'tanh', padding = 'valid')(conv_3x3_d2_1)
    conv_3x3_d2_3 = tf.keras.layers.Conv1D(filters = 128, kernel_size = 3, strides = 2, activation = 'tanh', padding = 'valid')(conv_3x3_d2_2)
    flat_3x3_d2 = tf.keras.layers.Flatten()(conv_3x3_d2_3)
    dense_3x3_d2_1 = tf.keras.layers.Dense(128)(flat_3x3_d2)
    drop_3x3_d2 = tf.keras.layers.Dropout(0.2)(dense_3x3_d2_1)
    dense_3x3_d2_2 = tf.keras.layers.Dense(128)(drop_3x3_d2)
# 3x1x2 dilated convolutions   
    conv_2x2_d3_1 = tf.keras.layers.Conv1D(filters = 128, kernel_size = 2, strides = 3, activation = 'tanh', padding = 'valid')(visible)
    conv_2x2_d3_2 = tf.keras.layers.Conv1D(filters = 128, kernel_size = 2, strides = 3, activation = 'tanh', padding = 'valid')(conv_2x2_d3_1)
    conv_2x2_d3_3 = tf.keras.layers.Conv1D(filters = 128, kernel_size = 2, strides = 3, activation = 'tanh', padding = 'valid')(conv_2x2_d3_2)
    flat_2x2_d3 = tf.keras.layers.Flatten()(conv_2x2_d3_3)
    dense_2x2_d3_1 = tf.keras.layers.Dense(128)(flat_2x2_d3)
    drop_2x2_d3 = tf.keras.layers.Dropout(0.2)(dense_2x2_d3_1)
    dense_2x2_d3_2 = tf.keras.layers.Dense(128)(drop_2x2_d3)
# 3x1x3 dilated convolutions
    conv_3x3_d3_1 = tf.keras.layers.Conv1D(filters = 128, kernel_size = 3, strides = 3, activation = 'tanh', padding = 'valid')(visible)
    conv_3x3_d3_2 = tf.keras.layers.Conv1D(filters = 128, kernel_size = 3, strides = 3, activation = 'tanh', padding = 'valid')(conv_3x3_d3_1)
    conv_3x3_d3_3 = tf.keras.layers.Conv1D(filters = 128, kernel_size = 3, strides = 3, activation = 'tanh', padding = 'valid')(conv_3x3_d3_2)
    flat_3x3_d3 = tf.keras.layers.Flatten()(conv_3x3_d3_3)
    dense_3x3_d3_1 = tf.keras.layers.Dense(128)(flat_3x3_d3)
    drop_3x3_d3 = tf.keras.layers.Dropout(0.2)(dense_3x3_d3_1)
    dense_3x3_d3_2 = tf.keras.layers.Dense(128)(drop_3x3_d3)
# concatenate convolutional branches and feed to an RNN    
    concat = tf.keras.layers.concatenate([dense_2x2_d2_2, dense_3x3_d2_2, dense_2x2_d3_2, dense_3x3_d3_2])
    dense_1 = tf.keras.layers.Dense(256)(concat)
    drop = tf.keras.layers.Dropout(0.2)(dense_1)
    dense_2 = tf.keras.layers.Dense(256)(drop)
    rv = tf.keras.layers.RepeatVector(maxlen_y + 1)(dense_2)
    gru1 = tf.keras.layers.CuDNNGRU(256, return_sequences = True)(rv)
    gru2 = tf.keras.layers.CuDNNGRU(256, return_sequences = True)(gru1)
    gru3 = tf.keras.layers.CuDNNGRU(256, return_sequences = True)(gru2)
    
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense((vocab_size_y), activation = 'softmax'))(gru3)
    
    model = tf.keras.Model(inputs = [visible], outputs = [output])
    model.compile(loss = 'categorical_crossentropy', optimizer = tf.train.AdamOptimizer(1e-4), metrics = ['accuracy'])
    return model

def decode(x):
    x = [[np.argmax(i) for i in j] for j in x]
    x = [id_y_2char[i] for i in x]
    print(x)

if __name__ == '__main__':
    initialize()
    # remove FASTA formatting
    with open('protein.fasta', 'r') as f:    
        protein_data = f.read().replace('\n', '').replace(')', ')\n').replace('>', '\n>')    
        protein_data = protein_data.split('\n')
    organize(protein_data)
    protein_df = pd.DataFrame(unnested)
    protein_df.columns = ['DrugBank ID', 'Sequence']
    compound_df = pd.read_csv('structure links 5.csv')[['DrugBank ID', 'SMILES']]
    # match drugs to protein sequences
    data = protein_df.merge(compound_df, on = 'DrugBank ID', how = 'outer').dropna().drop_duplicates()[['Sequence', 'SMILES']]
    # drugs that have atoms not in this set will be removed
    sub = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'Se', 'I'] 
    data = subset(data, elements, sub)
    # all atoms must have only one unique token
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
        # vocab list for tokenizing, padding, and one-hot encoding
        vocab_x, vocab_y  = sorted(set(''.join(data_x))), sorted(set(''.join(data_y)))  
        vocab_size_x, vocab_size_y = len(vocab_x), len(vocab_y)  
        char2id_x, char2id_y  = {u:i for i, u in enumerate(vocab_x)}, {u:i for i, u in enumerate(vocab_y)}
        maxlen_x, maxlen_y  = maxlength(data_x), maxlength(data_y)
        # decoding key
        id_y_2char = np.array(vocab_y)
        # train and test datasets
        x_data_train, y_data_train = data_x[:10000], data_y[:10000]
        x_data_test, y_data_test = data_x[10000:], data_y[10000:]
    x_data_train, x_data_test = expand(x_data_train, char2id_x, maxlen_x, vocab_size_x), expand(x_data_test, char2id_x, maxlen_x, vocab_size_x)
    y_data_train, y_data_test = expand(y_data_train, char2id_y, maxlen_y, vocab_size_y), expand(y_data_test, char2id_y, maxlen_y, vocab_size_y)
    model = build_model()
    # train on 4 GPUs
    strategy = tf.contrib.distribute.MirroredStrategy(num_gpus = 4)
    config = tf.estimator.RunConfig(train_distribute = strategy)
    estimator = tf.keras.estimator.model_to_estimator(keras_model = model, config = config, model_dir = 'logs/')
    # shuffles the dataset and feeds to to estimator in batches of 256 for 5000 epochs
    train_input_fn = lambda: tf.data.Dataset.from_tensor_slices((x_data_train, y_data_train)).shuffle(10000).batch(256).repeat(5000)
    eval_input_fn = lambda: tf.data.Dataset.from_tensor_slices((x_data_test, y_data_test)).shuffle(10000).batch(256).repeat(5000)
    train_spec = tf.estimator.TrainSpec(input_fn = train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn = eval_input_fn)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    # target is the MRSA MntC Manganese ion transporter
    target = expand(['MKKLVPLLLALLLLVAACGTGGKQSSDKSNGKLKVVTTNSILYDMAKNVGGDNVDIHSIVPVGQDPHEYEVKPKDIKKLTDADVILYNGLNLETGNGWFEKALEQAGKSLKDKKVIAVSKDVKPIYLNGEEGNKDKQDPHAWLSLDNGIKYVKTIQQTFIDNDKKHKADYEKQGNKYIAQLEKLNNDSKDSKDKFNDIPKEQRAMITSEGAFKYFSKQYGITPGYIWEINTEKQGTPEQMRQAIEFVKKHKLKHLLVETSVDKKAMESLSEETKKDIFGEVYTDSIGKEGTKGDSYYKMMKSNIETVHGSMK'], char2id_x, maxlen_x, vocab_size_x)
    target_input_fn = lambda: tf.data.Dataset.from_tensor_slices(target).batch(1)
    prediction = estimator.predict(input_fn = target_input_fn)
    # retrieving prediction from the estimator
    prediction = next(prediction)
    prediction = [prediction.popitem()[1]]
    print(prediction)
    decode(prediction)
