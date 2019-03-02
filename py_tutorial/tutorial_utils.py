from sklearn.feature_extraction import DictVectorizer
import json 
import numpy as np 

def load_x_y_from_json(file_name): 
    count_dicts = []
    y = []
    for line in open(file_name): 
        dd = json.loads(line)
        counts = dd['counts'].copy()
        cc = dd['class']
        count_dicts.append(counts); y.append(cc)
    return count_dicts, np.array(y)  

def prune_vocab(X_train, dv_vocab): 
    #remove words that occur in <5 docs 
    xx=X_train.copy()
    xx[xx>0]=1
    w_df = np.asarray(xx.sum(0)).flatten()
    new_vocab_mask = w_df >= 5
    print("Orig vocab %d, pruned %d" % (len(w_df), np.sum(new_vocab_mask)))
    X_train = X_train[:,new_vocab_mask]
    dv_vocab = dv_vocab[new_vocab_mask]
    return X_train, dv_vocab, new_vocab_mask

def get_train_data(fname):
    dv = DictVectorizer()
    train_count_dicts, y_train = load_x_y_from_json(fname)
    X_train = dv.fit_transform(train_count_dicts).toarray()
    dv_vocab = np.array(dv.feature_names_)
    X_train, dv_vocab, new_vocab_mask = prune_vocab(X_train, dv_vocab)
    # print("X,Y types:", type(X_train), type(y_train))
    print("Training X shape:",X_train.shape, "and Y shape:", y_train.shape)
    assert X_train.shape[0] == y_train.shape[0] 
    return X_train, y_train, dv, new_vocab_mask

def transform_test(test_count_dicts, vocab_mask, dict_vect): 
    X_test = dict_vect.transform(test_count_dicts).toarray()
    X_test = X_test[:,vocab_mask]
    return X_test

def get_test_group(fname, vocab_mask, dict_vect):
    """
    get test data (1 test group) 
    NOTE: the test group is the "inference" group in a real-word setting
    here we have labels on the test set, but in a real-word setting there 
    would most likely not be labels on the test set
    """
    test_count_dicts, y_test = load_x_y_from_json(fname)
    X_test = transform_test(test_count_dicts, vocab_mask, dict_vect)
    # print("X,Y types:", type(X_test), type(y_test))
    print("Testing X shape:", X_test.shape, "and Y shape:", y_test.shape)
    return X_test, y_test  

