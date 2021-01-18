# from sklearn.preprocessing import 
import os, sys
from tqdm import tqdm
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk import tokenize
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve,auc, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import joblib

sw = stopwords.words('portuguese')

def preprocess_data(content, model):
    sent_vec_ls = []
    word_ls =  tokenize.word_tokenize(content, language='portuguese') 
    word_ls = [i for i in word_ls if i not in sw]
    for w in word_ls:
        try:
            wv = model[w]
            sent_vec_ls.append(wv)
        except:
            continue
    mean_sent_ls = sum(sent_vec_ls) / (len(sent_vec_ls) + 1e-15)
    # return np.array(mean_sent_ls, dtype='float')
    try:
        mean_sent_ls = mean_sent_ls.tolist()
    except:
        return None
    if isinstance(mean_sent_ls, list):
        return mean_sent_ls

def load_pos_neg_data(pos_path, neg_path, model_path):
    pos_df = pd.read_csv(pos_path, sep='|', header=None, encoding='utf-8')
    neg_df = pd.read_csv(neg_path, sep='|', header=None, encoding='utf-8')
    
    pos_df.columns = ["ticket_id", "country_code", "organization_id", "content", "category_id_and_trans", "category_id"]
    neg_df.columns = ["ticket_id", "country_code", "organization_id", "content", "category_id_and_trans", "category_id"]

    pos_df = pos_df.dropna()
    neg_df = neg_df.dropna()

    model = KeyedVectors.load_word2vec_format(model_path)
    # pos / neg
    X_pos = []
    for content in tqdm(pos_df['content']):
        content_vec = preprocess_data(content, model)
        if content_vec == None:
            continue
        X_pos.append(content_vec)
    X_neg = []
    for content in tqdm(neg_df['content']):
        content_vec = preprocess_data(content, model)
        if content_vec == None:
            continue
        X_neg.append(content_vec)
    # train: vec,  test: 1/0
    X_total = X_pos + X_neg
    X_train = np.array(X_total, dtype='float')
    Y_train = np.concatenate((np.ones(len(X_pos)), np.zeros(len(X_neg))))
    return X_train, Y_train
    

def pac_model(X_train):
    pca = PCA(n_components=20).fit(X_train)
    # plt.plot(pca.explained_variance_,linewidth=2)
    # plt.show()
    X_reduced = pca.fit_transform(X_train)
    # joblib.dump(pca, 'tree_model/pca.pkl')
    print(X_reduced)
    print(X_reduced.shape)
    return X_reduced

def random_forest(X,  Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=66)
    model.fit(X_train, Y_train)
    print("准确率：%s" % model.score(X_test, Y_test))
    joblib.dump(model, 'tree_model/RandomForest.pkl')

    # eval
    Y_pred2 = model.predict(X_test)

    # print(Y_pred)
    print('........')
    print(Y_test.shape)
    # Y_pred2 = np.argmax(Y_pred, 1)
    print('.........')
    print(Y_pred2.shape)

    one_like_true = np.ones_like(Y_test)
    one_like_pred = np.ones_like(Y_pred2)
    zero_like_true = np.zeros_like(Y_test)
    zero_like_pred = np.zeros_like(Y_pred2)
    
    tp = np.sum(np.logical_and(np.equal(Y_test, one_like_true), np.equal(Y_pred2, one_like_pred)).astype(int))
    tn = np.sum(np.logical_and(np.equal(Y_test, zero_like_true), np.equal(Y_pred2, zero_like_pred)).astype(int))
    fp = np.sum(np.logical_and(np.equal(Y_test, zero_like_true), np.equal(Y_pred2, one_like_pred)).astype(int))
    fn = np.sum(np.logical_and(np.equal(Y_test, one_like_true), np.equal(Y_pred2, zero_like_pred)).astype(int))

    p_total = np.sum(Y_test == 1)
    n_total = np.sum(Y_test == 0)

    print("pos total: %d, neg total:%d" % (p_total, n_total))
    print("tp: %d, fp: %d, tn: %d, fn: %d" % (tp, fp, tn, fn))

    auc_v = (tp + tn) / (tp + fp + tn + fn + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    precision = tp / (tp + fp + 1e-5)     

    print("auccuray: %.2f" % auc_v)
    print("recall: %.2f" % recall)
    print("precision: %.2f" % precision)


    # print(a)
    # print("auccuray: %.2f" % accuracy_score(Y_test, Y_pred2))
    # print("recall: %.2f" % recall_score(Y_test, Y_pred2))
    # print("precision: %.2f" % precision_score(Y_test, Y_pred2))

    
    Y_pred = model.predict_proba(X_test)[:, 1]
    # Y_pred = Y_pred[:, 1]
    fpr,tpr,_ = roc_curve(Y_test, Y_pred)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label ='area=%.2f' %roc_auc)
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    path3 = sys.argv[3]
    X_train, Y_train = load_pos_neg_data(path1, path2, path3)
    X_train_reduce = pac_model(X_train)
    random_forest(X_train, Y_train)

    # load_data(path)

    # (((cancel)(.*?)( solicit(.*?)| corrida))|((sem)(.*?)( mi(.*?)| final(.*?)| passag(.*?)))|((mot(.*?)|ele)+( recus(.*?)| não most(.*?)| não veiu| não fez| cancel(.*?)| nao encer(.*?)| deu| não quis| embarcou| pegou| saiu sem))|((não|nao|não quis)+( est| vem| veio| vêm| cheg(.*?)| embar(.*?)| fiz| apare(.*?)| acont(.*?)| encontr(.*?)| solicit(.*?)| entr(.*?)| me bus(.*?)| final(.*?)| realizar)|((não)(.*?)( essa| feita| no carro| buscar)))|(no carro)|(quis fazer a corrida))
    
    # (cancel)(.*?)( solicit(.*?)| corrida)