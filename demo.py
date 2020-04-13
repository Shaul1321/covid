import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import faiss
import bert
from bert import BertEncoder
import pickle
import spike_queries
import sklearn
import time

@st.cache(allow_output_mutation=True)
def load_sents_and_ids():
    with st.spinner('Loading sentences and IDs...'):
        df = pd.read_csv("covid-all-sents-2.4.csv", sep = "\t")
        sents =  df["sentence_text"].tolist()
        ids = [hash(s) for s in sents]

        id2ind = {ids[i]:i for i,s in enumerate(sents)}
        ind2id = {i:ids[i] for i,s in enumerate(sents)}

        return df, sents, ids, id2ind, ind2id

@st.cache(allow_output_mutation=True)
def load_index(similarity, pooling):
    with st.spinner('Loading FAISS index...'):
        fname = "output.new." + pooling + ".index"
        index = faiss.read_index(fname)
        return index

@st.cache(allow_output_mutation=True)
def load_bert():
    with st.spinner('Loading BERT...'):
        model = bert.BertEncoder("cpu")
        return model

@st.cache(allow_output_mutation=True)        
def load_pca(pooling):

    fname = "output.new." + pooling + ".pca.pickle"
    with open(fname, "rb") as f:
    
        return pickle.load(f)
    
               
st.title('COVID-19 Similarity Search')

#a = st.empty()
mode = st.sidebar.radio("Mode", ("Sentence", "SPIKE-covid19"))
similarity = st.sidebar.selectbox('Similarity', ('dot product', "l2"))
pooling = st.sidebar.selectbox('Pooling', ('cls', 'mean-cls'))

if mode == "Sentence":
    filter_by_spike = True if st.sidebar.selectbox('Filter by SPIKE query?', ('False', 'True'))=="True" else False

df, sents, ids, id2ind, ind2id = load_sents_and_ids()
#sents =  df["sentence_text"].tolist()
#ids = [hash(s) for s in sents]
print("len sents", len(sents))
#print("Creating dicts...")
#id2ind = {ids[i]:i for i,s in enumerate(sents)}
#ind2id = {i:ids[i] for i,s in enumerate(sents)}
#print("Done.")

index = load_index(similarity, pooling)
bert = load_bert()
pca = load_pca(pooling)
st.write("Uses {}-dimensional vectors".format(pca.components_.shape[0]))

if mode == "Sentence" and filter_by_spike:

    #filter_query = st.text_input('Enter a SPIKE query to filter by', 'This [nsubj drug] treats [obj:l coronavirus].')

    query_type = st.radio("Query type", ("syntactic", "boolean", "token"))
    if query_type == "syntactic":
        filter_query = st.text_input('Query', 'The [subj:l coronavirus] [copula:w is] prevalent among [w:e bats].')
    elif query_type == "boolean":
       filter_query = st.text_input('Query', 'virus lemma=persist on')
    elif query_type == "token":
       filter_query = st.text_input('Query', 'novel coronavirus')


    filter_size = int(st.text_input('How many SPIKE search results?',  3000))
    results_df = spike_queries.perform_query(filter_query, dataset_name = "covid19", num_results = filter_size, query_type = query_type)
    results_sents = np.array(results_df["sentence_text"].tolist())
    results_ids = [hash(s) for s in results_sents]


if mode == "Sentence":

    input_sentence = st.text_input('Input sentence', 'The virus can spread rapidly via different transimission vectors.')
elif "SPIKE" in mode:

    query_type = st.radio("Query type", ("syntactic", "boolean", "token"))
    if query_type == "syntactic":
        input_query = st.text_input('Query', 'The [subj:l coronavirus] [copula:w is] prevalent among [w:e bats].')
    elif query_type == "boolean":
       input_query = st.text_input('Query', 'virus lemma=persist on')
    elif query_type == "token":
       input_query = st.text_input('Query', 'novel coronavirus')

    max_results = int(st.text_input("How many resuls?", 25))

            
show_results = True
start = st.button('Run')


if start:
 if mode == "Sentence":

    #input_sentence = st.text_input('Input sentence', 'The virus can spread rapidly via different transimission vectors.')
    encoding = pca.transform(bert.encode([input_sentence], [1], batch_size = 1, strategy = pooling, fname = "dummy.txt", write = False))#.squeeze()
    #st.write("filter by spike: ", filter_by_spike)
    #st.write(encoding.shape)
    #st.write(index.d)
    
    if not filter_by_spike:
    
        D,I = index.search(np.ascontiguousarray(encoding), 100)
    
    else:
        
        encoding_of_spike_results = np.array([index.reconstruct(id2ind[i]) for i in results_ids if i in id2ind])
        sims = sklearn.metrics.pairwise.cosine_similarity(encoding, encoding_of_spike_results)
        idx_sorted = sims.argsort()[0]
        spike_sents_sorted = results_sents[idx_sorted][::-1]
        I = np.array([[id2ind[hash(s)] for s in spike_sents_sorted if hash(s) in id2ind]])

 elif mode == "IDs":
    input_ids = st.text_input('Input ids', '39, 41, 49, 50, 112, 116, 119, 229, 286, 747')
    input_ids = [int(x) for x in input_ids.replace(" ", "").split(",")]
    st.write("First sentences corrsponding to those IDs:")
    l = range(min(10, len(input_ids) ) )
    query_sents = [sents[id2ind[input_ids[i]]] for i in l]
    st.table(query_sents) 
    encoding = np.array([index.reconstruct(id2ind[i]) for i in input_ids])
    encoding = np.mean(encoding, axis = 0)
    D,I = index.search(np.ascontiguousarray([encoding]), 100)


 elif "SPIKE" in mode:

    with st.spinner('Performing SPIKE query...'):
        results_df = spike_queries.perform_query(input_query, dataset_name = "covid19", num_results = max_results, query_type = query_type)
        results_sents = results_df["sentence_text"].tolist()
        results_ids = [hash(s) for s in results_sents] #results_df["sentence_id"].tolist()
        
        if len(results_sents) > 0:
            st.write("First sentences retrieved:")
            st.table(results_sents[:10])
            
            encoding = np.array([index.reconstruct(id2ind[i]) for i in results_ids if i in id2ind])
            encoding = np.mean(encoding, axis = 0)
            D,I = index.search(np.ascontiguousarray([encoding]), 100)
            
            #encoding = pca.transform(bert.encode(results_sents, [1]*len(results_sents), batch_size = 8, strategy = pooling, fname = "dummy.txt", write = False))#.squeeze()
            #encoding = np.mean(encoding, axis = 0)
            #D,I = index.search(np.ascontiguousarray([encoding]), 100)
        else:
            show_results = False
            st.write("No resutls found.")

 if show_results:
    st.write("Performed query of type '{}'. Similarity search results:".format(mode))
    results = [sents[i] for i in I.squeeze()]
    st.write(st.table(results))
    
