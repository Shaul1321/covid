import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import faiss
import bert
from bert import BertEncoder
import pickle
import spike_queries

@st.cache
def load_sents_and_ids():
    with st.spinner('Loading sentences and IDs...'):
        df = pd.read_csv("results.covid_dataset.all.tsv", sep = "\t")
        return df

@st.cache(allow_output_mutation=True)
def load_index(similarity, pooling):
    with st.spinner('Loading FAISS index...'):
        fname = "output." + pooling + ".index"
        index = faiss.read_index(fname)
        return index

@st.cache(allow_output_mutation=True)
def load_bert():
    with st.spinner('Loading BERT...'):
        model = bert.BertEncoder("cpu")
        return model
        
@st.cache
def load_pca(pooling):

    fname = "output." + pooling + ".pca.pickle"
    with open(fname, "rb") as f:
    
        return pickle.load(f)
    
               
st.title('COVID-19 Similarity Search')

#a = st.empty()
mode = st.sidebar.radio("Mode", ("Sentence", "IDs", "SPIKE-covid19", "SPIKE-pubmed"))
similarity = st.sidebar.selectbox('Similarity', ('dot product', "l2"))
pooling = st.sidebar.selectbox('Pooling', ('mean-cls', 'cls'))


df = load_sents_and_ids()
ids, sents = df["sentence_id"].tolist(), df["sentence_text"].tolist()
id2ind = {ids[i]:i for i,s in enumerate(sents)}
ind2id = {i:ids[i] for i,s in enumerate(sents)}


index = load_index(similarity, pooling)
bert = load_bert()
pca = load_pca(pooling)
st.write("Uses {}-dimensional vectors".format(pca.components_.shape[0]))

if mode == "Sentence":
    input_sentence = st.text_input('Input sentence', 'The virus can spread rapidly via different transimission vectors.')
    encoding = pca.transform(bert.encode([input_sentence], [1], batch_size = 1, strategy = pooling, fname = "dummy.txt", write = False))#.squeeze()
    D,I = index.search(np.ascontiguousarray(encoding), 100)

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
    #input_query = st.text_input('Query', '[subj:l coronavirus] is affecting different [obj:e animals]')
    input_query = st.text_input('Query', 'The [subj:l coronavirus] [copula:w is] prevalent among [w:e bats].')
    
    max_results = int(st.text_input("How many resuls?", 10))
    with st.spinner('Performing SPIKE query...'):
        results_df = spike_queries.perform_query(input_query, dataset_name = "covid19" if "covid19" in mode else "pubmed", num_results = max_results)
        results_sents = results_df["sentence_text"].tolist()
        st.write("First sentences retrieved:")
        st.table(results_sents[:10]) 
        encoding = pca.transform(bert.encode(results_sents, [1]*len(results_sents), batch_size = 8, strategy = pooling, fname = "dummy.txt", write = False))#.squeeze()
        encoding = np.mean(encoding, axis = 0)
        D,I = index.search(np.ascontiguousarray([encoding]), 100)
        
st.write("Performed query of type '{}'. Similarity search results:".format(mode))
results = [sents[i] for i in I.squeeze()]
st.write(st.table(results))
