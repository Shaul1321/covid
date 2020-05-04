git clone https://github.com/Shaul1321/covid.git
python3 run_bert.py ----input-filenam results.tsv --device cpu --pooling cls --output_fname output-cls.jsonl
python3 build_index.py --fname output-cls.jsonl --num_vecs_pca 100000 --pca_variance 0.985 --similarity_type cosine
