from FlagEmbedding import BGEM3FlagModel

model = None

def get_model():
    global model
    if model is None:
        model = BGEM3FlagModel('BAAI/bge-m3',  
                               use_fp16=True)
    return model

def encode_texts(chunks):
    """Encode a list of texts into embeddings using the BGEM3FlagModel.
    Args:
        transcripts (list of str): List of text strings to encode.
        
    Returns:
        list of list of float: List of embeddings corresponding to each input text.
    """
    model = get_model()
    embeddings = model.encode(chunks,
                              batch_size=16,
                              return_dense=True)
    return embeddings['dense_vecs']


