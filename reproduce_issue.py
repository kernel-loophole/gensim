
import logging
import io
import sys
import gensim
import gensim.models.word2vec_inner
print(f"Gensim file: {gensim.__file__}")
print(f"Word2Vec Inner file: {gensim.models.word2vec_inner.__file__}")
from gensim.models import Word2Vec

# Configure logging to capture warnings
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

def test_truncation():
    print("Testing truncation with sentence length > 10,000...")
    
    # Create a sentence with 10,005 tokens
    # Tokens 0-9999 are "a", tokens 10000-10004 are "b"
    # If truncated, "b" should not be in the vocabulary (or at least not trained effectively if min_count was higher, but here min_count=1)
    # Actually, build_vocab might see it, but train_batch_sg will truncate.
    
    long_sentence = ["a"] * 10000 + ["b"] * 5
    sentences = [long_sentence]
    
    model = Word2Vec(min_count=1, vector_size=10, window=5, workers=1, seed=42, sample=0)
    
    print("Building vocabulary...")
    model.build_vocab(sentences)
    
    # Check if 'b' is in vocab
    if 'b' in model.wv:
        print("'b' is in vocabulary (expected, as scanning vocab doesn't truncate usually)")
    else:
        print("'b' is NOT in vocabulary (unexpected for build_vocab)")


    print("Training model...")
    # Capture initial vector for 'b'
    b_vec_before = model.wv['b'].copy()
    
    # Capture stderr/stdout to check for warning (though logging goes to stderr usually)
    model.train(sentences, total_examples=model.corpus_count, epochs=1)
    
    b_vec_after = model.wv['b']
    
    import numpy as np
    if np.allclose(b_vec_before, b_vec_after):
        print("FAILURE: Vector for 'b' did NOT change. (Likely truncated)")
    else:
        print("SUCCESS: Vector for 'b' CHANGED. (Likely NOT truncated)")

    print("Train complete.")
    
    # In truncation, 'b' (at index 10000+) should be ignored during training.
    # However, since we initialized with min_count=1, 'b' has a vector. 
    # But it would not be updated during training if truncated.
    
    # We rely on seeing the warning message in the output.

if __name__ == "__main__":
    test_truncation()
