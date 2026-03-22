import sys
import os
import numpy as np
import time

# Add the project root to the system path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import scrape_text_from_url, make_cbow_data
from src.cbow import relu, relu_derivative, softmax, one_hot

def predict(context, W1, b1, W2, b2, word_to_ix, ix_to_word, V):
    """Generates a prediction using the trained weights."""
    x = np.zeros((V, 1))
    for w in context:
        x += one_hot(w, word_to_ix, V).reshape(V, 1)
    x /= len(context)

    h = relu(W1 @ x + b1)
    y_hat = softmax(W2 @ h + b2)
    return ix_to_word[np.argmax(y_hat)]

def main():
    ai_urls = [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Generative_artificial_intelligence",
        "https://en.wikipedia.org/wiki/History_of_artificial_intelligence",
    ]

    # 1. Corpus Aggregation & Vocabulary Building
    master_corpus = " "
    print("Starting corpus aggregation...")
    for url in ai_urls:
        print(f"Fetching: {url}")
        scraped_text = scrape_text_from_url(url)
        if scraped_text:
            master_corpus += scraped_text + " "
        time.sleep(1)

    tokens = master_corpus.split()
    vocab = sorted(set(tokens))
    V = len(vocab)
    print(f"\nVocabulary size: {V}")

    word_to_ix = {w: i for i, w in enumerate(vocab)}
    ix_to_word = {i: w for w, i in word_to_ix.items()}

    # 2. Dataset Generation
    window_size = 2
    cbow_data = make_cbow_data(tokens, window_size)
    print(f"CBOW samples: {len(cbow_data)}")

    # 3. Model Initialization
    N = 50
    W1 = np.random.randn(N, V) * 0.01
    b1 = np.zeros((N, 1))
    W2 = np.random.randn(V, N) * 0.01
    b2 = np.zeros((V, 1))

    lr = 0.05
    epochs = 100

    # 4. Training Loop
    print("\nStarting Training...")
    for epoch in range(epochs):
        loss_epoch = 0
        for context, target in cbow_data:
            # Forward pass
            x = np.zeros((V, 1))
            for w in context:
                x += one_hot(w, word_to_ix, V).reshape(V, 1)
            x /= len(context)

            z1 = W1 @ x + b1
            h = relu(z1)
            z2 = W2 @ h + b2
            y_hat = softmax(z2)

            y = one_hot(target, word_to_ix, V).reshape(V, 1)
            loss = -np.sum(y * np.log(y_hat + 1e-9))
            loss_epoch += loss

            # Backward pass
            dz2 = y_hat - y
            dW2 = dz2 @ h.T
            db2 = dz2

            dh = W2.T @ dz2
            dz1 = dh * relu_derivative(z1)
            dW1 = dz1 @ x.T
            db1 = dz1

            # Parameter update
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

        # Print loss every 10 epochs to keep the terminal output clean
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}, Loss: {loss_epoch/len(cbow_data):.4f}")

    # 5. Testing
    print("\n--- Testing Model ---")
    sample_context, sample_target = cbow_data[300]
    print("Context:", sample_context)
    print("True target:", sample_target)
    
    prediction = predict(sample_context, W1, b1, W2, b2, word_to_ix, ix_to_word, V)
    print("Predicted:", prediction)

if __name__ == "__main__":
    main()