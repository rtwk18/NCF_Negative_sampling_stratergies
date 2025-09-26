import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Lambda
from tensorflow.keras.utils import Sequence
from codecarbon import EmissionsTracker

# --- 1. CONFIGURATION ---
# IMPORTANT: Update this path to where you saved the ratings.dat file!
RATINGS_FILE_PATH = r'C:\Users\ritwi\OneDrive\Documents\ncf\ml-1m\ratings.dat' 

# Model & Training Hyperparameters
# NOTE: In-batch sampling works best with larger batch sizes
BATCH_SIZE = 1024 
EPOCHS = 5
EMBEDDING_SIZE = 32

# --- 2. DATA LOADING & PREPARATION ---
def load_and_prepare_data(file_path):
    """Loads and preprocesses the MovieLens 1M ratings data."""
    print("Loading and preparing data...")
    ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(
        file_path, 
        sep='::', 
        names=ratings_cols, 
        engine='python',
        encoding='latin-1'
    )
    
    # Remap user and movie IDs to start from 0
    ratings_df['user_id'] = ratings_df['user_id'].astype('category').cat.codes
    ratings_df['movie_id'] = ratings_df['movie_id'].astype('category').cat.codes
    
    num_users = ratings_df['user_id'].nunique()
    num_items = ratings_df['movie_id'].nunique()
    
    # We only need the positive pairs for this generator
    user_ids = ratings_df['user_id'].values.astype(np.int32)
    item_ids = ratings_df['movie_id'].values.astype(np.int32)

    print(f"Data loaded: {num_users} users, {num_items} items.")
    return user_ids, item_ids, num_users, num_items

# --- 3. DATA GENERATOR FOR IN-BATCH ---
class InBatchGenerator(Sequence):
    """Generates batches of positive (user, item) pairs."""
    def __init__(self, user_ids, item_ids, batch_size):
        self.user_ids, self.item_ids = user_ids, item_ids
        self.batch_size = batch_size
        self.indices = np.arange(len(self.user_ids))
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.user_ids) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        batch_users = self.user_ids[batch_indices]
        batch_items = self.item_ids[batch_indices]
        
        # Labels are the indices of the positive items for the loss function
        labels = tf.range(tf.shape(batch_users)[0])
        
        return (batch_users, batch_items), labels

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# --- 4. MODEL FOR IN-BATCH SAMPLING ---
def build_inbatch_model(num_users, num_items, embedding_size):
    """Builds a Matrix Factorization model for in-batch sampling."""
    print("Building In-Batch MF model...")
    
    # Input layers expect a single ID, hence shape=(1,)
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # Embedding layers
    user_embedding_layer = Embedding(num_users, embedding_size, name='user_embedding')
    item_embedding_layer = Embedding(num_items, embedding_size, name='item_embedding')

    # Get embeddings for the batch -> shape=(batch_size, 1, embedding_size)
    user_vecs = user_embedding_layer(user_input)
    item_vecs = item_embedding_layer(item_input)
    
    # Flatten to remove the middle dimension -> shape=(batch_size, embedding_size)
    user_vecs_flat = Flatten()(user_vecs)
    item_vecs_flat = Flatten()(item_vecs)
    
    # Compute dot product of all user-item pairs in the batch via matrix multiplication
    # Wrapped in a Lambda layer to use a TF function within the Keras model
    logits = Lambda(
        lambda tensors: tf.linalg.matmul(tensors[0], tensors[1], transpose_b=True),
        name='logits'
    )([user_vecs_flat, item_vecs_flat])
    
    model = Model(inputs=[user_input, item_input], outputs=logits)
    return model

# --- 5. MAIN EXECUTION ---
if __name__ == '__main__':
    user_ids, item_ids, num_users, num_items = load_and_prepare_data(RATINGS_FILE_PATH)
    
    model = build_inbatch_model(num_users, num_items, EMBEDDING_SIZE)
    
    # Use SparseCategoricalCrossentropy as we are predicting the index of the correct item
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.summary()
    
    print("Setting up In-Batch data generator...")
    train_generator = InBatchGenerator(user_ids, item_ids, BATCH_SIZE)

    # Define the output signature of the generator
    output_signature = (
        (tf.TensorSpec(shape=(None,), dtype=tf.int32), tf.TensorSpec(shape=(None,), dtype=tf.int32)),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
    tf_train_dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_signature=output_signature
    )
    
    # Define a function to reshape the generator's output to match the model's input
    def reshape_inputs(inputs, labels):
        user_ids, item_ids = inputs
        # Add a dimension to make shape (batch,) -> (batch, 1)
        return (tf.expand_dims(user_ids, axis=1), tf.expand_dims(item_ids, axis=1)), labels

    # Apply the reshape function and then repeat for multiple epochs
    tf_train_dataset = tf_train_dataset.map(reshape_inputs).repeat()
    
    # Set up the energy tracker
    tracker = EmissionsTracker(project_name="In-Batch_MF_Experiment")
    
    print("\nStarting training with In-Batch Negative Sampling...")
    tracker.start()
    
    model.fit(tf_train_dataset,
              epochs=EPOCHS,
              steps_per_epoch=len(train_generator),
              verbose=1)
    
    tracker.stop()
    print("\n--- Training Finished ---")
    print(f"In-Batch experiment finished. Check 'emissions.csv' for the energy report.")