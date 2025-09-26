import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Flatten, Concatenate, Dense, 
                                     Multiply, Dropout)
from tensorflow.keras.utils import Sequence
from codecarbon import EmissionsTracker

# --- 1. CONFIGURATION ---
# IMPORTANT: Update this path to where you saved the ratings.dat file!
RATINGS_FILE_PATH = r'C:\Users\ritwi\OneDrive\Documents\ncf\ml-1m\ratings.dat' 

# Model & Training Hyperparameters
BATCH_SIZE = 256
EPOCHS = 5
EMBEDDING_SIZE = 32
NEGATIVE_SAMPLES = 4 # Number of negative samples per positive sample
MLP_LAYERS = [64, 32, 16]

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
    
    # Create a user-item interaction matrix (for negative sampling lookup)
    user_item_matrix = set(zip(ratings_df['user_id'], ratings_df['movie_id']))

    # Get the user-item pairs for training
    user_ids = ratings_df['user_id'].values
    item_ids = ratings_df['movie_id'].values

    print(f"Data loaded: {num_users} users, {num_items} items.")
    return user_ids, item_ids, num_users, num_items, user_item_matrix

# --- 3. DATA GENERATOR WITH NEGATIVE SAMPLING ---
# --- 3. DATA GENERATOR WITH NEGATIVE SAMPLING (CORRECTED) ---
class NegativeSamplingGenerator(Sequence):
    """Keras Sequence for generating batches with negative samples."""
    def __init__(self, user_ids, item_ids, num_items, user_item_matrix, batch_size, neg_samples):
        self.user_ids, self.item_ids = user_ids, item_ids
        self.num_items = num_items
        self.user_item_matrix = user_item_matrix
        self.batch_size = batch_size
        self.neg_samples = neg_samples
        self.indices = np.arange(len(self.user_ids))
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.user_ids) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        # Get positive samples
        batch_users = self.user_ids[batch_indices]
        batch_items_pos = self.item_ids[batch_indices]
        
        # Prepare arrays for all samples
        all_users = np.repeat(batch_users, self.neg_samples + 1)
        all_items = []
        all_labels = []
        
        for i, user in enumerate(batch_users):
            item_pos = batch_items_pos[i]
            
            # Add the positive sample
            all_items.append(item_pos)
            all_labels.append(1)
            
            # Generate negative samples
            neg_count = 0
            while neg_count < self.neg_samples:
                item_neg = np.random.randint(0, self.num_items)
                if (user, item_neg) not in self.user_item_matrix:
                    all_items.append(item_neg)
                    all_labels.append(0)
                    neg_count += 1
                    
        # --- THIS IS THE FIXED LINE ---
        # We change the [list] to a (tuple) and specify dtype=np.int32
        return (
            (np.array(all_users, dtype=np.int32), np.array(all_items, dtype=np.int32)), 
            np.array(all_labels, dtype=np.int32)
        )

    def on_epoch_end(self):
        # Shuffle indices at the end of each epoch
        np.random.shuffle(self.indices)

# --- 4. NEUMF MODEL DEFINITION ---
def build_neumf_model(num_users, num_items, embedding_size, mlp_layers):
    """Builds the Neural Matrix Factorization (NeuMF) model."""
    print("Building NeuMF model...")
    # Input layers
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # GMF Path (Generalized Matrix Factorization)
    gmf_user_embedding = Embedding(num_users, embedding_size, name='gmf_user_embedding')(user_input)
    gmf_item_embedding = Embedding(num_items, embedding_size, name='gmf_item_embedding')(item_input)
    gmf_user_flat = Flatten()(gmf_user_embedding)
    gmf_item_flat = Flatten()(gmf_item_embedding)
    gmf_vector = Multiply()([gmf_user_flat, gmf_item_flat])

    # MLP Path (Multi-Layer Perceptron)
    mlp_user_embedding = Embedding(num_users, embedding_size, name='mlp_user_embedding')(user_input)
    mlp_item_embedding = Embedding(num_items, embedding_size, name='mlp_item_embedding')(item_input)
    mlp_user_flat = Flatten()(mlp_user_embedding)
    mlp_item_flat = Flatten()(mlp_item_embedding)
    mlp_vector = Concatenate()([mlp_user_flat, mlp_item_flat])
    
    for layer_size in mlp_layers:
        mlp_vector = Dense(layer_size, activation='relu')(mlp_vector)
        mlp_vector = Dropout(0.2)(mlp_vector)

    # Concatenate GMF and MLP paths
    final_vector = Concatenate()([gmf_vector, mlp_vector])
    
    # Output layer
    output = Dense(1, activation='sigmoid', name='output')(final_vector)
    
    model = Model(inputs=[user_input, item_input], outputs=output)
    return model

# --- 5. MAIN EXECUTION ---
# --- 5. MAIN EXECUTION (REVISED) ---
# --- 5. MAIN EXECUTION (REVISED) ---
if __name__ == '__main__':
    # Load data
    user_ids, item_ids, num_users, num_items, user_item_matrix = load_and_prepare_data(RATINGS_FILE_PATH)
    
    # Build model
    model = build_neumf_model(num_users, num_items, EMBEDDING_SIZE, MLP_LAYERS)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    
    # Set up data generator
    print("Setting up data generator...")
    train_generator = NegativeSamplingGenerator(
        user_ids, item_ids, num_items, user_item_matrix, BATCH_SIZE, NEGATIVE_SAMPLES
    )
    
    # --- START: THE FIX ---
    # Manually create a tf.data.Dataset and define the output signature.
    
    # 1. Define the exact shape and type of the data your generator yields.
    # The structure must match the output of your generator: ([users, items], labels)
    output_signature = (
        (tf.TensorSpec(shape=(None,), dtype=tf.int32),  # User IDs
         tf.TensorSpec(shape=(None,), dtype=tf.int32)), # Item IDs
        tf.TensorSpec(shape=(None,), dtype=tf.int32)   # Labels
    )

    # 2. Wrap your Keras Sequence in tf.data.Dataset.from_generator.
    tf_train_dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_signature=output_signature
    )
    # --- END: THE FIX ---

    # Set up energy tracker
    tracker = EmissionsTracker(project_name="RNS_NeuMF_Experiment")
    
    # Train the model
    print("\nStarting training with Random Negative Sampling (RNS)...")
    tracker.start()
    
    # 3. Pass the new tf.data.Dataset object to model.fit().
    model.fit(tf_train_dataset, # <-- Use the new dataset object
              epochs=EPOCHS,
              verbose=1)
    
    tracker.stop()
    print("\n--- Training Finished ---")
    print(f"RNS experiment finished. Check the 'emissions.csv' file for the energy report.")