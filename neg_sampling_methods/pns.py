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

# --- 2. DATA LOADING & PREPARATION (MODIFIED FOR PNS) ---
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
    
    # *** NEW: Calculate item popularity for PNS ***
    # Count how often each movie_id appears
    item_counts = ratings_df['movie_id'].value_counts()
    # Sort by movie_id to align with embedding indices
    item_popularity = item_counts.sort_index().values
    # Normalize to get a probability distribution
    item_popularity_dist = item_popularity / item_popularity.sum()
    
    # Create a user-item interaction matrix (for negative sampling lookup)
    user_item_matrix = set(zip(ratings_df['user_id'], ratings_df['movie_id']))

    # Get the user-item pairs for training
    user_ids = ratings_df['user_id'].values
    item_ids = ratings_df['movie_id'].values

    print(f"Data loaded: {num_users} users, {num_items} items.")
    # *** MODIFIED: Return the popularity distribution ***
    return user_ids, item_ids, num_users, num_items, user_item_matrix, item_popularity_dist

# --- 3. DATA GENERATOR (MODIFIED FOR PNS) ---
class PNS_Generator(Sequence):
    """Keras Sequence for generating batches with Popularity Negative Sampling."""
    # *** MODIFIED: Accept item_popularity_dist ***
    def __init__(self, user_ids, item_ids, num_items, user_item_matrix, 
                 batch_size, neg_samples, item_popularity_dist):
        self.user_ids, self.item_ids = user_ids, item_ids
        self.num_items = num_items
        self.user_item_matrix = user_item_matrix
        self.batch_size = batch_size
        self.neg_samples = neg_samples
        # *** NEW: Store the popularity distribution ***
        self.item_popularity_dist = item_popularity_dist
        self.indices = np.arange(len(self.user_ids))
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.user_ids) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        batch_users = self.user_ids[batch_indices]
        batch_items_pos = self.item_ids[batch_indices]
        
        all_users, all_items, all_labels = [], [], []
        
        for i, user in enumerate(batch_users):
            item_pos = batch_items_pos[i]
            
            # Add the positive sample
            all_users.append(user)
            all_items.append(item_pos)
            all_labels.append(1)
            
            # Generate negative samples
            neg_count = 0
            while neg_count < self.neg_samples:
                # *** MODIFIED: Sample using popularity distribution ***
                item_neg = np.random.choice(
                    a=self.num_items, 
                    p=self.item_popularity_dist
                )
                
                if (user, item_neg) not in self.user_item_matrix:
                    all_users.append(user)
                    all_items.append(item_neg)
                    all_labels.append(0)
                    neg_count += 1
        
        return (
            (np.array(all_users, dtype=np.int32), np.array(all_items, dtype=np.int32)), 
            np.array(all_labels, dtype=np.int32)
        )

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# --- 4. NEUMF MODEL DEFINITION (Unchanged) ---
def build_neumf_model(num_users, num_items, embedding_size, mlp_layers):
    """Builds the Neural Matrix Factorization (NeuMF) model."""
    print("Building NeuMF model...")
    # Input layers
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # GMF Path
    gmf_user_embedding = Embedding(num_users, embedding_size, name='gmf_user_embedding')(user_input)
    gmf_item_embedding = Embedding(num_items, embedding_size, name='gmf_item_embedding')(item_input)
    gmf_vector = Multiply()([Flatten()(gmf_user_embedding), Flatten()(gmf_item_embedding)])

    # MLP Path
    mlp_user_embedding = Embedding(num_users, embedding_size, name='mlp_user_embedding')(user_input)
    mlp_item_embedding = Embedding(num_items, embedding_size, name='mlp_item_embedding')(item_input)
    mlp_vector = Concatenate()([Flatten()(mlp_user_embedding), Flatten()(mlp_item_embedding)])
    
    for layer_size in mlp_layers:
        mlp_vector = Dense(layer_size, activation='relu')(mlp_vector)
        mlp_vector = Dropout(0.2)(mlp_vector)

    # Concatenate GMF and MLP paths
    final_vector = Concatenate()([gmf_vector, mlp_vector])
    
    # Output layer
    output = Dense(1, activation='sigmoid', name='output')(final_vector)
    
    model = Model(inputs=[user_input, item_input], outputs=output)
    return model

# --- 5. MAIN EXECUTION (MODIFIED FOR PNS) ---
# --- 5. MAIN EXECUTION (CORRECTED FOR MULTIPLE EPOCHS) ---
if __name__ == '__main__':
    # Load data including popularity distribution
    user_ids, item_ids, num_users, num_items, user_item_matrix, item_pop_dist = load_and_prepare_data(RATINGS_FILE_PATH)
    
    # Build model
    model = build_neumf_model(num_users, num_items, EMBEDDING_SIZE, MLP_LAYERS)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Set up PNS data generator
    print("Setting up PNS data generator...")
    train_generator = PNS_Generator(
        user_ids, item_ids, num_items, user_item_matrix, 
        BATCH_SIZE, NEGATIVE_SAMPLES, item_pop_dist
    )
    
    # Define the output signature
    output_signature = (
        (tf.TensorSpec(shape=(None,), dtype=tf.int32),
         tf.TensorSpec(shape=(None,), dtype=tf.int32)),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )

    # --- THIS IS THE FIX ---
    # Wrap in tf.data.Dataset and add .repeat()
    tf_train_dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_signature=output_signature
    ).repeat()

    # Update project name for tracker
    tracker = EmissionsTracker(project_name="PNS_NeuMF_Experiment")
    
    # Train the model
    print("\nStarting training with Popularity-based Negative Sampling (PNS)...")
    tracker.start()
    
    # The PNS_Generator's __len__ will automatically define steps_per_epoch
    model.fit(tf_train_dataset,
              epochs=EPOCHS,
              steps_per_epoch=len(train_generator), # Explicitly set steps_per0epoch
              verbose=1)
    
    tracker.stop()
    print("\n--- Training Finished ---")
    print(f"PNS experiment finished. Check 'emissions.csv' for the energy report.")