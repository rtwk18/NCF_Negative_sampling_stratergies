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
RATINGS_FILE_PATH = r'C:\Users\ritwi\OneDrive\Documents\ncf\ml-1m\ratings.dat' 

# Model & Training Hyperparameters
BATCH_SIZE = 256 # Note: A smaller batch size might be needed if memory becomes an issue
EPOCHS = 5
EMBEDDING_SIZE = 32
NEGATIVE_SAMPLES = 4
MLP_LAYERS = [64, 32, 16]

# --- NEW: DNS Hyperparameter ---
# Number of random candidates to check to find the 'hardest' negative
DNS_CANDIDATES = 100 

# --- 2. DATA LOADING & PREPARATION (Unchanged from PNS) ---
def load_and_prepare_data(file_path):
    print("Loading and preparing data...")
    ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(file_path, sep='::', names=ratings_cols, engine='python', encoding='latin-1')
    
    ratings_df['user_id'] = ratings_df['user_id'].astype('category').cat.codes
    ratings_df['movie_id'] = ratings_df['movie_id'].astype('category').cat.codes
    
    num_users = ratings_df['user_id'].nunique()
    num_items = ratings_df['movie_id'].nunique()
    
    user_item_matrix = set(zip(ratings_df['user_id'], ratings_df['movie_id']))
    user_ids = ratings_df['user_id'].values
    item_ids = ratings_df['movie_id'].values

    print(f"Data loaded: {num_users} users, {num_items} items.")
    return user_ids, item_ids, num_users, num_items, user_item_matrix

# --- 3. DATA GENERATOR (NEW FOR DNS) ---
class DNS_Generator(Sequence):
    """Keras Sequence for Dynamic Negative Sampling."""
    def __init__(self, user_ids, item_ids, num_items, user_item_matrix, batch_size, neg_samples, num_candidates):
        self.user_ids, self.item_ids = user_ids, item_ids
        self.num_items = num_items
        self.user_item_matrix = user_item_matrix
        self.batch_size = batch_size
        self.neg_samples = neg_samples
        self.num_candidates = num_candidates
        self.indices = np.arange(len(self.user_ids))
        self.model = None # Placeholder for the model
        np.random.shuffle(self.indices)

    def set_model(self, model):
        """Inject the model reference into the generator."""
        self.model = model

    def __len__(self):
        return int(np.ceil(len(self.user_ids) / self.batch_size))

    def __getitem__(self, idx):
        if self.model is None:
            raise RuntimeError("Model has not been set in DNS_Generator. Call set_model(model) first.")

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
            
            # Generate negative samples dynamically
            for _ in range(self.neg_samples):
                # 1. Generate random candidates
                candidates = np.random.randint(0, self.num_items, size=self.num_candidates)
                
                # 2. Predict scores for candidates
                user_array = np.full(self.num_candidates, user, dtype=np.int32)
                preds = self.model.predict([user_array, candidates], batch_size=self.num_candidates, verbose=0).flatten()

                # 3. Find the 'hardest' negative (highest score)
                # We loop to ensure we don't accidentally pick a true positive
                while True:
                    best_candidate_idx = np.argmax(preds)
                    item_neg = candidates[best_candidate_idx]
                    
                    if (user, item_neg) not in self.user_item_matrix:
                        all_users.append(user)
                        all_items.append(item_neg)
                        all_labels.append(0)
                        break # Found a valid hard negative
                    else:
                        # This candidate was a true positive, invalidate it and try again
                        preds[best_candidate_idx] = -np.inf

        return (
            (np.array(all_users, dtype=np.int32), np.array(all_items, dtype=np.int32)), 
            np.array(all_labels, dtype=np.int32)
        )

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# --- 4. NEUMF MODEL DEFINITION (Unchanged) ---
def build_neumf_model(num_users, num_items, embedding_size, mlp_layers):
    print("Building NeuMF model...")
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    gmf_user_embedding = Embedding(num_users, embedding_size, name='gmf_user_embedding')(user_input)
    gmf_item_embedding = Embedding(num_items, embedding_size, name='gmf_item_embedding')(item_input)
    gmf_vector = Multiply()([Flatten()(gmf_user_embedding), Flatten()(gmf_item_embedding)])
    mlp_user_embedding = Embedding(num_users, embedding_size, name='mlp_user_embedding')(user_input)
    mlp_item_embedding = Embedding(num_items, embedding_size, name='mlp_item_embedding')(item_input)
    mlp_vector = Concatenate()([Flatten()(mlp_user_embedding), Flatten()(mlp_item_embedding)])
    for layer_size in mlp_layers:
        mlp_vector = Dense(layer_size, activation='relu')(mlp_vector)
        mlp_vector = Dropout(0.2)(mlp_vector)
    final_vector = Concatenate()([gmf_vector, mlp_vector])
    output = Dense(1, activation='sigmoid', name='output')(final_vector)
    model = Model(inputs=[user_input, item_input], outputs=output)
    return model

# --- 5. MAIN EXECUTION (MODIFIED FOR DNS) ---
if __name__ == '__main__':
    user_ids, item_ids, num_users, num_items, user_item_matrix = load_and_prepare_data(RATINGS_FILE_PATH)
    
    model = build_neumf_model(num_users, num_items, EMBEDDING_SIZE, MLP_LAYERS)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    print("Setting up DNS data generator...")
    train_generator = DNS_Generator(
        user_ids, item_ids, num_items, user_item_matrix, 
        BATCH_SIZE, NEGATIVE_SAMPLES, DNS_CANDIDATES
    )
    
    # *** IMPORTANT: Inject model into generator ***
    train_generator.set_model(model)
    
    output_signature = (
        (tf.TensorSpec(shape=(None,), dtype=tf.int32), tf.TensorSpec(shape=(None,), dtype=tf.int32)),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
    tf_train_dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_signature=output_signature
    ).repeat()

    tracker = EmissionsTracker(project_name="DNS_NeuMF_Experiment")
    
    print("\nStarting training with Dynamic Negative Sampling (DNS)... (This will be slower)")
    tracker.start()
    
    model.fit(tf_train_dataset,
              epochs=EPOCHS,
              steps_per_epoch=len(train_generator),
              verbose=1)
    
    tracker.stop()
    print("\n--- Training Finished ---")
    print(f"DNS experiment finished. Check 'emissions.csv' for the energy report.")