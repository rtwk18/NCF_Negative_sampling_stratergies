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
BATCH_SIZE = 256
EPOCHS = 5
EMBEDDING_SIZE = 32
MLP_LAYERS = [64, 32, 16]
NEGATIVE_SAMPLES = 4 

# --- AHNS Hyperparameters ---
NUM_RANDOM_CANDIDATES = 50
NUM_POPULAR_CANDIDATES = 50

# --- 2. DATA LOADING & PREPARATION (with Popularity) ---
def load_and_prepare_data(file_path):
    print("Loading and preparing data...")
    ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(file_path, sep='::', names=ratings_cols, engine='python', encoding='latin-1')
    
    ratings_df['user_id'] = ratings_df['user_id'].astype('category').cat.codes
    ratings_df['movie_id'] = ratings_df['movie_id'].astype('category').cat.codes
    
    num_users = ratings_df['user_id'].nunique()
    num_items = ratings_df['movie_id'].nunique()
    
    item_counts = ratings_df['movie_id'].value_counts()
    item_popularity = item_counts.sort_index().values
    item_popularity_dist = item_popularity / item_popularity.sum()
    
    user_item_matrix = set(zip(ratings_df['user_id'], ratings_df['movie_id']))
    user_ids = ratings_df['user_id'].values
    item_ids = ratings_df['movie_id'].values

    print(f"Data loaded: {num_users} users, {num_items} items.")
    return user_ids, item_ids, num_users, num_items, user_item_matrix, item_popularity_dist

# --- 3. DATA GENERATOR FOR AHNS ---
class AHNS_Generator(Sequence):
    def __init__(self, user_ids, item_ids, num_items, user_item_matrix, batch_size, neg_samples, 
                 item_pop_dist, num_rand_cand, num_pop_cand):
        self.user_ids, self.item_ids = user_ids, item_ids
        self.num_items = num_items
        self.user_item_matrix = user_item_matrix
        self.batch_size = batch_size
        self.neg_samples = neg_samples
        self.item_pop_dist = item_pop_dist
        self.num_rand_cand = num_rand_cand
        self.num_pop_cand = num_pop_cand
        self.indices = np.arange(len(self.user_ids))
        self.model = None
        np.random.shuffle(self.indices)

    def set_model(self, model):
        self.model = model

    def __len__(self):
        return int(np.ceil(len(self.user_ids) / self.batch_size))

    def __getitem__(self, idx):
        if self.model is None:
            raise RuntimeError("Model has not been set. Call set_model(model) first.")

        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_users = self.user_ids[batch_indices]
        batch_items_pos = self.item_ids[batch_indices]
        
        all_users, all_items, all_labels = [], [], []

        for i, user in enumerate(batch_users):
            item_pos = batch_items_pos[i]
            
            # Add positive sample
            all_users.append(user)
            all_items.append(item_pos)
            all_labels.append(1)

            # Get score for the positive item
            pos_score = self.model.predict([np.array([user]), np.array([item_pos])], verbose=0)[0][0]

            for _ in range(self.neg_samples):
                # 1. Create candidate pool
                rand_cand = np.random.randint(0, self.num_items, self.num_rand_cand)
                pop_cand = np.random.choice(self.num_items, self.num_pop_cand, p=self.item_pop_dist)
                candidates = np.unique(np.concatenate([rand_cand, pop_cand]))
                
                # 2. Score candidates
                user_array = np.full(len(candidates), user)
                candidate_scores = self.model.predict([user_array, candidates], batch_size=len(candidates), verbose=0).flatten()

                # 3. Adaptive selection
                valid_indices = [idx for idx, c in enumerate(candidates) if (user, c) not in self.user_item_matrix]
                valid_scores = candidate_scores[valid_indices]
                valid_candidates = candidates[valid_indices]
                
                # Find candidates harder than random but easier than the positive
                hard_indices = np.where(valid_scores < pos_score)[0]

                if len(hard_indices) > 0:
                    # Pick the best among the valid hard ones
                    best_hard_idx = np.argmax(valid_scores[hard_indices])
                    final_neg_item = valid_candidates[hard_indices[best_hard_idx]]
                elif len(valid_indices) > 0:
                    # Fallback: if all are 'too hard', pick the easiest of the valid ones
                    final_neg_item = valid_candidates[np.argmin(valid_scores)]
                else:
                    # Rare fallback: if all candidates were true positives, get a random one
                    while True:
                        final_neg_item = np.random.randint(0, self.num_items)
                        if (user, final_neg_item) not in self.user_item_matrix:
                            break
                
                all_users.append(user)
                all_items.append(final_neg_item)
                all_labels.append(0)

        return (np.array(all_users, dtype=np.int32), np.array(all_items, dtype=np.int32)), np.array(all_labels, dtype=np.int32)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# --- 4. NEUMF MODEL DEFINITION (Standard) ---
def build_neumf_model(num_users, num_items, embedding_size, mlp_layers):
    print("Building NeuMF model...")
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    gmf_user_embedding = Embedding(num_users, embedding_size)(user_input)
    gmf_item_embedding = Embedding(num_items, embedding_size)(item_input)
    gmf_vector = Multiply()([Flatten()(gmf_user_embedding), Flatten()(gmf_item_embedding)])
    mlp_user_embedding = Embedding(num_users, embedding_size)(user_input)
    mlp_item_embedding = Embedding(num_items, embedding_size)(item_input)
    mlp_vector = Concatenate()([Flatten()(mlp_user_embedding), Flatten()(mlp_item_embedding)])
    for layer_size in mlp_layers:
        mlp_vector = Dense(layer_size, activation='relu')(mlp_vector)
        mlp_vector = Dropout(0.2)(mlp_vector)
    final_vector = Concatenate()([gmf_vector, mlp_vector])
    output = Dense(1, activation='sigmoid')(final_vector)
    model = Model(inputs=[user_input, item_input], outputs=output)
    return model

# --- 5. MAIN EXECUTION ---
if __name__ == '__main__':
    user_ids, item_ids, num_users, num_items, user_item_matrix, item_pop_dist = load_and_prepare_data(RATINGS_FILE_PATH)
    
    model = build_neumf_model(num_users, num_items, EMBEDDING_SIZE, MLP_LAYERS)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    print("Setting up AHNS data generator...")
    train_generator = AHNS_Generator(
        user_ids, item_ids, num_items, user_item_matrix, BATCH_SIZE, NEGATIVE_SAMPLES,
        item_pop_dist, NUM_RANDOM_CANDIDATES, NUM_POPULAR_CANDIDATES
    )
    
    train_generator.set_model(model)
    
    tracker = EmissionsTracker(project_name="AHNS_NeuMF_Experiment")
    
    print("\nStarting training with AHNS... (This will be slow)")
    tracker.start()
    
    model.fit(train_generator,
              epochs=EPOCHS,
              verbose=1)
    
    tracker.stop()
    print("\n--- Training Finished ---")
    print(f"AHNS experiment finished. Check 'emissions.csv' for the energy report.")