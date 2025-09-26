import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Lambda, Dot, Subtract
from tensorflow.keras.utils import Sequence
from codecarbon import EmissionsTracker

# --- 1. CONFIGURATION ---
RATINGS_FILE_PATH = r'C:\Users\ritwi\OneDrive\Documents\ncf\ml-1m\ratings.dat' 
BATCH_SIZE = 1024 
EPOCHS = 5
EMBEDDING_SIZE = 32
MIX_ALPHA = 0.2 # The mixing coefficient for generating negatives

# --- 2. DATA LOADING & PREPARATION ---
def load_and_prepare_data(file_path):
    print("Loading and preparing data...")
    ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(file_path, sep='::', names=ratings_cols, engine='python', encoding='latin-1')
    
    ratings_df['user_id'] = ratings_df['user_id'].astype('category').cat.codes
    ratings_df['movie_id'] = ratings_df['movie_id'].astype('category').cat.codes
    
    num_users = ratings_df['user_id'].nunique()
    num_items = ratings_df['movie_id'].nunique()
    
    user_item_matrix = set(zip(ratings_df['user_id'], ratings_df['movie_id']))
    user_ids = ratings_df['user_id'].values.astype(np.int32)
    item_ids = ratings_df['movie_id'].values.astype(np.int32)

    print(f"Data loaded: {num_users} users, {num_items} items.")
    return user_ids, item_ids, num_users, num_items, user_item_matrix

# --- 3. DATA GENERATOR FOR MIXGCF (CORRECTED) ---
class MixGCF_Generator(Sequence):
    """Generates (user, positive_item, negative_item) triplets."""
    def __init__(self, user_ids, item_ids, num_items, user_item_matrix, batch_size):
        self.user_ids, self.item_ids = user_ids, item_ids
        self.num_items = num_items
        self.user_item_matrix = user_item_matrix
        self.batch_size = batch_size
        self.indices = np.arange(len(self.user_ids))
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.user_ids) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        batch_users = self.user_ids[batch_indices]
        batch_items_pos = self.item_ids[batch_indices]
        batch_items_neg = []

        for user in batch_users:
            while True:
                neg_item = np.random.randint(0, self.num_items)
                if (user, neg_item) not in self.user_item_matrix:
                    batch_items_neg.append(neg_item)
                    break
        
        # --- FIX: The inputs must be a TUPLE, not a list ---
        inputs = (
            batch_users, 
            batch_items_pos, 
            np.array(batch_items_neg, dtype=np.int32)
        )
        dummy_y = np.zeros(len(batch_users), dtype=np.float32)
        
        return inputs, dummy_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# --- 4. MODEL & LOSS FOR MIXGCF ---
def build_mixgcf_model(num_users, num_items, embedding_size, alpha):
    """Builds a model that performs mixing internally and uses BPR loss."""
    print("Building MixGCF model...")
    
    user_input = Input(shape=(1,), name='user_input', dtype='int32')
    pos_item_input = Input(shape=(1,), name='pos_item_input', dtype='int32')
    neg_item_input = Input(shape=(1,), name='neg_item_input', dtype='int32')
    
    user_embedding = Embedding(num_users, embedding_size, name='user_embedding')
    item_embedding = Embedding(num_items, embedding_size, name='item_embedding')

    u_vec = Flatten()(user_embedding(user_input))
    pos_i_vec = Flatten()(item_embedding(pos_item_input))
    neg_i_vec = Flatten()(item_embedding(neg_item_input))
    
    synthetic_neg_vec = Lambda(lambda x: alpha * x[0] + (1 - alpha) * x[1], name='mix')([pos_i_vec, neg_i_vec])
    
    pos_score = Dot(axes=-1, name='pos_score')([u_vec, pos_i_vec])
    neg_score = Dot(axes=-1, name='neg_score')([u_vec, synthetic_neg_vec])
    
    score_diff = Subtract(name='score_difference')([pos_score, neg_score])
    
    loss = Lambda(lambda x: tf.reduce_mean(tf.nn.softplus(-x)), name='bpr_loss')(score_diff)
    
    model = Model(inputs=[user_input, pos_item_input, neg_item_input], outputs=loss)
    return model

# --- 5. MAIN EXECUTION ---
if __name__ == '__main__':
    user_ids, item_ids, num_users, num_items, user_item_matrix = load_and_prepare_data(RATINGS_FILE_PATH)
    
    model = build_mixgcf_model(num_users, num_items, EMBEDDING_SIZE, MIX_ALPHA)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=lambda y_true, y_pred: y_pred 
    )
    model.summary()
    
    print("Setting up MixGCF data generator...")
    train_generator = MixGCF_Generator(user_ids, item_ids, num_items, user_item_matrix, BATCH_SIZE)
    
    tracker = EmissionsTracker(project_name="MixGCF_Experiment")
    
    print("\nStarting training with MixGCF...")
    tracker.start()
    
    model.fit(train_generator,
              epochs=EPOCHS,
              verbose=1)
    
    tracker.stop()
    print("\n--- Training Finished ---")
    print(f"MixGCF experiment finished. Check 'emissions.csv' for the energy report.")