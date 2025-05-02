import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from database import engine
import random

class NewProductRecommendationModel:
    def __init__(self):
        self.df = None
        self.model = None
        self.user2idx = {}
        self.idx2cat = {}
        self.n_users = 0
        self.n_categories = 0

    def load_data(self):
        query = """
        SELECT DISTINCT
            c.customer_id,
            cat.category_name
        FROM orders o
        JOIN order_details od ON o.order_id = od.order_id
        JOIN products p ON od.product_id = p.product_id
        JOIN categories cat ON p.category_id = cat.category_id
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE c.customer_id IS NOT NULL
        """
        df = pd.read_sql(query, engine)
        self.df = df
        user_ids = df['customer_id'].unique().tolist()
        category_names = df['category_name'].unique().tolist()
        self.user2idx = {u: i for i, u in enumerate(user_ids)}
        cat2idx = {c: i for i, c in enumerate(category_names)}
        self.idx2cat = {i: c for c, i in cat2idx.items()}
        df['user'] = df['customer_id'].map(self.user2idx)
        df['item'] = df['category_name'].map(cat2idx)
        df['label'] = 1

        negatives = []
        for u in df['user'].unique():
            bought = set(df[df['user'] == u]['item'])
            while len(bought) < len(category_names):
                i = random.randint(0, len(category_names) - 1)
                if i not in bought:
                    negatives.append([u, i, 0])
                    bought.add(i)
        neg_df = pd.DataFrame(negatives, columns=["user", "item", "label"])
        self.all_df = pd.concat([df[['user', 'item', 'label']], neg_df], ignore_index=True)
        self.n_users = len(user_ids)
        self.n_categories = len(category_names)

    def train_model(self):
        X = self.all_df[['user', 'item']].values
        y = self.all_df['label'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        embedding_dim = 32

        user_input = tf.keras.Input(shape=(1,))
        item_input = tf.keras.Input(shape=(1,))
        user_vec = tf.keras.layers.Embedding(self.n_users, embedding_dim)(user_input)
        item_vec = tf.keras.layers.Embedding(self.n_categories, embedding_dim)(item_input)
        user_vec = tf.keras.layers.Flatten()(user_vec)
        item_vec = tf.keras.layers.Flatten()(item_vec)
        concat = tf.keras.layers.Concatenate()([user_vec, item_vec])
        dense = tf.keras.layers.Dense(64, activation='relu')(concat)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

        self.model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=0)

    def recommend(self, customer_id_str, top_n=3):
        if customer_id_str not in self.user2idx:
            return ["❌ Bilinmeyen kullanıcı."]

        user_idx = self.user2idx[customer_id_str]
        all_category_indices = set(range(self.n_categories))
        previously_bought = set(self.df[self.df['user'] == user_idx]['item'])

        new_category_candidates = list(all_category_indices - previously_bought)

        if not new_category_candidates:
            return [f"ℹ️ {customer_id_str} tüm kategorilerde alışveriş yapmış. Yeni öneri yapılamaz."]

        user_array = np.full(len(new_category_candidates), user_idx)
        item_array = np.array(new_category_candidates)

        predictions = self.model.predict([user_array, item_array], verbose=0)
        top_indices = predictions.reshape(-1).argsort()[-top_n:][::-1]

        results = []
        for i in top_indices:
            category = self.idx2cat[new_category_candidates[i]]
            probability = predictions[i][0]
            results.append({"category": category, "probability": round(float(probability), 4)})

        return results
    
def segment_new_product_purchase_potential():
    model = NewProductRecommendationModel()
    model.load_data()
    model.train_model()
    return model.recommend("ALFKI", top_n=3)