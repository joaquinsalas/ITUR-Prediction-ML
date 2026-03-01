#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ===========================================================
# SWIN TRANSFORMER + ALPHAEARTH 64 + 3 VARIABLES EXTRA (67 CANALES)
# REGRESIÓN ITUR – CÓDIGO COMPLETO
# ===========================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# In[3]:


# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------
import numpy as np
import pandas as pd
import cv2
import albumentations as albu
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.saving import register_keras_serializable

from sklearn.metrics import mean_squared_error, r2_score


# In[4]:


# -----------------------------------------------------------
# PARÁMETROS GENERALES
# -----------------------------------------------------------
batch_size = 32
input_height, input_width = 48, 48
channels_total = 67
learning_rate = 1e-4
epochs = 100

base_path = "/mnt/data-r1/MarivelZea/2025.11.01_ITUR_AlphaEarth/partition/"

# -----------------------------------------------------------
# 1. CARGA DE PARTICIONES
# -----------------------------------------------------------
x_train = np.load(base_path + "x_train.npy")
x_val   = np.load(base_path + "x_val.npy")
x_test  = np.load(base_path + "x_test.npy")

y_train = np.load(base_path + "y_train.npy").astype("float32")
y_val   = np.load(base_path + "y_val.npy").astype("float32")
y_test  = np.load(base_path + "y_test.npy").astype("float32")

print("Shapes imágenes:")
print(x_train.shape, x_val.shape, x_test.shape)


# In[5]:


# -----------------------------------------------------------
# 2. CARGA DE 3 VARIABLES ADICIONALES (CAR_SER_VI, DILOCCON50, P_USOSUEPV)
# -----------------------------------------------------------
df = pd.read_csv("AlphaEarth_Aguascalientes_ITUR_con_norm_2.csv")

extra_all = df[["CAR_SER_VI", "DILOCCON50_log_norm", "P_USOSUEPV_norm"]].to_numpy(np.float32)

ind_train = np.load(base_path + "ind_train.npy")
ind_val   = np.load(base_path + "ind_val.npy")
ind_test  = np.load(base_path + "ind_test.npy")

extra_train = extra_all[ind_train]
extra_val   = extra_all[ind_val]
extra_test  = extra_all[ind_test]

print("Extra variables:", extra_train.shape)


# In[6]:


## ===========================================================
# 3. FUNCIONES AUXILIARES
# ===========================================================

def resize_img(img, shape):
    """
    img: (H0, W0, C)
    shape: (H, W)
    """
    return cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


# ===========================================================
# 4. GENERADOR DE DATOS (64 + 3 = 67 CANALES)
# ===========================================================

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, extra_set,
                 batch_size=batch_size,
                 dim=(input_height, input_width),
                 augment=False):
        self.x = x_set        # (N, H0, W0, 64)
        self.y = y_set        # (N,)
        self.extra = extra_set  # (N, 3)
        self.batch_size = batch_size
        self.dim = dim        # (H, W)
        self.augment = augment
        self.indexes = np.arange(len(self.x))

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        X_batch = []
        for ID in batch_idx:

            # ============================
            # 1) REDIMENSIONAR IMAGEN
            # ============================
            img = self.x[ID]                  # (H0, W0, 64)
            img_resized = resize_img(img, self.dim)  # (H, W, 64)

            # ============================
            # 2) CAPA CONSTANTE DE 3 VARIABLES
            # ============================
            extras = self.extra[ID]                 # (3,)
            extras_layer = np.ones(
                (self.dim[0], self.dim[1], 3),
                dtype=np.float32
            ) * extras.reshape(1, 1, 3)

            # ============================
            # 3) CONCATENAR (64 + 3 = 67)
            # ============================
            img_full = np.concatenate([img_resized, extras_layer], axis=-1)

            # ============================
            # 4) AUMENTACIÓN (OPCIONAL)
            # ============================
            if self.augment:
                img_full = self.apply_augmentation(img_full)

            X_batch.append(img_full)

        X_batch = np.array(X_batch)  # (batch, H, W, 67)
        y_batch = self.y[batch_idx]

        return X_batch, y_batch

    # ===========================================================
    # FUNCIONES DE AUMENTACIÓN
    # ===========================================================
    def apply_augmentation(self, img):
        aug = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomRotate90(p=0.5),
            albu.RandomBrightnessContrast(p=0.3),
        ])
        return aug(image=img)["image"]

train_gen = DataGenerator(x_train, y_train, extra_train, augment=True)
val_gen   = DataGenerator(x_val,   y_val,   extra_val,   augment=False)
test_gen  = DataGenerator(x_test,  y_test,  extra_test,  augment=False)


# In[7]:


# ===========================================================
# 5. CAPAS SWIN TRANSFORMER (como en tu código original)
# ===========================================================

@register_keras_serializable()
class MLPBlock(layers.Layer):
    def __init__(self, hidden_units, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.mlp_layers = []
        for units in hidden_units:
            self.mlp_layers.append(layers.Dense(units, activation=tf.nn.gelu))
            self.mlp_layers.append(layers.Dropout(dropout_rate))

    def call(self, inputs):
        x = inputs
        for layer in self.mlp_layers:
            x = layer(x)
        return x

@register_keras_serializable()
class WindowAttention(layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)

    def call(self, x):
        B = tf.shape(x)[0]
        N = tf.shape(x)[1]
        C = tf.shape(x)[2]

        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = tf.unstack(qkv, 3)

        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

@register_keras_serializable()
class SwinTransformerBlock(layers.Layer):
    def __init__(self, dim, input_resolution, num_heads,
                 window_size=7, shift_size=0, mlp_ratio=4.0,
                 drop=0., attn_drop=0., **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(dim, window_size, num_heads,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = layers.Dropout(drop)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLPBlock([mlp_hidden_dim, dim], dropout_rate=drop)

    def call(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# In[8]:


# ===========================================================
# 6. MODELO SWIN COMPLETO
# ===========================================================

def build_swin_model(input_shape=(48, 48, 67), embed_dim=64, num_heads=4, num_blocks=4):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(embed_dim, kernel_size=4, strides=4)(inputs)
    H_p = input_shape[0] // 4
    W_p = input_shape[1] // 4
    x = layers.Reshape((H_p * W_p, embed_dim))(x)

    for i in range(num_blocks):
        x = SwinTransformerBlock(
            dim=embed_dim,
            input_resolution=(H_p, W_p),
            num_heads=num_heads,
            window_size=6,
            shift_size=0 if i % 2 == 0 else 3,
            mlp_ratio=4.0
        )(x)

    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation="linear")(x)

    return keras.Model(inputs, outputs, name="swin_regression_67c")

model = build_swin_model()
model.summary()


# In[9]:


# ===========================================================
# 7. CALLBACKS (EARLY STOP + BEST WEIGHTS + ONECYCLE)
# ===========================================================

steps_per_epoch = len(train_gen)
total_steps = steps_per_epoch * epochs

class OneCycleLR(Callback):
    def __init__(self, max_lr, total_steps, div_factor=25, pct_start=0.3, final_div_factor=1e4):
        super().__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.div_factor = div_factor
        self.pct_start = pct_start
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.min_lr = self.initial_lr / final_div_factor
        self.step = 0
        self.phase1 = int(total_steps * pct_start)

    def on_train_batch_begin(self, batch, logs=None):
        self.step += 1

        # Fase 1: warm-up
        if self.step <= self.phase1:
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (self.step / self.phase1)

        # Fase 2: cooldown
        else:
            lr = self.max_lr - (self.max_lr - self.min_lr) * (
                (self.step - self.phase1) / (self.total_steps - self.phase1)
            )

        # ====== ASIGNACIÓN CORREGIDA DE LR ======
        opt = self.model.optimizer
        if isinstance(opt.learning_rate, tf.Variable):
            opt.learning_rate.assign(lr)
        else:
            opt.learning_rate = tf.keras.backend.cast_to_floatx(lr)

onecycle = OneCycleLR(max_lr=1e-3, total_steps=total_steps)

best_model_path = "best_swin_67bands.keras"

checkpoint = ModelCheckpoint(
    best_model_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
    mode="min"
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min",
    restore_best_weights=True,
    verbose=1
)


# In[10]:


# ===========================================================
# 8. COMPILACIÓN Y ENTRENAMIENTO
# ===========================================================
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[onecycle, checkpoint, early_stop],
    verbose=1
)


# In[11]:


# ===========================================================
# 9. GUARDAR HISTORIA Y MODELO FINAL
# ===========================================================
pd.DataFrame(history.history).to_csv("training_history_swin67.csv", index=False)
model.save("model_final_swin67.keras")


# In[12]:


print("\n🔎 Cargando mejor modelo...")
best_model = tf.keras.models.load_model(best_model_path)

print("\n🚀 Predicciones...")
y_pred = best_model.predict(test_gen).flatten()
y_true = y_test

mse = mean_squared_error(y_true, y_pred)
r2  = r2_score(y_true, y_pred)

print(f"\n📊 MSE Test: {mse:.4f}")
print(f"📈 R² Test: {r2:.4f}")


# In[13]:


plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.5)
mi, ma = min(y_true), max(y_true)
plt.plot([mi,ma],[mi,ma],'r--')
plt.xlabel("Real")
plt.ylabel("Predicho")
plt.title("Swin 67 bandas – Real vs Predicho")
plt.grid()
plt.savefig("scatter_swin67.png", dpi=300)
plt.show()


# In[14]:


df_pred = pd.DataFrame({"real": y_true, "prediction": y_pred})
df_pred.to_csv("predictions_swin67.csv", index=False)

print("📁 CSV guardado: predictions_swin67.csv")


# In[ ]:




