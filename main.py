import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Concatenate, Layer
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K

# ========== ⚙️ Configuration ==========
noisy_path = 'data/noisy'
clean_path = 'data/clean'
SAMPLE_RATE = 22050
DURATION = 5.0
SAMPLES = int(SAMPLE_RATE * DURATION)

# ========== 🧩 Custom layer to match length ==========
class MatchLength1D(Layer):
    def call(self, inputs):
        x, ref = inputs
        x_len = K.shape(x)[1]
        ref_len = K.shape(ref)[1]
        diff = x_len - ref_len

        def crop():
            return x[:, :ref_len, :]

        def pad():
            return tf.pad(x, [[0, 0], [0, ref_len - x_len], [0, 0]])

        return tf.cond(diff > 0, crop, pad)

# ========== 🧠 U-Net Model Definition ==========
def build_unet(input_shape):
    inputs = Input(shape=input_shape)

    c1 = Conv1D(16, 3, activation='relu', padding='same')(inputs)
    p1 = MaxPooling1D(2)(c1)

    c2 = Conv1D(32, 3, activation='relu', padding='same')(p1)
    p2 = MaxPooling1D(2)(c2)

    c3 = Conv1D(64, 3, activation='relu', padding='same')(p2)

    u1 = UpSampling1D(2)(c3)
    u1 = MatchLength1D()([u1, c2])
    m1 = Concatenate()([u1, c2])
    c4 = Conv1D(32, 3, activation='relu', padding='same')(m1)

    u2 = UpSampling1D(2)(c4)
    u2 = MatchLength1D()([u2, c1])
    m2 = Concatenate()([u2, c1])
    c5 = Conv1D(16, 3, activation='relu', padding='same')(m2)

    outputs = Conv1D(1, 1, activation='tanh')(c5)

    return Model(inputs, outputs)

# ========== 🔄 Load Audio Pairs ==========
def load_audio_pairs(noisy_dir, clean_dir):
    X, Y = []
    files = os.listdir(noisy_dir)
    for file in files:
        noisy_file = os.path.join(noisy_dir, file)
        clean_file = os.path.join(clean_dir, file)
        if os.path.exists(clean_file):
            noisy, _ = librosa.load(noisy_file, sr=SAMPLE_RATE)
            clean, _ = librosa.load(clean_file, sr=SAMPLE_RATE)
            noisy = librosa.util.fix_length(noisy, size=SAMPLES)
            clean = librosa.util.fix_length(clean, size=SAMPLES)
            X.append(noisy)
            Y.append(clean)
    return np.array(X), np.array(Y)

# ========== 🚀 Train the Model ==========
print("🔄 Loading audio files...")
X, Y = load_audio_pairs(noisy_path, clean_path)
X = X.reshape(-1, SAMPLES, 1)
Y = Y.reshape(-1, SAMPLES, 1)

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

print("🧠 Building model...")
model = build_unet((SAMPLES, 1))
model.compile(optimizer='adam', loss='mae')

print("🚀 Training...")
model.fit(x_train, y_train, epochs=10, batch_size=8, validation_data=(x_val, y_val))

# ========== 💾 Save the Model ==========
os.makedirs("model", exist_ok=True)
model.save('model/U-Net.h5')
print("✅ Model saved to: model/U-Net.h5")
