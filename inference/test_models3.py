import tensorflow as tf

model = tf.keras.models.load_model(
    "models/cough-classification-lstm/INPUT_model_path"
)

print(model.summary())