import tensorflow as tf
import gradio as gr

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
x_train=x_train/255.0
x_test=x_test/255.0

model=tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),tf.keras.layers.Dense(128,activation="relu"),tf.keras.layers.Dense(10,activation="softmax")])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10)

Train on 60000 samples, validate on 10000 samples
Epoch 1/10
60000/60000 [==============================] - 10s 173us/sample - loss: 0.2631 - val_loss: 0.1375
Epoch 2/10
60000/60000 [==============================] - 7s 123us/sample - loss: 0.1158 - val_loss: 0.1019
Epoch 3/10
60000/60000 [==============================] - 7s 119us/sample - loss: 0.0805 - val_loss: 0.0928
Epoch 4/10
60000/60000 [==============================] - 7s 123us/sample - loss: 0.0603 - val_loss: 0.0740
Epoch 5/10
60000/60000 [==============================] - 7s 117us/sample - loss: 0.0459 - val_loss: 0.0730
Epoch 6/10
60000/60000 [==============================] - 8s 137us/sample - loss: 0.0368 - val_loss: 0.0772
Epoch 7/10
60000/60000 [==============================] - 7s 124us/sample - loss: 0.0293 - val_loss: 0.0741
Epoch 8/10
60000/60000 [==============================] - 7s 122us/sample - loss: 0.0231 - val_loss: 0.0793
Epoch 9/10
60000/60000 [==============================] - 7s 122us/sample - loss: 0.0195 - val_loss: 0.0787
Epoch 10/10
60000/60000 [==============================] - 7s 117us/sample - loss: 0.0161 - val_loss: 0.0797

#CREATING GUI
def classify(image):
 prediction=model.predict(image).tolist()[0]
 return{str(i):prediction[i] for i in range(10)}
sketchpad=gr.inputs.Sketchpad()
label=gr.outputs.Label(num_top_classes=3)
interface=gr.Interface(classify,sketchpad,label,live=True,capture_session=True)

#LAUNCH IT
interface.launch()
