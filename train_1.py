import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


X = np.load("features.npy")  
y = np.load("labels.npy")    


num_classes = len(set(y))  
y = to_categorical(y - 1, num_classes=num_classes)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(200, 40)),  
    LSTM(64, return_sequences=False),  
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')  
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)


model.save("model.h5")
print("✅ Model training complete! Model saved as 'model.h5'")
