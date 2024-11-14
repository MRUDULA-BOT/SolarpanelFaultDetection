from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from keras import backend as K

# Create the base CNN model (feature extractor)
def create_base_model(input_shape):
    model_input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(model_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    return Model(inputs=model_input, outputs=x)

# Create the embedding model
def create_triple_net(input_shape):
    base_model = create_base_model(input_shape)
    
    # Create triplet loss model
    anchor = Input(shape=input_shape)
    positive = Input(shape=input_shape)
    negative = Input(shape=input_shape)

    # Get the embeddings for each of the 3 inputs
    anchor_embedding = base_model(anchor)
    positive_embedding = base_model(positive)
    negative_embedding = base_model(negative)
    
    # Create the triplet loss function
    def triplet_loss(y_true, y_pred):
        margin = 1.0
        positive_dist = K.sum(K.square(anchor_embedding - positive_embedding), axis=-1)
        negative_dist = K.sum(K.square(anchor_embedding - negative_embedding), axis=-1)
        return K.mean(K.maximum(positive_dist - negative_dist + margin, 0.0))
    
    # Define the model
    model = Model(inputs=[anchor, positive, negative], outputs=[anchor_embedding, positive_embedding, negative_embedding])

    model.compile(optimizer='adam', loss=triplet_loss)
    return model

# Input shape (same as your X data)
input_shape = (40, 24, 1)

# Create the model
model = create_triple_net(input_shape)

# Summarize the model
model.summary()
