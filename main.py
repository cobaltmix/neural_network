from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
import numpy as np

# Define the training data (sentences) and their labels
sentences = [
    'I love spending time with my friends and family.',
    'This weather is absolutely beautiful.',
    'Im feeling incredibly grateful right now.',
    'The movie was a pleasant surprise.',
    'Im impressed with your performance.',
    'I enjoy helping others and making a difference.',
    'This food tastes delicious.',
    'I appreciate your positive attitude.',
    'I cant believe how much I love this new album.',
    'The traffic is flowing smoothly today.',
    'Im really delighted by all the laughter.',
    'This project is going exceptionally well.',
    'I enjoy being around optimistic people.',
    'Im happy with the progress weve made.',
    'The service at this restaurant is outstanding.',
    'Im extremely satisfied with the quality of this product.',
    'I can find something positive about this situation.',
    'Im interested in hearing your stories.',
    'This book is thought-provoking and inspiring.',
    'Im so pleased with the level of progress.',
    'I enjoy creating a clean and organized living space.',
    'This job brings me fulfillment and purpose.',
    'I cant believe how much value I received from this purchase.',
    'I appreciate your positive outlook on life.',
    'The customer service was exceptional.',
    'Im extremely happy with the outcome of this project.',
    'I cant help but admire this level of competence.',
    'This music is uplifting and soothing.',
    'Im so grateful for the peace and harmony in my life.',
    'I cant believe how kind and considerate people are.',
    'This party is a great success.',
    'Im really impressed by your behavior.',
    'I cant get enough of your presence.',
    'This situation has immense potential for growth.',
    'Im interested in hearing what you have to say.',
    'Im delighted by the open and clear communication.',
    'I love the way you always listen to me.',
    'This movie is entertaining and captivating.',
    'I cant wait to spend another minute here.',
    'Im filled with joy by your actions.',
    'This meal is absolutely delightful.',
    'Im completely inspired by your honesty.',
    'I cant believe how smoothly this event was organized.',
    'Im grateful for your constructive criticism.',
    'I can find many redeeming qualities in this product.',
    'Im overjoyed with the way you treated me.',
    'I love this feeling of accomplishment.',
    'Im impressed with your creativity.',
    'I cant help but appreciate the way you always find solutions.',
    'This vacation was a complete success.',
    'I feel horrible!',
    'This is so bad!',
    'This cannot get any worse.',
    'I hate this result!',
    'The movie was a complete disappointment.',
    'Im not impressed with your performance.',
    'I hate dealing with difficult customers.',
    'This food tastes awful.',
    'Im tired of your constant complaining.',
    'I cant believe how much I dislike this new album.',
    'The traffic is unbearable today.',
    'Im really annoyed by all the noise.',
    'This project is a complete disaster.',
    'I cant stand being around negative people.',
    'Im sick and tired of your excuses.',
    'The service at this restaurant is terrible.',
    'Im extremely dissatisfied with the quality of this product.',
    'I cant find anything positive about this situation.',
    'Im not interested in your boring stories.',
    'This book is a waste of time.',
    'Im so frustrated with the lack of progress.',
    'I despise doing household chores.',
    'This job is nothing but stress and frustration.',
    'I cant believe how much money I wasted on this purchase.',
    'Im sick of your constant negativity.',
    'The customer service was absolutely appalling.',
    'Im extremely unhappy with the outcome of this project.',
    'I cant tolerate this level of incompetence.',
    'This music is grating on my nerves.',
    'Im so fed up with this constant chaos.',
    'I cant believe how rude that person was.',
    'This party is a total disaster.',
    'Im really disappointed in your behavior.',
    'I cant stand the sight of you.',
    'This situation is completely hopeless.',
    'Im not interested in what you have to say.',
    'Im furious about the lack of communication.',
    'I hate the way you always interrupt me.',
    'This movie is a waste of time and money.',
    'I cant bear the thought of spending another minute here.',
    'Im disgusted by your actions.',
    'This meal is absolutely disgusting.',
    'Im completely fed up with your lies.',
    'I cant believe how poorly this event was organized.',
    'Im sick and tired of your constant criticism.',
    'I cant find any redeeming qualities in this product.',
    'Im furious with the way you treated me.',
    'I hate this feeling of helplessness.',
    'Im not impressed with your excuses.',
    'I cant stand the way you always complain.',
    'This vacation was a complete disaster.',
]
labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 1 represents positive, 0 represents negative

# Tokenize the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1

# Convert sentences to integer sequences
input_sequences = tokenizer.texts_to_sequences(sentences)

# Find the maximum sequence length
max_sequence_length = max(len(sequence) for sequence in input_sequences)

# Pad sequences manually
padded_sequences = []
for sequence in input_sequences:
    pad_length = max_sequence_length - len(sequence)
    padded_sequence = sequence + [0] * pad_length
    padded_sequences.append(padded_sequence)

# Convert padded sequences to numpy array
padded_sequences = np.array(padded_sequences)

# Create the neural network model
embedding_size = 16
model = Sequential()
model.add(Embedding(total_words, embedding_size, input_length=max_sequence_length))
model.add(LSTM(16))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=1000, verbose=1)

# Test the model with new sentences
test_sentences = [
    'Wow, the traffic is really clearing up!',
    'This apple has some rotten aftertaste'
]
test_sequences = tokenizer.texts_to_sequences(test_sentences)

# Pad test sequences manually
padded_test_sequences = []
for sequence in test_sequences:
    pad_length = max_sequence_length - len(sequence)
    padded_sequence = sequence + [0] * pad_length
    padded_test_sequences.append(padded_sequence)

# Convert padded test sequences to numpy array
padded_test_sequences = np.array(padded_test_sequences)

predictions = model.predict(padded_test_sequences)

for i, sentence in enumerate(test_sentences):
    sentiment = 'positive' if predictions[i] > 0.5 else 'negative'
    print(f'Sentence: {sentence} ')
    print(f'Sentiment: {sentiment}\n')
