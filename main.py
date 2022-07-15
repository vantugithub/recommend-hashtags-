import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
from transformers import BertTokenizer

app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False

model = tf.keras.models.load_model('sentiment_model')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

labels_ = ['art',
           'cryptocurrency',
           'drama',
           'eatclean',
           'fashion',
           'filmphotography',
           'food',
           'netflix',
           'skincare',
           'smartphone',
           'sport',
           'spotify',
           'technology',
           'travel']


def prepare_data(input_text, tokenizer):
    token = tokenizer.encode_plus(
        input_text,
        max_length=256,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='tf'
    )
    return {
        'input_ids': tf.cast(token.input_ids, tf.float64),
        'attention_mask': tf.cast(token.attention_mask, tf.float64)
    }


def make_prediction(model, processed_data, classes=None):
    if classes is None:
        classes = labels_
    probs = model.predict(processed_data)[0]
    # for pro in probs:
    #     print(pro)
    ind = np.argpartition(probs, -2)[-2:]
    top_1_labels = classes[ind[1]]
    top_2_labels = classes[ind[0]]
    return [top_1_labels, top_2_labels]


@app.route('/classify/<string:hashtag>')
def classify(hashtag):
    processed_data = prepare_data(hashtag, tokenizer)
    results = make_prediction(model, processed_data=processed_data)
    print(results)
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)
