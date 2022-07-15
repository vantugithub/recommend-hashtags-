import re
import pandas as pd
from langdetect import detect
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertModel


def get_hashtag(text):
    text = text.split()
    text_ = ""
    for tx in text:
        if "#" in tx:
            text_ += tx + " "
    text_ = text_.split('#')
    rs = ""
    for tx in text_:
        tx = tx.replace(" ", "")
        if len(tx) > 0:
            rs += tx + " "
    rs = remove_emoji(rs)
    rs_lis = rs.split()
    result = ""
    for i in rs_lis:
        if len(i) > 0:
            try:
                dectect_language = str(detect(i))
                if checkLanguage(dectect_language) is False:
                    continue
                else:
                    result += i + " "
            except:
                continue
    return result


def checkLanguage(language):
    list_language_code = ["th", "ar", "ko", "ur", "ja", "zh-cn", "fa", "tr", "zh-tw", "ru"]
    for i in list_language_code:
        if str(language) == i:
            return False
    return True


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def remove_special_characters(string):
    encoded_string = string.encode("ascii", "ignore")
    decode_string = encoded_string.decode()
    string = re.sub("\s\s+", " ", decode_string)
    string = string.replace('_', '')
    return re.sub(r'\W+', ' ', string)


def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist


def listToString(s):
    str1 = " "
    return (str1.join(s))


def get_hashtag_final(df, fileName):
    df['hashtags'] = df['hashtags'].apply(lambda x: get_hashtag(str(x)))
    df.to_csv(r'' + str(fileName), index=False, header=True)
    print(fileName)


def generate_training_data(df, ids, masks, tokenizer):
    for i, text in tqdm(enumerate(df['hashtags'])):
        tokenized_text = tokenizer.encode_plus(
            str(text),
            max_length=256,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='tf'
        )
        ids[i, :] = tokenized_text.input_ids
        masks[i, :] = tokenized_text.attention_mask
    return ids, masks


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


# Read data
df = pd.read_csv('data.csv', encoding="utf-8")

# Tokenization
BERT_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

df2 = df.copy()
labels_ = unique_list(df2['labels'])
labels_.sort()

df2['labels'] = df2['labels'].astype('category')
df2['labels'] = df2['labels'].cat.codes

df2 = df2.sample(frac=1).reset_index(drop=True)

token = tokenizer.encode_plus(
    df2['hashtags'].iloc[0],
    max_length=256,
    truncation=True,
    padding='max_length',
    add_special_tokens=True,
    return_tensors='tf'
)

X_input_ids = np.zeros((len(df2), 256))
X_attn_masks = np.zeros((len(df2), 256))

X_input_ids, X_attn_masks = generate_training_data(df2, X_input_ids, X_attn_masks, tokenizer)
labels = np.zeros((len(df2), len(labels_)))
labels[np.arange(len(df2)), df2['labels'].values] = 1

dataset = tf.data.Dataset.from_tensor_slices((X_input_ids, X_attn_masks, labels))


def SentimentDatasetMapFunction(input_ids, attn_masks, labels):
    return {
               'input_ids': input_ids,
               'attention_mask': attn_masks
           }, labels


dataset = dataset.map(SentimentDatasetMapFunction)
dataset = dataset.shuffle(10000).batch(16, drop_remainder=True)
p = 0.8
train_size = int(
    (len(df2) // 16) * p)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

model = TFBertModel.from_pretrained('bert-base-cased')

input_ids = tf.keras.layers.Input(shape=(256,), name='input_ids', dtype='int32')
attn_masks = tf.keras.layers.Input(shape=(256,), name='attention_mask', dtype='int32')

bert_embds = model.bert(input_ids, attention_mask=attn_masks)[1]

intermediate_layer = tf.keras.layers.Dense(512, activation='relu', name='intermediate_layer')(bert_embds)
output_layer = tf.keras.layers.Dense(len(labels_), activation='softmax', name='output_layer')(intermediate_layer)

sentiment_model = tf.keras.Model(inputs=[input_ids, attn_masks], outputs=output_layer)
sentiment_model.summary()

optim = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6)
loss_func = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

sentiment_model.compile(optimizer=optim, loss=loss_func, metrics=[acc])

hist = sentiment_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=2
)

sentiment_model.save('sentiment_model')
