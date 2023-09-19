import pandas as pd
from tensorflow.keras.models import load_model
from transformers import DistilBertTokenizerFast, TFDistilBertModel

df_test = pd.read_csv('test.csv')
df_test['full_text'].replace(r'\s+|\\n', ' ', regex=True, inplace=True)

model = load_model('model/model.h5', custom_objects={'TFDistilBertModel': TFDistilBertModel})
tokenizer = DistilBertTokenizerFast.from_pretrained('./model')

encoded = tokenizer.batch_encode_plus(
    df_test['full_text'].tolist(),
    add_special_tokens=False,
    max_length=512,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='tf',)

result = model.predict((encoded['input_ids'], encoded['attention_mask']))
submission = pd.read_csv('sample_submission.csv')

submission['cohesion'] = result[0]
submission['syntax'] = result[1]
submission['vocabulary'] = result[2]
submission['phraseology'] = result[3]
submission['grammar'] = result[4]
submission['conventions'] = result[5]

submission.to_csv('submission.csv', index=None)