import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple
import json

import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.notebook import tqdm, trange

from pathlib import Path

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# Configs
logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

class Args():
    def __init__(self):
        self.output_dir = 'output-small-save'
        self.model_type = 'gpt2'
        self.model_name_or_path = 'gpt2-large'
        self.config_name = 'gpt2-large'
        self.tokenizer_name = 'gpt2-large'
        self.cache_dir = 'cached'
        self.block_size = 512
        self.do_train = True
        self.do_eval = True
        self.evaluate_during_training = False
        self.per_gpu_train_batch_size = 4
        self.per_gpu_eval_batch_size = 4
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 3
        self.max_steps = -1
        self.warmup_steps = 0
        self.logging_steps = 1000
        self.save_steps = 3500
        self.save_total_limit = None
        self.eval_all_checkpoints = False
        self.no_cuda = False
        self.overwrite_output_dir = True
        self.overwrite_cache = True
        self.should_continue = False
        self.seed = 42
        self.local_rank = -1
        self.fp16 = False
        self.fp16_opt_level = 'O1'

args = Args()

def clauseprediction(utterance):

    max_len = max([len(x.split()) for x in utterance])
    max_len

    # Emotion Identification Model

    import numpy as np
    import pandas as pd
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    #!pip install contractions
    import contractions
    import re
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    import matplotlib.pyplot as plt
    #import seaborn as sns
    #from wordcloud import WordCloud

    train_data = pd.read_csv(r'C:\Users\acer\Projects\Paper 1\Emotion\train.txt', names=['text', 'emotion'], sep=';')
    val_data = pd.read_csv(r'C:\Users\acer\Projects\Paper 1\Emotion\val.txt', names=['text', 'emotion'], sep=';')
    test_data = pd.read_csv(r'C:\Users\acer\Projects\Paper 1\Emotion\test.txt', names=['text', 'emotion'], sep=';')

    data = {'Train Data': train_data, 'Validation Data': val_data, 'Test Data': test_data}
    def preprocess(sentence):
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        sentence = re.sub('[^A-z]', ' ', sentence)
        negative = ['not', 'neither', 'nor', 'but', 'however', 'although', 'nonetheless', 'despite', 'except',
                            'even though', 'yet']
        stop_words = [z for z in stop_words if z not in negative]
        preprocessed_tokens = [lemmatizer.lemmatize(contractions.fix(temp.lower())) for temp in sentence.split() if temp not in stop_words] #lemmatization
        return ' '.join([x for x in preprocessed_tokens]).strip()

    train_data['text'] = train_data['text'].apply(lambda x: preprocess(x))
    val_data['text'] = val_data['text'].apply(lambda x: preprocess(x))
    test_data['text'] = test_data['text'].apply(lambda x: preprocess(x))

    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    train_x, train_y = ros.fit_resample(np.array(train_data['text']).reshape(-1, 1), np.array(train_data['emotion']).reshape(-1, 1))
    train = pd.DataFrame(list(zip([x[0] for x in train_x], train_y)), columns = ['text', 'emotion'])

    from sklearn import preprocessing
    le = preprocessing.OneHotEncoder()
    y_train= le.fit_transform(np.array(train['emotion']).reshape(-1, 1)).toarray()
    y_test= le.fit_transform(np.array(test_data['emotion']).reshape(-1, 1)).toarray()
    y_val= le.fit_transform(np.array(val_data['emotion']).reshape(-1, 1)).toarray()

    from transformers import RobertaTokenizerFast
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    def roberta_encode(data,maximum_length) :
      input_ids = []
      attention_masks = []


      for i in range(len(data.text)):
          encoded = tokenizer.encode_plus(data.text[i],
            add_special_tokens=True,
            max_length=maximum_length,
            pad_to_max_length=True,

            return_attention_mask=True,

          )

          input_ids.append(encoded['input_ids'])
          attention_masks.append(encoded['attention_mask'])
      return np.array(input_ids),np.array(attention_masks)

    max_len = max([len(x.split()) for x in train_data['text']])
    train_input_ids,train_attention_masks = roberta_encode(train, max_len)
    test_input_ids,test_attention_masks = roberta_encode(test_data, max_len)
    val_input_ids,val_attention_masks = roberta_encode(val_data, max_len)

    def create_model(bert_model, max_len):
        input_ids = tf.keras.Input(shape=(max_len,),dtype='int32')
        attention_masks = tf.keras.Input(shape=(max_len,),dtype='int32')

        output = bert_model([input_ids,attention_masks])
        output = output[1]

        output = tf.keras.layers.Dense(6, activation='softmax')(output)
        model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)
        model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    from transformers import TFRobertaModel
    roberta_model = TFRobertaModel.from_pretrained('roberta-base')
    model = create_model(roberta_model, max_len)
    model.summary()


    def roberta_inference_encode(data,maximum_length) :
        input_ids = []
        attention_masks = []



        encoded = tokenizer.encode_plus(
        data,
        add_special_tokens=True,
        max_length=maximum_length,
        pad_to_max_length=True,

        return_attention_mask=True

        )

        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        return np.array(input_ids),np.array(attention_masks)

    model = create_model(roberta_model, 43)
    model.load_weights(r'C:\Users\acer\Projects\Paper 1\Emotion\my_checkpoint')
    def inference(text_sentence, max_len):
        preprocessed_text = preprocess(text_sentence)
        input_ids, attention_masks = roberta_inference_encode(preprocessed_text, maximum_length = max_len)

        result = model.predict([input_ids, attention_masks])
        #le.categories_[0] = ['anger' 'fear' 'joy' 'love' 'sadness' 'surprise']
        result = pd.DataFrame(dict(zip(list(le.categories_[0]), [round(x*100, 2)for x in result[0]])).items(), columns = ['Category', 'Confidence'])
        #plot_result(result)
        return result

    # Initial emotion of each clause

    query_final=[]
    conf_final = []
    for text in utterance:
        res = inference(text, max_len)
        mx_val=res['Confidence'].argmax()
        query=res['Category'][mx_val]
        conf_final.append(res['Confidence'])
        query_final.append(query)

    # Emotion of the entire conversation

    text = ' '.join(utterance)
    result1 = inference(text, max_len)
    xyz=result1['Confidence'].argmax()
    overallquery=result1['Category'][xyz]

    # Final emotion of each clause

    for i in range(0,len(utterance)):
        maxval = max(conf_final[i])
        for j in range(0,6):
            if conf_final[i][j] == maxval:
                if conf_final[i][j]<70:
                    query_final[i] = overallquery

    # Emotion Score of each clause

    anger = []
    love = []
    joy = []
    fear = []
    sad = []
    sur = []
    def dataappend(result):
        anger.append(result['Confidence'][0])
        fear.append(result['Confidence'][1])
        joy.append(result['Confidence'][2])
        love.append(result['Confidence'][3])
        sad.append(result['Confidence'][4])
        sur.append(result['Confidence'][5])

    for text in utterance:
        result = inference(text, max_len)
        #print(result)
        dataappend(result)

    # Causal Likelihood Score

    import numpy as np

    def cosine(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    from sentence_transformers import SentenceTransformer
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

    import numpy as np
    def WordEmbeddingSimilarity(query_final, utterance):
        wes = []
        for (q,s) in zip(query_final,utterance):
            sim = cosine(sbert_model.encode([q])[0], sbert_model.encode([s])[0])
            wes.append(sim)
        return wes
    wes = WordEmbeddingSimilarity(query_final, utterance)

    # Word/Synonym present

    synonymsjoy = sorted(['Thanks', 'Nice', 'comfort', 'great', 'playfulness', 'blessedness', 'help', 'warm', 'cheer', 'mirth', 'merriment', 'glee', 'flying high', 'breathtaking', 'anniversary', 'easy', 'merry', 'exultant', 'enjoying', 'favorite', 'perky', 'wins', 'pride', 'gratified', 'elation', 'big fan', 'wow', 'fabulous', 'gleeful', 'chipper', 'excited', 'peppy', 'light', 'pride and joy', 'solace', 'contented', 'good', 'paradise', 'walking on air', 'intoxicated', 'cheerful', 'masterpiece', 'jubilance', 'sunny', 'exulting', 'hilarity', 'jolly', 'very well', 'well-being', 'fruition', 'nice', 'better', 'gorgeous', 'joyous', 'popular', 'terrific', 'on cloud nine', 'smart', 'refreshment', 'cute', 'like', 'geniality', 'liveliness', 'sparkling', 'thanks', 'fantastic', 'joviality', 'chirpy', 'thrilled', 'optimism', 'victory', 'pleased', 'funny', 'sport', 'euphoria', 'dear', 'treat', 'thank you', 'delighted', 'ecstasy', 'gratification', 'good cheer', 'jubilant', 'prosperity', 'charm', 'peaceful', 'regalement', 'frolic', 'transport', 'right', 'pleasant', 'good humor', 'overjoyed', 'stunning', 'satisfaction', 'engaged', 'prize', 'congratulation', 'elated', 'reward', 'beatitude', 'glad', 'animation', 'revelry', 'kind', 'rich', 'indulgence', 'blissful', 'exhilaration', 'alleviation', 'interesting', 'ecstatic', 'luxury', 'enjoyed', 'felicity', 'gay', 'goodbye', 'gem', 'laughter', 'looking good', 'blest', 'blithe', 'cheeriness', 'playful', 'festivity', 'laugh', 'exciting', 'delight', 'ravishment', 'exuberance', 'laughing', 'neat', 'tickled pink', 'excellent', "let's go", 'good luck', 'pleasure', 'fun', 'rapture', 'awesome', 'content', 'loved', 'jubilation', 'welcome', 'fine', 'energy', 'joyful', 'oh my god', 'contentment', 'loves', 'luck', 'best', 'friend', 'relaxed', 'amazing', 'diversion', 'amusement', 'Thank you', 'heaven', 'peace of mind', 'joy', 'lightheartedness', 'so cool', 'sweet', 'jewel', 'sanctity', 'enjoy', 'lively', 'greateful', 'captivated', 'hopefulness', 'incredible', 'wonderful', 'cheerfulness', 'bliss', 'marry', 'tickled', 'proud', 'fine day', 'party', 'congratulations', 'happy', 'satisfied', 'grateful', 'festival', 'christmas', 'good spirits', 'impressed', 'lovely', 'enjoyment', 'treasure', 'convivial', 'humor', 'eudemonia', 'love', 'fastest', 'gladness', 'enchantment', 'beautiful', 'relieved', 'delirium', 'gaiety', 'vivacity', 'upbeat', 'blessed', 'perfect day', 'delectation', 'absolutely', 'new', 'perfect', 'accurate', 'rejoicing', 'seventh heaven', 'lucky', 'charming', "can't complain", 'mirthful', 'hope', 'delicious', 'exultation'])
    synonymsanger = sorted(['accident', 'acerbity', 'acrimony', 'affronted', 'agitation', 'alone', 'anger', 'angry', 'animosity', 'annoyance', 'annoyed', 'antagonism', 'antagonized', 'apoplexy', 'arrest', 'asperity', 'awful', 'bad', 'bitter', 'bitterness', 'blank', 'blood boil', 'blow up', 'blowup', 'bluster', 'breathing down my neck', 'cat fit', 'chafed', 'chagrin', 'cheated', 'choler', 'choleric', 'complaining', 'conniption', 'convulsed', 'convulsion', 'crazy', 'cross', 'crossed the line', 'cruel', 'dander', 'dare you', 'dirty', 'disappointing', 'disapprobation', 'disgusting', 'displeased', 'displeasure', 'distemper', 'divorce', "don 't ", "don't", 'dying', 'end', 'enmity', 'enraged', 'eruption', 'exacerbated', 'exasperated', 'exasperation', 'excitement', 'expensive', 'explode', 'explosion', 'ferment', 'ferocious', 'ferocity', 'fierce', 'fiery', 'fired', 'fireworks', 'frenzy', 'frustrating', 'fuming', 'furious', 'furor', 'fury', 'gall', 'galled', 'get out', 'go away', 'gross', 'had it', 'hate', 'hateful', 'hatred', 'heat', 'heated', 'hemorrhage', 'hissy fit', 'horrible', 'hot', 'how dare you', 'huff', 'huffy', 'hysterics', 'idiot', 'ill humor', 'ill temper', 'ill-tempered', 'impassioned', 'impatience', 'impose', 'impossible', 'inappropriate', 'incensed', 'inconsiderate', 'indignant', 'indignation', 'inflamed', 'infuriated', 'infuriation', 'irascibility', 'irascible', 'irate', 'ire', 'ireful', 'irritability', 'irritable', 'irritated', 'irritation', "isn't fair", 'kill', 'mad', 'maddened', 'madness', 'mania', 'miff', 'missing the point', 'mistake', 'must go', 'nettled', 'not nice', 'obsession', 'offended', 'outburst', 'outrage', 'outraged', 'overpaid', 'paroxysm', 'passion', 'peevishness', 'petulance', 'pigsty', 'pique', 'piqued', 'pissed', 'pissed off', 'problem', 'provoked', 'put out', 'rage', 'raging', 'rampage', 'rankling', 'raving', 'resentful', 'resentment', 'respect', 'riled', 'screaming', 'sick', 'slow burn', 'sore', 'soreness', 'sorry', 'spasm', 'spleen', 'splenetic', 'squall', 'stew', 'storm', 'storming', 'stupid', 'sucks', 'sulky', 'sullen', 'tantrum', 'temper', 'terrible', 'tiff', 'too far', 'too late', 'umbrage', 'uncomfortable', 'unpleasant', 'uproar', 'upset', 'uptight', 'vehemence', 'very sorry', 'vexation', 'vexed', 'violence', 'wingding', 'wiped out', 'worst', 'wrath', 'wrathful'])
    synonymssad = sorted(['agitated', 'all torn up', 'amazed', 'anguish', 'antsy', 'apprehensive', 'ashamed', 'awful', 'bad', 'badly', 'bereaved', 'bitter', 'blahs', 'blank', 'bleakness', 'blew', 'blue', 'blue devils', 'blue funk', 'bored', 'boring', 'breaking up', 'broken heart', 'broken up', 'bummed out', 'bummer', 'capsized', 'chaotic', 'cheerless', 'cheerlessness', 'come apart', 'confused', 'dejected', 'dejection', 'depressed', 'depressing', 'despairing', 'despondency', 'despondent', 'disappointed', 'disconcerted', 'disconsolate', 'disconsolateness', 'dismal', 'dismals', 'dismayed', 'disordered', 'dispiritedness', 'disquieted', 'distress', 'distressed', 'doleful', 'dolefulness', 'dolor', "don't", "don't know", 'down', 'down in dumps', 'down in the mouth', 'downcast', 'downcastness', 'downer', 'dragged', 'dysphoria', 'end', 'fools', 'forlorn', 'forlornness', 'frantic', 'funk', 'get away', 'gloominess', 'gloomy', 'glum', 'grief', 'grief-stricken', 'grieved', 'grieving', 'hate', 'heartache', 'heartbreak', 'heartbroken', 'heartsick', 'heavy heart', 'heavy-hearted', 'homesick', 'hopelessness', 'horrible', 'hurt', 'hurting', 'hurts', 'idiot', 'ill', 'in disarray', 'in doldrums', 'in grief', 'in the dumps', 'jittery', 'jumpy', 'languishing', 'let-down', 'letdown', 'listlessness', 'lonely', 'lost my job', 'low', 'low-spirited', 'lugubrious', 'melancholy', 'misery', 'mistakes', 'moodiness', 'mopes', 'morbid', 'morose', 'mournful', 'mournfulness', 'mourning', 'muddled', 'not happy', 'nothing to do', 'out of fashion', 'out of sorts', 'overturned', 'overwrought', 'part', 'pensive', 'pessimistic', 'poignancy', 'poor', 'problem', 'psyched-out', 'put out', 'rattled', 'rude', 'ruffled', 'sacked', 'sad', 'shocked', 'shook-up', 'sick', 'sick at heart', 'somber', 'sorrow', 'sorrowful', 'sorrowfulness', 'sorry', 'spilled', 'stupid', 'terrible', 'the blues', 'the dumps', 'thrown', 'tipped over', 'toppled', 'tribulation', 'trouble', 'troubled', 'tumbled', 'unfortunately', 'unglued', 'unhappy', 'unsettled', 'unzipped', 'upset', 'upside-down', 'very sorry', 'weeping', 'wistful', 'woe', 'woebegone', 'worried'])
    synonymslove = sorted(['admirable', 'admire', 'adorable', 'adore', 'adulation', 'affection', 'allegiance', 'alluring', 'amity', 'amorousness', 'amour', 'appreciate', 'appreciation', 'ardency', 'ardor', 'attachment', 'attractive', 'be crazy about', 'be entertained', 'be fond of', 'be gone on', 'be mad for', 'be nuts about', 'be pleased', 'be serious about', 'be smitten with', 'be stuck on', 'be sweet on', 'be wild about', 'beauteous', 'beautiful', 'bewitching', 'captivating', 'case', 'charming', 'cherish', 'cherishing', 'comely', 'congratulations', 'cotton to', 'crush', 'dainty', 'delectable', 'delicate', 'delicious', 'delight', 'delight in', 'delightful', 'devotedness', 'devotion', 'dig', 'dote on', 'drink in', 'eat up', 'emotion', 'enchanting', 'enchantment', 'engaging', 'enjoy', 'enjoyable', 'enjoyment', 'esteem', 'exalt', 'exquisite', 'fair', 'fall for', 'fancy', 'fervor', 'fidelity', 'flame', 'flip over', 'fondness', 'freak out on', 'friendship', 'get a charge out of', 'get a kick out of', 'get high on', 'glorify', 'go', 'go for', 'good-looking', 'gorgeous', 'graceful', 'gratifying', 'great', 'handsome', 'hankering', 'happy', 'have a ball', 'have a good time', 'have fun', 'help', 'honor', 'idolatry', 'idolize', 'inclination', 'infatuation', 'involvement', 'knockout', 'like', 'live a little', 'live it up', 'love', 'loves', 'lovesome', 'lucky', 'lust', 'luxuriate in', 'mad for', 'mind', 'paint the town', 'partiality', 'passion', 'piety', 'pleasant', 'pleasing', 'pretty', 'prize', 'pulchritudinous', 'rapture', 'rare', 'regard', 'rejoice in', 'relish', 'respect', 'revel in', 'revere', 'reverence', 'savor', 'scrumptious', 'sentiment', 'soft spot', 'splendid', 'stunning', 'sweet', 'take joy in', 'tenderness', 'thank you', 'thoughtful', 'thrill to', 'treasure', 'venerate', 'weakness', 'winning', 'wonderful', 'worship', 'yearning', 'zeal'])
    synonymsfear = sorted(['CRASH ?', 'abashed', 'abhorrence', 'afraid', 'aghast', 'agitation', 'agony', 'alarm', 'alarmed', 'alert', 'angst', 'anguish', 'anxiety', 'anxious', 'apprehension', 'apprehensive', 'apprehensiveness', 'aroused', 'aversion', 'awe', 'be careful', 'blanched', 'blow', 'break my leg', 'broke', 'bête noire', "can ' t believe", 'chickenheartedness', 'cold feet', 'cold sweat', 'collapse', 'concern', 'confusion', 'consternation', 'cowardice', 'cowardly', 'cowed', 'creeps', 'damage', 'daunted', 'derangement', 'despair', 'discomposure', 'discouraged', 'disheartened', 'dismay', 'dismayed', 'disquietude', 'distress', 'distressed', 'disturbance', 'disturbed', 'doubt', 'dread', 'faint-hearted', 'faintheartedness', 'fearful', 'fearfulness', 'foreboding', 'fright', 'frightened', 'frozen', 'funk', 'get away', 'have cold feet', 'having cold feet', 'horrified', 'horror', 'hurt', 'in awe', 'injury', 'intimidated', 'jitters', 'jolt', 'love', 'misgiving', 'nervous', 'nightmare', 'ordeal', 'outburst', 'panic', 'panic-stricken', 'panicked', 'panicky', 'perplexed', 'perturbed', 'petrified', 'phobia', 'presentiment', 'qualm', 'rattled', 'recreancy', 'reverence', 'revulsion', 'run scared', 'safety', 'scare', 'scared', 'scared stiff', 'scared to death', 'shaken', 'shock', 'shocked', 'shooting', 'spooked', 'startled', 'strain', 'stress', 'stunned', 'suffering', 'suspicion', 'suspicious', 'terrified', 'terror', 'terror-stricken', 'timid', 'timidity', 'timorous', 'torture', 'traumatization', 'trembling', 'trepidation', 'unease', 'uneasiness', 'upheaval', 'upset', 'worried', 'worry', 'wound'])
    synonymssurp = sorted(['abruptness', 'accident', 'accountancy ?', 'affect', 'affright', 'agitate', 'alarm', 'amaze', 'amazement', 'amazing', 'amazing !', 'astonish', 'astonishment', 'astound', 'astoundment', 'attack', 'awe', 'beautiful', 'believe', 'bewilder', 'bewilderment', 'blank', 'blow away', "blow one's mind", 'bolt', 'bombshell', 'bowl over', "can't believe", 'consternate', 'consternation', 'crazy', 'curiosity', 'curveball', 'dare you !', 'daze', 'deal', 'depressed', 'disappointment', 'disillusion', 'dumbfound', 'electrify', 'epiphany', 'eureka', 'eye-opener', 'flabbergast', 'floor', 'fortune', 'fright', 'frightened', 'give a turn', 'godsend', 'goodness !', 'great', 'great !', 'hope', 'impossible !', 'impress', 'incredulity', 'incredible', 'interesting', 'joking', 'jolt', 'jump', 'kick', 'kidding', 'kidding !', 'kidding ?', 'laugh', 'look !', 'love', 'make jump', 'marvel', 'miracle', 'miscalculation', 'move', 'nice', 'no kidding !', 'nuts !', 'oh dear', 'perplex', 'phenomenon', 'pleasure', 'popular', 'portent', 'precipitance', 'precipitation', 'precipitousness', 'prodigy', 'put one away', 'rarity', 'real ?', 'really', 'really ?', 'revelation', 'ridiculous !', 'rock', 'scare', 'scare to death', 'serious !', 'shake up', 'shock', 'shocked', 'soon ?', 'sorry', 'spook', 'spring', 'spring something on', 'stagger', 'startle', 'strike', 'stun', 'stupefaction', 'stupefy', 'stupid', 'suddenness', 'surprise', 'surprised', 'take aback', 'terrible', 'terrified', 'terrify', 'terrorize', 'thank you', 'thunderbolt', 'too far !', 'touch', 'unbelievable !', 'unexpected', 'unforeseen', 'watch !', 'welcome', 'whammy', 'what !', 'what ?', 'why !', 'why ?', 'wonder', 'wonderful', 'wonderment', 'worried', 'wow', 'wow !', 'wrong ?', 'yeah ?'])

    def Word_or_Syn(utterance, query_final):
        wordpresent = []
        for (test_string,qu) in zip(utterance,query_final):
            if qu == 'joy':
                synn = synonymsjoy
            elif qu == 'sadness':
                synn = synonymssad
            elif qu == 'anger':
                synn = synonymsanger
            elif qu == 'love':
                synn = synonymslove
            elif qu == 'fear':
                synn = synonymsfear
            else:
                synn = synonymssurp
            res = [ele for ele in synn if(ele in test_string)]
            if res:
                wordpresent.append(1)
            else:
                wordpresent.append(0)
        return wordpresent

    wordpresent = Word_or_Syn(utterance, query_final)

    # Combining it all to make a df

    import pandas as pd

    df = pd.DataFrame()
    df = pd.DataFrame(columns = ['Clauses', 'Anger','Joy', 'Word Embedding Similarity', 'Word/Synonym', 'Love', 'Fear', 'Sadness','Surprise','Query'])
    for (i,j,k,l,m,a,b,c,d,e) in zip(utterance,wes,anger,wordpresent,joy,fear,love,sur,sad,query_final):
        df = df.append({'Clauses' : i, 'Word Embedding Similarity':j, 'Word/Synonym':l, 'Anger' :k, 'Joy' : m, 'Fear': a, 'Love': b, 'Surprise':c, 'Sadness':d, 'Query':e}, ignore_index = True)

    df['Clauses'] = df['Clauses'].astype(str)
    df['Anger'] = df['Anger'].astype(float)
    df['Joy'] = df['Joy'].astype(float)
    df['Word Embedding Similarity'] = df['Word Embedding Similarity'].astype(float)
    df['Word/Synonym'] = df['Word/Synonym'].astype(int)
    df['Love'] = df['Love'].astype(float)
    df['Fear'] = df['Fear'].astype(float)
    df['Sadness'] = df['Sadness'].astype(float)
    df['Surprise'] = df['Surprise'].astype(float)
    df['Query'] = df['Query'].astype(str)
    df.dtypes


    # Mix Text Generation

    import pandas as pd
    sen_w_feats = []
    for index, row in df.iterrows():
        combined = ""
        combined += "{:} shows {:} emotion. The similarity between the clause and {:} emotion is {:}. The clause has {:} level of anger, {:} level of fear, {:} level of love, {:} level of joy, {:} level of sadness and {:} level of surprise. It has {:} word that shows {:} emotion.".format(row["Clauses"], row["Query"], row["Query"],row["Word Embedding Similarity"],row["Anger"], row["Fear"], row["Love"], row["Joy"], row["Sadness"], row["Surprise"], row["Word/Synonym"],row["Query"])
        sen_w_feats.append(combined)
    df['callfeats'] = sen_w_feats
    df

    # Feature Extraction

    import os
    import time
    import datetime
    import random
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.model_selection import train_test_split
    from transformers import TFAutoModel, AutoTokenizer, TFBertForSequenceClassification,AutoConfig
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.layers import Maximum,LayerNormalization,GlobalMaxPooling2D,Average,Dot, Dense, Input, GlobalAveragePooling1D, BatchNormalization, Activation, Concatenate, Flatten, Dropout, Conv1D, MaxPooling1D, Add, Lambda, GlobalAveragePooling2D, Reshape, RepeatVector, UpSampling1D
    from tensorflow.keras.models import Model
    from keras.layers import LSTM, Bidirectional
    from official import nlp
    import official.nlp.optimization
    data = df.Clauses.values
    import re
    for i in range(data.shape[0]):
        data[i] = re.sub(r'<url>','HTTPURL',data[i])
        data[i] = re.sub(r'<user>','@USER',data[i])

    from transformers import BertTokenizer
    # using the low level BERT for our task.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    MAX_LENGTH  = 128
    MAX_LENGTH_MIX = 256
    def TexttoEmb(data, tokenizer, MAX_LENGTH):
        input_ids       = []
        attention_masks = []

        for sent in tqdm(data):
            encoded_dict = tokenizer.encode_plus(
                                sent,
                                add_special_tokens = True,
                                max_length = MAX_LENGTH,
                                pad_to_max_length = True,
                                return_attention_mask = True,
                                return_tensors = 'np',
                                truncation = True,
                           )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        id_tvt       = np.concatenate(input_ids)
        mask_tvt      = np.concatenate(attention_masks)

        return id_tvt

    id_train = TexttoEmb(data, tokenizer, MAX_LENGTH)

    for i in range(0,df.shape[0]):
        df['Word Embedding Similarity'][i] = df['Word Embedding Similarity'][i]*1000
    df.head(5)

    for i in df.columns:
        #print(new_df[i].dtypes)
        if df[i].dtypes == 'float64':
            df[i] = df[i].astype(int)
    df

    def MixTexttoEmb(data, tokenizer, MAX_LENGTH):
        input_ids       = []
        attention_masks = []

        for sent in tqdm(data):
            encoded_dict = tokenizer.encode_plus(
                                sent,
                                add_special_tokens = True,
                                max_length = MAX_LENGTH,
                                pad_to_max_length = True,
                                return_attention_mask = True,
                                return_tensors = 'np',
                                truncation = True,
                           )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        id_tvt        = np.concatenate(input_ids)
        mask_tvt      = np.concatenate(attention_masks)

        return id_tvt

    mtrain_sent      = df.callfeats.values
    mid_train = MixTexttoEmb(mtrain_sent, tokenizer, MAX_LENGTH_MIX)

    def lofl(a):
        lst = []
        for i in range(0,len(a)):
            lst.append([])
            lst[i].append(a[i])
        return lst

    num_train_1 = lofl(df['Word Embedding Similarity'].to_numpy().tolist())
    num_train_2 = lofl(df['Anger'].to_numpy().tolist())
    num_train_3 = lofl(df['Love'].to_numpy().tolist())
    num_train_4 = lofl(df['Joy'].to_numpy().tolist())
    num_train_5 = lofl(df['Fear'].to_numpy().tolist())
    num_train_6 = lofl(df['Surprise'].to_numpy().tolist())
    num_train_7 = lofl(df['Sadness'].to_numpy().tolist())
    num_train_8 = lofl(df['Word/Synonym'].to_numpy().tolist())

    def NumtoEmb(data,tokenizer, MAX_LENGTH):
        num_input_ids       = []
        num_attention_masks = []
        for sent in tqdm(data):
            encoded_dict = tokenizer.encode_plus(
                                sent,
                                add_special_tokens = False,
                                max_length = MAX_LENGTH,
                                pad_to_max_length = True,
                                return_attention_mask = True,
                                return_tensors = 'np',
                                truncation = True,
                           )
            num_input_ids.append(encoded_dict['input_ids'])
            num_attention_masks.append(encoded_dict['attention_mask'])


        num_id_tvt        = np.concatenate(num_input_ids)
        num_mask_tvt      = np.concatenate(num_attention_masks)

        return num_id_tvt

    num_train_idwe= NumtoEmb(num_train_1,tokenizer, MAX_LENGTH)
    num_train_idc= NumtoEmb(num_train_2,tokenizer, MAX_LENGTH)
    num_train_idl= NumtoEmb(num_train_3,tokenizer, MAX_LENGTH)
    num_train_idp= NumtoEmb(num_train_4,tokenizer, MAX_LENGTH)
    num_train_idne= NumtoEmb(num_train_5,tokenizer, MAX_LENGTH)
    num_train_idnu= NumtoEmb(num_train_6,tokenizer, MAX_LENGTH)
    num_train_idpos= NumtoEmb(num_train_7,tokenizer, MAX_LENGTH)
    num_train_idsyn= NumtoEmb(num_train_8,tokenizer, MAX_LENGTH)

    MODEL       = 'bert-base-uncased'
    MODEL_NAME  = 'bert-base-uncased'
    N_LABELS    = 1
    transformer = TFAutoModel.from_pretrained(MODEL,output_attentions=False,output_hidden_states=True,return_dict =True)
    transformer.trainable = False

    def attfun(inpAttImg_query, inpAttImg_key):
        head = 1
        att_layers = 1
        att_hid = 32
        for layer in range(1):
          for _ in range(head):
            img_key = Dense(att_hid/head, use_bias=False)(inpAttImg_key) #change to tanh?
            text_query = Dense(att_hid/head, use_bias=False)(inpAttImg_query)
            img_value = Dense(att_hid/head, use_bias=False)(inpAttImg_key)

            attention = Dot(axes=1)([text_query, img_key])
            attention = Lambda(lambda x: x[0]/x[1])([attention,np.sqrt(att_hid/head)])
            attention = Activation("softmax")(attention)
            attention = Dense(att_hid/head, use_bias=False)(attention)
            head_att_img = Dot(axes=1)([attention, img_value])

        coatt = tf.keras.layers.Dense(32)(head_att_img)
        return coatt

    def create_model():
        input_ids = Input(shape=(128,), dtype=tf.float32, name='input_ids')
        num_input_ids_1 = Input(shape=(128,), dtype=tf.float32, name='num_input_ids_1')
        num_input_ids_2 = Input(shape=(128,), dtype=tf.float32, name='num_input_ids_2')
        num_input_ids_3 = Input(shape=(128,), dtype=tf.float32, name='num_input_ids_3')
        num_input_ids_4 = Input(shape=(128,), dtype=tf.float32, name='num_input_ids_4')
        num_input_ids_5 = Input(shape=(128,), dtype=tf.float32, name='num_input_ids_5')
        num_input_ids_6 = Input(shape=(128,), dtype=tf.float32, name='num_input_ids_6')
        num_input_ids_7 = Input(shape=(128,), dtype=tf.float32, name='num_input_ids_7')
        num_input_ids_8 = Input(shape=(128,), dtype=tf.float32, name='num_input_ids_8')
        minput_ids = Input(shape=(256,), dtype=tf.float32, name='mixinput_ids')

        head = 1
        att_layers = 1
        att_hid = 32
        final_hid = 32
        coatt1 = attfun(input_ids, num_input_ids_1)
        coatt2 = attfun(input_ids, num_input_ids_2)
        coatt3 = attfun(input_ids, num_input_ids_3)
        coatt4 = attfun(input_ids, num_input_ids_4)
        coatt5 = attfun(input_ids, num_input_ids_5)
        coatt6 = attfun(input_ids, num_input_ids_6)
        coatt7 = attfun(input_ids, num_input_ids_7)
        coatt8 = attfun(input_ids, num_input_ids_8)

        att = []
        att.append(coatt1)
        att.append(coatt2)
        att.append(coatt3)
        att.append(coatt4)
        att.append(coatt5)
        att.append(coatt6)
        att.append(coatt7)
        att.append(coatt8)
        finalatt = Concatenate(axis=1)(att)
        finalatt = tf.keras.layers.Dense(32)(finalatt)

        mixnew = attfun(minput_ids, minput_ids)
        # mixnew
        x0 = tf.keras.layers.Dense(32)(input_ids)
        x1 = tf.keras.layers.Dense(32)(num_input_ids_1)
        x2 = tf.keras.layers.Dense(32)(num_input_ids_2)
        x3 = tf.keras.layers.Dense(32)(num_input_ids_3)
        x4 = tf.keras.layers.Dense(32)(num_input_ids_4)
        x5 = tf.keras.layers.Dense(32)(num_input_ids_5)
        x6 = tf.keras.layers.Dense(32)(num_input_ids_6)
        x7 = tf.keras.layers.Dense(32)(num_input_ids_7)
        x8 = tf.keras.layers.Dense(32)(num_input_ids_8)

        merge = []
        merge.append(x0)
        merge.append(finalatt)
        merge.append(x1)
        merge.append(x2)
        merge.append(x3)
        merge.append(x4)
        merge.append(x5)
        merge.append(x6)
        merge.append(x7)
        merge.append(x8)
        merge.append(mixnew)
        # merge

        final = Concatenate(axis=1)(merge)
        # final

        drop_rate = 0.2
        N_LABELS = 1
        l_merge = Dropout(drop_rate)(final)
        l_merge = Dense(final_hid)(l_merge)
        l_merge = BatchNormalization()(l_merge)
        l_merge = Activation('relu')(l_merge)
        l_merge = Dropout(drop_rate)(l_merge)
        out = Dense(N_LABELS, activation='sigmoid')(l_merge)
        out
        model = Model(inputs=[input_ids, num_input_ids_1, num_input_ids_2, num_input_ids_3, num_input_ids_4, num_input_ids_5, num_input_ids_6, num_input_ids_7, num_input_ids_8, minput_ids], outputs=out)
    #num_input_ids_1, num_input_ids_2, num_input_ids_3, num_input_ids_4, num_input_ids_5, num_input_ids_6, num_input_ids_7, num_input_ids_8
        return model

    model = create_model()
    model.load_weights(r'C:\Users\acer\Projects\checkpoint\weights\my_checkpoint')
    newpred = [id_train, num_train_idwe, num_train_idc, num_train_idl, num_train_idp, num_train_idne, num_train_idnu, num_train_idpos, num_train_idsyn, mid_train]
    predictions = model.predict(newpred, verbose=1)
    from iteration_utilities import deepflatten
    predictions = list(deepflatten(predictions, depth=1))

    theta = 0.5
    clauses = []
    for i in range(0,len(utterance)):
        if predictions[i]>theta:
            #print("Cause")
            clauses.append(i)

    print(clauses)
    return clauses, conf_final, overallquery, query_final, wes, wordpresent

def causeprediction(utterance, clauses, conf_final, overallquery, query_final, wes, wordpresent, clausepred):
    # Length of each clause

    def lengthfeature (utterance):
        listtosort = utterance.copy()
        length = []
        for i in listtosort:
            leng = len(i)
            length.append(leng)
        #listtosort.sort(key = len, reverse=True)
        #print(length)
        return length
    length = lengthfeature (utterance)

    max_len = max([len(x.split()) for x in utterance])
    max_len

    # Presence of Conjunction/Preposition

    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    stop_words = set(stopwords.words('english'))
    def Conj_or_Prepo_Present(utterance):
        conjprep = []
        for txt in utterance:
            #print(txt)
            count = 0
            tokenized = sent_tokenize(txt)
            for i in tokenized:
                wordsList = nltk.word_tokenize(i)
                tagged = nltk.pos_tag(wordsList)
    #             print(tagged)
                for j in range(0,len(tagged)):
                    if tagged[j][1] == 'CC' or tagged[j][1] == 'IN':
                        count= count+1
    #                     print(txt, tagged[j][0], "-->", tagged[j][1])
                        j = j+2
    #                 print(count)
            if count>0:
                conjprep.append(1)
            else:
                conjprep.append(0)
        return conjprep
    conjprep = Conj_or_Prepo_Present(utterance)

    # Finding the Conjunction/Preposition present

    word = []
    for i in range(0,len(conjprep)):
        if conjprep[i] == 1:
            x = utterance[i]
    #         print(x)
            tokenized = sent_tokenize(x)
            for i in tokenized:
                wordsList = nltk.word_tokenize(i)
                tagged = nltk.pos_tag(wordsList)
                for j in range(0,len(tagged)):
                    if tagged[j][1] == 'CC':
                        #print(x, tagged[j][0], "-->", tagged[j][1])
                        word.append(tagged[j][0])
                        break
                    elif tagged[j][1] == 'IN':
                        #print(x, tagged[j][0], "-->", tagged[j][1])
                        word.append(tagged[j][0])
                        break
        else:
            word.append('None')

    # Sentiment of each clause

    import numpy as np
    import pandas as pd
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    #!pip install contractions
    import contractions
    import re
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    import matplotlib.pyplot as plt
    # import seaborn as sns
    # from wordcloud import WordCloud

    train_data = pd.read_csv(r'C:\Users\acer\Projects\Paper 1\Sentiment\sentrain.txt', names=['text', 'sentiment'], sep=';')
    val_data = pd.read_csv(r'C:\Users\acer\Projects\Paper 1\Sentiment\senval.txt', names=['text', 'sentiment'], sep=';')
    test_data = pd.read_csv(r'C:\Users\acer\Projects\Paper 1\Sentiment\sentest.txt', names=['text', 'sentiment'], sep=';')
    train_data.head()

    data = {'Train Data': train_data, 'Validation Data': val_data, 'Test Data': test_data}

    def preprocess(sentence):
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        sentence = re.sub('[^A-z]', ' ', sentence)
        negative = ['not', 'neither', 'nor', 'but', 'however', 'although', 'nonetheless', 'despite', 'except',
                            'even though', 'yet']
        stop_words = [z for z in stop_words if z not in negative]
        preprocessed_tokens = [lemmatizer.lemmatize(contractions.fix(temp.lower())) for temp in sentence.split() if temp not in stop_words] #lemmatization
        return ' '.join([x for x in preprocessed_tokens]).strip()

    train_data['text'] = train_data['text'].apply(lambda x: preprocess(x))
    val_data['text'] = val_data['text'].apply(lambda x: preprocess(x))
    test_data['text'] = test_data['text'].apply(lambda x: preprocess(x))

    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    train_x, train_y = ros.fit_resample(np.array(train_data['text']).reshape(-1, 1), np.array(train_data['sentiment']).reshape(-1, 1))
    train = pd.DataFrame(list(zip([x[0] for x in train_x], train_y)), columns = ['text', 'sentiment'])

    from sklearn import preprocessing
    le = preprocessing.OneHotEncoder()
    y_train= le.fit_transform(np.array(train['sentiment']).reshape(-1, 1)).toarray()
    y_test= le.fit_transform(np.array(test_data['sentiment']).reshape(-1, 1)).toarray()
    y_val= le.fit_transform(np.array(val_data['sentiment']).reshape(-1, 1)).toarray()

    from transformers import RobertaTokenizerFast
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    def roberta_encode(data,maximum_length) :
      input_ids = []
      attention_masks = []


      for i in range(len(data.text)):
          encoded = tokenizer.encode_plus(

            data.text[i],
            add_special_tokens=True,
            max_length=maximum_length,
            pad_to_max_length=True,

            return_attention_mask=True,

          )

          input_ids.append(encoded['input_ids'])
          attention_masks.append(encoded['attention_mask'])
      return np.array(input_ids),np.array(attention_masks)

    max_len = max([len(x.split()) for x in train_data['text']])
    train_input_ids,train_attention_masks = roberta_encode(train, max_len)
    test_input_ids,test_attention_masks = roberta_encode(test_data, max_len)
    val_input_ids,val_attention_masks = roberta_encode(val_data, max_len)

    def create_model(bert_model, max_len):
        input_ids = tf.keras.Input(shape=(max_len,),dtype='int32')
        attention_masks = tf.keras.Input(shape=(max_len,),dtype='int32')

        output = bert_model([input_ids,attention_masks])
        output = output[1]

        output = tf.keras.layers.Dense(3, activation='softmax')(output)
        model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)
        model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    from transformers import TFRobertaModel
    roberta_model = TFRobertaModel.from_pretrained('roberta-base')
    model = create_model(roberta_model, max_len)
    model.summary()

    def roberta_inference_encode(data,maximum_length) :
        input_ids = []
        attention_masks = []



        encoded = tokenizer.encode_plus(
        data,
        add_special_tokens=True,
        max_length=maximum_length,
        pad_to_max_length=True,

        return_attention_mask=True

        )

        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        return np.array(input_ids),np.array(attention_masks)

    model = create_model(roberta_model, 43)
    model.load_weights(r'C:\Users\acer\Projects\Paper 1\Sentiment\my_checkpoint_senti')
    def inference(text_sentence, max_len):
        preprocessed_text = preprocess(text_sentence)
        input_ids, attention_masks = roberta_inference_encode(preprocessed_text, maximum_length = max_len)
        result = model.predict([input_ids, attention_masks])
        #le.categories_[0] = ['anger' 'fear' 'joy' 'love' 'sadness' 'surprise']
        result = pd.DataFrame(dict(zip(list(le.categories_[0]), [round(x*100, 2)for x in result[0]])).items(), columns = ['Category', 'Confidence'])
        #plot_result(result)
        return result

    # Sentiment of each clause

    neg = []
    pos = []
    neu = []
    def dataappend(result):
        neg.append(result['Confidence'][0])
        neu.append(result['Confidence'][1])
        pos.append(result['Confidence'][2])
    for text in utterance:
        result = inference(text, max_len)
        #print(result)
        dataappend(result)

    # Calculation of Position

    pt=pd.DataFrame(0, index=np.arange(len(utterance)), columns=['Position'])
    pt

    j = clausepred
    for i in range(0,len(utterance)):
        if i == j:
            pt['Position'][i] = 0
        elif i<j:
            pt['Position'][i] = i-j
        else:
            pt['Position'][i] = i-j
    pt

    position = pt['Position'].tolist()

    # Combining it all to make a df

    import pandas as pd

    df = pd.DataFrame()
    df = pd.DataFrame(columns = ['Clauses', 'Conjunction or Preposition','Length of the clasue', 'Word Embedding Similarity', 'Word/Synonym', 'Positive', 'Negative', 'Neutral','Position','Query','word'])
    for (i,j,k,l,m,a,b,c,d,e,f) in zip(utterance,wes,conjprep,wordpresent,length,pos,neg,neu,position,query_final,word):
        df = df.append({'Clauses' : i, 'Word Embedding Similarity':j, 'Word/Synonym':l, 'Conjunction or Preposition' :k, 'Length of the clasue' : m, 'Positive': a, 'Negative': b, 'Neutral':c, 'Position':d, 'Query':e, 'word':f}, ignore_index = True)

    df['Clauses'] = df['Clauses'].astype(str)
    df['Conjunction or Preposition'] = df['Conjunction or Preposition'].astype(int)
    df['Length of the clasue'] = df['Length of the clasue'].astype(int)
    df['Word Embedding Similarity'] = df['Word Embedding Similarity'].astype(float)
    df['Word/Synonym'] = df['Word/Synonym'].astype(int)
    df['Positive'] = df['Positive'].astype(float)
    df['Negative'] = df['Negative'].astype(float)
    df['Neutral'] = df['Neutral'].astype(float)
    df['Position'] = df['Position'].astype(int)
    df['Query'] = df['Query'].astype(str)
    df['word'] = df['word'].astype(str)
    df.dtypes

    df

    # Mix Text Generation

    import pandas as pd
    sen_w_feats = []
    for index, row in df.iterrows():
        combined = ""
        combined += "{:} shows {:} emotion. The similarity between the clause and {:} emotion is {:}. It has {:} ({:}) connector words in it. The clause has {:} level of positivity, {:} level of negativity and {:} level of neutrality. It is at {:} position with length {:}. It has {:} word that shows {:} emotion.".format(row["Clauses"], row["Query"], row["Query"],row["Word Embedding Similarity"],row["Conjunction or Preposition"], row["word"], row["Positive"], row["Negative"], row["Neutral"], row["Position"], row["Length of the clasue"],row["Word/Synonym"],row["Query"])
        sen_w_feats.append(combined)
    df['callfeats'] = sen_w_feats
    df

    # Feature Extraction

    import os
    import time
    import datetime
    import random
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.model_selection import train_test_split
    from transformers import TFAutoModel, AutoTokenizer, TFBertForSequenceClassification,AutoConfig
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.layers import Maximum,LayerNormalization,GlobalMaxPooling2D,Average,Dot, Dense, Input, GlobalAveragePooling1D, BatchNormalization, Activation, Concatenate, Flatten, Dropout, Conv1D, MaxPooling1D, Add, Lambda, GlobalAveragePooling2D, Reshape, RepeatVector, UpSampling1D
    from tensorflow.keras.models import Model
    from keras.layers import LSTM, Bidirectional
    from official import nlp
    import official.nlp.optimization
    data = df.Clauses.values
    import re
    for i in range(data.shape[0]):
        data[i] = re.sub(r'<url>','HTTPURL',data[i])
        data[i] = re.sub(r'<user>','@USER',data[i])

    from transformers import BertTokenizer
    # using the low level BERT for our task.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    MAX_LENGTH  = 128
    MAX_LENGTH_MIX = 256
    def TexttoEmb(data, tokenizer, MAX_LENGTH):
        input_ids       = []
        attention_masks = []

        for sent in tqdm(data):
            encoded_dict = tokenizer.encode_plus(
                                sent,
                                add_special_tokens = True,
                                max_length = MAX_LENGTH,
                                pad_to_max_length = True,
                                return_attention_mask = True,
                                return_tensors = 'np',
                                truncation = True,
                           )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        id_tvt       = np.concatenate(input_ids)
        mask_tvt      = np.concatenate(attention_masks)

        return id_tvt

    id_train = TexttoEmb(data, tokenizer, MAX_LENGTH)

    for i in range(0,df.shape[0]):
        df['Word Embedding Similarity'][i] = df['Word Embedding Similarity'][i]*1000
    df.head(5)

    for i in df.columns:
        #print(new_df[i].dtypes)
        if df[i].dtypes == 'float64':
            df[i] = df[i].astype(int)
    df

    def MixTexttoEmb(data, tokenizer, MAX_LENGTH):
        input_ids       = []
        attention_masks = []

        for sent in tqdm(data):
            encoded_dict = tokenizer.encode_plus(
                                sent,
                                add_special_tokens = True,
                                max_length = MAX_LENGTH,
                                pad_to_max_length = True,
                                return_attention_mask = True,
                                return_tensors = 'np',
                                truncation = True,
                           )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        id_tvt        = np.concatenate(input_ids)
        mask_tvt      = np.concatenate(attention_masks)

        return id_tvt

    mtrain_sent      = df.callfeats.values
    mid_train = MixTexttoEmb(mtrain_sent, tokenizer, MAX_LENGTH_MIX)

    def lofl(a):
        lst = []
        for i in range(0,len(a)):
            lst.append([])
            lst[i].append(a[i])
        return lst

    num_train_1 = lofl(df['Word Embedding Similarity'].to_numpy().tolist())
    num_train_2 = lofl(df['Conjunction or Preposition'].to_numpy().tolist())
    num_train_3 = lofl(df['Length of the clasue'].to_numpy().tolist())
    num_train_4 = lofl(df['Positive'].to_numpy().tolist())
    num_train_5 = lofl(df['Negative'].to_numpy().tolist())
    num_train_6 = lofl(df['Neutral'].to_numpy().tolist())
    num_train_7 = lofl(df['Position'].to_numpy().tolist())
    num_train_8 = lofl(df['Word/Synonym'].to_numpy().tolist())

    def NumtoEmb(data,tokenizer, MAX_LENGTH):
        num_input_ids       = []
        num_attention_masks = []
        for sent in tqdm(data):
            encoded_dict = tokenizer.encode_plus(
                                sent,
                                add_special_tokens = False,
                                max_length = MAX_LENGTH,
                                pad_to_max_length = True,
                                return_attention_mask = True,
                                return_tensors = 'np',
                                truncation = True,
                           )
            num_input_ids.append(encoded_dict['input_ids'])
            num_attention_masks.append(encoded_dict['attention_mask'])


        num_id_tvt        = np.concatenate(num_input_ids)
        num_mask_tvt      = np.concatenate(num_attention_masks)

        return num_id_tvt

    num_train_idwe= NumtoEmb(num_train_1,tokenizer, MAX_LENGTH)
    num_train_idc= NumtoEmb(num_train_2,tokenizer, MAX_LENGTH)
    num_train_idl= NumtoEmb(num_train_3,tokenizer, MAX_LENGTH)
    num_train_idp= NumtoEmb(num_train_4,tokenizer, MAX_LENGTH)
    num_train_idne= NumtoEmb(num_train_5,tokenizer, MAX_LENGTH)
    num_train_idnu= NumtoEmb(num_train_6,tokenizer, MAX_LENGTH)
    num_train_idpos= NumtoEmb(num_train_7,tokenizer, MAX_LENGTH)
    num_train_idsyn= NumtoEmb(num_train_8,tokenizer, MAX_LENGTH)

    MODEL       = 'bert-base-uncased'
    MODEL_NAME  = 'bert-base-uncased'
    N_LABELS    = 1
    transformer = TFAutoModel.from_pretrained(MODEL,output_attentions=False,output_hidden_states=True,return_dict =True)
    transformer.trainable = False

    def attfun(inpAttImg_query, inpAttImg_key):
        head = 1
        att_layers = 1
        att_hid = 32
        for layer in range(1):
          for _ in range(head):
            img_key = Dense(att_hid/head, use_bias=False)(inpAttImg_key) #change to tanh?
            text_query = Dense(att_hid/head, use_bias=False)(inpAttImg_query)
            img_value = Dense(att_hid/head, use_bias=False)(inpAttImg_key)

            attention = Dot(axes=1)([text_query, img_key])
            attention = Lambda(lambda x: x[0]/x[1])([attention,np.sqrt(att_hid/head)])
            attention = Activation("softmax")(attention)
            attention = Dense(att_hid/head, use_bias=False)(attention)
            head_att_img = Dot(axes=1)([attention, img_value])

        coatt = tf.keras.layers.Dense(32)(head_att_img)
        return coatt

    def create_model():
        input_ids = Input(shape=(128,), dtype=tf.float32, name='input_ids')
        num_input_ids_1 = Input(shape=(128,), dtype=tf.float32, name='num_input_ids_1')
        num_input_ids_2 = Input(shape=(128,), dtype=tf.float32, name='num_input_ids_2')
        num_input_ids_3 = Input(shape=(128,), dtype=tf.float32, name='num_input_ids_3')
        num_input_ids_4 = Input(shape=(128,), dtype=tf.float32, name='num_input_ids_4')
        num_input_ids_5 = Input(shape=(128,), dtype=tf.float32, name='num_input_ids_5')
        num_input_ids_6 = Input(shape=(128,), dtype=tf.float32, name='num_input_ids_6')
        num_input_ids_7 = Input(shape=(128,), dtype=tf.float32, name='num_input_ids_7')
        num_input_ids_8 = Input(shape=(128,), dtype=tf.float32, name='num_input_ids_8')
        minput_ids = Input(shape=(256,), dtype=tf.float32, name='mixinput_ids')

        head = 1
        att_layers = 1
        att_hid = 32
        final_hid = 32
        coatt1 = attfun(input_ids, num_input_ids_1)
        coatt2 = attfun(input_ids, num_input_ids_2)
        coatt3 = attfun(input_ids, num_input_ids_3)
        coatt4 = attfun(input_ids, num_input_ids_4)
        coatt5 = attfun(input_ids, num_input_ids_5)
        coatt6 = attfun(input_ids, num_input_ids_6)
        coatt7 = attfun(input_ids, num_input_ids_7)
        coatt8 = attfun(input_ids, num_input_ids_8)

        att = []
        att.append(coatt1)
        att.append(coatt2)
        att.append(coatt3)
        att.append(coatt4)
        att.append(coatt5)
        att.append(coatt6)
        att.append(coatt7)
        att.append(coatt8)
        finalatt = Concatenate(axis=1)(att)
        finalatt = tf.keras.layers.Dense(32)(finalatt)

        mixnew = attfun(minput_ids, minput_ids)
        # mixnew
        x0 = tf.keras.layers.Dense(32)(input_ids)
        x1 = tf.keras.layers.Dense(32)(num_input_ids_1)
        x2 = tf.keras.layers.Dense(32)(num_input_ids_2)
        x3 = tf.keras.layers.Dense(32)(num_input_ids_3)
        x4 = tf.keras.layers.Dense(32)(num_input_ids_4)
        x5 = tf.keras.layers.Dense(32)(num_input_ids_5)
        x6 = tf.keras.layers.Dense(32)(num_input_ids_6)
        x7 = tf.keras.layers.Dense(32)(num_input_ids_7)
        x8 = tf.keras.layers.Dense(32)(num_input_ids_8)

        merge = []
        merge.append(x0)
        merge.append(finalatt)
        merge.append(x1)
        merge.append(x2)
        merge.append(x3)
        merge.append(x4)
        merge.append(x5)
        merge.append(x6)
        merge.append(x7)
        merge.append(x8)
        merge.append(mixnew)
        # merge

        final = Concatenate(axis=1)(merge)
        # final

        drop_rate = 0.2
        N_LABELS = 1
        l_merge = Dropout(drop_rate)(final)
        l_merge = Dense(final_hid)(l_merge)
        l_merge = BatchNormalization()(l_merge)
        l_merge = Activation('relu')(l_merge)
        l_merge = Dropout(drop_rate)(l_merge)
        out = Dense(N_LABELS, activation='sigmoid')(l_merge)
        out
        model = Model(inputs=[input_ids, num_input_ids_1, num_input_ids_2, num_input_ids_3, num_input_ids_4, num_input_ids_5, num_input_ids_6, num_input_ids_7, num_input_ids_8, minput_ids], outputs=out)
    #num_input_ids_1, num_input_ids_2, num_input_ids_3, num_input_ids_4, num_input_ids_5, num_input_ids_6, num_input_ids_7, num_input_ids_8
        return model

    model = create_model()
    model.load_weights(r'C:\Users\acer\Projects\checkpoint\weights\my_checkpoint')
    newpred = [id_train, num_train_idwe, num_train_idc, num_train_idl, num_train_idp, num_train_idne, num_train_idnu, num_train_idpos, num_train_idsyn, mid_train]
    predictions = model.predict(newpred, verbose=1)
    from iteration_utilities import deepflatten
    predictions = list(deepflatten(predictions, depth=1))

    theta = 0.5
    causes = []
    for i in range(0,len(utterance)):
        if predictions[i]>theta:
            #print("Cause")
            causes.append(i)
        else:
            print('Not')

    print(causes)
    return causes

clauses, conf_final, overallquery, query_final, wes, wordpresent = clauseprediction(utterance)
causelist = []
for i in clauses:
    causes = causeprediction(utterance, clauses, conf_final, overallquery, query_final, wes, wordpresent, i)
    causelist.append(causes)

pairs = []

for i in range(0,len(clauses)):
    takenclause = clauses[i]
    couldbecauses = causelist[i]
    for j in couldbecauses:
        pairs_1 = []
        if j<=takenclause:
#             print(takenclause,j)
            pairs_1.append(takenclause)
            pairs_1.append(j)
            print(pairs_1)
        if not pairs_1:
            continue
        else:
            pairs.append(pairs_1)

def construct_conv(row, tokenizer, eos = True):
    flatten = lambda l: [item for sublist in l for item in sublist]
    # conv = list(reversed([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row]))
    # row = convert_to_representation(row)
    conv = list([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row])
    # print(conv)
    conv = flatten(conv)
    return conv

class ConversationDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, df, block_size=512):

        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)

        directory = args.cache_dir
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size)
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            # df = convert_to_representation(df)
            # print(df['convo'][10])
            for _, row in df.iterrows():
                # print(row)
                conv = construct_conv(row, tokenizer)
                # print(conv)
                self.examples.append(conv)

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

  def load_and_cache_examples(args, tokenizer, df_trn, df_val, evaluate=False):
    return ConversationDataset(tokenizer, args, df_val if evaluate else df_trn)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate, drop_last = True
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model.resize_token_embeddings(len(tokenizer))
    # add_special_tokens_(model, tokenizer)


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs, labels = (batch, batch)
            if inputs.shape[1] > 1024: continue
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

# Evaluation of some model

def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, df_trn, df_val, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, df_trn, df_val, evaluate=True)
    os.makedirs(eval_output_dir, exist_ok=True)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate, drop_last = True
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result

def main(df_trn, df_val):
    args = Args()

    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
        and not args.should_continue
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    model = AutoModelWithLMHead.from_pretrained(
        args.model_name_or_path,
        from_tf=False,
        config=config,
        cache_dir=args.cache_dir,
    )
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, df_trn, df_val, evaluate=False)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelWithLMHead.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = AutoModelWithLMHead.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, df_trn, df_val, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results

main(trn_df, val_df)
