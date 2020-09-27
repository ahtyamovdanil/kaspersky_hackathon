import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('/rapids/notebooks/my_data/BMSTU_hack/')

import torch
import biGRU_model
from deeppavlov.core.data.simple_vocab import SimpleVocabulary
import numpy as np

gru = torch.load('/rapids/notebooks/my_data/BMSTU_hack/models/biGRU')
device = torch.device('cpu')
vocab = SimpleVocabulary(save_path="/rapids/notebooks/my_data/BMSTU_hack/models/vocab.dict")
gru = biGRU_model.BiGRU(vocab.count, embedding_dim=10, hidden_size=50, device='cpu') 
gru.load_state_dict(torch.load('/rapids/notebooks/my_data/BMSTU_hack/models/biGRU', map_location=device))


from tbot import config
import telebot

bot = telebot.TeleBot(config.token)

@bot.message_handler(content_types=["text"])
def repeat_all_messages(message): 
    label = biGRU_model.predict_proba(gru, vocab, np.array([[message.text]]), device='cpu')
    label = '\U0000274C' if label > 0.5 else '\U00002705'
    bot.send_message(message.chat.id, label)

if __name__ == '__main__':
     bot.polling(none_stop=True)