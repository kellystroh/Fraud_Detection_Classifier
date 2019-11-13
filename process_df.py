import pandas as pd 
from collections import Counter
import numpy as np

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords, wordnet
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

import re
import ast


data_raw = pd.read_csv('data/raw_data.csv', index_col=0)

# create fraud column as target
target = pd.DataFrame()
target["fraud"] = data_raw["account"].str.contains("fraud")

def drop_leaky_cols(df):
    # eliminate columns that won't appear in test data
    df.drop(['account', 'payout_date', 'xt', 'order_count', 'payout_count', 'sale_duration2'], axis=1, inplace=True)

def clean_html(sentence):
    cleanr = re.compile('<.*?>')
    text=re.sub("rsquo","",sentence)
    text=re.sub("nbsp","",text)
    text=re.sub("ndash","",text)
    text=re.sub("\r"," ",text)
    text=re.sub("\n"," ",text)
    text=re.sub("amp","",text)
    cleantext = re.sub(cleanr, ' ', text)
    return cleantext

def clean_urls(sentence):
    cleantext = re.sub(r"www\..*?($|\s)", '', sentence)
    cleantext = re.sub(r"http://.*?($|\s)", '', cleantext)
    return cleantext

def clean_emails(sentence):
    cleantext = re.sub(r"(?<!\S)\w*@.*?\.com(?!\S)", '', sentence)
    return cleantext

def clean_punc(word):
    cleaned = re.sub(r"'s", r'', word)
    cleaned = re.sub(r'[?|!|\'|#-]', r'', cleaned)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    return cleaned

def scrub_desc(df):
    #stop = stopwords.words('english')

    sentence_list = []
    
    for sentence in df['description'].fillna('').values:
        filtered_words = []
        sentence = clean_html(sentence)
        sentence = clean_urls(sentence)
        sentence = clean_emails(sentence)
        sentence = clean_punc(sentence)
        sentence = sentence.lower()
        for word in sentence.split():
            for cleaned_word in clean_punc(word).split():
                    filtered_words.append(cleaned_word)

        strl = ' '.join(filtered_words)
        sentence_list.append(strl)

    df['clean_desc'] = sentence_list
    df.clean_desc.fillna('', inplace = True)

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def word_count_fields(df):
    word_count_name = []
    for row in df['name'].fillna('').values:
        row = clean_punc(row)
        row = row.split(' ')
        word_count = len(row)
        word_count_name.append(word_count)
    df['word_count_name'] = word_count_name

    word_count_org = []
    for row in df['org_name'].fillna('').values:
        row = clean_punc(row)
        row = row.split(' ')
        word_count = len(row)
        word_count_org.append(word_count)
    df['word_count_org'] = word_count_org

def to_lemma(df):
    # takes text, returns lemmatized words
    # remove nonsense
    stop = stopwords.words('english')
    lem_tokens = []
    lem_docs = []
    for text in df['clean_desc'].fillna('').values:
        if type(text) != str:
            text = str(text)
        tokens = word_tokenize(text)
        #print(tokens)
        filtered_tokens = []
        for w in tokens:
            if w.isalpha() and w not in stop and len(w) > 2:
                filtered_tokens.append(w)
        # for lemmatization, need to pass part of speech
        pos = nltk.pos_tag(filtered_tokens)
        # lemmatization with pos 
        lem = WordNetLemmatizer()
        lemmed_words = []
        lemmed_sentence = ''
        for w,p in pos:
            p_new = get_wordnet_pos(p)
            lemmed_words.append(lem.lemmatize(w,p_new))
            lemmed_sentence += (w + ' ')
        lem_tokens.append(lemmed_words)
        lem_docs.append(lemmed_sentence)
    df['lem_tokens'] = lem_tokens 
    df['lem_text'] = lem_docs

### FILL NA : FB, Twitter, Delivery_Method, Sale_Duration
def fill_na_cols(df):
    sd = df["sale_duration"].mean()
    fb = df.org_facebook.mean()
    tw = df.org_twitter.mean()
    
    fill_values = {'org_facebook': fb, 
                'org_twitter': tw, 
                'sale_duration':sd, 
                'delivery_method':86, 
                'average_ticket_price':0,
                'country': 'X',
                'payout_method': 'X'}

    df.fillna(value=fill_values, inplace=True)
    df[['org_facebook', 'org_twitter', 'sale_duration']].astype(int, inplace=True)


def ticket_groups(df): 
    
    tg_col = df['ticket_groups']
    num_tiers = []
    num_tickets = []
    average_price = []
    min_price = []
    max_price = []
    sum_ticket = []
    
    for row in tg_col:
        row = ast.literal_eval(row)
        num_tiers.append(len(row))
        quant = []
        price = []
        for dct in row:
            quant.append(dct['quantity_total'])
            price.append(dct['cost'])
        arr = np.array([quant, price])
        if arr[0].sum() != 0:
            average_price.append(np.round((arr[0] * arr[1]).sum()/arr[0].sum(), 2))
            num_tickets.append(int(arr[0].sum()))
            min_price.append(arr[1].min())
            max_price.append(arr[1].max())
            sum_ticket.append((arr[0] * arr[1]).sum())
        else: 
            average_price.append(0)
            num_tickets.append(0)
            min_price.append(0)
            max_price.append(0)
            sum_ticket.append(0)
    df["num_tiers"] = num_tiers
    df["tickets_available"] = num_tickets
    df["average_ticket_price"] = average_price
    df['min_price'] = min_price
    df['max_price'] = max_price
    df['potential_revenue'] = sum_ticket
    
def past_payouts(df):
    
    pp_col_value = df.past_payouts
    avg_payouts = []
    num_payouts = []
    
    for row in pp_col_value:
        row = ast.literal_eval(row)
        num_payouts.append(len(row))

        if len(row) > 0:
            amount = 0
            for dct in row:
                
                amount += dct['amount']
            avg_payouts.append(np.round(amount/len(row), 2))

        else:
            avg_payouts.append(0)
    df["avg_past_payouts"] = avg_payouts
    df["num_past_payouts"] = num_payouts

def add_new_columns(df):
    # make column for whether 'description' includes a url 
    df['desc_has_link'] = df.description.str.contains('href')
    df.desc_has_link.fillna(False, inplace=True)

    # make column for whether 'org_desc' includes a url 
    df['org_has_link'] = df.org_desc.str.contains('href')
    df.org_has_link.fillna(False, inplace=True)

    # make column for email suffix code
    df['email_suffix'] = df.email_domain.str.strip().str.lower().str.extract(r'([^.]+$)')
    top9 = list(data.email_suffix.value_counts().index[:9])
    email_suffix_list = list(df.email_suffix.value_counts().index)
    email_suffix_dict = {}
    for idx in range(len(email_suffix_list)):
        if email_suffix_list[idx] in top9:
            email_suffix_dict[email_suffix_list[idx]] = idx
        else:
            email_suffix_dict[email_suffix_list[idx]] = 9
    df['email_suffix_code'] = df.copy().email_suffix.apply(lambda x: email_suffix_dict[x])
    
    # make country_match_code
    df['country_match'] = df.country == df.venue_country

    # make length of text columns
    ### Create column for the length of entry in the organization description (org_desc) field. 
    df['org_desc_len'] = df.org_desc.fillna('').str.len()

    ### Create column for the length of entry in the venue_name field. 
    df['venue_name_len'] = df['venue_name'].fillna('').str.len()

    ### Create column for the length of entry in the org_name field. 
    df['org_name_len'] = df['org_name'].fillna('').str.len()


if __name__ == '__main__':
    
    target.to_csv('data/target.csv')
    drop_leaky_cols(data_raw)
    data = data_raw.copy()

    ticket_groups(data)
    fill_na_cols(data)
    
    scrub_desc(data)
    to_lemma(data)
    
    add_new_columns(data)
    past_payouts(data)
    
    data.drop(['description', 'email_domain', 'date_created', 'end_date', 'start_date', 'date_published',
           'org_desc', 'name','org_name', 'past_payouts', 'ticket_groups', 'venue_address',
           'venue_country', 'venue_latitude', 'venue_longitude','venue_name', 'venue_state',
           'email_suffix', 'object_id', 'payee_name', ], axis=1, inplace=True)
    
    data.to_csv('data/clean_data.csv')