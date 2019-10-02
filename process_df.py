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


data_raw = pd.read_csv('data/raw_data.csv', index_col=0)

# create fraud column as target
target = pd.DataFrame()
target["fraud"] = data_raw["account"].str.contains("fraud")

def drop_leaky_cols(df):
    # eliminate columns that won't appear in test data
    df.drop(['account', 'payout_date', 'xt', 'order_count', 'payout_count', 'sale_duration2'], axis=1, inplace=True)


def make_cat_cols(df):
    ### 'listed' column : replace 'y' with 1, replace 'n' with 0
    df.listed.replace({"y":1, "n": 0}, inplace=True)
    #df.listed = df.listed.astype('category')

    ### 'payout_type' column : "ACH" = 0, "CHECK" = 1, "" = 2
    df.payout_method.fillna('x', inplace=True)
    df.payout_method.replace({"ACH": 0, "CHECK": 1, "x": 2}, inplace=True)
    #df.payout_type = df.payout_type.astype('category', inplace=True)

    ### 'currency' column : [AUD, CAD, EUR, GBP, MXN, NZD, USD]
    currency_list = ['AUD', 'CAD', 'EUR', 'GBP', 'MXN', 'NZD', 'USD']
    currency_codes = {k:v for v, k in enumerate(currency_list)}
    df.currency.replace({k:v for v, k in enumerate(currency_list)}, inplace=True)

    ### 'country' column : 
    country_dict = {k:v for v, k in enumerate(list(df.country.unique()))}
    df.country.replace(country_dict, inplace=True)
    df.country.fillna(71, inplace=True)

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
    cleaned = re.sub(r'[?|!|\'|#]', r'', cleaned)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    return cleaned

def scrub_desc(df):
    stop = stopwords.words('english')

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

def stem_tokens(df):
    stop = stopwords.words('english')
    sno = SnowballStemmer('english')
    token_col = []
    for sentence in df['clean_desc'].fillna('').values:
        # print(sentence)
        # print(type(sentence))
        tokens = []
        for word in sentence.split(' '):
            #print(word)
            if ((word.isalpha()) and (len(word) > 2) and (word not in stop)):
                s = sno.stem(word)
                tokens.append(s)
            else:
                continue
        token_col.append(tokens)
    df['stem_tokens'] = token_col


def to_lemma(df):
    # takes text, returns lemmatized words
    # remove nonsense
    stop = stopwords.words('english')
    lem_list = []
    lem_sent_col = []
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
        lem_list.append(lemmed_words)
        lem_sent_col.append(lemmed_sentence)
    df['lem_tokens'] = lem_list 
    df['lem_text'] = lem_sent_col
    

'''def lem_text(df):
    nrows = len(df)
    lemmatized_text_list = []
    wordnet_lemmatizer = WordNetLemmatizer()

    for row in range(0, nrows):
        
        # Create an empty list containing lemmatized words
        lemmatized_list = []
        
        # Save the text and its words into an object
        text = df.loc[row]['clean_desc']
        text_words = text.split(" ")

        # Iterate through every word to lemmatize
        for word in text_words:
            lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
            
        # Join the list
        lemmatized_text = " ".join(lemmatized_list)
        
        # Append to the list containing the texts
        lemmatized_text_list.append(lemmatized_text)
    df['lem_text'] = lemmatized_text_list'''

### FILL NA : FB, Twitter, Delivery_Method, Sale_Duration
def fill_na_cols(df):
    sd = df["sale_duration"].mean()
    fb = df.org_facebook.mean()
    tw = df.org_twitter.mean()
    
    fill_values = {'org_facebook': fb, 
                'org_twitter': tw, 
                'sale_duration':sd, 
                'delivery_method':86, 
                'average_ticket_price':0}

    df.fillna(value=fill_values, inplace=True)
    df[['org_facebook', 'org_twitter', 'sale_duration']].astype(int, inplace=True)


def ticket_types(df): 
    import ast
    
    tt_col = df['ticket_groups']
    num_tiers = []
    num_tickets = []
    average_price = []
    
    for row in tt_col:
        row = ast.literal_eval(row)
        num_tiers.append(len(row))
        quant = []
        price = []
        for dct in row:
            quant.append(dct['quantity_total'])
            price.append(dct['cost'])
        arr = np.array([quant, price])
        average_price.append(np.round((arr[0] * arr[1]).sum()/arr[1].sum(), 2))
        num_tickets.append(int(arr[1].sum()))
        
    df["num_tiers"] = num_tiers
    df["tickets_available"] = num_tickets
    df["average_ticket_price"] = average_price 
    
def previous_payouts(df):
    import ast
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
    df["avg_previous_payouts"] = avg_payouts
    df["num_previous_payouts"] = num_payouts

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
    
    ticket_types(data)
    fill_na_cols(data)
#    scrub_desc(data)
#    stem_tokens(data)
#    to_lemma(data)
    make_cat_cols(data)
    add_new_columns(data)
    previous_payouts(data)

    data.to_csv('data/clean_data.csv')

'''
Columns used in original model:

[["body_length", "channels","delivery_method","fb_published",
"has_logo", "listed", "name_length",  "object_id", "org_facebook",
"org_twitter", "payout_type", "sale_duration", "show_map", "user_created",
"num_tiers", "tickets_available", "average_ticket_price", "user_age"]]

'''