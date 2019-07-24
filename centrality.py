import string
import networkx as nx
import pandas as pd
import sys
import spacy
import textacy.vsm
from textacy.vsm import Vectorizer
import time
import os
import re
import csv
import unicodedata
import numpy as np
from io import StringIO
from datetime import datetime
from random import uniform


# Happy Emoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
 
# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
 
# all emoticons (happy + sad)
emoticons = emoticons_happy.union(emoticons_sad)

data = pd.read_csv('user.csv', header=None)
temp_user = str(data[1][4])
cont = 0
cont_e = 0
reply = 0
rt = 0
lk = 0
quote = 0
na = 0
param = 50
param_two = 100
ctt = True
 
def clean_tweets(tweet):
    # remove stock market tickers like $GE
    tweet = tweet[2:-1]
    tweet = re.sub(r'\$\w*', '', tweet)
 
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
 
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'@', '', tweet)
    tweet = re.sub(r'\\x..', '', tweet)
    tweet = re.sub(r'\\n', ' ', tweet)
    tweet = tweet.lower()
    


    

    # tweet = re.sub(r'\\x\s\S', '', tweet)
    # tweet = re.sub(r'\\x\d\D', '', tweet)
    # tweet = re.sub(r'\\x\S\s', '', tweet)
    # tweet = re.sub(r'\\x\D\d', '', tweet)
    # tweet = re.sub(r'\\x\d\d', '', tweet)
    # tweet = re.sub(r'\\x\s\s', '', tweet)


 
    return tweet

def calculate_centrality_followee(data):
	global cont, cont_e, ctt, reply, quote, rt, like, param, na, param_two
	for i, r in data.iterrows():
		if (na > param):
			print("Completed")
			break
		else:
			duration = 1000000000000000000
			s = ''
			min_interval = 10000000000000
			dfb = next(iter(data_one[data_one['origin_id']==r['id_str']].index), 'no match')
			if dfb == "no match":
				na = na + 1	
				print ('No Action # %d' % na)
				loc_ind = 0
				for ia, ra in data_one.iterrows():
					duration_raw = ra['created_at'] - r['created_at']
					# print (ra['created_at'])
					# print (r['created_at'])
					# print (duration_raw)
					duration = duration_raw.total_seconds()
					if duration > 0:
						if duration < min_interval:
							min_interval = duration
							loc_ind = ia
				if min_interval == 10000000000000:
					part_data = data_one
				else:
					if (loc_ind < param_two):
						part_data = data_one.iloc[0:loc_ind-1]
					else:
						part_data = data_one.iloc[loc_ind-(param_two-1):loc_ind-1]
				if r['status'] == 'like':
					s = r['origin_full_text']
				else:
					s = r['full_text']
			else:
				part_data = pd.DataFrame()
				s = 'null'


			for index, row in part_data.iterrows():
				if not row['full_text'] == 'null':
					docs = clean_tweets(row['full_text'])
					doc = nlp(docs)
					corpus.add_doc(doc)
					token_list = []
					for token in doc:		
						if any(c.isalpha() for c in str(token)) and not token.is_stop:
							token_list.append(token.lemma_)
							if str(token.lemma_) not in G:		
								G.add_node(str(token.lemma_), decay = 0, checked = 'n')
					for i_two in range(len(token_list)):
						count = i_two + 1
						while count < len(token_list):
							if G.has_edge(token_list[i_two], token_list[count]):
								G[token_list[i_two]][token_list[count]]['weight'] += 1
							else: 
								G.add_edge(token_list[i_two],token_list[count], weight=1)
							count = count + 1
				else :
					pass

			if (s == 'null'):
				pass
			else:
				docs_two = clean_tweets(s)
				doc_two = nlp(docs_two)
				corpus.add_doc(doc_two)

				token_list = []
				for token in doc_two:		
					if any(c.isalpha() for c in str(token)) and not token.is_stop:
						token_list.append(token.lemma_)
						if str(token.lemma_) not in G:
							G.add_node(str(token.lemma_),decay = 1,checked = 'y')
							#G.add_node(str(token.lemma_.lower()), weight = 1)
						else:
							attrs = {str(token.lemma_): {'decay': 1, 'checked':'y'}}
							nx.set_node_attributes(G, attrs)
					# for i_two in range(len(token_list)):
					# 	count = i_two + 1
					# 	while count < len(token_list):
					# 		if G.has_edge(token_list[i_two], token_list[count]):
					# 			G[token_list[i_two]][token_list[count]]['weight'] += 1
					# 		else: 
					# 			G.add_edge(token_list[i_two],token_list[count], weight=1)
					# 		count = count + 1
				for token in doc_two:
					if any(c.isalpha() for c in str(token)) and not token.is_stop:
						# print (G[str(token.lemma_)])	
						for n in G.neighbors(str(token.lemma_)):
							if G.node[n]['checked'] == 'n':
								G.node[n]['checked'] = 'y'
								# print (G.node[n]['checked'])
								G.node[n]['decay'] = G.node[str(token.lemma_)]['decay']*0.5
								# print (G.node[n]['decay'])

				for i_two in range(len(token_list)):
					count = i_two + 1
					while count < len(token_list):
						if G.has_edge(token_list[i_two], token_list[count]):
							G[token_list[i_two]][token_list[count]]['weight'] += 1
						else: 
							G.add_edge(token_list[i_two],token_list[count], weight=1)
						count = count + 1
			
				vectorizer = textacy.vsm.Vectorizer(apply_idf=True, norm=None, idf_type='standard')
				tokenized_docs = [doc._.to_terms_list(ngrams=1, named_entities=True, as_strings=True, filter_stops=True, normalize='lemma') for doc in corpus]
				doc_term_matrix = vectorizer.fit_transform(tokenized_docs)
				a = doc_term_matrix.toarray()
				

				start_time = time.time()
				try:
					centrality1 = nx.degree_centrality(G)
				except:
					centrality1 = 0
					print ("error")
				print("Calculating degree_centrality")
				t1 = time.time()
				elapsed_time1 = t1 - start_time
				print ("Time elapsed: %s seconds" % (elapsed_time1) )

				dfbase = pd.DataFrame(list(centrality1.items()), columns = ['node', 'degree_centrality'])
				#centrality_list = ['eigenvector_centrality']
				centrality_list = ['eigenvector_centrality','closeness_centrality','betweenness_centrality','load_centrality','subgraph_centrality','harmonic_centrality']
			



				temp_time = time.time() 
				for ct in centrality_list:
					try:
						print ("Calculating %s" % ct)
						centrality = getattr(nx, ct)(G)
						print ("Time elapsed: %s seconds" % (time.time() - temp_time) )
						temp_time = time.time()
						dfbase[ct]=dfbase['node'].map(centrality)
					except Exception as e:
						print ("error")
						dfbase[ct]=0
				#for token in doc_two:
					#print(token.text)
				token_list_two = pd.DataFrame()
				# print (dfbase['node'])
				# print (dfbase['degree_centrality'])
				for sent in doc_two.sents:
					for token in sent:
						# print (token)
						if any(c.isalpha() for c in str(token)) and not token.is_stop:
							rr = dfbase.loc[dfbase['node'] == token.lemma_]

							# print (rr)

							if not rr.empty :
								# print (rr)
								token_list_two = token_list_two.append(rr)


				if not token_list_two.empty:

					centrality_list_full = ['degree_centrality','eigenvector_centrality','closeness_centrality','betweenness_centrality','load_centrality','subgraph_centrality','harmonic_centrality']
				

					for itm in centrality_list_full:
						kwargs = {('weighted_%s' % itm) : lambda x: 0.0}
						token_list_two = token_list_two.assign(**kwargs)

					
					for i, row in token_list_two.iterrows():
						vb = a.item(((int(a.size/a[0].size))-1),(vectorizer.vocabulary_terms.get(row['node'])))
						for it in centrality_list_full:
							# print (vec.vocabulary_terms)
							# print (row['node'])
							va = row[it]
							# print ('vb')
							# print (vb)
							c = va*vb
							# print ('c')
							# print (c)
							#print(va)
							#print (vb)
							token_list_two.set_value(i, ('weighted_%s' % it), c)
							

							

							#print (('weighted_%s' % it))
							#print (row[('weighted_%s' % it)])
						
					#print ("token_list")
					#print (token_list)
					# print (token_list_two['weighted_degree_centrality'])

					tk_sum = pd.DataFrame()
					if r['status'] == 'like':
						tk_sum = tk_sum.assign(node = [r['origin_id']])
					else:
						tk_sum = tk_sum.assign(node = [r['id_str']])




					
					tk_sum = tk_sum.assign(screen_name = [r['screen_name']])
					tk_sum = tk_sum.assign(created_at = [r['created_at']])
					tk_sum = tk_sum.assign(full_text = [r['full_text']])
					tk_sum = tk_sum.assign(truncated = [r['truncated']])
					tk_sum = tk_sum.assign(user_description = [r['len(user.description)']])
					tk_sum = tk_sum.assign(user_followers_count = [r['user.followers_count']])
					tk_sum = tk_sum.assign(user_friends_count= [r['user.friends_count']])
					tk_sum = tk_sum.assign(user_listed_count = [r['user.listed_count']])
					tk_sum = tk_sum.assign(user_statuses_count = [r['user.statuses_count']])
					tk_sum = tk_sum.assign(user_favourites_count = [r['user.favourites_count']])
					tk_sum = tk_sum.assign(user_created_at = [r['user.created_at']])
					tk_sum = tk_sum.assign(entities_hashtags = [r['entities.hashtags']])
					tk_sum = tk_sum.assign(entities_user_mentions = [r['entities.user_mentions']])
					tk_sum = tk_sum.assign(entities_symbols = [r['entities.symbols']])
					tk_sum = tk_sum.assign(entities_polls = [r['bool(entities.polls)']])
					tk_sum = tk_sum.assign(entities_media_type = [r['entities.media_type']])
					tk_sum = tk_sum.assign(retweet_coun = [r['retweet_count']])
					tk_sum = tk_sum.assign(favorite_count = [r['favorite_count']])
					tk_sum = tk_sum.assign(origin_id = [r['origin_id']])
					tk_sum = tk_sum.assign(origin_created_at = [r['origin_created_at']])
					tk_sum = tk_sum.assign(origin_full_text = [r['origin_full_text']])
					tk_sum = tk_sum.assign(origin_user = [r['origin_user']])
					tk_sum = tk_sum.assign(origin_user_len_description = [r['origin_user_len_description']])
					tk_sum = tk_sum.assign(origin_user_followers_count = [r['origin_user_followers_count']])
					tk_sum = tk_sum.assign(origin_user_friends_count = [r['origin_user_friends_count']])
					tk_sum = tk_sum.assign(origin_user_listed_count = [r['origin_user_listed_count']])
					tk_sum = tk_sum.assign(origin_user_statuses_count = [r['origin_user_statuses_count']])
					tk_sum = tk_sum.assign(origin_user_favourites_count = [r['origin_user_favourites_count']])
					tk_sum = tk_sum.assign(origin_user_created_at = [r['origin_user_created_at']])
					tk_sum = tk_sum.assign(origin_entities_hashtags = [r['origin_entities_hashtags']])
					tk_sum = tk_sum.assign(origin_entities_user_mentions = [r['origin_entities_user_mentions']])
					tk_sum = tk_sum.assign(origin_entities_symbols = [r['origin_entities_symbols']])
					tk_sum = tk_sum.assign(origin_entities_polls = [r['origin_entities_polls']])
					tk_sum = tk_sum.assign(origin_entities_media_type = [r['origin_entities_media_type']])
					tk_sum = tk_sum.assign(origin_retweet_count = [r['origin_retweet_count']])
					tk_sum = tk_sum.assign(origin_favorite_count = [r['origin_favorite_count']])
					tk_sum = tk_sum.assign(origin_status = [r['origin_status']])


					try:
						tk_sum = tk_sum.assign(min_interval = duration)
					except:
						tk_sum = tk_sum.assign(min_interval = 100000000000000000)

					for itm in centrality_list_full:
						# print (token_list[('weighted_%s' % itm)])
						kwargs = {itm : lambda x: [token_list_two[('%s' % itm)].sum()]}
						try:
							tk_sum = tk_sum.assign(**kwargs)
						except:
							kwargs = {itm : lambda x: 0.0}
							tk_sum = tk_sum.assign(**kwargs)

					for itm in centrality_list_full:
						# print (token_list[('weighted_%s' % itm)])
						# print (token_list_two[('weighted_%s' % itm)].sum())
						kwargs = {('weighted_%s' % itm) : lambda x: [token_list_two[('weighted_%s' % itm)].sum()]}
						try:
							tk_sum = tk_sum.assign(**kwargs)
						except:
							kwargs = {('weighted_%s' % itm)  : lambda x: 0.0}
							tk_sum = tk_sum.assign(**kwargs)

					

					
					try:
						tk_sum = tk_sum.assign(status = ['No action'])
					except:
						tk_sum = tk_sum.assign(status = ['unknown'])
					# if ind == 1:
					# 	try:
					# 		tk_sum = tk_sum.assign(status = ['no action'])
					# 	except:
					# 		tk_sum = tk_sum.assign(status = ['unknown'])
					#print (tk_sum)
					# .reset_index()
					# token_list.sort_values(by=list(token_list.columns),axis=0)
					# token_list.reset_index()
					#tk = token_list.iloc[[0]]
					# tk = tk.assign(status = r['status'])
					cont = cont + 1
					print ("%d interactions collected" % cont)


					with open(os.path.join(tweetnetworks,'%s_interaction.csv' % temp_user), 'a') as f:
						writer = csv.writer(f)
						#tk.to_csv(f, header=False,index=False)
						tk_sum.to_csv(f, header=False,index=False)
				else:
					cont_e = cont_e + 1
					print ("%dth interaction with zero centralities" % cont_e)
				part_data = pd.DataFrame()















def calculate_centrality_original(data):
	global cont, cont_e, ctt, reply, quote, rt, lk, param, na, param_two
	for i, r in data.iterrows():
		if i == 0:
			pass
		else:
			if r['status'] == "reply":
				if reply > param:
					ctt = False
				else:
					print ('reply # %d' % reply)
				reply = reply + 1		
			elif r['status'] == "quote":
				if quote > param:
					ctt = False
				else:
					print ('quote # %d' % quote)
				quote = quote + 1
			elif r['status'] == "retweet":
				if rt > param:
					ctt = False
				else:
					print ('retweet # %d' % rt)
				rt = rt + 1
			elif r['status'] == "like":
				if lk > param:
					ctt = False
				else:
					print ('like # %d' % lk)
				lk = lk + 1	
			elif (r['status'] == 'original'):
				ctt = False
			else:
				ctt = True
			# if (lk > param) and (rt > param) and (quote > param) and (reply > param) and (na > param):
			# 	print("Completed")
			# 	break
			if (not ctt):
				pass
				ctt = True
			else:
				print (i)
				duration = 1000000000000000000
				s = ''
				if not r['status'] == 'original' and not r['origin_full_text'] == 'null':
					if (i < param_two):
						part_data = data_one.iloc[0:i-1]
					else:
						part_data = data_one.iloc[i-(param_two-1):i-1]
					duration_raw = r['created_at'] - datetime.strptime(r['origin_created_at'], '%Y-%m-%d %H:%M:%S') 
					duration = duration_raw.total_seconds()
					if duration == 0:
						duration = uniform(0, 10800)
				s = r['origin_full_text']


				for index, row in part_data.iterrows():
					if not row['full_text'] == 'null':
						docs = clean_tweets(row['full_text'])
						doc = nlp(docs)
						corpus.add_doc(doc)
						token_list = []
						for token in doc:		
							if any(c.isalpha() for c in str(token)) and not token.is_stop:
								token_list.append(token.lemma_)
								if str(token.lemma_) not in G:		
									G.add_node(str(token.lemma_), decay = 0, checked = 'n')
						for i_two in range(len(token_list)):
							count = i_two + 1
							while count < len(token_list):
								if G.has_edge(token_list[i_two], token_list[count]):
									G[token_list[i_two]][token_list[count]]['weight'] += 1
								else: 
									G.add_edge(token_list[i_two],token_list[count], weight=1)
								count = count + 1
					else :
						pass

				if (s == 'null'):
					pass
				else:
					docs_two = clean_tweets(s)
					doc_two = nlp(docs_two)
					corpus.add_doc(doc_two)

					token_list = []
					for token in doc_two:		
						if any(c.isalpha() for c in str(token)) and not token.is_stop:
							token_list.append(token.lemma_)
							if str(token.lemma_) not in G:
								G.add_node(str(token.lemma_),decay = 1,checked = 'y')
								#G.add_node(str(token.lemma_.lower()), weight = 1)
							else:
								attrs = {str(token.lemma_): {'decay': 1, 'checked':'y'}}
								nx.set_node_attributes(G, attrs)
						# for i_two in range(len(token_list)):
						# 	count = i_two + 1
						# 	while count < len(token_list):
						# 		if G.has_edge(token_list[i_two], token_list[count]):
						# 			G[token_list[i_two]][token_list[count]]['weight'] += 1
						# 		else: 
						# 			G.add_edge(token_list[i_two],token_list[count], weight=1)
						# 		count = count + 1
					for token in doc_two:
						if any(c.isalpha() for c in str(token)) and not token.is_stop:
							# print (G[str(token.lemma_)])	
							for n in G.neighbors(str(token.lemma_)):
								if G.node[n]['checked'] == 'n':
									G.node[n]['checked'] = 'y'
									# print (G.node[n]['checked'])
									G.node[n]['decay'] = G.node[str(token.lemma_)]['decay']*0.5
									# print (G.node[n]['decay'])

					for i_two in range(len(token_list)):
						count = i_two + 1
						while count < len(token_list):
							if G.has_edge(token_list[i_two], token_list[count]):
								G[token_list[i_two]][token_list[count]]['weight'] += 1
							else: 
								G.add_edge(token_list[i_two],token_list[count], weight=1)
							count = count + 1
				
					vectorizer = textacy.vsm.Vectorizer(apply_idf=True, norm=None, idf_type='standard')
					tokenized_docs = [doc._.to_terms_list(ngrams=1, named_entities=True, as_strings=True, filter_stops=True, normalize='lemma') for doc in corpus]
					doc_term_matrix = vectorizer.fit_transform(tokenized_docs)
					a = doc_term_matrix.toarray()
					

					start_time = time.time()
					try:
						centrality1 = nx.degree_centrality(G)
					except:
						centrality1 = 0
						print ("error")
					print("Calculating degree_centrality")
					t1 = time.time()
					elapsed_time1 = t1 - start_time
					print ("Time elapsed: %s seconds" % (elapsed_time1) )

					dfbase = pd.DataFrame(list(centrality1.items()), columns = ['node', 'degree_centrality'])
					#centrality_list = ['eigenvector_centrality']
					centrality_list = ['eigenvector_centrality','closeness_centrality','betweenness_centrality','load_centrality','subgraph_centrality','harmonic_centrality']
				



					temp_time = time.time() 
					for ct in centrality_list:
						try:
							print ("Calculating %s" % ct)
							centrality = getattr(nx, ct)(G)
							print ("Time elapsed: %s seconds" % (time.time() - temp_time) )
							temp_time = time.time()
							dfbase[ct]=dfbase['node'].map(centrality)
						except Exception as e:
							print ("error")
							dfbase[ct]=0
					#for token in doc_two:
						#print(token.text)
					token_list_two = pd.DataFrame()
					# print (dfbase['node'])
					# print (dfbase['degree_centrality'])
					for sent in doc_two.sents:
						for token in sent:
							# print (token)
							if any(c.isalpha() for c in str(token)) and not token.is_stop:
								rr = dfbase.loc[dfbase['node'] == token.lemma_]

								# print (rr)

								if not rr.empty :
									# print (rr)
									token_list_two = token_list_two.append(rr)


					if not token_list_two.empty:

						centrality_list_full = ['degree_centrality','eigenvector_centrality','closeness_centrality','betweenness_centrality','load_centrality','subgraph_centrality','harmonic_centrality']
					

						for itm in centrality_list_full:
							kwargs = {('weighted_%s' % itm) : lambda x: 0.0}
							token_list_two = token_list_two.assign(**kwargs)

						
						for i, row in token_list_two.iterrows():
							vb = a.item(((int(a.size/a[0].size))-1),(vectorizer.vocabulary_terms.get(row['node'])))
							for it in centrality_list_full:
								# print (vec.vocabulary_terms)
								# print (row['node'])
								va = row[it]
								# print ('vb')
								# print (vb)
								c = va*vb
								# print ('c')
								# print (c)
								#print(va)
								#print (vb)
								token_list_two.set_value(i, ('weighted_%s' % it), c)
								

								

								#print (('weighted_%s' % it))
								#print (row[('weighted_%s' % it)])
							
						#print ("token_list")
						#print (token_list)
						# print (token_list_two['weighted_degree_centrality'])

						tk_sum = pd.DataFrame()
						if r['status'] == 'like':
							tk_sum = tk_sum.assign(node = [r['origin_id']])
						else:
							tk_sum = tk_sum.assign(node = [r['id_str']])




						
						tk_sum = tk_sum.assign(screen_name = [r['screen_name']])
						tk_sum = tk_sum.assign(created_at = [r['created_at']])
						tk_sum = tk_sum.assign(full_text = [r['full_text']])
						tk_sum = tk_sum.assign(truncated = [r['truncated']])
						tk_sum = tk_sum.assign(user_description = [r['len(user.description)']])
						tk_sum = tk_sum.assign(user_followers_count = [r['user.followers_count']])
						tk_sum = tk_sum.assign(user_friends_count= [r['user.friends_count']])
						tk_sum = tk_sum.assign(user_listed_count = [r['user.listed_count']])
						tk_sum = tk_sum.assign(user_statuses_count = [r['user.statuses_count']])
						tk_sum = tk_sum.assign(user_favourites_count = [r['user.favourites_count']])
						tk_sum = tk_sum.assign(user_created_at = [r['user.created_at']])
						tk_sum = tk_sum.assign(entities_hashtags = [r['entities.hashtags']])
						tk_sum = tk_sum.assign(entities_user_mentions = [r['entities.user_mentions']])
						tk_sum = tk_sum.assign(entities_symbols = [r['entities.symbols']])
						tk_sum = tk_sum.assign(entities_polls = [r['bool(entities.polls)']])
						tk_sum = tk_sum.assign(entities_media_type = [r['entities.media_type']])
						tk_sum = tk_sum.assign(retweet_coun = [r['retweet_count']])
						tk_sum = tk_sum.assign(favorite_count = [r['favorite_count']])
						tk_sum = tk_sum.assign(origin_id = [r['origin_id']])
						tk_sum = tk_sum.assign(origin_created_at = [r['origin_created_at']])
						tk_sum = tk_sum.assign(origin_full_text = [r['origin_full_text']])
						tk_sum = tk_sum.assign(origin_user = [r['origin_user']])
						tk_sum = tk_sum.assign(origin_user_len_description = [r['origin_user_len_description']])
						tk_sum = tk_sum.assign(origin_user_followers_count = [r['origin_user_followers_count']])
						tk_sum = tk_sum.assign(origin_user_friends_count = [r['origin_user_friends_count']])
						tk_sum = tk_sum.assign(origin_user_listed_count = [r['origin_user_listed_count']])
						tk_sum = tk_sum.assign(origin_user_statuses_count = [r['origin_user_statuses_count']])
						tk_sum = tk_sum.assign(origin_user_favourites_count = [r['origin_user_favourites_count']])
						tk_sum = tk_sum.assign(origin_user_created_at = [r['origin_user_created_at']])
						tk_sum = tk_sum.assign(origin_entities_hashtags = [r['origin_entities_hashtags']])
						tk_sum = tk_sum.assign(origin_entities_user_mentions = [r['origin_entities_user_mentions']])
						tk_sum = tk_sum.assign(origin_entities_symbols = [r['origin_entities_symbols']])
						tk_sum = tk_sum.assign(origin_entities_polls = [r['origin_entities_polls']])
						tk_sum = tk_sum.assign(origin_entities_media_type = [r['origin_entities_media_type']])
						tk_sum = tk_sum.assign(origin_retweet_count = [r['origin_retweet_count']])
						tk_sum = tk_sum.assign(origin_favorite_count = [r['origin_favorite_count']])
						tk_sum = tk_sum.assign(origin_status = [r['origin_status']])


						try:
							tk_sum = tk_sum.assign(min_interval = duration)
						except:
							tk_sum = tk_sum.assign(min_interval = 100000000000000000)

						for itm in centrality_list_full:
							# print (token_list[('weighted_%s' % itm)])
							kwargs = {itm : lambda x: [token_list_two[('%s' % itm)].sum()]}
							try:
								tk_sum = tk_sum.assign(**kwargs)
							except:
								kwargs = {itm : lambda x: 0.0}
								tk_sum = tk_sum.assign(**kwargs)

						for itm in centrality_list_full:
							# print (token_list[('weighted_%s' % itm)])
							# print (token_list_two[('weighted_%s' % itm)].sum())
							kwargs = {('weighted_%s' % itm) : lambda x: [token_list_two[('weighted_%s' % itm)].sum()]}
							try:
								tk_sum = tk_sum.assign(**kwargs)
							except:
								kwargs = {('weighted_%s' % itm)  : lambda x: 0.0}
								tk_sum = tk_sum.assign(**kwargs)

						

						
						try:
							tk_sum = tk_sum.assign(status = [r['status']])
						except:
							tk_sum = tk_sum.assign(status = ['unknown'])
						# if ind == 1:
						# 	try:
						# 		tk_sum = tk_sum.assign(status = ['no action'])
						# 	except:
						# 		tk_sum = tk_sum.assign(status = ['unknown'])
						#print (tk_sum)
						# .reset_index()
						# token_list.sort_values(by=list(token_list.columns),axis=0)
						# token_list.reset_index()
						#tk = token_list.iloc[[0]]
						# tk = tk.assign(status = r['status'])
						cont = cont + 1
						print ("%d interactions collected" % cont)


						with open(os.path.join(tweetnetworks,'%s_interaction.csv' % temp_user), 'a') as f:
							writer = csv.writer(f)
							#tk.to_csv(f, header=False,index=False)
							tk_sum.to_csv(f, header=False,index=False)
					else:
						cont_e = cont_e + 1
						print ("%dth interaction with zero centralities" % cont_e)
					part_data = pd.DataFrame()


if __name__ == '__main__':
	tweetnetworks = 'collection'
	if not os.path.exists(tweetnetworks):
	    os.makedirs(tweetnetworks)
	
	with open(os.path.join(tweetnetworks,'%s_interaction.csv' % temp_user), 'a') as f:
		writer = csv.writer(f)
		#writer.writerow(['node','degree_centrality','eigenvector_centrality','status'])
		writer.writerow([
					'node',
					'screen_name', 
					'created_at',
					'full_text',
					'truncated',
					'len(user.description)',
					'user.followers_count',
					'user.friends_count',
					'user.listed_count',
					'user.statuses_count',
					'user.favourites_count',
					'user.created_at',
					'entities.hashtags',
					'entities.user_mentions',
					'entities.symbols',
					'bool(entities.polls)',
					'entities.media_type',
					'retweet_count',
					'favorite_count',
					'origin_id',
					'origin_created_at',
					'origin_full_text',
					'origin_user',
					'origin_user_len_description',
					'origin_user_followers_count',
					'origin_user_friends_count',
					'origin_user_listed_count',
					'origin_user_statuses_count',
					'origin_user_favourites_count',
					'origin_user_created_at',
					'origin_entities_hashtags',
					'origin_entities_user_mentions',
					'origin_entities_symbols',
					'origin_entities_polls',
					'origin_entities_media_type',
					'origin_retweet_count',
					'origin_favorite_count',
					'origin_status',
					'min_interval',

					'degree_centrality',
					'eigenvector_centrality',
					'closeness_centrality',
					'betweenness_centrality',
					'load_centrality',
					'subgraph_centrality',
					'harmonic_centrality',
					'weighted_degree_centrality',
					'weighted_eigenvector_centrality',
					'weighted_closeness_centrality',
					'weighted_betweenness_centrality',
					'weighted_load_centrality',
					'weighted_subgraph_centrality',
					'weighted_harmonic_centrality',
					'status'])

	data_one = pd.read_csv(os.path.join(tweetnetworks,'%s_tweets.csv' % temp_user))
	data_one = data_one.replace(np.nan, 'null', regex=True)
	# data_one['created_at'] = data_one.apply(lambda row: row['origin_created_at'] if row['created_at'] == 'null' else row['created_at'],axis=1)
	data_one['created_at'] = pd.to_datetime(data_one['created_at'])
	data_one = data_one.sort_values('created_at', ascending=True)
	data_one = data_one.reset_index()
	
	# print (data_one['created_at'])

	data_two = pd.read_csv(os.path.join(tweetnetworks,'%s_followee_tweets.csv' % temp_user))
	data_two = data_two.replace(np.nan, 'null', regex=True)
	# data_two['created_at'] = data_two.apply(lambda row: row['origin_created_at'] if row['created_at'] == 'null' else row['created_at'],axis=1)
	data_two['created_at'] = pd.to_datetime(data_one['created_at'])
	data_two = data_two.sort_values('created_at', ascending=True)
	data_two = data_two.reset_index()
	

	spacy.prefer_gpu()
	nlp = spacy.load("en_core_web_sm")
	G = nx.Graph()
	corpus = textacy.Corpus(nlp)

	calculate_centrality_original(data_one)
	calculate_centrality_followee(data_two)
	
			