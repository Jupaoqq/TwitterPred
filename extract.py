from datetime import datetime, date
import time
import tweepy #https://github.com/tweepy/tweepy
import csv
import pandas as pd
import operator
import os
from pathlib import Path

tweets = 'collection'
global_timestamp = []
reply_one_hundred_check = []

data = pd.read_csv('user.csv', header=None)
consumer_key = str(data[1][0])
consumer_secret = str(data[1][1])
access_key = str(data[1][2])
access_secret = str(data[1][3])

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth,wait_on_rate_limit=True, wait_on_rate_limit_notify=True,compression=True)


if __name__ == '__main__':	
	if not os.path.exists(tweets):
		os.makedirs(tweets)

	temp_user = str(data[1][4])
	ts = pd.read_csv((os.path.join(tweets,'%s_tweets.csv' % temp_user)))
	global_timestamp = pd.to_datetime(ts['created_at'])
	id_list = pd.read_csv('tweet.csv')
	all_ids = id_list['ID']

	
	with open(os.path.join(tweets,'test_tweets.csv'), 'a') as f:
		writer = csv.writer(f)
		writer.writerow(['screen_name', 
					'id_str',
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
					'status',
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
					'origin_status'
					,'min_interval'])





one_hundred_ids = []
all_tweets = []
id_objects = []
id_c = 0
for ids in all_ids:
	one_hundred_ids.append(ids)
	id_c = id_c + 1
	if id_c == 100:
		all_tweets.extend(api.statuses_lookup(id_ = one_hundred_ids, map_ = True, tweet_mode="extended"))
		print("ID look up API call")
		id_c = 0
		one_hundred_ids = []
if not id_c == 0:
	all_tweets.extend(api.statuses_lookup(id_ = one_hundred_ids, map_ = True,tweet_mode="extended"))
	print("ID look up API call")
	id_c = 0
	one_hundred_id_id = []

for tweet in all_tweets:
	print (tweet.created_at)

ct = 0
for tweet in all_tweets:
	print ("collect info for tweet %s " % ct)
	print (tweet.full_text)
	ct = ct+1
	if hasattr(tweet, 'retweeted_status'):
		tweet.status = 'retweet'	
		tweet.origin_id = tweet.retweeted_status.id_str
		tweet.origin_created_at = tweet.retweeted_status.created_at
		tweet.origin_full_text = tweet.retweeted_status.full_text.encode("utf-8")
		tweet.origin_user = tweet.retweeted_status.user.screen_name
		tweet.origin_favorite_count = tweet.retweeted_status.favorite_count
		tweet.origin_retweet_count = tweet.retweeted_status.retweet_count
		#print (tweet.retweeted_status.user)
		tweet.origin_user_len_description = len(tweet.retweeted_status.user.description)
		tweet.origin_user_followers_count = tweet.retweeted_status.user.followers_count
		tweet.origin_user_friends_count = tweet.retweeted_status.user.friends_count
		tweet.origin_user_listed_count = tweet.retweeted_status.user.listed_count
		tweet.origin_user_statuses_count = tweet.retweeted_status.user.statuses_count
		tweet.origin_user_favourites_count = tweet.retweeted_status.user.favourites_count
		tweet.origin_user_created_at = tweet.retweeted_status.user.created_at


		tweet.entities_hashtags = ''
		tweet.entities_user_mentions = ''
		tweet.entities_symbols = ''
		tweet.entities_polls = False
		tweet.entities_media_type = ''
		for a in tweet.entities['hashtags']:
			tweet.entities_hashtags = tweet.entities_hashtags + " " + a['text']
		for b in tweet.entities['user_mentions']:
			tweet.entities_user_mentions = tweet.entities_user_mentions + " " +b['screen_name']
		for c in tweet.entities['symbols']:
			tweet.entities_symbols = tweet.entities_symbols + " " + c['text']
		try:
			if tweet.entities['polls']:
				tweet.entities_polls = True
		except Exception:
			pass
		try:
			if tweet.entities['media']:
				tweet.entities_media_type = tweet.entities['media']['type']
		except Exception:
			pass
		tweet.origin_entities_hashtags = ''
		tweet.origin_entities_user_mentions = ''
		tweet.origin_entities_symbols = ''
		tweet.origin_entities_polls = False
		tweet.origin_entities_media_type = ''
		tweet.origin_status = ''
		for a in tweet.retweeted_status.entities['hashtags']:
			tweet.origin_entities_hashtags = tweet.origin_entities_hashtags + " " + a['text']
		for b in tweet.retweeted_status.entities['user_mentions']:
			tweet.origin_entities_user_mentions = tweet.origin_entities_user_mentions + " " + b['screen_name']
		for c in tweet.retweeted_status.entities['symbols']:
			tweet.origin_entities_symbols = tweet.origin_entities_symbols + " " + c['text']
		try:
			if tweet.retweeted_status.entities['polls']:
				tweet.origin_entities_polls = True
		except Exception:
			pass
		try:
			if tweet.retweeted_status.entities['media']:
				tweet.origin_entities_media_type = tweet.retweeted_status.entities['media']['type']
		except Exception:
			pass
		if hasattr(tweet.retweeted_status, 'retweeted_status'):
			tweet.origin_status = 'retweet'
		elif tweet.retweeted_status.in_reply_to_status_id_str is not None:
			tweet.retweeted_status.origin_status = 'reply'
		elif hasattr(tweet.retweeted_status, 'quoted_status'):
			tweet.origin_status = 'quote'
		else:
			tweet.origin_status = 'original'

	elif tweet.in_reply_to_status_id_str is not None:
		tweet.status = 'reply'
		tweet.origin_id = tweet.in_reply_to_status_id
		tweet.origin_created_at = ''
		tweet.origin_full_text = ''
		tweet.origin_user = ''
		tweet.origin_favorite_count = ''
		tweet.origin_retweet_count = ''
		tweet.origin_user_len_description = ''
		tweet.origin_user_followers_count = ''
		tweet.origin_user_friends_count = ''
		tweet.origin_user_listed_count = ''
		tweet.origin_user_statuses_count = ''
		tweet.origin_user_favourites_count = ''
		tweet.origin_user_created_at = ''
		tweet.entities_hashtags = ''
		tweet.entities_user_mentions = ''
		tweet.entities_symbols = ''
		tweet.entities_polls = False
		tweet.entities_media_type = ''
		tweet.origin_entities_hashtags = ''
		tweet.origin_entities_user_mentions = ''
		tweet.origin_entities_symbols = ''
		tweet.origin_entities_polls = ''
		tweet.origin_entities_media_type = ''
		tweet.origin_status = ''

	if tweet.favorite_count is None:
		tweet.favorite_count = 0

		reply_one_hundred_check.append(tweet)
	elif hasattr(tweet, 'quoted_status'):
		tweet.status = 'quote'
		tweet.origin_id = tweet.quoted_status.id_str
		tweet.origin_created_at = tweet.quoted_status.created_at
		tweet.origin_full_text = tweet.quoted_status.full_text.encode("utf-8")
		tweet.origin_user = tweet.quoted_status.user.screen_name
		tweet.origin_favorite_count = tweet.quoted_status.favorite_count
		tweet.origin_retweet_count = tweet.quoted_status.retweet_count
		tweet.origin_user_len_description = len(tweet.quoted_status.user.description)
		tweet.origin_user_followers_count = tweet.quoted_status.user.followers_count
		tweet.origin_user_friends_count = tweet.quoted_status.user.friends_count
		tweet.origin_user_listed_count = tweet.quoted_status.user.listed_count
		tweet.origin_user_statuses_count = tweet.quoted_status.user.statuses_count
		tweet.origin_user_favourites_count = tweet.quoted_status.user.favourites_count
		tweet.origin_user_created_at = tweet.quoted_status.user.created_at

		tweet.entities_hashtags = ''
		tweet.entities_user_mentions = ''
		tweet.entities_symbols = ''
		tweet.entities_polls = False
		tweet.entities_media_type = ''
		for a in tweet.entities['hashtags']:
			tweet.entities_hashtags = tweet.entities_hashtags + " " + a['text']
		for b in tweet.entities['user_mentions']:
			tweet.entities_user_mentions = tweet.entities_user_mentions + " " +b['screen_name']
		for c in tweet.entities['symbols']:
			tweet.entities_symbols = tweet.entities_symbols + " " + c['text']
		try:
			if tweet.entities['polls']:
				tweet.entities_polls = True
		except Exception:
			pass
		try:
			if tweet.entities['media']:
				tweet.entities_media_type = tweet.entities['media']['type']
		except Exception:
			pass
		tweet.origin_entities_hashtags = ''
		tweet.origin_entities_user_mentions = ''
		tweet.origin_entities_symbols = ''
		tweet.origin_entities_polls = False
		tweet.origin_entities_media_type = ''
		tweet.origin_status = ''
		for a in tweet.quoted_status.entities['hashtags']:
			tweet.origin_entities_hashtags = tweet.origin_entities_hashtags + " " + a['text']
		for b in tweet.quoted_status.entities['user_mentions']:
			tweet.origin_entities_user_mentions = tweet.origin_entities_user_mentions + " " + b['screen_name']
		for c in tweet.quoted_status.entities['symbols']:
			tweet.origin_entities_symbols = tweet.origin_entities_symbols + " " + c['text']
		try:
			if tweet.quoted_status.entities['polls']:
				tweet.origin_entities_polls = True
		except Exception:
			pass
		try:
			if tweet.quoted_status.entities['media']:
				tweet.origin_entities_media_type = tweet.quoted_status.entities['media']['type']
		except Exception:
			pass

		if hasattr(tweet.quoted_status, 'retweeted_status'):
			tweet.origin_status = 'retweet'
		elif tweet.quoted_status.in_reply_to_status_id_str is not None:
			tweet.origin_status = 'reply'
		elif hasattr(tweet.quoted_status, 'quoted_status'):
			tweet.origin_status = 'quote'
		else:
			tweet.origin_status = 'original'


	else:
		tweet.status = 'original'
		tweet.origin_id = ''
		tweet.origin_created_at = ''
		tweet.origin_full_text = ''
		tweet.origin_user = ''
		tweet.origin_favorite_count = ''
		tweet.origin_retweet_count = ''
		tweet.origin_user_len_description = ''
		tweet.origin_user_followers_count = ''
		tweet.origin_user_friends_count = ''
		tweet.origin_user_listed_count = ''
		tweet.origin_user_statuses_count = ''
		tweet.origin_user_favourites_count = ''
		tweet.origin_user_created_at = ''

		tweet.entities_hashtags = ''
		tweet.entities_user_mentions = ''
		tweet.entities_symbols = ''
		tweet.entities_polls = False
		tweet.entities_media_type = ''
		for a in tweet.entities['hashtags']:
			tweet.entities_hashtags = tweet.entities_hashtags + " " + a['text']
		for b in tweet.entities['user_mentions']:
			tweet.entities_user_mentions = tweet.entities_user_mentions + " " +b['screen_name']
		for c in tweet.entities['symbols']:
			tweet.entities_symbols = tweet.entities_symbols + " " + c['text']
		try:
			if tweet.entities['polls']:
				tweet.entities_polls = True
		except Exception:
			pass
		try:
			if tweet.entities['media']:
				tweet.entities_media_type = tweet.entities['media']['type']
		except Exception:
			pass
		tweet.origin_entities_hashtags = ''
		tweet.origin_entities_user_mentions = ''
		tweet.origin_entities_symbols = ''
		tweet.origin_entities_polls = False
		tweet.origin_entities_media_type = ''
		tweet.origin_status = ''

	if tweet.favorite_count is None:
		tweet.favorite_count = 0


reply_c = 0
one_hundred_reply_id = []
reply_objects = []
for replies in reply_one_hundred_check:
	one_hundred_reply_id.append(replies.in_reply_to_status_id)
	reply_c = reply_c + 1
	if reply_c == 100:
		reply_objects.extend(api.statuses_lookup(id_ = one_hundred_reply_id, map_ = True, tweet_mode="extended"))
		print("reply look up API call")
		reply_c = 0
		one_hundred_reply_id = []
if not reply_c == 0:
	reply_objects.extend(api.statuses_lookup(id_ = one_hundred_reply_id, map_ = True,tweet_mode="extended"))
	print("reply look up API call")
	reply_c = 0
	one_hundred_reply_id = []
for tweet in all_tweets:
	if tweet.in_reply_to_status_id_str is not None:
		for rp in reply_objects:
			if tweet.in_reply_to_status_id == rp.id: 
				if hasattr (rp, 'full_text'):
					tweet.origin_created_at = rp.created_at
					tweet.origin_full_text = rp.full_text.encode("utf-8")
					tweet.origin_user = rp.user.screen_name
					tweet.origin_favorite_count = rp.favorite_count
					tweet.origin_retweet_count = rp.retweet_count
					tweet.origin_user_len_description = len(rp.user.description)
					tweet.origin_user_followers_count = rp.user.followers_count
					tweet.origin_user_friends_count = rp.user.friends_count
					tweet.origin_user_listed_count = rp.user.listed_count
					tweet.origin_user_statuses_count = rp.user.statuses_count
					tweet.origin_user_favourites_count = rp.user.favourites_count
					tweet.origin_user_created_at = rp.user.created_at



					tweet.entities_hashtags = ''
					tweet.entities_user_mentions = ''
					tweet.entities_symbols = ''
					tweet.entities_polls = False
					tweet.entities_media_type = ''
					for a in tweet.entities['hashtags']:
						tweet.entities_hashtags = tweet.entities_hashtags + " " + a['text']
					for b in tweet.entities['user_mentions']:
						tweet.entities_user_mentions = tweet.entities_user_mentions + " " +b['screen_name']
					for c in tweet.entities['symbols']:
						tweet.entities_symbols = tweet.entities_symbols + " " + c['text']
					try:
						if tweet.entities['polls']:
							tweet.entities_polls = True
					except Exception:
						pass
					try:
						if tweet.entities['media']:
							tweet.entities_media_type = tweet.entities['media']['type']
					except Exception:
						pass
					tweet.origin_entities_hashtags = ''
					tweet.origin_entities_user_mentions = ''
					tweet.origin_entities_symbols = ''
					tweet.origin_entities_polls = False
					tweet.origin_entities_media_type = ''
					tweet.origin_status = ''
					for a in tweet.rp.entities['hashtags']:
						tweet.origin_entities_hashtags = tweet.origin_entities_hashtags + " " + a['text']
					for b in tweet.rp.entities['user_mentions']:
						tweet.origin_entities_user_mentions = tweet.origin_entities_user_mentions + " " + b['screen_name']
					for c in tweet.rp.entities['symbols']:
						tweet.origin_entities_symbols = tweet.origin_entities_symbols + " " + c['text']
					try:
						if tweet.rp.entities['polls']:
							tweet.origin_entities_polls = True
					except Exception:
						pass
					try:
						if tweet.rp.entities['media']:
							tweet.origin_entities_media_type = tweet.rp.entities['media']['type']
					except Exception:
						pass

					if hasattr(rp, 'retweeted_status'):
						tweet.origin_status = 'retweet'
					elif rp.in_reply_to_status_id_str is not None:
						tweet.origin_status = 'reply'
					elif hasattr(rp, 'quoted_status'):
						tweet.origin_status = 'quote'
					else:
						tweet.origin_status = 'original'


	for tweet in all_tweets:
		tweet.min_interval = 10000000000000
		for t in global_timestamp:
			duration_raw = tweet.created_at - t
			duration = duration_raw.total_seconds()
			if duration > 0:
				if duration < tweet.min_interval:
					tweet.min_interval = duration
	#write the csv	
	outtweets = [[tweet.user.screen_name, 
				tweet.id_str,
				tweet.created_at,
				tweet.full_text.encode("utf-8"),
				tweet.truncated,
				len(tweet.user.description),
				tweet.user.followers_count,
				tweet.user.friends_count,
				tweet.user.listed_count,
				tweet.user.statuses_count,
				tweet.user.favourites_count,
				tweet.user.created_at,
				tweet.entities_hashtags,
				tweet.entities_user_mentions,
				tweet.entities_symbols,
				bool(tweet.entities_polls),
				tweet.entities_media_type,
				tweet.retweet_count,
				tweet.favorite_count,
				tweet.status,
				tweet.origin_id,
				tweet.origin_created_at,
				tweet.origin_full_text,
				tweet.origin_user,
				tweet.origin_user_len_description,
				tweet.origin_user_followers_count,
				tweet.origin_user_friends_count,
				tweet.origin_user_listed_count,
				tweet.origin_user_statuses_count,
				tweet.origin_user_favourites_count,
				tweet.origin_user_created_at,
				tweet.origin_entities_hashtags,
				tweet.origin_entities_user_mentions,
				tweet.origin_entities_symbols,
				tweet.origin_entities_polls,
				tweet.origin_entities_media_type,
				tweet.origin_retweet_count,
				tweet.origin_favorite_count,
				tweet.origin_status
				] for tweet in all_tweets]

with open(os.path.join(tweets,'test_tweets.csv'), 'a') as f:
	writer = csv.writer(f)
	writer.writerows(outtweets)
