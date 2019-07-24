from datetime import datetime, date
import time
import tweepy #https://github.com/tweepy/tweepy
import csv
import pandas as pd
import operator
import os
from pathlib import Path

tweets = 'collection'
#Twitter API credentials
data = pd.read_csv('user.csv', header=None)
consumer_key = str(data[1][0])
consumer_secret = str(data[1][1])
access_key = str(data[1][2])
access_secret = str(data[1][3])

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth,wait_on_rate_limit=True, wait_on_rate_limit_notify=True,compression=True)

global_username = ''
all_users = []
global_timestamp = []
all_users_object = []



temp_users = []
user = ''
count_helper = 0
followers_page_count = 0
ten_count = 0

def get_all_tweets(screen_name):
	# print (screen_name)
	
	#initialize a list to hold all the tweepy Tweets
	alltweets = []	
	new_tweets = []
	reply_one_hundred_check = []
	#make initial request for most recent tweets (200 is the maximum allowed count)
	for page in tweepy.Cursor(api.user_timeline, screen_name = screen_name, tweet_mode="extended", count=200).pages():
		try:
			print ("get user timeline API call")
			alltweets.extend(page)
			print ("...%s tweets downloaded so far for user %s" % (len(alltweets),screen_name))
		except Exception:
			pass
	#transform the tweepy tweets into a 2D array that will populate the csv
	ct = 0
	for tweet in alltweets:
		print ("check action type tweet %s for user %s" % (ct,screen_name))
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
	for tweet in alltweets:
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
						for a in rp.entities['hashtags']:
							tweet.origin_entities_hashtags = tweet.origin_entities_hashtags + " " + a['text']
						for b in rp.entities['user_mentions']:
							tweet.origin_entities_user_mentions = tweet.origin_entities_user_mentions + " " + b['screen_name']
						for c in rp.entities['symbols']:
							tweet.origin_entities_symbols = tweet.origin_entities_symbols + " " + c['text']
						try:
							if rp.entities['polls']:
								tweet.origin_entities_polls = True
						except Exception:
							pass
						try:
							if rp.entities['media']:
								tweet.origin_entities_media_type = rp.entities['media']['type']
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

	for tweet in alltweets:
		if not screen_name == global_username:
			tweet.min_interval = 10000000000000
			for t in global_timestamp:
				duration_raw = tweet.created_at - t
				duration = duration_raw.total_seconds()
				if duration > 0:
					if duration < tweet.min_interval:
						tweet.min_interval = duration


	# tweet.entities.hashtags.text
	# tweet.entities.media_type
	# tweet.entities.user_mentions.screen_name
	# tweet.entities.polls
	# tweet.coordinates


	if screen_name == global_username:
		for tweet in alltweets:
			if tweet.origin_user not in all_users:
				all_users.extend([tweet.origin_user])
			global_timestamp.append(tweet.created_at)
		#write the csv	
		outtweets = [[screen_name, 
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
					] for tweet in alltweets]

		with open(os.path.join(tweets,'%s_tweets.csv' % screen_name), 'a') as f:
			writer = csv.writer(f)
			writer.writerows(outtweets)
	else:
		#write the csv	
		outtweets = [[screen_name, 
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
					tweet.origin_status,
					tweet.min_interval] for tweet in alltweets]
		with open(os.path.join(tweets,'%s_followee_tweets.csv' % temp_user), 'a') as f:
			writer = csv.writer(f)
			writer.writerows(outtweets)
	
pass

def get_all_liked_tweets(screen_name):
	
	#initialize a list to hold all the tweepy Tweets
	alltweets = []	
	new_tweets1 = []
	#make initial request for most recent tweets (200 is the maximum allowed count)
	# try: 
	# 	new_tweets1 = api.favorites(screen_name = screen_name,count=200)
	# 	#print("get user fav_timeline API call")
	# except Exception:
 #         pass

	if screen_name == temp_users[0]:
		ua = api.lookup_users(screen_names = [screen_name])
		u = ua[0]
	else:
		u = lambda : None
		u.followers_count = ''
		u.friends_count = ''
		u.listed_count = ''
		u.statuses_count = ''
		u.favourites_count = ''
		u.created_at = ''
		for a in all_users_object:
			if a.screen_name == screen_name:
				u.followers_count = a.followers_count
				u.friends_count = a.friends_count
				u.listed_count = a.listed_count
				u.statuses_count = a.statuses_count
				u.favourites_count = a.favourites_count
				u.created_at = a.created_at
	
	for page in tweepy.Cursor(api.favorites, screen_name = screen_name, tweet_mode="extended", count=200).pages():
		try:
			print ("get user likes timeline API call")
			alltweets.extend(page)
			print ("...%s liked tweets downloaded so far for user %s" % (len(alltweets),screen_name))
		except Exception:
			pass
	#transform the tweepy tweets into a 2D array that will populate the csv
	#save most recent tweets
	alltweets.extend(new_tweets1)
	
	#transform the tweepy tweets into a 2D array that will populate the csv	
	for tweet in alltweets:

		if not hasattr(u, 'description'):
			u.description = ''
		if not hasattr(u, 'followers_count'):
			u.followers_count = ''
		if not hasattr(u, 'friends_count'):
			u.friends_count = ''
		if not hasattr(u, 'listed_count'):
			u.listed_count = ''
		if not hasattr(u, 'statuses_count'):
			u.statuses_count = ''
		if not hasattr(u, 'favourites_count'):
			u.favourites_count = ''
		if not hasattr(u, 'created_at'):
			u.created_at = ''
		if hasattr(tweet, 'retweeted_status'):
			tweet.origin_status = 'retweet'
		elif tweet.in_reply_to_status_id_str is not None:
			tweet.origin_status = 'reply'
		elif hasattr(tweet, 'quoted_status'):
			tweet.origin_status = 'quote'
		else:
			tweet.origin_status = 'original'
		if tweet.favorite_count is None:
			tweet.favorite_count = ''
		if not screen_name == global_username:
			tweet.min_interval = 10000000000000
			for t in global_timestamp:
				duration_raw = tweet.created_at - t
				duration = duration_raw.total_seconds()
				if duration > 0:
					if duration < tweet.min_interval:
						tweet.min_interval = duration
		
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

	if screen_name == global_username:
		if tweet.user.screen_name not in all_users:
			all_users.extend([tweet.user.screen_name])
			global_timestamp.append(tweet.created_at)
		#write the csv	
		outtweets = [[screen_name, 
					tweet.id_str,
					tweet.created_at,
					'   ',
					tweet.truncated,
					len(u.description),
					u.followers_count,
					u.friends_count,
					u.listed_count,
					u.statuses_count,
					u.favourites_count,
					u.created_at,
					tweet.entities_hashtags,
					tweet.entities_user_mentions,
					tweet.entities_symbols,
					bool(tweet.entities_polls),
					tweet.entities_media_type,
					tweet.retweet_count,
					tweet.favorite_count,
					'like',
					tweet.id_str,
					tweet.created_at,
					tweet.full_text.encode("utf-8"),
					tweet.user.screen_name,
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
					tweet.origin_status
		] for tweet in alltweets]
	
		with open(os.path.join(tweets,'%s_tweets.csv' % screen_name), 'a') as f:
			writer = csv.writer(f)
			writer.writerows(outtweets)
	else:
		#write the csv	
		outtweets = [[screen_name, 
					tweet.id_str,
					tweet.created_at,
					tweet.full_text.encode("utf-8"),
					tweet.truncated,
					len(u.description),
					u.followers_count,
					u.friends_count,
					u.listed_count,
					u.statuses_count,
					u.favourites_count,
					u.created_at,
					tweet.entities_hashtags,
					tweet.entities_user_mentions,
					tweet.entities_symbols,
					bool(tweet.entities_polls),
					tweet.entities_media_type,
					tweet.retweet_count,
					tweet.favorite_count,
					'like',
					tweet.id_str,
					tweet.created_at,
					tweet.full_text.encode("utf-8"),
					tweet.user.screen_name,

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
					tweet.origin_status,
					tweet.min_interval] for tweet in alltweets]
		with open(os.path.join(tweets,'%s_followee_tweets.csv' % temp_user), 'a') as f:
			writer = csv.writer(f)
			writer.writerows(outtweets)
	
pass




def get_quoted_tweets_root(tweet):
	#print("get tweet status API call (quote)")
	if tweet.is_quote_status == True:
		if not hasattr(tweet, 'quoted_status'):
			if (hasattr(api.get_status(id = tweet.id),'quoted_status')):
				#print("get tweet status API call (quote)")
				return get_quoted_tweets_root(api.get_status(id = tweet.id).quoted_status)
			else:
				return tweet
		else:
			return get_quoted_tweets_root(tweet.quoted_status)

	else:
		return tweet


def get_replied_tweets_root(tweet):
	#print("get tweet status API call (reply)")
	if tweet.in_reply_to_status_id is not None:
		try:
			return get_replied_tweets_root(api.get_status(id = tweet.in_reply_to_status_id))
		except Exception:
			return tweet
	else:
		return tweet





if __name__ == '__main__':
	#pass in the username of the account you want to download
	# for attr in dir(api):
	#     print("obj.%s = %r" % (attr, getattr(api, attr)))
	# pprint(vars(api.get_status))
		
	if not os.path.exists(tweets):
		os.makedirs(tweets)
	temp_user = str(data[1][4])
	temp_users = [temp_users]
	global_username = temp_user
	with open(os.path.join(tweets,'%s_tweets.csv' % temp_user), 'a') as f:
		writer = csv.writer(f)
		writer.writerow([
					'screen_name', 
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
					])
	with open(os.path.join(tweets,'%s_followee_tweets.csv' %temp_user), 'a') as f:
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
	get_all_tweets(temp_user)
	get_all_liked_tweets(temp_user)


	one_hundred_ids = []
	one_hundred_count = 0
	for name in all_users:	
		one_hundred_ids.append(name)
		one_hundred_count = one_hundred_count + 1
		if one_hundred_count  == 100:
			all_users_object.extend(api.lookup_users(screen_names = one_hundred_ids))
			one_hundred_count = 0
			one_hundred_ids = []
			# print (index)
	if not one_hundred_count == 0:
		all_users_object.extend(api.lookup_users(screen_names = one_hundred_ids))
		# print (one_hundred_count)
		one_hundred_count = 0
		one_hundred_ids = []


	for users in all_users:
		if users:
			if not users.lower() == global_username:
				get_all_tweets(users)
				get_all_liked_tweets(users)
