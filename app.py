import streamlit as st 
import numpy as np 
import pandas as pd 
import requests
import tensorflow as tf 
from tensorflow import keras
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random 
# Web scraping 
from bs4 import BeautifulSoup 
from urllib.request import urlopen 
import torch
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import DistilBertTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline
from pathlib import Path

SAMPLE_SIZE = 100

@st.cache
def load_embedded_data(embedding_data_path="healthFactTrainData.pkl"):
	with open(embedding_data_path, 'rb') as file:
		try:
			while True:
				embedding_data = pickle.load(file)
		except EOFError:
			pass
	return embedding_data

def sentence_splitter(article):

	return re.split("(?<=[.!?])\s+", article)
	

# Code from this function was used from a tutorial by Jesse E. Agbe
def get_text(raw_url):
	page = urlopen(raw_url)
	soup = BeautifulSoup(page)
	fetched = ' '.join(map(lambda p:p.text, soup.find_all('p')))
	return fetched

def load_tokenizer(filename):
	with open(filename, 'rb') as handle:
		tokenizer = pickle.load(handle)

	return tokenizer

def preprocess_text(tokenizer,text):
  """
  Args:
	tokenizer: keras tokenizer object
	text(list): list of strings of texts to make predictions on
  
  Return
	padded_sequence (np.array): shape [n_examples, maxlen] 
  """
  encoded_docs = tokenizer.texts_to_sequences(text)
  padded_sequence = pad_sequences(encoded_docs, maxlen=300)
  return padded_sequence

def preprocess_bulk_text(tokenizer,textList):
	"""
	Args:
	tokenizer: keras tokenizer object
	text(list(list)): list of list of strings of texts to make predictions on

	Return
	padded_sequence (np.array): shape [n_examples, maxlen] 
	"""
	padded_sequence_list = []
	for text in textList:
		encoded_docs = tokenizer.texts_to_sequences(text)
		padded_sequence = pad_sequences(encoded_docs, maxlen=300)

		padded_sequence_list.append(padded_sequence)

	X = np.concatenate(padded_sequence_list, axis=0)

	return X

@st.cache
def load_testing_data(filename_data):
	with open(filename_data, 'rb') as handle:
		data = pickle.load(handle)

	return data

def predictionRoutine(data, tokenizer, model):
	"""
	INPUTS:
	@data(pd.DataFrame) - The data for the lower split, exxpected number 
	of rows is SAMPLE_SIZE
	@tokenizer(????) - Tokenizer 
	@model (???) - model to use

	OUTPUTS:
	predicted label, true label numpy  arrays

	"""

	preprocessedText = preprocess_text(tokenizer, data["main_text"])
	pred = model.predict(preprocessedText)
	test_pred_class = np.argmax(pred, axis = -1)
	test_true_class= data["label"]

	return test_pred_class, test_true_class

def generateAccDfLSTM(lowerData, upperData, tokenizer, model, sampleSize = SAMPLE_SIZE):

	# lowerToPredict = lowerData.sample(sampleSize)
	# upperToPredict = upperData.sample(sampleSize)

	# #To ensure that we have at least one of each class
	# while len(set(list(lowerToPredict["label"])))!=4:
	# 	lowerToPredict = lowerData.sample(SAMPLE_SIZE)

	# while len(set(list(upperToPredict["label"])))!=4:
	# 	upperToPredict = upperData.sample(SAMPLE_SIZE)

	labels = ['False', 'Mixture', 'True', 'Unproven'] 
	lowerToPredict = lowerData
	upperToPredict = upperData

	y_lower_pred, y_lower_true = predictionRoutine(lowerToPredict, tokenizer, model)
	y_upper_pred, y_upper_true = predictionRoutine(upperToPredict, tokenizer, model)

	acc_lower = np.sum(y_lower_pred == y_lower_true)/len(y_lower_true)
	acc_upper = np.sum(y_upper_pred == y_upper_true)/len(y_upper_true)

	totalaccuracyDF = pd.DataFrame(data = {"LowSplit TotalAcc": [acc_lower], "UppSplit TotalAcc": [acc_upper]})

	lowerLabelNum = []
	upperLabelNum = []
	lowerAccuracy = []
	upperAccuracy = []

	for i in range(4):
		lowerLabelNum.append(len(y_lower_true[y_lower_true==i]))
		upperLabelNum.append(len(y_upper_true[y_upper_true==i]))
		
		y_lower_pred_one_label = y_lower_pred[y_lower_true==i]
		y_lower_true_one_label = y_lower_true[y_lower_true==i]
		accLowOneLabel = np.sum(y_lower_pred_one_label==y_lower_true_one_label)/len(y_lower_true_one_label)
		lowerAccuracy.append(accLowOneLabel)

		y_upper_pred_one_label = y_upper_pred[y_upper_true==i]
		y_upper_true_one_label = y_upper_true[y_upper_true==i]
		accUpperOneLabel = np.sum(y_upper_pred_one_label==y_upper_true_one_label)/len(y_upper_true_one_label)
		upperAccuracy.append(accUpperOneLabel)
	
	accuracyDf = pd.DataFrame(data = {"Labels": labels,
									  "LowSplit Acc": lowerAccuracy,
								      "LowSplit Num": lowerLabelNum,
								      "UppSplit Acc": upperAccuracy,
								      "UppSplit Num": upperLabelNum})



	return accuracyDf, totalaccuracyDF 

def displayPredictedDf(pred):
	# Prediction and print dataframe of probabilities 

	outcomes = ["Article predicted to contain false information.", "Article predicted to contain some true and some false information.", "Article predicted to contain true information.", "Article predicted to contain unproven information."]    

	pred_df = pd.DataFrame(data=pred)
	pred_df.columns =['False', 'Mixture', 'True', 'Unproven'] 
	test_class = np.argmax(pred, axis = -1)
#                st.dataframe(pred_df.style.highlight_max(axis=1, color='green'))

	if test_class == 0:
		st.dataframe(pred_df.style.highlight_max(axis=1, color='red'))
		st.error(outcomes[np.int(test_class)])
	elif test_class == 1:
		st.dataframe(pred_df.style.highlight_max(axis=1, color='yellow'))
		st.warning(outcomes[np.int(test_class)])
	elif test_class == 2:
		st.dataframe(pred_df.style.highlight_max(axis=1, color='green'))
		st.success(outcomes[np.int(test_class)])
	elif test_class == 3:
		st.dataframe(pred_df.style.highlight_max(axis=1, color='grey'))
		st.info(outcomes[np.int(test_class)])

def main():
	
	bert_tokenizer = DistilBertTokenizer.from_pretrained('bert_tokenizer')
	bert_fine_tuned_model = AutoModelForSequenceClassification.from_pretrained('bert_fine_tuned')
	pipeline = TextClassificationPipeline(model = bert_fine_tuned_model, tokenizer = bert_tokenizer, return_all_scores = True)
	
	#Loading in lstm tokenizer and model
	tokenizer = load_tokenizer('tokenizer.pkl')
	model_lstm = keras.models.load_model('lstm_token300_dim32_softmax.h5')

	#Loading in sentence similarity embedding model
	claim_model_bert = torch.load("claim_model_bert")


	#Dictionaries relating number label to written label
	label2class = {'False': 0, 'Mixture': 1, 'True': 2, 'Unproven': 3}
	class2label = {0: 'False', 1: 'Mixture', 2: 'True', 3: 'Unproven'}

	#Loading in the testing data and removing erroneous results
	testingData = pd.read_csv('TestDataWithSlicingDF')
	testingData = testingData[testingData["label"]!=-1]

	st.title("Public Health Fake News Analyzer")

	options = st.sidebar.selectbox("Choose a page", ["About","Fake News Prediction", "Similarity Matching", "Testing"])


	if options == "About":

		st.write("Creators: Alex Gui, Vivek Lam and Sathya Chitturi")
		st.write("Dataset Source: https://arxiv.org/abs/2010.09926")

		st.write(" Due to the nature and popularity of social networking sites, misinformation can propagate rapidly leading to widespread dissemination of misleading and even harmful information. A plethora of misinformation can make it hard for the public to understand what claims hold merit and which are baseless. This machine learning tool allows users to quickly learn whether an article contains fake news. The tabs in this website correspond to predictive fake news detection, user claim similarity matching, and model performance evaluation.")

		st.error("Disclaimer: This project is still a work in progress. The best source of information regarding fake news comes directly from verified fact-checkers.")

###### Tab 1 #####
	if options == "Fake News Prediction":

		#Setting other buttons states in different tabs to false
		st.subheader("Prediction on article")
		st.markdown(Path("predTab.md").read_text())
		

		user_input_type = st.selectbox("Select a method to input data", ['Text Box', 'URL'])

		if user_input_type == 'Text Box':
			text = st.text_area("Enter Text", "Type Here", key="predText")
			if text != 'Type Here':

				model_selected = st.selectbox("Select a model", ['Baseline LSTM', 'Fine-tuned BERT'])

				if model_selected == 'Baseline LSTM':

					if st.button("Analyze"):
						X = preprocess_text(tokenizer,[text])
						pred = model_lstm.predict(X)
						displayPredictedDf(pred)

				if model_selected == 'Fine-tuned BERT':

					if st.button("Analyze"):
						#truncates text so that the model will run
						text = text[:2000]
						pred = pipeline(text)
						pred = np.expand_dims(np.array([pred[0][0]["score"], pred[0][1]["score"], pred[0][2]["score"], pred[0][3]["score"]]), axis=0)
						displayPredictedDf(pred)

		elif user_input_type == 'URL':
			raw_url = st.text_input("Enter URL", "Type Here", key="predURL")
			if raw_url != 'Type Here':

				try:
				    text = get_text(raw_url)
				    textLoaded = True
				except:
				    st.error("Cannot parse url")
				    textLoaded = False

				if textLoaded: 
					model_selected = st.selectbox("Select a model", ['Baseline LSTM', 'Fine-tuned BERT'])

					if model_selected == 'Baseline LSTM':

						if st.button("Analyze"):

							X = preprocess_text(tokenizer,[text])
							pred = model_lstm.predict(X)
							displayPredictedDf(pred)


					if model_selected == 'Fine-tuned BERT':

						if st.button("Analyze"):
							#truncates text so that the model will run
							text = text[:2000]
							pred = pipeline(text)
							pred = np.expand_dims(np.array([pred[0][0]["score"], pred[0][1]["score"], pred[0][2]["score"], pred[0][3]["score"]]), axis=0)
							displayPredictedDf(pred)


########### Tab 2 #############
	if options == "Similarity Matching":

		st.subheader("Similar Claim Finder")
		st.markdown(Path("simTab.md").read_text())

		user_input_type = st.selectbox("Select a method to input data", ['Text Box', 'URL'])

		if user_input_type == 'Text Box':
			text = st.text_area("Enter Text", "Type Here", key="simText")
			if text != 'Type Here':

				model_selected = st.selectbox("Select a model", ['BERT Similarity'])

				if model_selected == 'BERT Similarity':

					if st.button("Analyze Claim"):

						article_split = sentence_splitter(text)
						# Take first 20 sentences 
						processed_article = (article_split)

						embedding_data = load_embedded_data("healthFactTrainData.pkl")

						matched_claims = []

						for sent in processed_article:
							claim_embed = claim_model_bert.encode(sent)
							sim_scores = cosine_similarity(embedding_data['claim_embedding'],claim_embed.reshape(1,-1))
							top = np.flip(np.argsort(sim_scores.flatten()))[:3]

							for idx in top:

								if round(sim_scores[idx].item(),3) > 0.8:
									if embedding_data['label'][idx] != -1:
										matched_claims.append([round(sim_scores[idx].item(),3), sent, embedding_data['claim'][idx], class2label[embedding_data['label'][idx]]])

						df = pd.DataFrame(matched_claims, columns = ['Similarity Score', 'Trigger Sentence in Article', 'Claim in Training Data', 'Claim Label'])  
						df = df.sort_values(by=['Similarity Score'],ascending=False)
						st.table(df.assign(hack='').set_index('hack'))


		elif user_input_type == 'URL':
			raw_url = st.text_input("Enter URL", "Type Here", key="simURL")
			if raw_url != 'Type Here':

				try:
				    text = get_text(raw_url)
				    textLoaded = True
				except:
				    st.error("Cannot parse url")
				    textLoaded = False


				if textLoaded: 
					model_selected = st.selectbox("Select a model", ['BERT Similarity'])

					if model_selected == 'BERT Similarity':

						if st.button("Analyze Claim"):

							article_split = sentence_splitter(text)
							# Take first 20 sentences 
							processed_article = (article_split)

							embedding_data = load_embedded_data("healthFactTrainData.pkl")

							matched_claims = []

							for sent in processed_article:
								claim_embed = claim_model_bert.encode(sent)
								sim_scores = cosine_similarity(embedding_data['claim_embedding'],claim_embed.reshape(1,-1))
								top = np.flip(np.argsort(sim_scores.flatten()))[:3]

								for idx in top:
									if round(sim_scores[idx].item(),3) > 0.8:
										if embedding_data['label'][idx] != -1:
											matched_claims.append([round(sim_scores[idx].item(),3), sent, embedding_data['claim'][idx], class2label[embedding_data['label'][idx]]])

							df = pd.DataFrame(matched_claims, columns = ['Similarity Score', 'Trigger Sentence in Article', 'Claim in Training Data', 'Claim Label'])  
							df = df.sort_values(by=['Similarity Score'],ascending=False)
							st.table(df.assign(hack='').set_index('hack'))

###### Tab 3 ######
	if options == "Testing":

		st.subheader("Prediction on testing data")
		st.markdown(Path("testTab.md").read_text())


		model_selected = st.selectbox("Select a model", ['Baseline LSTM'])

		if model_selected == 'Baseline LSTM':
			st.write("{0} Accuracy on whole Test Dataset: 66%".format(model_selected))

			user_input = st.selectbox("Slicing Type", ["Word Count", "Year Published", \
				"Average Sentence Length", "Percentage Punctuation"])

			if user_input == "Word Count":
				wordCountSplit = st.slider(label="Word Count Split", min_value=200, max_value=1300, step=1)

				lowerSplitWC = testingData[testingData["word_counts"]<=wordCountSplit]
				upperSplitWC = testingData[testingData["word_counts"]>wordCountSplit]

				if st.button("Generate Split Statistics"):

					accDf, totalaccuracyDF  = generateAccDfLSTM(lowerSplitWC, upperSplitWC, tokenizer, model_lstm, sampleSize = SAMPLE_SIZE)
					st.write(totalaccuracyDF)
					st.write(accDf)

			elif user_input == "Year Published":
				yearPublishedSplit = st.slider(label="Year Published Split", min_value=2010, max_value=2019, step=1)
				
				#Removing instances where the year is unavailable
				testingDataYP = testingData[testingData["year_published"]!=0]

				lowerSplitYP = testingDataYP[testingDataYP["year_published"]<=yearPublishedSplit]
				upperSplitYP = testingDataYP[testingDataYP["year_published"]>yearPublishedSplit]

				if st.button("Generate Split Statistics"):
					accDf, totalaccuracyDF = generateAccDfLSTM(lowerSplitYP, lowerSplitYP, tokenizer, model_lstm, sampleSize = SAMPLE_SIZE)
					st.write(totalaccuracyDF)
					st.write(accDf)


			elif user_input == "Average Sentence Length":
				avgSentLenSplit = st.slider(label="Average Sentence Length Split", min_value=101, max_value=199, step=1)
				
				#Removing instances where the year is unavailable
				testingDataASL = testingData[testingData["average_sentence_length"]!=0]

				lowerSplitASL = testingDataASL[testingDataASL["average_sentence_length"]<=avgSentLenSplit]
				upperSplitASL = testingDataASL[testingDataASL["average_sentence_length"]>avgSentLenSplit]
				
				if st.button("Generate Split Statistics"):
					accDf, totalaccuracyDF = generateAccDfLSTM(lowerSplitASL, upperSplitASL, tokenizer, model_lstm, sampleSize = SAMPLE_SIZE)
					st.write(totalaccuracyDF)
					st.write(accDf)


			elif user_input == "Percentage Punctuation":
				fracPuncSplit = st.slider(label="Percentage Punctuation Split", min_value=.12, max_value=.8, step=.01)
				
				#Removing instances where the year is unavailable
				testingDataFPS = testingData[testingData["percentage_punc_to_word"]!=1]

				lowerSplitFPS = testingDataFPS[testingDataFPS["percentage_punc_to_word"]<=fracPuncSplit]
				upperSplitFPS = testingDataFPS[testingDataFPS["percentage_punc_to_word"]>fracPuncSplit]

				if st.button("Generate Split Statistics"):
					accDf, totalaccuracyDF = generateAccDfLSTM(lowerSplitFPS, upperSplitFPS, tokenizer, model_lstm, sampleSize = SAMPLE_SIZE)
					st.write(totalaccuracyDF)
					st.write(accDf)

if __name__ == '__main__':

	main()
