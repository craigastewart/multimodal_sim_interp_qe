from xgboost import XGBRegressor
import numpy as np
import argparse, os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy import stats

data_dir = os.getcwd() + "/data/"
seeds = [632, 358, 201, 219, 638]
lang_dict = {'fr/':1, 'it/':2, 'ja/ja_b/':3, 'ja/ja_a/':4, 'ja/ja_s/':5}
lang_str = ''
feature_str = ''

# Trim to only features that are of mean or higher importance in at least 3/5 settings
qpp_feat_idx = [0, 1, 2, 3, 5, 6, 7, 9, 10, 12, 14, 15, 16, 17, 20, 22]
acl_feat_idx = [23, 27]
text_feat_idx = [0, 1, 2, 3, 5, 6, 7, 9, 10, 12, 14, 15, 16, 17, 20, 22, 23, 27]
interp_aux_idx = [1, 5, 7, 17, 18, 19, 22, 23, 29, 34, 37, 41]
src_aux_idx = [18, 20, 22, 28, 29, 37, 41]

parser = argparse.ArgumentParser()
parser.add_argument("-fr","--french", action="store_true", help="Use french language setting", required=False)
parser.add_argument("-it","--italian", action="store_true", help="Use italian language setting", required=False)
parser.add_argument("-jb","--japanese_b", action="store_true", help="Use japanese B rank language setting", required=False)
parser.add_argument("-ja","--japanese_a", action="store_true", help="Use japanese A rank language setting", required=False)
parser.add_argument("-js","--japanese_s", action="store_true", help="Use japanese S rank language setting", required=False)
parser.add_argument("-jt","--joint_training", action="store_true", help="Training and evaluation on all language settings", required=False)
parser.add_argument("-tf","--text_features", action="store_true", help="Use text features", required=False)
parser.add_argument("-ia","--interp_audio", action="store_true", help="Use interpreter audio features", required=False)
parser.add_argument("-sa","--src_audio", action="store_true", help="Use source audio features", required=False)
parser.add_argument("-tr","--trimmed", action="store_true", help="Use trimmed model", required=False)
parser.add_argument("-ss","--subsample", help="Subsample parameter value for XGBRegressor", default=1, required=False)
parser.add_argument("-lr","--learning_rate", help="Learning rate value for XGBRegressor", default=0.1, required=False)
parser.add_argument("-la","--lambda", help="Lambda parameter (L2 regularization) value for XGBRegressor", default=1, required=False)
parser.add_argument("-ne","--n_estimators", help="Number of estimators for XGBRegressor", default=100, required=False)
parser.add_argument("-cb","--colsample_bytree", help="colsample_bytree parameter (analogous to max_features) value for XGBRegressor", default=1, required=False)
parser.add_argument("-t","--test", action="store_true", help="Evaluate on test set", required=False)
parser.add_argument("-a","--asr", action="store_true", help="Use ASR transcripts", required=False)
parser.add_argument("-tu","--tuned", action="store_true", help="Use predefined tuned parameter settings for XGBRegressor", required=False)
args = vars(parser.parse_args())

if args["tuned"]:
	# Pre-tuned parameters
	subsample = 0.8
	learning_rate = 0.09
	colsample_bytree = 0.6
	reg_lambda = 0.95
	n_estimators = 400

else:
	subsample = float(args["subsample"])
	learning_rate = float(args["learning_rate"])
	colsample_bytree = float(args["colsample_bytree"])
	reg_lambda = float(args["lambda"])
	n_estimators = int(args["n_estimators"])

if not args["text_features"] and not args["interp_audio"] and not args["src_audio"]:
	raise IOError("No features selected, expected flag (-tf/-ia/-sa)")

if args["text_features"]:
	feature_str += 't'
if args["interp_audio"]:
	feature_str += 'i'
if args["src_audio"]:
	feature_str += 's'

if args["joint_training"]:
	args["french"] = True
	args["italian"] = True
	args["japanese_b"] = True
	args["japanese_a"] = True
	args["japanese_s"] = True


def get_languages(arguments):
	langs = []
	if arguments["french"]:
		langs.append("fr/")
	if arguments["italian"]:
		langs.append("it/")
	if arguments["japanese_b"]:
		langs.append("ja/ja_b/")
	if arguments["japanese_a"]:
		langs.append("ja/ja_a/")
	if arguments["japanese_s"]:
		langs.append("ja/ja_s/")
	if len(langs) == 0:
		raise IOError("No langauge setting selected, expected flag (-fr/-it/-jb/-ja/-js)")
	return langs


def compile_data(languages):
	X_data = None
	y_data = None
	if len(languages) == 1:
		path = data_dir + languages[0]
		lang_label = lang_dict[languages[0]]
		features = []
		if args["text_features"]:
			if args["asr"]:
				if args["trimmed"]:
					features.append(np.loadtxt(path + "text_features_asr.tsv", delimiter='\t')[:, text_feat_idx])
				else:
					features.append(np.loadtxt(path + "text_features_asr.tsv", delimiter='\t'))
			else:
				if args["trimmed"]:
					features.append(np.loadtxt(path + "text_features.tsv", delimiter='\t')[:, text_feat_idx])
				else:
					features.append(np.loadtxt(path + "text_features.tsv", delimiter='\t'))
		if args["interp_audio"]:
			if args["trimmed"]:
				features.append(np.loadtxt(path + "interp_audio.tsv", delimiter='\t')[:, interp_aux_idx])
			else:
				features.append(np.loadtxt(path + "interp_audio.tsv", delimiter='\t'))
		if args["src_audio"]:
			if args["trimmed"]:
				features.append(np.loadtxt(path + "src_audio.tsv", delimiter='\t')[:, src_aux_idx])
			else:
				features.append(np.loadtxt(path + "src_audio.tsv", delimiter='\t'))
		ind_labels = np.arange(1, features[0].shape[0]+1)
		ind_labels = np.reshape(ind_labels, (-1, 1))
		lang_labels = np.zeros((features[0].shape[0],1)) + lang_label
		features.append(ind_labels)
		features.append(lang_labels)
		X_data = np.hstack(features)
		y_data = np.loadtxt(path + "meteor_scores.tsv", delimiter='\t')
	else:
		X_datas = []
		y_datas = []
		for lang in languages:
			path = data_dir + lang
			lang_label = lang_dict[lang]
			features = []
			if args["text_features"]:
				if args["asr"]:
					if args["trimmed"]:
						features.append(np.loadtxt(path + "text_features_asr.tsv", delimiter='\t')[:, text_feat_idx])
					else:
						features.append(np.loadtxt(path + "text_features_asr.tsv", delimiter='\t'))
				else:
					if args["trimmed"]:
						features.append(np.loadtxt(path + "text_features.tsv", delimiter='\t')[:, text_feat_idx])
					else:
						features.append(np.loadtxt(path + "text_features.tsv", delimiter='\t'))
			if args["interp_audio"]:
				if args["trimmed"]:
					features.append(np.loadtxt(path + "interp_audio.tsv", delimiter='\t')[:, interp_aux_idx])
				else:
					features.append(np.loadtxt(path + "interp_audio.tsv", delimiter='\t'))
			if args["src_audio"]:
				if args["trimmed"]:
					features.append(np.loadtxt(path + "src_audio.tsv", delimiter='\t')[:, src_aux_idx])
				else:
					features.append(np.loadtxt(path + "src_audio.tsv", delimiter='\t'))
			ind_labels = np.arange(1, features[0].shape[0]+1)
			ind_labels = np.reshape(ind_labels, (-1, 1))
			lang_labels = np.zeros((features[0].shape[0],1)) + lang_label
			features.append(ind_labels)
			features.append(lang_labels)
			X_datas.append(np.hstack(features))
			y_datas.append(np.loadtxt(path + "meteor_scores.tsv", delimiter='\t'))
		X_data = np.vstack(X_datas)
		y_data = np.hstack(y_datas)
	return X_data, y_data


def shuffle_data(X_data, y_data):
	np.random.seed(7)
	permutation = np.random.permutation(X_data.shape[0])
	X_shuf = X_data[permutation]
	y_shuf = y_data[permutation]
	return X_shuf, y_shuf, permutation


def normalize(X_train, X_test):
	train_mean = X_train.mean(axis=0)
	train_std = X_train.std(axis=0)
	if not train_std.all():
		train_std += 1e-8
	X_train = (X_train - train_mean)/train_std
	X_test = (X_test - train_mean)/train_std
	return np.nan_to_num(X_train), np.nan_to_num(X_test)


def main():
	langs = get_languages(args)
	X_data, y_data = compile_data(langs)
	X_data, y_data, permutation = shuffle_data(X_data, y_data)
	kf = KFold(n_splits=10)

	preds = []
	truths = []
	joint_preds = {'1':[], '2':[], '3':[], '4':[], '5':[]}
	joint_truths = {'1':[], '2':[], '3':[], '4':[], '5':[]}

	for train_index, test_index in kf.split(X_data):
		X_train, X_split = X_data[train_index], X_data[test_index]
		y_train, y_split = y_data[train_index], y_data[test_index]
		X_dev, X_test, y_dev, y_test, dev_indices, test_indices = train_test_split(X_split, y_split, test_index, test_size=0.5, random_state=47)
		if not args["test"]:
			X_test = X_dev
			y_test = y_dev
			test_indices = dev_indices

		# Need to keep a copy of data before normalization to track zero length utterances
		X_train_raw = X_train
		X_test_raw = X_test

		X_train, X_test = normalize(X_train, X_test)

		for seed in seeds:
			clf = XGBRegressor(random_state=seed, subsample=subsample, learning_rate=learning_rate, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, n_estimators=n_estimators)
			clf.fit(X_train, y_train, eval_metric='mae')
			y_hat = clf.predict(X_test)

			# Adjust prediction for zero length utterances
			if args["text_features"]:
				for i in range(0, len(y_hat)):
					if X_test_raw[i][0] == 0 or X_test_raw[i][1] == 0:
						y_hat[i] = 0.0

			for i in range(len(y_hat)):
				key = str(int(X_test_raw[i][X_test_raw.shape[1]-1]))
				joint_preds[key].append(y_hat[i])
				joint_truths[key].append(y_test[i])

			preds.append(y_hat)
			truths.append(y_test)
			with open('feature_importances.csv', 'ab') as feat_file:
				np.savetxt(feat_file, np.reshape(clf.feature_importances_, (-1, 1)).transpose(), delimiter=',')

	fr_result = None
	it_result = None
	jb_result = None
	ja_result = None
	js_result = None

	if len(joint_preds['1']) > 0:
		fr_result = stats.pearsonr(joint_preds['1'], joint_truths['1'])[0]
	if len(joint_preds['2']) > 0:
		it_result = stats.pearsonr(joint_preds['2'], joint_truths['2'])[0]
	if len(joint_preds['3']) > 0:
		jb_result = stats.pearsonr(joint_preds['3'], joint_truths['3'])[0]
	if len(joint_preds['4']) > 0:
		ja_result = stats.pearsonr(joint_preds['4'], joint_truths['4'])[0]
	if len(joint_preds['5']) > 0:
		js_result = stats.pearsonr(joint_preds['5'], joint_truths['5'])[0]

	preds = np.hstack(preds)
	truths = np.hstack(truths)
	result = stats.pearsonr(preds, truths)[0]
	out_str = ''
	lang_str = ''
	for lang in langs:
		lang_str += lang
	out_str += "{},{},{},{},{},{},{},{}".format(lang_str, feature_str, learning_rate, subsample, colsample_bytree, reg_lambda, n_estimators, result)
	with open('log.csv', 'a') as log_file:
		log_file.write(out_str + '\n')
	print("{} :: {} :: {}".format(lang_str, feature_str, result))

	if fr_result:
		print("FR :: {}\n".format(fr_result))
	if it_result:
		print("IT :: {}\n".format(it_result))
	if jb_result:
		print("JB :: {}\n".format(jb_result))
	if ja_result:
		print("JA :: {}\n".format(ja_result))
	if js_result:
		print("JS :: {}\n".format(js_result))

if __name__ == "__main__":
	main()
