from xgboost import XGBRegressor
import numpy as np
import argparse, os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy import stats

data_dir = os.getcwd() + "/data/"
seeds = [632, 358, 201, 219, 638]

parser = argparse.ArgumentParser()
parser.add_argument("-fr","--french", action="store_true", help="Use french language setting", required=False)
parser.add_argument("-it","--italian", action="store_true", help="Use italian language setting", required=False)
parser.add_argument("-jb","--japanese_b", action="store_true", help="Use japanese B rank language setting", required=False)
parser.add_argument("-ja","--japanese_a", action="store_true", help="Use japanese A rank language setting", required=False)
parser.add_argument("-js","--japanese_s", action="store_true", help="Use japanese S rank language setting", required=False)
parser.add_argument("-tf","--text_features", action="store_true", help="Use text features", required=False)
parser.add_argument("-ia","--interp_audio", action="store_true", help="Use interpreter audio features", required=False)
parser.add_argument("-sa","--src_audio", action="store_true", help="Use source audio features", required=False)
parser.add_argument("-t","--test", action="store_true", help="Evaluate on test set", required=False)
args = vars(parser.parse_args())

if not args["text_features"] and not args["interp_audio"] and not args["src_audio"]:
	raise IOError("No features selected, expected flag (-tf/-ia/-sa)")


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
		features = []
		if args["text_features"]:
			features.append(np.loadtxt(path + "text_features.tsv", delimiter='\t'))
		if args["interp_audio"]:
			features.append(np.loadtxt(path + "interp_audio.tsv", delimiter='\t'))
		if args["src_audio"]:
			features.append(np.loadtxt(path + "src_audio.tsv", delimiter='\t'))
		X_data = np.hstack(features)
		y_data = np.loadtxt(path + "meteor_scores.tsv", delimiter='\t')
	else:
		X_datas = []
		y_datas = []
		for lang in languages:
			path = data_dir + lang
			features = []
			if args["text_features"]:
				features.append(np.loadtxt(path + "text_features.tsv", delimiter='\t'))
			if args["interp_audio"]:
				features.append(np.loadtxt(path + "interp_audio.tsv", delimiter='\t'))
			if args["src_audio"]:
				features.append(np.loadtxt(path + "src_audio.tsv", delimiter='\t'))
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
	X_data, y_data = compile_data(get_languages(args))
	X_data, y_data, permutation = shuffle_data(X_data, y_data)
	kf = KFold(n_splits=10)
	results = []

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
		split_results = []
		for seed in seeds:
			clf = XGBRegressor(random_state=seed, subsample=0.5)
			clf.fit(X_train, y_train, eval_metric='mae')
			y_hat = clf.predict(X_test)

			# Adjust prediction for zero length utterances
			if args["text_features"]:
				for i in range(0, len(y_hat)):
					if X_test_raw[i][0] == 0 or X_test_raw[i][1] == 0:
						y_hat[i] = 0.0

			pearson = stats.pearsonr(y_test, y_hat)[0]
			split_results.append(pearson)
		results.append(np.mean(split_results))
		print(split_results)
	print(np.mean(results))
	# print(results)


if __name__ == "__main__":
	main()
