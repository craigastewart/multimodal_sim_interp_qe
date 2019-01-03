from xgboost import XGBRegressor
import numpy as np
import argparse, os

data_dir = os.getcwd() + "/data/"

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
	raise IOError("No features selected, expected flag (-t/-i/-s)")


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


def main():
	X_data, y_data = compile_data(get_languages(args))
	print(X_data.shape)
	print(y_data.shape)


if __name__ == "__main__":
	main()
