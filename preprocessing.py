#!/usr/bin/env python
# -*- coding: utf-8 -*-
# data preprocessing part written by python
import cPickle
import h5py
import numpy as np

trainset, testset, dicts = cPickle.load(open("data/atis.pkl"))
word2idx, label2idx = dicts['words2idx'], dicts['labels2idx']
word2idx = sorted(word2idx.items(), key=lambda d: d[1])
label2idx = sorted(label2idx.items(), key=lambda d: d[1])

def buildWordDict():
	with file("data/words.dict","w") as f:
		for (k,v) in word2idx:
			f.write(k)
			f.write("\n")
	f.close()

def buildLabelDict():
	with file("data/labels.dict","w") as f:
		for (k,v) in label2idx:
			f.write(k)
			f.write("\n")
	f.close()

def buildTrainSet():
	wordlist = trainset[0]
	labellist = trainset[2]
	lens = [len(wlist) for wlist in wordlist]
	max_len = max(lens)
	wordlist = [wlist.tolist() + [-1]*(max_len-len(wlist)) for wlist in wordlist]
	labellist = [llist.tolist() + [-1]*(max_len-len(llist)) for llist in labellist]
	f = h5py.File("data/trainset.hdf5","w")
	f["words"] = np.array(wordlist)
	f["labels"] = np.array(labellist)
	f.close()
	
	
if __name__ == "__main__":
	pass
	#buildWordDict()
	#buildLabelDict()
	#buildTrainSet()

