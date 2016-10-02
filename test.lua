require 'nngraph'
require 'nn'
require 'torch'
require 'optim'
require 'hdf5'
require 'Embedding'
require 'Dataset' 
local RNN = require 'RNN'
local model_utils = require 'model_utils'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Test a slot filling model based Elman-RNN')
cmd:text()
cmd:text('Options')

cmd:option("-rnn_size",200,"")
cmd:option("-win",15,"")
cmd:option("-datafile","data/testset.hdf5","")
cmd:option("-worddict","data/words.dict","")
cmd:option("-labeldict","data/labels.dict","")
cmd:option("-modelfile","data/slu.t7","")

opt = cmd:parse(arg)
-- dataset
local dataset = Dataset(opt)
opt.word_size = dataset.words.size
opt.label_size = dataset.labels.size
opt.source_length = dataset.source:size(2)
opt.target_length = dataset.target:size(2)
print("word vocabulary size is ".. opt.word_size)
print("label vocabulary size is ".. opt.label_size)
print("source and target length is "..opt.source_length)

protos = torch.load(opt.modelfile)

function feval()
	x_all = dataset.source
	y_all = dataset.target

	local h0 = torch.zeros(opt.rnn_size)
	local embeddings = {}
	local h = {[0]=h0}
	local predictions = {}
	local loss = 0.0
	local count = 0

	for i=1,x_all:size()[1] do -- outter loop
		x = x_all[i]:clone()
		y = y_all[i]:clone()
		for t=1,x:size()[1] do -- inner loop
			if x[t] ~= -1 then
				x[t] = x[t] + 2
				y[t] = y[t] + 1
				local input = torch.Tensor({x[t]})
       			embeddings[t] = protos.embed:forward(input):resize(opt.rnn_size*opt.win)
        		h[t] = protos.rnn:forward{embeddings[t],h[t-1]}
        		predictions[t] = protos.softmax:forward(h[t])
        		loss = loss + protos.criterion:forward(predictions[t],y[t])
        		count = count + 1
        	end
		end
	end

	loss = loss / count
	print("loss: " .. loss)

end

feval()









