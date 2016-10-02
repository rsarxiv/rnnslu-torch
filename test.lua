require 'nngraph'
require 'nn'
require 'torch'
require 'optim'
require 'hdf5'
require 'Embedding'
require 'Dataset' 
local RNN = require 'RNN'

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
	local correct = 0
	for i=1,x_all:size()[1] do -- outter loop
		_x = x_all[i]:clone()
		x = {}
		for t=1,_x:size()[1] do
			if _x[t] ~= -1 then
				table.insert(x,_x[t])
			end
		end
		x = contextWin(x,opt)
		y = y_all[i]:clone()
		x = x:add(2)
		y = y:add(1)
		for t=1,x:size()[1] do -- inner loop
       		embeddings[t] = protos.embed:forward(x[t]):resize(opt.rnn_size*opt.win)
        	h[t] = protos.rnn:forward{embeddings[t],h[t-1]}
        	predictions[t] = protos.softmax:forward(h[t])
        	_,pindex = predictions[t]:topk(1,true)
        	if pindex[1] == y[t] then
        		correct = correct + 1
        	end
        	-- loss = loss + protos.criterion:forward(predictions[t],y[t])
        	count = count + 1
		end
	end
	loss = 1.0 * correct / count
	print("loss: " .. loss)

end

function contextWin(tt,opt)
-- win = opt.win
  local start = {}
  local step = math.floor(opt.win / 2)
  for i=1,step do
    table.insert(start,-1)
  end
  local s = torch.Tensor(start)
  local e = s:clone()
  local t = torch.Tensor(tt)
  local list = s:cat(t):cat(e)
  local result = {}
  local size = #tt
  for i=1,size do
    local a = list:narrow(1,i,opt.win)
    table.insert(result,a:totable())
  end
  return torch.Tensor(result)
end

feval()









