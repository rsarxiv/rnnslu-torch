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
cmd:text('Train a slot filling model based Elman-RNN')
cmd:text()
cmd:text('Options')

cmd:option("-rnn_size",200,"")
cmd:option("-learning_rate",2e-3,"")
cmd:option("-decay_rate",0.95,"")
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option("-seed",123,"")
cmd:option("-max_epochs",1,"")
cmd:option("-print_every",100,"")
cmd:option("-win",15,"")
cmd:option("-datafile","data/trainset.hdf5","")
cmd:option("-worddict","data/words.dict","")
cmd:option("-labeldict","data/labels.dict","")
cmd:option("-savefile","data/slu.t7","")
cmd:option("-save_every",1000,"")

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

-- define model one time step, then clone them

-- x,y = dataset:nextBatch()

local protos = {}
protos.embed = Embedding(opt.word_size + 2,opt.rnn_size) -- word_size + 1 including -1 
protos.rnn = RNN.rnn(opt)
-- -- output
protos.softmax = nn.Sequential():add(nn.Linear(opt.rnn_size, opt.label_size)):add(nn.LogSoftMax())
-- -- criterion 
protos.criterion = nn.ClassNLLCriterion() 
-- put the above things into one flattened parameters tensor
local params, grad_params = model_utils.combine_all_parameters(protos.embed, protos.rnn,protos.softmax)
params:uniform(-0.08, 0.08)

print('number of parameters in the model: ' .. params:nElement())

-- make a bunch of clones, AFTER flattening, as that reallocates memory
local clones = {}
for name,proto in pairs(protos) do
    print('cloning '..name)
    clones[name] = model_utils.clone_many_times(proto, opt.source_length, not proto.parameters)
end

-- initial state (zero initially)
local h0 = torch.zeros(opt.rnn_size)
-- decoder final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
local dfinalstate_h = h0:clone()

-- do fwd/bwd and return loss, grad_params
function feval()

    grad_params:zero()
    ------------------ get minibatch -------------------
    x,y = dataset:nextBatch()
    x = x:add(2) -- clear the index 0 and -1, start the index from 1
    y = y:add(1) -- start the index from 1
    ------------------- forward pass -------------------
    local embeddings = {}            -- embeddings
    local h = {[0]=h0} -- internal hidden h
    local predictions = {}           -- softmax outputs
    local loss =  0
    --forward pass
    for t=1,x:size()[1] do
        embeddings[t] = clones.embed[t]:forward(x[t]):resize(opt.rnn_size*opt.win)
        h[t] = clones.rnn[t]:forward{embeddings[t],h[t-1]}
        predictions[t] = clones.softmax[t]:forward(h[t])
        loss = loss + clones.criterion[t]:forward(predictions[t],y[t])
    end

    loss = loss / x:size()[1]
    -- print("loss = "..loss)

    ---------------- backward pass -------------------

    local dh_embeddings = {}                              -- d loss / d input embeddings
    local dh = {}--{[opt.source_length]=dencfinalstate_h}

    for t=y:size()[1],1,-1 do
    	local doutput_t = clones.criterion[t]:backward(predictions[t],y[t])
    	if t == y:size()[1] then
        	dh[t] = clones.softmax[t]:backward(h[t],doutput_t)
    	else
    		dh[t]:add(clones.softmax[t]:backward(h[t],doutput_t))
    	end
    	dh_embeddings[t],dh[t-1] = unpack(clones.rnn[t]:backward(embeddings[t],dh[t]))
    	clones.embed[t]:backward(x[t],dh_embeddings[t])
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    h0:copy(h[#h])
    -- -- clip gradient element-wise
    grad_params:clamp(-1, 1)

    return loss, grad_params

end

-- optimization stuff
local losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * dataset.source:size(1)
print("totally needs training iterations "..iterations)
for i = 1, iterations do
    local epoch = i / dataset.source:size(1)
    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, params, optim_state) -- rmsprop
    local time = timer:time().real
    losses[#losses + 1] = loss[1]
    -- exponential learning rate decay
    if i % dataset.source:size(1) == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end
    if i % opt.save_every == 0 then
    	print("saving model...")
        torch.save(opt.savefile, protos)
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, loss[1], grad_params:norm() / params:norm(), time))
    end
end









