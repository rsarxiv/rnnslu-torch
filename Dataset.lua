--[[

Dataset Class

]]--

local Dataset = torch.class("Dataset")

function Dataset:__init(opt)
  local dataset = hdf5.open(opt.datafile,"r")
  self.source = dataset:read("words"):all()
  self.target  = dataset:read("labels"):all()
  self.batchId = 1
  self.words = self:buildWordDict(opt)
  self.labels = self:buildLabelDict(opt)
end

function Dataset:buildWordDict(opt)
  local id2word = {}
  local word2id = {}
  local f = torch.DiskFile(opt.worddict, "r")
    f:quiet()
    local word =  f:readString("*l") -- read file by line
    while word ~= '' do
        id2word[#id2word+1] = word
        word2id[word] = #id2word
        word = f:readString("*l")
    end
    return {["id2word"]=id2word,["word2id"]=word2id,["size"]=#id2word}
end

function Dataset:buildLabelDict(opt)
  local id2label = {}
  local label2id = {}
  local f = torch.DiskFile(opt.labeldict, "r")
    f:quiet()
    local label =  f:readString("*l") -- read file by line
    while label ~= '' do
        id2label[#id2label+1] = label
        label2id[label] = #id2label
        label = f:readString("*l")
    end
    return {["id2label"]=id2label,["label2id"]=label2id,["size"]=#id2label}
end

function Dataset:contextWin(tt,opt)
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
    local a = list:range(i,opt.win+i-1)
    print(a)
    --table.insert(result,a)
  end
  return torch.Tensor(result)
end

function Dataset:nextBatch()
   local source = {}
   local target = {}
   local size = self.source[self.batchId]:size()[1]
   for i=1,size do
      if self.source[self.batchId][i] ~= -1 then
        table.insert(source,self.source[self.batchId][i])
        table.insert(target,self.target[self.batchId][i])
      end
   end

   self.batchId = self.batchId + 1
   return self:contextWin(source,opt)
end