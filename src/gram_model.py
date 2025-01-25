import numpy as np
import torch
import torch.nn.functional as F
import random

## Inputs
  ## dataset - of the form where it will have a column labeled 'text' which is a string of text
  ## dataString - all data rows in dataset have been combined into a single string
  ## n1 - the first split of the dataset into a training block
  ## n2 - the second split of the dataset to change remaining into a dev and test set
## Returns
  ## X_train - list of trigrams that appear
  ## Y_train - the character that appears after the trigram in X_train
  ## ctoi - a dictionary where the key is the character and the value is the number index
  ## itoc - a flipped dictionary of ctoi where the key is the index and the value is the character
def buildDataset(dataset, dataString, block_size):
    ctoi = {ch: i for i, ch in enumerate(sorted(set(dataString)))}
    itoc = {v:k for k, v in ctoi.items()}

    stop_index = ctoi['.']

    X_train, Y_train = _build_dataset(dataset['text'], block_size, stop_index, ctoi)

    return X_train, Y_train, ctoi, itoc

def _build_dataset(words, block_size, stop_index, ctoi):
  x = []
  y = []

  for word in words:
    context = [stop_index] * block_size
    for ch in word:
      idx = ctoi[ch]
      x.append(context)
      y.append(idx)
      context = context[1:] + [idx]

  X = torch.tensor(x)
  Y = torch.tensor(y)
  print(X.shape, Y.shape)
  return X, Y


class gramModel():
  def __init__(self, block_size, vocab_size, generator = torch.Generator().manual_seed(2147483647), stop_char = ".") -> None:
    self.vocab_size = vocab_size
    self.block_size = block_size
    self.embd = torch.randn((vocab_size, 10), generator = generator)
    self.W1 = torch.randn((10*block_size, 100), generator = generator)
    self.B1 = torch.randn((100), generator = generator)
    self.W2 = torch.randn((100, 200), generator = generator)
    self.B2 = torch.randn((200), generator = generator)
    self.W3 = torch.randn((200, vocab_size), generator = generator)
    self.B3 = torch.randn((vocab_size), generator = generator)

    self.generator = generator


    self.parameters = [self.embd, self.W1, self.B1, self.W2, self.B2, self.W3, self.B3]
    for p in self.parameters:
      p.requires_grad = True

  def train(self, X_train, Y_train, learning_rate = 0.1, epochs = 1000, verbose = False):
    loss_t = []
    step_t = []

    for step in range(epochs):
      idx = torch.randint(0, X_train.shape[0], (64,))

      embd_input = self.embd[X_train][idx]
      self.h1 = torch.tanh((embd_input.view(-1, 10*self.block_size) @ self.W1) + self.B1)
      self.h2 = torch.tanh((self.h1 @ self.W2) + self.B2)
      logits = (self.h2 @ self.W3) + self.B3
      loss = F.cross_entropy(logits, Y_train[idx])

      for para in self.parameters:
        para.grad = None

      loss.backward()

      for para in self.parameters:
        para.data += -learning_rate * para.grad

      step_t.append(step)
      loss_t.append(loss.item())
    if verbose:
      return step_t, loss_t

  def get_prob(self, startingChar):
    context = startingChar
    embd_input = self.embd[torch.tensor([context])]
    h1 = torch.tanh((embd_input.view(1,-1) @ self.W1) + self.B1)
    h2 = torch.tanh((h1 @ self.W2 )+ self.B2)
    logits = (h2 @ self.W3 )+ self.B3
    prob = F.softmax(logits, dim=1)

    return prob

  def forward(self, startingChar, itoc) -> None:
    gen_text = []
    context = startingChar

    embd_input = self.embd[torch.tensor([context])]
    h1 = torch.tanh((embd_input.view(1,-1) @ self.W1) + self.B1)
    h2 = torch.tanh((h1 @ self.W2 )+ self.B2)
    logits = (h2 @ self.W3 )+ self.B3
    prob = F.softmax(logits, dim=1)
    idx_list = torch.multinomial(prob, num_samples=3, replacement=False, generator=self.generator)
    for idx in idx_list.squeeze().tolist():
      gen_text.append(itoc[idx])

    return gen_text

## needs to be given both a starting text to predict from
## and the index -> char dictionary
  def __call__(self, startChar, itoc) -> None:
    return self.forward(startChar)
  
def processTestInput(ctoi, text, block_size):
  if len(text) < block_size:
    arr = [ctoi['.']] * (block_size - len(text))
    arr.extend([ctoi.get(ch, ctoi['.']) for ch in text])
    return arr

  if len(text) > block_size:
    text = text[-block_size:]
  return [ctoi.get(ch, ctoi['.']) for ch in text]

