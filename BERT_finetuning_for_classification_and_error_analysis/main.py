import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import numpy as np

torch.random.manual_seed(8942764)
torch.cuda.manual_seed(8942764)
np.random.seed(8942764)

device = 'cuda:0'

!pip install transformers
!pip install datasets

from transformers import AutoTokenizer, BertModel
from datasets import load_dataset

dataset = load_dataset("christinacdl/clickbait_notclickbait_dataset")
dataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print(dataset["train"][8])
print(dataset["validation"][6])
print(dataset["test"][0])

print('Original: ', dataset['train'][8]['text'])

print('Tokenized: ', tokenizer.tokenize(dataset['train'][8]['text']))

print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dataset['train'][8]['text'])))

def tokenize(batch):
  sentences = [x['text'] for x in batch]
  labels = torch.LongTensor([x['label'] for x in batch])
  new_batch = dict(tokenizer(sentences, padding=True, truncation=True, return_tensors="pt"))
  new_batch['label'] = labels
  return new_batch

@torch.no_grad()
def evaluate(model, dataset, batch_size, device, collate_fn=None):
  model = model.eval().to(device)
  dataloader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=collate_fn)
  lossfn = nn.NLLLoss()

  loss_history = []
  acc_history = []
  for i, batch in enumerate(dataloader):
      batch = {k:v.to(device) for k,v in batch.items() if isinstance(v, torch.Tensor)}
      y = batch.pop('label')

      logits = model(**batch)
      loss = lossfn(logits, y)

      pred = logits.argmax(1)
      acc = (pred == y).float().mean()
      loss_history.append(loss.item())
      acc_history.append(acc.item())
  return np.mean(loss_history), np.mean(acc_history)

@torch.no_grad()
def evaluate(model, dataset, batch_size, device, collate_fn=None):
  model = model.eval().to(device)
  dataloader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=collate_fn)
  lossfn = nn.NLLLoss()

  loss_history = []
  acc_history = []
  for i, batch in enumerate(dataloader):
      batch = {k:v.to(device) for k,v in batch.items() if isinstance(v, torch.Tensor)}
      y = batch.pop('label')

      logits = model(**batch)
      loss = lossfn(logits, y)

      pred = logits.argmax(1)
      acc = (pred == y).float().mean()
      loss_history.append(loss.item())
      acc_history.append(acc.item())
  return np.mean(loss_history), np.mean(acc_history)

def train(model,
          train_dataset,
          val_dataset,
          num_epochs,
          batch_size,
          optimizer_cls,
          lr,
          weight_decay,
          device,
          collate_fn=None):
  model = model.train().to(device)
  dataloader = DataLoader(train_dataset, batch_size, shuffle=True,
                          collate_fn=collate_fn)

  if optimizer_cls == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay=weight_decay)
  elif optimizer_cls == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)

  train_loss_history = []
  train_acc_history = []
  val_loss_history = []
  val_acc_history = []

  lossfn = nn.NLLLoss()
  for e in range(num_epochs):
    epoch_loss_history = []
    epoch_acc_history = []
    for i, batch in enumerate(dataloader):
      batch = {k:v.to(device) for k,v in batch.items() if isinstance(v, torch.Tensor)}
      y = batch.pop('label')

      logits = model(**batch)
      loss = lossfn(logits, y)

      pred = logits.argmax(1)
      acc = (pred == y).float().mean()

      epoch_loss_history.append(loss.item())
      epoch_acc_history.append(acc.item())

      if (i % 100 == 0):
        print(f'epoch: {e}\t iter: {i}\t train_loss: {np.mean(epoch_loss_history):.3e}\t train_accuracy:{np.mean(epoch_acc_history):.3f}')
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    val_loss, val_acc = evaluate(model, val_dataset, batch_size, device, collate_fn=collate_fn)

    train_loss_history.append(np.mean(epoch_loss_history))
    train_acc_history.append(np.mean(epoch_acc_history))
    val_loss_history.append(val_loss.item())
    val_acc_history.append(val_acc.item())
    print(f'epoch: {e}\t train_loss: {train_loss_history[-1]:.3e}\t train_accuracy:{train_acc_history[-1]:.3f}\t val_loss: {val_loss_history[-1]:.3e}\t val_accuracy:{val_acc_history[-1]:.3f}')

  return model, (train_loss_history, train_acc_history, val_loss_history, val_acc_history)

class BertForTextClassification(nn.Module):
  def __init__(self, bert_pretrained_config_name, num_classes, freeze_bert=False):
    super().__init__()
    self.bert = BertModel.from_pretrained(bert_pretrained_config_name)
    self.bert.requires_grad_(not freeze_bert)
    self.classifier = nn.Sequential(
        nn.Linear(self.bert.config.hidden_size, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
        nn.LogSoftmax(dim=-1)
    )

  def forward(self, **bert_kwargs):
     output=self.bert(**bert_kwargs)
     cls_embed = output.pooler_output
     logits = self.classifier(cls_embed)
     return logits

bert_cls = BertForTextClassification('bert-base-uncased', 2, freeze_bert=True)

print(f'num_trainable_params={sum([p.numel() for p in bert_cls.parameters() if p.requires_grad])}\n')

bert_cls, bert_cls_logs = train(bert_cls, dataset['train'], dataset['validation'],
                                num_epochs=8, batch_size=32, optimizer_cls='Adam',
                                lr=1e-3, weight_decay=1e-4, device=device,
                                collate_fn=tokenize)

print('\n')
print('Starting test run')
test_loss, test_acc=evaluate(bert_cls,dataset['test'],batch_size=32, device=device, collate_fn=tokenize)
print(f'Test Complete.\t Test Loss: {test_loss:.3e}\t Test Accuracy: {test_acc:.3f}')