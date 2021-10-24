import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data = [
    ("me gusta comer en la cafeteria".split(), "SPANISH"),
    ("Give it to me".split(), "ENGLISH"),
    ("No creo que sea una buena idea".split(), "SPANISH"),
    ("No it is not a good idea to get lost at sea".split(), "ENGLISH")
]
test_data = [
    ("Yo creo que si".split(), "SPANISH"),
    ("it is lost on me".split(), "ENGLISH")
]

# print(data)

word_to_index = {}

for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)


# print(word_to_index)


class BoWClassfier(nn.Module):
    def __init__(self, vocab_size, num_labels):
        super(BoWClassfier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim=1)


def make_bow_vector(sentense, word_to_index):
    vec = torch.zeros(len(word_to_index))
    for word in sentense:
        vec[word_to_index[word]] += 1
    return vec.view(1, -1)


label_to_index = {"ENGLISH": 0, "SPANISH": 1}


def make_target(label, label_to_index):
    return torch.LongTensor(label_to_index[label])


VOCAB_SIZE = len(word_to_index)
NUM_LABELS = 2
model = BoWClassfier(VOCAB_SIZE, NUM_LABELS)

# for param in model.parameters():
#     print(param)

# print(model)

with torch.no_grad():
    sample = data[0]
    bow_vector = make_bow_vector(sample[0], word_to_index)
    log_probs = model(bow_vector)
    print(log_probs)
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_function = nn.NLLLoss()
for epoch in range(5):
    for elem in data:
        model.zero_grad()
        bow_vec = make_bow_vector(elem[0], word_to_index)
        target = make_target(elem[1], label_to_index)
        log_probs = model(bow_vec)
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()
with torch.no_grad():
    sample = test_data[0]
    bow_vec = make_bow_vector(sample[0], word_to_index)
    log_probs = model(bow_vec)
    print(log_probs)
print(next(model.parameters())[:, word_to_index["creo"]])
