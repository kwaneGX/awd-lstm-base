import torch
import matplotlib.pyplot as plt
# import seaborn as sns


def set_writer(wr):
    global writer
    writer = wr


def set_dict(idx2word, si=0):
    global vocab
    global seq_idx
    vocab = idx2word
    seq_idx = si


def set_sequence(ids):
    global sequence
    global seq_probs
    sequence = list(map(lambda x: vocab[x], ids))
    seq_probs = []


def set_prob(att_probs):
    if len(att_probs) < 15:
        tmp = torch.cat((att_probs, torch.zeros(15-len(att_probs)).cuda()))
        seq_probs.append(tmp)
    else:
        seq_probs.append(att_probs)


def visualize_sequence():
    global seq_idx
    global prev_seq

    idx = 0
    if seq_idx == 0:
        input = sequence[idx:idx+15]
        attended = ['<empty context>'] + sequence[idx:idx+14]
        probs = seq_probs[idx:idx+15]
        draw_boxplot(attended, input, probs, idx)
        idx+=15

    while idx < 70:
        input = sequence[idx:idx+15]
        if idx != 0:
            attended = sequence[idx-15:idx+15]
        else:
            attended = prev_seq + sequence[:idx+15]
        probs = seq_probs[idx:idx+15]
        for nzpad, prob in enumerate(probs):
            probs[nzpad] = torch.cat((torch.zeros(nzpad).cuda(), prob, torch.zeros(15-nzpad).cuda()))
        draw_boxplot(attended, input, probs, idx)
        idx+=15

    prev_seq = sequence[-15:]
    print('trap')

def draw_boxplot(cols, rows, data, idx):
    global seq_idx
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(list(map(lambda x: (x * 100).int().cpu().numpy(), data)),
                linewidths=.5, annot=True, fmt="d", xticklabels=cols, yticklabels=rows)

    plt.savefig('./misc/seq%d_%d.png' % (seq_idx, idx), bbox_inches='tight')
    plt.close()


def inc_seq_idx():
    global seq_idx
    seq_idx += 1

