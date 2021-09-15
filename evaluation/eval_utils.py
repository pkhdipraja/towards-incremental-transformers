import torch
import numpy as np

from collections import defaultdict
from model.model_utils import add_null_tokens


def get_partial_outputs(cfgs, loader, model, task, label_size, token2idx):
    """
    Get incremental outputs for non-recurrent model.
    """
    results = defaultdict(dict)
    model.eval()

    with torch.no_grad():
        if task == 'labelling':
            for idx, (seq, tag) in enumerate(loader):  # We use batch size of 1
                labels_mask = (tag != 0)
                seq_len = torch.sum(labels_mask).item()
                active_tokens = torch.zeros(seq.size(1), dtype=torch.long).unsqueeze(0)
                active_mask = torch.zeros(seq.size(1), dtype=torch.bool).unsqueeze(0)

                # To store increasing prefix
                predictions = np.empty((seq_len, seq_len))
                predictions.fill(np.inf)

                # To store edits
                changes = np.zeros((seq_len, seq_len))

                # Add null tokens for linear transformers with delayed outputs
                if cfgs.DELAY > 0:
                    seq = add_null_tokens(seq, cfgs.DELAY, token2idx['NULL'])

                    # Fill active_tokens for d times where d is delay
                    for d in range(cfgs.DELAY):
                        active_tokens[:, d] = seq[:, d]

                # Split sequence into partial inputs
                for length in range(1, seq_len+1):
                    active_tokens[:, length-1+cfgs.DELAY] = seq[:, length-1+cfgs.DELAY]
                    active_mask[:, length-1+cfgs.DELAY] = True
                    if cfgs.MODEL == 'incremental-transformers':
                        out = model(active_tokens, valid=True)
                    else:
                        out = model(active_tokens)
                    out = torch.argmax(out, dim=2)

                    # Save partial outputs
                    predictions[length-1][:length] = out[active_mask].numpy()

                    if length == 1:
                        changes[length-1][0] = 1
                    else:
                        changes[length-1] = predictions[length-1] != predictions[length-2]

                active_tag = tag[labels_mask].view(1, -1)
                accuracy = (predictions[-1] == active_tag.numpy()).sum() / seq_len

                results['partial_outputs'][idx] = predictions
                results['log_changes'][idx] = changes
                results['accuracy'][idx] = accuracy

        elif task == 'classification':
            for idx, (seq, label) in enumerate(loader):  # We use batch size of 1
                tokens_mask = (seq != 0)
                seq_len = torch.sum(tokens_mask).item()
                active_tokens = torch.zeros(seq.size(1), dtype=torch.long).unsqueeze(0)
                active_mask = torch.zeros(seq.size(1), dtype=torch.bool).unsqueeze(0)

                # To store increasing prefix
                predictions = np.zeros((seq_len, 1))

                # To store edits
                changes = np.zeros((seq_len, 1))

                # Add null tokens for linear transformers with delayed outputs
                if cfgs.DELAY > 0:
                    seq = add_null_tokens(seq, cfgs.DELAY, token2idx['NULL'])

                    # Fill active_tokens for d times where d is delay
                    for d in range(cfgs.DELAY):
                        active_tokens[:, d] = seq[:, d]

                # Split sequence into partial inputs
                for length in range(1, seq_len+1):
                    active_tokens[:, length-1+cfgs.DELAY] = seq[:, length-1+cfgs.DELAY]
                    active_mask[:, length-1+cfgs.DELAY] = True
                    if cfgs.MODEL in ['transformers', 'linear-transformers']:
                        out = model(active_tokens)
                    elif cfgs.MODEL == 'linear-transformers-causal':
                        out, _ = model(active_tokens)
                    elif cfgs.MODEL == 'incremental-transformers':
                        out, _ = model(active_tokens, sub_mask=False)

                    out = torch.argmax(out, dim=1)

                    # Save partial outputs
                    predictions[length-1] = out[-1].item()

                    if length == 1:
                        changes[length-1][0] = 1
                    else:
                        changes[length-1] = predictions[length-1] != predictions[length-2]

                label = label.view(1, -1)
                accuracy = (predictions[-1] == label.numpy()).sum()

                results['partial_outputs'][idx] = predictions
                results['log_changes'][idx] = changes
                results['accuracy'][idx] = accuracy

        else:
            raise NotImplementedError('Task type does not exist')

    return results


def get_partial_outputs_recurrent(cfgs, loader, model, task, label_size, token2idx):
    """
    Get incremental outputs for recurrent model.
    """
    results = defaultdict(dict)
    model.eval()

    with torch.no_grad():
        if task == 'labelling':
            for idx, (seq, tag) in enumerate(loader):  # We use batch size of 1
                labels_mask = (tag != 0)
                active_seq = seq[labels_mask].view(1, -1)

                seq_len = active_seq.size(1)

                # To store increasing prefix
                predictions = np.empty((seq_len, seq_len))
                predictions.fill(np.inf)

                # To store edits
                changes = np.zeros((seq_len, seq_len))

                # Split sequence into partial inputs
                for length in range(1, seq_len+1):
                    active_tokens = active_seq[:, :length]
                    memory = None

                    for i in range(active_tokens.size(1)):
                        out_i, memory = model(active_tokens[:, i:i+1],
                                              i=i,
                                              memory=memory,
                                              )

                        out_i = torch.argmax(out_i, dim=1)

                        # Save partial outputs
                        predictions[length-1][i] = out_i.item()

                        if length == 1:
                            changes[length-1][0] = 1
                        else:
                            changes[length-1][i] = predictions[length-1][i] != predictions[length-2][i]

                active_tag = tag[labels_mask].view(1, -1)
                accuracy = (predictions[-1] == active_tag.numpy()).sum() / seq_len

                results['partial_outputs'][idx] = predictions
                results['log_changes'][idx] = changes
                results['accuracy'][idx] = accuracy

        elif task == 'classification':
            for idx, (seq, label) in enumerate(loader):  # We use batch size of 1
                tokens_mask = (seq != 0)
                active_seq = seq[tokens_mask].view(1, -1)

                seq_len = active_seq.size(1)

                # To store increasing prefix
                predictions = np.zeros((seq_len, 1))

                # To store edits
                changes = np.zeros((seq_len, 1))

                # Split sequence into partial inputs
                for length in range(1, seq_len+1):
                    active_tokens = active_seq[:, :length]
                    memory = None
                    outs = torch.zeros((1, cfgs.HIDDEN_SIZE))

                    for i in range(active_tokens.size(1)):
                        out_i, memory = model(active_tokens[:, i:i+1],
                                              i=i,
                                              memory=memory,
                                              )

                        outs += out_i

                        # logits_i = model(outs, i=length, recurrent=True, valid=True)
                        logits_i = model.out_proj(outs/(i+1))
                        logits_i = torch.argmax(logits_i, dim=1)

                        # Save partial outputs
                        predictions[length-1] = logits_i.item()

                        if length == 1:
                            changes[length-1][0] = 1
                        else:
                            changes[length-1] = predictions[length-1] != predictions[length-2]

                label = label.view(1, -1)
                accuracy = (predictions[-1] == label.numpy()).sum()

                results['partial_outputs'][idx] = predictions
                results['log_changes'][idx] = changes
                results['accuracy'][idx] = accuracy

        else:
            raise NotImplementedError('Task type does not exist!')

    return results
