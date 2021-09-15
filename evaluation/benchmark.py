import numpy as np
import torch
import csv

from collections import defaultdict


def speed_benchmark_normal(cfgs, loader, model, task, label_size, token2idx):
    """
    Get total runtime for non-recurrent model.
    """
    model.eval()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        if task == 'labelling':
            start.record()
            for idx, (seq, tag) in enumerate(loader):  # We use batch size of 1
                labels_mask = (tag != 0)
                seq_len = torch.sum(labels_mask).item()
                active_tokens = torch.zeros(seq.size(1),
                                            dtype=torch.long).unsqueeze(0)

                for length in range(seq_len):
                    active_tokens[:, length] = seq[:, length]
                    if cfgs.MODEL == 'incremental-transformers':
                        out = model(active_tokens, valid=True)
                    else:
                        out = model(active_tokens)
                    out = torch.argmax(out, dim=2)

            end.record()
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end)/1000

        elif task == 'classification':
            start.record()
            for idx, (seq, label) in enumerate(loader):  # We use batch size of 1
                tokens_mask = (seq != 0)
                seq_len = torch.sum(tokens_mask).item()
                active_tokens = torch.zeros(seq.size(1),
                                            dtype=torch.long).unsqueeze(0)

                for length in range(seq_len):
                    active_tokens[:, length] = seq[:, length]
                    if cfgs.MODEL in ['transformers', 'linear-transformers']:
                        out = model(active_tokens)
                    elif cfgs.MODEL == 'linear-transformers-causal':
                        out, _ = model(active_tokens)
                    elif cfgs.MODEL == 'incremental-transformers':
                        out, _ = model(active_tokens, sub_mask=False)

                    out = torch.argmax(out, dim=1)

            end.record()
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end)/1000

        else:
            raise NotImplementedError('Task type does not exist')

    return elapsed_time


def speed_benchmark_recurrent(cfgs, loader, model, task, label_size, token2idx):
    """
    Get total runtime for recurrent model.
    """
    model.eval()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        if task == 'labelling':
            start.record()
            for idx, (seq, tag) in enumerate(loader):  # We use batch size of 1
                labels_mask = (tag != 0)
                active_seq = seq[labels_mask].view(1, -1)
                seq_len = active_seq.size(1)

                memory = None
                for i in range(seq_len):
                    out_i, memory = model(
                        active_seq[:, i],
                        i=i,
                        memory=memory
                    )

                    out_i = torch.argmax(out_i, dim=1)

            end.record()
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end)/1000

        elif task == 'classification':
            start.record()
            for idx, (seq, label) in enumerate(loader):  # We use batch size of 1
                tokens_mask = (seq != 0)
                active_seq = seq[tokens_mask].view(1, -1)
                seq_len = active_seq.size(1)

                memory = None
                outs = torch.zeros((1, cfgs.HIDDEN_SIZE))
                for i in range(seq_len):
                    out_i, memory = model(
                        active_seq[:, i],
                        i=i,
                        memory=memory
                    )

                    outs += out_i

                    logits_i = model.out_proj(outs/(i+1))
                    logits_i = torch.argmax(logits_i, dim=1)

            end.record()
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end)/1000

        else:
            raise NotImplementedError('Task type does not exist')

    return elapsed_time


def speed_benchmark(cfgs, loader, model, task, label_size, token2idx, recurrent):
    if not recurrent:
        elapsed = speed_benchmark_normal(cfgs, loader, model,
                                         task, label_size, token2idx)
    else:
        elapsed = speed_benchmark_recurrent(cfgs, loader, model,
                                            task, label_size, token2idx)

    return elapsed


def incr_speed_benchmark_normal(cfgs, loader, model, task, label_size, token2idx):
    """
    Get processing time for varying sentence length with
    non-recurrent models (restart-incremental).
    """
    model.eval()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    process_time_list = []

    with torch.no_grad():
        if task == 'labelling':

            seq, tag = next(iter(loader))  # Only do this for 1 example, with maximum length.
            seq_len = seq.size(1)
            assert seq_len == cfgs.MAX_TOKEN, "Number of tokens in the sentence is not equal to max token."

            for current_length in range(1, seq_len+1):
                current_seq = seq[:, :current_length]
                current_seq_len = current_seq.size(1)
                active_tokens = torch.zeros(current_seq_len,
                                            dtype=torch.long).unsqueeze(0)

                start.record()

                for restart_length in range(current_seq_len):
                    active_tokens[:, restart_length] = current_seq[:, restart_length]
                    if cfgs.MODEL == 'incremental-transformers':
                        out = model(active_tokens, valid=True)
                    else:
                        out = model(active_tokens)
                    out = torch.argmax(out, dim=2)

                end.record()
                torch.cuda.synchronize()
                elapsed_time = start.elapsed_time(end)/1000
                process_time_list.append(elapsed_time)

        elif task == 'classification':

            seq, label = next(iter(loader))  # Only do this for 1 example, with maximum length.
            seq_len = seq.size(1)
            assert seq_len == cfgs.MAX_TOKEN, "Number of tokens in the sentence is not equal to max token."

            for current_length in range(1, seq_len+1):
                current_seq = seq[:, :current_length]
                current_seq_len = current_seq.size(1)
                active_tokens = torch.zeros(current_seq_len,
                                            dtype=torch.long).unsqueeze(0)

                start.record()

                for restart_length in range(current_seq_len):
                    active_tokens[:, restart_length] = current_seq[:, restart_length]
                    if cfgs.MODEL in ['transformers', 'linear-transformers']:
                        out = model(active_tokens)
                    elif cfgs.MODEL == 'linear-transformers-causal':
                        out, _ = model(active_tokens)
                    elif cfgs.MODEL == 'incremental-transformers':
                        out, _ = model(active_tokens, sub_mask=False)  
                    out = torch.argmax(out, dim=1)

                end.record()
                torch.cuda.synchronize()
                elapsed_time = start.elapsed_time(end)/1000
                process_time_list.append(elapsed_time)

        else:
            raise NotImplementedError('Task type does not exist')

    return process_time_list


def incr_speed_benchmark_recurrent(cfgs, loader, model, task, label_size, token2idx):
    """
    Get processing time for varying sentence length with recurrent models.
    """
    model.eval()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    process_time_list = []

    with torch.no_grad():
        if task == 'labelling':
            seq, tag = next(iter(loader))  # Only do this for 1 example, with maximum length.
            seq_len = seq.size(1)
            assert seq_len == cfgs.MAX_TOKEN, "Number of tokens in the sentence is not equal to max token."

            for current_length in range(1, seq_len+1):
                current_seq = seq[:, :current_length]
                current_seq_len = current_seq.size(1)

                memory = None
                start.record()

                for i in range(current_seq_len):
                    out_i, memory = model(
                       current_seq[:, i],
                       i=i,
                       memory=memory
                    )

                    out_i = torch.argmax(out_i, dim=1)

                end.record()
                torch.cuda.synchronize()
                elapsed_time = start.elapsed_time(end)/1000
                process_time_list.append(elapsed_time)

        elif task == 'classification':
            seq, label = next(iter(loader))  # Only do this for 1 example, with maximum length.
            seq_len = seq.size(1)
            assert seq_len == cfgs.MAX_TOKEN, "Number of tokens in the sentence is not equal to max token."

            for current_length in range(1, seq_len+1):
                current_seq = seq[:, :current_length]
                current_seq_len = current_seq.size(1)

                memory = None
                outs = torch.zeros((1, cfgs.HIDDEN_SIZE))

                start.record()

                for i in range(current_seq_len):
                    out_i, memory = model(
                       current_seq[:, i],
                       i=i,
                       memory=memory
                    )

                    outs += out_i
                    logits_i = model.out_proj(outs/(i+1))
                    logits_i = torch.argmax(logits_i, dim=1)

                end.record()
                torch.cuda.synchronize()
                elapsed_time = start.elapsed_time(end)/1000
                process_time_list.append(elapsed_time)

        else:
            raise NotImplementedError('Task type does not exist')

    return process_time_list


def incr_speed_benchmark(cfgs, loader, model, task, label_size, token2idx, recurrent):
    """
    Get runtime performance for increasing sentence
    length (from 0 to MAX_TOKEN) and write it to .tsv file.
    """

    if not recurrent:
        elapsed = incr_speed_benchmark_normal(cfgs, loader, model, task,
                                              label_size, token2idx)
        filename = 'timing_increasing_len_non_recurrent_' + task

    else:
        elapsed = incr_speed_benchmark_recurrent(cfgs, loader, model, task,
                                                 label_size, token2idx)
        filename = 'timing_increasing_len_recurrent_' + task

    with open(filename + '.tsv', 'w', newline='') as f:
        tsv_out = csv.writer(f, delimiter='\t')
        tsv_out.writerow(elapsed)
