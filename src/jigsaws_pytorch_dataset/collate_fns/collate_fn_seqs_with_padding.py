import torch

def collate_fn_seqs_with_padding(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             'label' can be a tensor (integer or one-hot) or a list of strings
             and 'length' is a scalar.
    """
    features_list, labels_list, lengths = zip(*data)
    max_len = max(lengths)
    n_ftrs = features_list[0].size(1)
    
    features = torch.zeros((len(data), max_len, n_ftrs))
    lengths = torch.tensor(lengths)

    for i in range(len(data)):
        j, k = features_list[i].size(0), features_list[i].size(1)
        features[i] = torch.cat([features_list[i], torch.zeros((max_len - j, k))])

    # Handle different label formats
    first_label = labels_list[0]
    if isinstance(first_label, torch.Tensor):
        # Labels are tensors (integer or one-hot)
        if first_label.dim() == 1:
            # Integer labels, pad with 0 (assuming 0 is a safe padding value, e.g., for 'G0')
            labels = torch.zeros((len(data), max_len), dtype=torch.long)
            for i, label_seq in enumerate(labels_list):
                labels[i, :len(label_seq)] = label_seq
        elif first_label.dim() == 2:
            # One-hot labels
            num_classes = first_label.size(1)
            labels = torch.zeros((len(data), max_len, num_classes), dtype=torch.float)
            for i, label_seq in enumerate(labels_list):
                labels[i, :len(label_seq), :] = label_seq
    else:
        # Raw string labels (list of lists of strings)
        labels = labels_list

    return features.float(), labels, lengths.long()