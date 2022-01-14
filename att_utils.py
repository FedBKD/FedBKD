import torch
import copy


def igfl_attention(client_delta, server_delta=None, client_delta_last=None, strategy="self"):
    """

    :param client_delta: tensor [S, dim], S is the number of clients, dim is the number of parameters in the model
    :param server_delta: tensor [S], aggregation res from client_delta (possibly weighted by Num of data per client)
    :param client_delta_last: tensor [S, dim], used in "time" strategy
    :param strategy: chosen from ["self", "global", "time"]
    :return: attention: [S, S] where element_i_j represents the attention score client i pay to client j
    """

    S, *dim = client_delta.size()
    client_delta = torch.reshape(client_delta, (S, -1))

    if strategy not in ["self", "global", "time"]:
        raise ValueError("strategy must in ['self', 'global', 'time']")
    elif strategy == "self":
        q = client_delta  # [S, dim]
    elif strategy == "global":
        if not server_delta:
            server_delta = torch.mean(client_delta, dim=0)  # [dim]
        q = torch.unsqueeze(server_delta, 0).expand(S, -1)  # [S, dim]
    else:
        if not client_delta_last:
            raise ValueError("client_delta_last must be given in 'time' strategy")
        else:
            q = client_delta_last
    inner_product = torch.mm(q, torch.transpose(client_delta, 0, 1))
    softmax = torch.nn.Softmax(dim=1)
    attention = softmax(inner_product)
    return inner_product, attention


def igfl_server_aggregate(w_locals, idxs_users, layer_keys, w_glob=None, strategy="self"):
    """

    :param w_locals: w_locals[idx][key] represents the parameter tensor of layer key of client idx model
    :param idxs_users: clients idxs sampled
    :param layer_keys: keys of all layers
    :param w_glob:
    :return: w_glob_hat
    """
    w_glob_hat = {k: None for k in layer_keys}
    # attention_score_by_layer = {k: None for k in layer_keys}
    for key in layer_keys:
        client_delta = []
        for idx in idxs_users:
            client_delta.append(w_locals[idx][key])

        client_delta = torch.stack(client_delta)  # [S, *dim]
        S, *layer_dim = client_delta.size()
        client_delta = torch.reshape(client_delta, (S, -1))

        _, attention = igfl_attention(client_delta, server_delta=w_glob, strategy=strategy)  # [S, S]
        client_delta_hat = torch.matmul(attention, client_delta)  # [S, -1]
        client_delta_hat = torch.reshape(client_delta_hat, (S, *layer_dim))  # [S, *dim]

        w_glob_hat[key] = torch.mean(client_delta_hat, dim=0)  # [*dim]
        # attention_score_by_layer[key] = score.tolist()

    return w_glob_hat


def get_para_property(w_locals, idxs_users, layer_keys, args,w_glob=None, strategy="self"):
    """

    :param w_locals: w_locals[idx][key] represents the parameter tensor of layer key of client idx model
    :param idxs_users: clients idxs sampled
    :param layer_keys: keys of all layers
    :param w_glob:
    :return:
    """
    record_inner_product = {k: None for k in layer_keys}
    record_norm_2 = {k: None for k in layer_keys}
    record_cosine_matrix = {k: None for k in layer_keys}
    cuda0 = torch.device('cuda:' + str(args.gpu))
    for key in layer_keys:
        client_delta = []
        for idx in idxs_users:
            client_delta.append(w_locals[idx][key])

        # print("client_delta.device:")
        # print(client_delta.device)
        client_delta = [torch.tensor(ele).to(cuda0) for ele in client_delta]

        # client_delta = client_delta.to(cuda0)
        client_delta = torch.stack(client_delta)  # [S, *dim]
        S, *layer_dim = client_delta.size()
        client_delta = torch.reshape(client_delta, (S, -1))

        inner_product, _ = igfl_attention(client_delta, server_delta=w_glob, strategy=strategy)  # [S, S]
        record_inner_product[key] = inner_product.tolist()

        norm_2 = torch.linalg.norm(client_delta, ord=2, dim=1)  # [S]
        record_norm_2[key] = norm_2.tolist()

        norm_2_matrix = torch.matmul(torch.diag(norm_2), torch.unsqueeze(norm_2, 0).expand(S, -1))  # [S, S]
        cosine_matrix = torch.div(inner_product, norm_2_matrix)
        record_cosine_matrix[key] = cosine_matrix.tolist()


    return record_inner_product, record_norm_2, record_cosine_matrix


def igfl_personalize(w_locals, idxs_users, layer_keys, w_glob=None, strategy="self"):
    """

    :param w_locals: w_locals[idx][key] represents the parameter tensor of layer key of client idx model
    :param idxs_users: clients idxs sampled
    :param layer_keys: keys of all layers
    :param w_glob:
    :return: w_locals_hat
    """
    w_glob_hat = {k: None for k in layer_keys}
    w_locals_hat = copy.deepcopy(w_locals)

    for key in layer_keys:
        client_delta = []
        for idx in idxs_users:
            client_delta.append(w_locals[idx][key])

        client_delta = torch.stack(client_delta)  # [S, *dim]
        S, *layer_dim = client_delta.size()
        client_delta = torch.reshape(client_delta, (S, -1))

        attention = igfl_attention(client_delta, server_delta=w_glob, strategy=strategy)  # [S, S]
        client_delta_hat = torch.matmul(attention, client_delta)  # [S, -1]
        client_delta_hat = torch.reshape(client_delta_hat, (S, *layer_dim))  # [S, *dim]
        for i, idx in enumerate(idxs_users):
            w_locals_hat[idx][key] = client_delta_hat[i]

    return w_locals_hat