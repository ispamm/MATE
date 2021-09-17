from math import sqrt
import os
import torch
import higher
import numpy as np
from torch_geometric.data import Data, DataLoader
import torch_geometric as ptgeom
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.models.model_selector import model_selector
from torch_geometric.nn import MessagePassing
from ExplanationEvaluation.datasets.ground_truth_loaders import load_dataset_ground_truth


def init_mask(x, edge_index):
    (N, F), E = x.size(), edge_index.size(1)
    std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
    return  torch.nn.Parameter(torch.randn(E, device=x.device) * std)

def set_mask(model, edge_mask):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module.__edge_mask__ = edge_mask


def clear_mask(model):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module.__edge_mask__ = None


def exploss(masked_pred, original_pred, edge_mask, reg_coefs):
    """
    Returns the loss score based on the given mask.
    :param masked_pred: Prediction based on the current explanation
    :param original_pred: Predicion based on the original graph
    :param edge_mask: Current explanaiton
    :param reg_coefs: regularization coefficients
    :return: loss
    """
    size_reg = reg_coefs[0]
    entropy_reg = reg_coefs[1]
    EPS = 1e-15

    # Regularization losses
    mask = torch.sigmoid(edge_mask)
    size_loss = torch.sum(mask) * size_reg
    mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
    mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

    # Explanation loss
    cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)

    return cce_loss + size_loss + mask_ent_loss


def create_data_list(graphs, features, labels, mask):
    """
    Convert the numpy data to torch tensors and save them in a list.
    :params graphs: edge indecs of the graphs
    :params features: features for every node
    :params labels: ground truth labels
    :params mask: mask, used to filter the data
    :retuns: list; contains the dataset
    """
    indices = np.argwhere(mask).squeeze()
    data_list = []
    for i in indices:
        x = torch.tensor(features[i])
        edge_index = torch.tensor(graphs[i])
        y = torch.tensor(labels[i].argmax())
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list


def evaluate(out, labels):
    """
    Calculates the accuracy between the prediction and the ground truth.
    :param out: predicted outputs of the explainer
    :param labels: ground truth of the data
    :returns: int accuracy
    """
    preds = out.argmax(dim=1)
    correct = preds == labels
    acc = int(correct.sum()) / int(correct.size(0))
    return acc


def store_checkpoint(paper, dataset, model, train_acc, val_acc, test_acc, epoch=-1):
    """
    Store the model weights at a predifined location.
    :param paper: str, the paper 
    :param dataset: str, the dataset
    :param model: the model who's parameters we whish to save
    :param train_acc: training accuracy obtained by the model
    :param val_acc: validation accuracy obtained by the model
    :param test_acc: test accuracy obtained by the model
    :param epoch: the current epoch of the training process
    :retunrs: None
    """
    save_dir = f"./checkpoints/{paper}/{dataset}"
    checkpoint = {'model_state_dict': model.state_dict(),
                  'train_acc': train_acc,
                  'val_acc': val_acc,
                  'test_acc': test_acc}
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if epoch == -1:
        torch.save(checkpoint, os.path.join(save_dir, f"best_model"))
    else:
        torch.save(checkpoint, os.path.join(save_dir, f"model_{epoch}"))


def load_best_model(best_epoch, paper, dataset, model, eval_enabled):
    """
    Load the model parameters from a checkpoint into a model
    :param best_epoch: the epoch which obtained the best result. use -1 to chose the "best model"
    :param paper: str, the paper 
    :param dataset: str, the dataset
    :param model: the model who's parameters overide
    :param eval_enabled: wheater to activate evaluation mode on the model or not
    :return: model with pramaters taken from the checkpoint
    """
    print(best_epoch)
    if best_epoch == -1:
        checkpoint = torch.load(f"./checkpoints/{paper}/{dataset}/best_model")
    else:
        checkpoint = torch.load(f"./checkpoints/{paper}/{dataset}/model_{best_epoch}")
    model.load_state_dict(checkpoint['model_state_dict'])

    if eval_enabled: model.eval()

    return model

def meta_train_node(_dataset, _paper, args, device):
    """
    Train a explainer to explain node classifications
    :param _dataset: the dataset we wish to use for training
    :param _paper: the paper we whish to follow, chose from "GNN" or "PG"
    :param args: a dict containing the relevant model arguements
    """
    graph, features, labels, train_mask, val_mask, test_mask = load_dataset(_dataset)
    model = model_selector(_paper, _dataset, False, True).to(device)
    reg_coefs=(args.reg_size, args.reg_ent)

    x = torch.tensor(features).to(device)
    edge_index = torch.tensor(graph).to(device)
    labels = torch.tensor(labels).to(device)

    # Define graph
    print(model)
    for p in model.parameters():
        p.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    meta_opt = torch.optim.Adam(model.parameters(), 
                                    lr=args.lr)      
    inner_opt = torch.optim.Adam(model.parameters(),
                                    lr=args.inner_lr)

    best_val_acc = 0.0
    best_epoch = 0
    
    _, indices = load_dataset_ground_truth(_dataset)
    l = list(indices)
    for epoch in range(0, args.epochs):
        if True:
            idx = torch.randint(0, len(l), (1,))
            node_idx = torch.tensor([l[idx]])
        else:
            node_idx = torch.randint(0, x.size(0), (1,))

        sub_index = ptgeom.utils.k_hop_subgraph(node_idx, 3, edge_index)[1]
        model.eval()
        with torch.no_grad():
            original_pred = model(x, sub_index)[node_idx]
            pred_label = original_pred.argmax(dim=1)
        
        edge_mask = init_mask(x, sub_index)
        opt_exp = torch.optim.Adam([edge_mask], lr=0.03)
        for _ in range(30):
            opt_exp.zero_grad()
            set_mask(model, edge_mask)
            masked_pred = model(x, sub_index)[node_idx]
            exp_loss = exploss(masked_pred, pred_label, edge_mask, reg_coefs)
            exp_loss.backward()
            opt_exp.step()                
        
        model.train()
        meta_opt.zero_grad()
        edge_mask.requires_grad = False
        set_mask(model, edge_mask)
        with higher.innerloop_ctx(
            model, inner_opt, copy_initial_weights=False
            ) as (fnet, diffopt):
            for _ in range(args.T):
                masked_pred = fnet(x, sub_index)[node_idx]
                exp_loss = exploss(masked_pred, pred_label, edge_mask, reg_coefs)
                params = diffopt.step(exp_loss)                
 
        clear_mask(model)      
        with torch.no_grad():
            model.conv1.weight.copy_(params[-9])
            model.conv1.bias.copy_(params[-8])
            model.conv2.weight.copy_(params[-6])
            model.conv2.bias.copy_(params[-5])
            model.conv3.weight.copy_(params[-4])
            model.conv3.bias.copy_(params[-3])
            model.lin.weight.copy_(params[-2])
            model.lin.bias.copy_(params[-1])

        out = model(x, edge_index)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max)
        meta_opt.step()
        
        
        if args.eval_enabled: model.eval()
        with torch.no_grad():
            out = model(x, edge_index)

        # Evaluate train
        train_acc = evaluate(out[train_mask], labels[train_mask])
        test_acc = evaluate(out[test_mask], labels[test_mask])
        val_acc = evaluate(out[val_mask], labels[val_mask])

        print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")

        if val_acc > best_val_acc: # New best results
            print("Val improved")
            best_val_acc = val_acc
            best_epoch = epoch
            store_checkpoint(_paper, _dataset, model, train_acc, val_acc, test_acc, best_epoch)

        if epoch - best_epoch > args.early_stopping and best_val_acc > 0.99:
            break

    model = load_best_model(best_epoch, _paper, _dataset, model, args.eval_enabled)
    out = model(x, edge_index)

    # Train eval
    train_acc = evaluate(out[train_mask], labels[train_mask])
    test_acc = evaluate(out[test_mask], labels[test_mask])
    val_acc = evaluate(out[val_mask], labels[val_mask])
    print(f"final train_acc:{train_acc}, val_acc: {val_acc}, test_acc: {test_acc}")

    store_checkpoint(_paper, _dataset, model, train_acc, val_acc, test_acc)


def meta_train_graph(_dataset, _paper, args, device):
    """
    Train a explainer to explain graph classifications
    :param _dataset: the dataset we wish to use for training
    :param _paper: the paper we whish to follow, chose from "GNN" or "PG"
    :param args: a dict containing the relevant model arguements
    """
    graphs, features, labels, train_mask, val_mask, test_mask = load_dataset(_dataset)
    train_set = create_data_list(graphs, features, labels, train_mask)
    val_set = create_data_list(graphs, features, labels, val_mask)
    test_set = create_data_list(graphs, features, labels, test_mask)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    model = model_selector(_paper, _dataset, False).to(device)
    
    reg_coefs=(args.reg_size, args.reg_ent)

    # Define graph
    criterion = torch.nn.CrossEntropyLoss()
    meta_opt = torch.optim.Adam(model.parameters(), 
                                    lr=args.lr)      
    inner_opt = torch.optim.Adam(model.parameters(),
                                    lr=args.inner_lr)


    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(0, args.epochs):
        for data in train_loader:
            graph_idx = torch.randint(0, len(graphs), (1,)).to(device)
            x = torch.tensor(features[graph_idx]).to(device)
            edge_index = torch.tensor(graphs[graph_idx]).to(device)

            model.eval()
            with torch.no_grad():
                original_pred = model(x, edge_index)
                pred_label = original_pred.argmax(dim=1)

            sub_index = edge_index[:, (edge_index[0] != edge_index[1])]
            edge_mask = init_mask(x, sub_index)
            opt_exp = torch.optim.Adam([edge_mask], lr=0.03)
            for _ in range(30):
                opt_exp.zero_grad()
                set_mask(model, edge_mask)
                masked_pred = model(x, sub_index)
                exp_loss = exploss(masked_pred, pred_label, edge_mask, reg_coefs)
                exp_loss.backward()
                opt_exp.step()                

            model.train()
            meta_opt.zero_grad()
            edge_mask.requires_grad = False
            set_mask(model, edge_mask)
            with higher.innerloop_ctx(
                model, inner_opt, copy_initial_weights=False
                ) as (fnet, diffopt):
                for _ in range(args.T):
                    masked_pred = fnet(x, sub_index)
                    exp_loss = exploss(masked_pred, pred_label, edge_mask, reg_coefs)
                    params = diffopt.step(exp_loss)    

            clear_mask(model)      
            with torch.no_grad():
                model.conv1.weight.copy_(params[-9])
                model.conv1.bias.copy_(params[-8])
                model.conv2.weight.copy_(params[-6])
                model.conv2.bias.copy_(params[-5])
                model.conv3.weight.copy_(params[-4])
                model.conv3.bias.copy_(params[-3])
                model.lin.weight.copy_(params[-2])
                model.lin.bias.copy_(params[-1])

            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max)
            meta_opt.step()
            
        model.eval()
        # Evaluate train
        with torch.no_grad():
            train_sum = 0
            loss = 0
            for data in train_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                loss += criterion(out, data.y)
                preds = out.argmax(dim=1)
                train_sum += (preds == data.y).sum()
            train_acc = int(train_sum) / int(len(train_set))
            train_loss = float(loss) / int(len(train_loader))

            eval_data = next(iter(test_loader)).to(device) # Loads all test samples
            out = model(eval_data.x, eval_data.edge_index, eval_data.batch)
            test_acc = evaluate(out, eval_data.y)

            eval_data = next(iter(val_loader)).to(device) # Loads all eval samples
            out = model(eval_data.x, eval_data.edge_index, eval_data.batch)
            val_acc = evaluate(out, eval_data.y)

        print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")

        if val_acc > best_val_acc:  # New best results
            print("Val improved")
            best_val_acc = val_acc
            best_epoch = epoch
            store_checkpoint(_paper, _dataset, model, train_acc, val_acc, test_acc, best_epoch)

        # Early stopping
        if epoch - best_epoch > args.early_stopping:
            break

    model = load_best_model(best_epoch, _paper, _dataset, model, args.eval_enabled)

    with torch.no_grad():
        train_sum = 0
        for data in train_loader:
            data= data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            preds = out.argmax(dim=1)
            train_sum += (preds == data.y).sum()
        train_acc = int(train_sum) / int(len(train_set))

        eval_data = next(iter(test_loader)).to(device)
        out = model(eval_data.x, eval_data.edge_index, eval_data.batch)
        test_acc = evaluate(out, eval_data.y)

        eval_data = next(iter(val_loader)).to(device)
        out = model(eval_data.x, eval_data.edge_index, eval_data.batch)
        val_acc = evaluate(out, eval_data.y)

    print(f"final train_acc:{train_acc}, val_acc: {val_acc}, test_acc: {test_acc}")

    store_checkpoint(_paper, _dataset, model, train_acc, val_acc, test_acc)
