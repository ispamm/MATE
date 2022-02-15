from math import sqrt
import os
import torch
import higher
import numpy as np
from torch_geometric.data import Data, DataLoader
import torch_geometric as ptgeom
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from torch_geometric.nn import MessagePassing
from ExplanationEvaluation.datasets.ground_truth_loaders import load_dataset_ground_truth

from ExplanationEvaluation.utils.graph import index_edge

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


def exploss(masked_pred, original_pred, edge_mask, reg_coefs=(0.01, 1.0)):
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
    print(best_epoch)
    if best_epoch == -1:
        checkpoint = torch.load(f"./checkpoints/{paper}/{dataset}/best_model")
    else:
        checkpoint = torch.load(f"./checkpoints/{paper}/{dataset}/model_{best_epoch}")
    model.load_state_dict(checkpoint['model_state_dict'])

    if eval_enabled: model.eval()

    return model


def train_node(model, dataset, edge_index, x, labels, train_mask, val_mask, test_mask):

    # Define graph
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(0, 100):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
        # Evaluate train
        train_acc = evaluate(out[train_mask], labels[train_mask])
        test_acc = evaluate(out[test_mask], labels[test_mask])
        val_acc = evaluate(out[val_mask], labels[val_mask])

        print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")
        if val_acc > best_val_acc: # New best results
            best_val_acc = val_acc
            best_epoch = epoch
            #store_checkpoint("GNN", dataset, model, train_acc, val_acc, test_acc, best_epoch)

        if epoch - best_epoch > 100: # or best_val_acc > 0.99:
            break
    #model = load_best_model(best_epoch, "GNN", dataset, model, True)
    #store_checkpoint("GNN", dataset, model, 0, 0, 0)

    return model



def meta_train_node(model, dataset, edge_index, x, labels, train_mask, val_mask, test_mask):
    """
    Train a explainer to explain node classifications
    :param _dataset: the dataset we wish to use for training
    :param _paper: the paper we whish to follow, chose from "GNN" or "PG"
    :param args: a dict containing the relevant model arguements
    """
    criterion = torch.nn.CrossEntropyLoss()
    meta_opt = torch.optim.Adam(model.parameters(), 
                                    lr=0.003)      
    inner_opt = torch.optim.Adam(model.parameters(),
                                    lr=0.0001)
    best_val_acc = 0.0
    best_epoch = 0
    _, indices = load_dataset_ground_truth("syn1")
    l = list(indices)
    for epoch in range(0, 100):
        idx = torch.randint(0, len(l), (1,))
        node_idx = torch.tensor([l[idx]])
        
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
            exp_loss = exploss(masked_pred, pred_label, edge_mask)
            exp_loss.backward()
            opt_exp.step()                
        
        model.train()
        meta_opt.zero_grad()
        edge_mask.requires_grad = False
        set_mask(model, edge_mask)
        with higher.innerloop_ctx(
            model, inner_opt, copy_initial_weights=False
            ) as (fnet, diffopt):
            for _ in range(3):
                masked_pred = fnet(x, sub_index)[node_idx]
                #exp_loss = exploss(masked_pred, pred_label, edge_mask)
                exp_loss = criterion(masked_pred, pred_label)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        meta_opt.step()
        
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
        # Evaluate train
        train_acc = evaluate(out[train_mask], labels[train_mask])
        test_acc = evaluate(out[test_mask], labels[test_mask])
        val_acc = evaluate(out[val_mask], labels[val_mask])

        print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")
        if val_acc > best_val_acc: # New best results
            best_val_acc = val_acc
            best_epoch = epoch
            #store_checkpoint("GNN", dataset, model, train_acc, val_acc, test_acc, best_epoch)

        if epoch - best_epoch > 100: # or best_val_acc > 0.99:
            break

    #model = load_best_model(best_epoch, "GNN", dataset, model, True)
    #store_checkpoint("GNN", dataset, model, 0, 0, 0)
    return model


class GNNExplainer(torch.nn.Module):

    def __init__(self, model_to_explain, graphs, features, index, epochs=1000, lr=0.005, reg_coefs=(0.05, 1.0)):
        super().__init__()
        self.model_to_explain = model_to_explain.eval()
        for p in self.model_to_explain.parameters():
            p.requires_grad = False

        self.epochs = epochs
        self.lr = lr
        self.size_reg = reg_coefs[0]
        self.entropy_reg = reg_coefs[1]
        self.index = int(index)
        self.features = features
        self.graph = ptgeom.utils.k_hop_subgraph(self.index, 3, graphs)[1]
        
        (N, _), E = self.features.size(), self.graph.size(1)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)#.retain_grad()

    
    def _set_masks(self):
        for module in self.model_to_explain.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def _clear_masks(self):
        for module in self.model_to_explain.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None


    def _loss(self, masked_pred, original_pred):       
        EPS = 1e-15

        # Regularization losses
        mask = torch.sigmoid(self.edge_mask)
        size_loss = torch.sum(mask) * self.size_reg
        mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        mask_ent_loss = self.entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)

        return cce_loss + size_loss + mask_ent_loss
    

    def predict(self):
        self._clear_masks()
        with torch.no_grad():
            original_pred = self.model_to_explain(self.features, self.graph)[self.index]
            self.pred_label = original_pred.argmax(dim=-1).detach()

    def explain(self):
        # Prepare model for new explanation run
        self._clear_masks()
        # Similar to the original paper we only consider a subgraph for explaining
        with torch.no_grad():
            original_pred = self.model_to_explain(self.features, self.graph)[self.index]
            self.pred_label = original_pred.argmax(dim=-1).detach()
       
        self._set_masks()
        optimizer = torch.optim.Adam([self.edge_mask], lr=self.lr)

        losses = []
        # Start training loop
        for e in range(0, self.epochs):
            optimizer.zero_grad()

            masked_pred = self.model_to_explain(self.features, self.graph)[self.index]
            loss = self._loss(masked_pred.unsqueeze(0), self.pred_label.unsqueeze(0))
           
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())
        # Retrieve final explanation
        mask = torch.sigmoid(self.edge_mask)
        expl_graph_weights = torch.zeros(self.graph.size(1))
        for i in range(0, self.edge_mask.size(0)): # Link explanation to original graph
            pair = self.graph.T[i]
            t = index_edge(self.graph, pair)
            expl_graph_weights[t] = mask[i]

        return self.graph, expl_graph_weights, losses

    def forward(self, model):
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask
        return (model(self.features, self.graph)[self.index]).unsqueeze(0)