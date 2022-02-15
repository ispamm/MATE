import os
import torch
import numpy as np

from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.datasets.ground_truth_loaders import load_dataset_ground_truth
from ExplanationEvaluation.evaluation.AUCEvaluation import AUCEvaluation
from ExplanationEvaluation.models.model_selector import model_selector
from ExplanationEvaluation.tasks.replication import get_classification_task, to_torch_graph
from dig.xgraph.method import SubgraphX

from ExplanationEvaluation.utils.plotting import plot

def find_closest_node_result(results, max_nodes):
    results = sorted(results, key=lambda x: len(x.coalition))
    result_node = results[0]
    for result_idx in range(len(results)):
        x = results[result_idx]
        if len(x.coalition) <= max_nodes and x.P > result_node.P:
            result_node = x
    return result_node

def graph_build_split(edge_index, node_mask: np.array):
    row, col = edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
    return edge_mask

def pipeline():
    dataset_name = "syn1"

    if dataset_name == "syn1" or dataset_name == "syn2" or dataset_name == "syn3":
        thres_snip=12
        thres_min=100
    elif dataset_name == "syn4":
        thres_snip=24
        thres_min=100
    elif dataset_name == "ba2":
        indices = [527]
        thres_snip=5
        thres_min=-1
    else:
        indices= [ 85 ]
        thres_snip=2
        thres_min=-1
    # Load complete dataset

    graphs, features, labels, _, _, _ = load_dataset(dataset_name)
    task = get_classification_task(graphs)

    device = torch.device('cuda', index=1)

    num_classes = len(np.unique(labels))
    features = torch.tensor(features).to(device)
    labels = torch.tensor(labels).to(device)
    graphs = to_torch_graph(graphs, task, device)

    # Load pretrained models
    model, _ = model_selector("GNN",
                              dataset_name,
                              pretrained=True,
                              return_checkpoint=True)
    model.eval()

    # Get ground_truth for every node
    explanation_labels, indices = load_dataset_ground_truth(dataset_name)

    model = model.to(device)
    explanation_saving_dir = os.path.join("/home/indro/MATE/ExplanationEvaluation/subgraphx",
                                          dataset_name)

    torch.manual_seed(0)
    np.random.seed(1)
    #indices = indices[::4]
   
    if task=='graph':
        
        subgraphx = SubgraphX(model,
                        num_classes,
                        device,
                        explain_graph=(task=="graph"),
                        verbose=True,
                        c_puct=10.0,
                        rollout=20,
                        high2low=True,
                        min_atoms=5,
                        expand_atoms=20,
                        reward_method="mc_l_shapley",
                        subgraph_building_method="split",
                        save_dir=explanation_saving_dir)

        for i in indices:
            i = 527
            print(i)
            saved_MCTSInfo_list = None
            prediction = model(features[i], graphs[i]).argmax(-1)
            explain_result, _ = \
                subgraphx.explain(features[i], graphs[i],
                                max_nodes=100,
                                label=prediction,
                                saved_MCTSInfo_list=saved_MCTSInfo_list)

            explain_result = subgraphx.read_from_MCTSInfo_list(explain_result)
            result = find_closest_node_result(explain_result, 100)
            mask = torch.zeros(result.data.x.shape[0]).type(torch.float32)
            mask[result.coalition] = 1.0
            explanation = graph_build_split(result.data.edge_index, mask)
            plot(result.data.edge_index.cpu(), explanation.float().cpu(), labels, i, thres_min, thres_snip, dataset_name, None, explanation_labels)

    else:
        
        subgraphx = SubgraphX(model,
                        num_classes,
                        device,
                        explain_graph=(task=="graph"),
                        verbose=True,
                        c_puct=10.0,
                        rollout=20,
                        high2low=True,
                        min_atoms=5,
                        expand_atoms=20,
                        reward_method="Nc_mc_l_shapley",
                        subgraph_building_method="split",
                        save_dir=explanation_saving_dir)

        predictions = model(features, graphs).argmax(-1)
        for i in indices:
            saved_MCTSInfo_list = None
            explain_result, _ = \
                subgraphx.explain(features, graphs,
                                node_idx=i,
                                max_nodes=100,
                                label=predictions[i].item(),
                                saved_MCTSInfo_list=saved_MCTSInfo_list)

            explain_result = subgraphx.read_from_MCTSInfo_list(explain_result)
            result = find_closest_node_result(explain_result, 100)
            mask = torch.zeros(result.data.x.shape[0]).type(torch.float32)
            mask[result.coalition] = 1.0
            explanation = graph_build_split(result.data.edge_index, mask)
            subset = subgraphx.mcts_state_map.subset
            subgraph_y = labels[subset].to('cpu')
            node_idx =subgraphx.mcts_state_map.new_node_idx
            plot(result.data.edge_index.cpu(), explanation.float().cpu(), subgraph_y, node_idx, thres_min, thres_snip, dataset_name)


if __name__ == '__main__':
    pipeline()