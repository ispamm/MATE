import os
import torch
import numpy as np

from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.datasets.ground_truth_loaders import load_dataset_ground_truth
from ExplanationEvaluation.evaluation.AUCEvaluation import AUCEvaluation
from ExplanationEvaluation.models.model_selector import model_selector
from ExplanationEvaluation.tasks.replication import get_classification_task, to_torch_graph
from dig.xgraph.method import SubgraphX


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
    dataset_name = "mutag"
    # Load complete dataset

    graphs, features, labels, _, _, _ = load_dataset(dataset_name)
    task = get_classification_task(graphs)

    device = torch.device('cuda', index=2)

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

    auc_scores = []
    auc_evaluation = AUCEvaluation(task, explanation_labels, indices[::10])
    for seed in range(0,2):
        explanations = []
        torch.manual_seed(seed)
        np.random.seed(seed)
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
                            #reward_method="Nc_mc_l_shapley",
                            reward_method="mc_l_shapley",
                            subgraph_building_method="split",
                            save_dir=explanation_saving_dir)
        if task=='graph':
            for i in indices[::10]:
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
                obj = [result.data.edge_index.cpu(), explanation.float().cpu()]
                explanations.append(obj)
        else:
            predictions = model(features, graphs).argmax(-1)


            for i in indices[::5]:
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
                obj = [result.data.edge_index.cpu(), explanation.float().cpu()]
                explanations.append(obj)

        auc_score = auc_evaluation.get_score(explanations)    
        auc_scores.append(auc_score)

    auc = np.mean(auc_scores)
    auc_std = np.std(auc_scores)

    print(dataset_name)
    print(auc)
    print(auc_std)

if __name__ == '__main__':
    pipeline()