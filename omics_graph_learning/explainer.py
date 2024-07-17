# # GNN Explainer!
# if args.model != "MLP":
#     with open(
#         "/ocean/projects/bio210019p/stevesho/data/preprocess/graphs/explainer_node_ids.pkl",
#         "rb",
#     ) as file:
#         ids = pickle.load(file)
#     explain_path = "/ocean/projects/bio210019p/stevesho/data/preprocess/explainer"
#     explainer = Explainer(
#         model=model,
#         algorithm=GNNExplainer(epochs=200),
#         explanation_type="model",
#         node_mask_type="attributes",
#         edge_mask_type="object",
#         model_config=dict(mode="regression", task_level="node", return_type="raw"),
#     )

#     data = data.to(device)
#     for index in random.sample(ids, 5):
#         explanation = explainer(data.x, data.edge_index, index=index)

#         print(f"Generated explanations in {explanation.available_explanations}")

#         path = f"{explain_path}/feature_importance_{savestr}_{best_validation}.png"
#         explanation.visualize_feature_importance(path, top_k=10)
#         print(f"Feature importance plot has been saved to '{path}'")

#         path = f"{explain_path}/subgraph_{savestr}_{best_validation}.pdf"
#         explanation.visualize_graph(path)
#         print(f"Subgraph visualization plot has been saved to '{path}'")
