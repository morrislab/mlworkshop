from sklearn.tree import export_graphviz

def draw_tree_to_file(tree, feature_names, file):
	with open(file, 'w') as f:
		export_graphviz(tree, out_file=f,
				feature_names=feature_names)
