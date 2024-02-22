"""
Define a graph over genes according to the KEGG BRITE orthology.

The KEGG BRITE orthology specifies four progressively coarser levels of terms that describe biological phenomena. The
lowest level, D, defines a graph over genes (and other molecules, but we only use genes). Nodes (genes) in level D
may have connections to one or more nodes in the next level, C. The nodes in level C represent KEGG pathways,
which map functional relationships between genes in level D. Nodes in level B and A correspond to terms that describe
biological phenomena at a more general leve. Nodes in each level of the BRITE orthology can be related to one or more
nodes in the next level up. Therefore, we use the KEGG BRITE orthology to specify a graph coarsening strategy for
modeling a latent representation of gene expression data using graph neural networks.
"""
import argparse
import Bio.KEGG.REST
from . import biopathgraph as bpg
import pickle
import os
import re

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Create dictionary objects that define graphs of each level of the "
                                                 "KEGG BRITE orthology")
    parser.add_argument("-o", "--output_dir",
                        help="The path to the directory where the BRITE graph will be written",
                        default=os.path.join(os.getcwd(), "data"),
                        type=str)
    args = vars(parser.parse_args())
    output_dir = args["output_dir"]

    # Download the full BRITE orthology
    print("Downloading KEGG BRITE orthology", end="... ")
    with Bio.KEGG.REST.kegg_get("br:hsa00001") as kegg_query:
        brite_content = kegg_query.read()
    print("Done")

    # Create a lists of BRITE entries extracted from the full BRITE orthology
    print("Converting KEGG BRITE orthology to list object", end="... ")
    brite_list = bpg.list_from_brite(brite_content)
    print("Done")

    print("Assembling node dictionaries for BRITE orthology levels", end="... ")
    level_a_dict = bpg.brite_level_dict(brite_list, level="A")
    level_b_dict = bpg.brite_level_dict(brite_list, level="B")
    level_c_dict = bpg.brite_level_dict(brite_list, level="C")
    level_d_dict = bpg.brite_level_dict(brite_list, level="D")
    # Restrict level_c_dict to nodes that are associated with pathways that are present in humans
    # Explanation: the BRITE orthology lists several entries that are not present in humans and are therefore not
    # associated with a KEGG hsa ID (in the "accession" slot of the entry dictionary). Examples include pathways that
    # are only present in plants or prokaryotes. Some of these entries have children in level D, but we do not care
    # about these ancestries because they are not relevant to human biology. This loop removes the irrelevant entries
    # from level C.
    for node, node_data in level_c_dict.copy().items():
        if node_data["accession"] is None:
            level_c_dict.pop(node)
    print("Done")

    # Construct the adjacency dictionaries
    # Notes: The edges between nodes in level D of the BRITE orthology are derived from relations in KGML data about
    # nodes in level C. The KGML data also contain information about edges between nodes in level C: if a node's KGML
    # data node (i.e. pathway) at level C, an edge is created between them. Constructing the level C and level D
    # adjacency dictionaries simultaneously during a single loop over the nodes in level C is most efficient,
    # but it still takes several minutes to complete.
    level_c_adj = dict()
    level_d_adj = dict()
    for node in tqdm(level_c_dict.values(), total=len(level_c_dict), desc="Creating level C and D adjacency dicts"):
        if node["accession"] is not None:
            # There are two types of accessions: those prefixed with PATH and those prefixed with BR. PATH denotes a
            # pathway map. BR denotes collections of related molecules, e.g. "Ion channels." This loop only adds
            # edges to PATH: nodes. BR: nodes have degree zero.
            if re.search(r"^PATH:", node["accession"]) is not None:
                accession = bpg.re_strip(node["accession"], "^[A-Z]*:")
                # print(f"Updating adjacency list from {accession}")
                level_d_adj, level_c_adj = bpg.adjacency_from_path(accession, level_d_adj, level_c_adj)
    level_c_adj = bpg.make_undirected(level_c_adj)  # The `adjacency_from_path` function gives undirected edges

    # Restrict the level D adjacency list to entries that correspond to gene-encoded molecules
    # Explanation: Even though the BRITE hsa orthology only lists gene-encoded molecules at level D, the KGML data
    # that the adjacency dictionaries are derived from often contain relations that involve non- gene-encoded
    # molecules. This loop removes edges to or from non- gene-encoded molecules. Note that KGML still records edges
    # between two genes when there is a non- gene-encoded intermediary between them, so removing the intermediary
    # from the adjacency should only minimally affect our knowledge of the relationship it mediates between the
    # gene-encoded proteins, if it has an effect at all.
    reg_exp = re.compile(r"^hsa\d+")
    for node, neighbors in level_d_adj.copy().items():
        res = reg_exp.search(node)
        if res is None:
            level_d_adj.pop(node)  # Remove node entries that don't correspond to an hsa gene ID
        else:
            for neighbor in neighbors.copy():
                res = reg_exp.search(neighbor[0])
                if res is None:
                    level_d_adj[node].remove(neighbor)  # Remove neighbors that don't correspond to an hsa gene ID

    # Make a version of the adjacency list that does not include edge information for use with networkx
    level_d_adj_no_edge_type = dict()
    for node, neighbors in level_d_adj.items():
        level_d_adj_no_edge_type[node] = set(neighbor[0] for neighbor in neighbors)

    # The code below creates an adjacency list for BRITE level B.
    # Notes: For any two nodes in level B with children in level C, make an edge if any child of node 1 is adjacent
    # to any child of node 2. The edges made are directed. For example, the following relationships will be
    # created (if level C adjacency dict is undirected):
    #
    # B:    (09176 DrugResist) -----------------------------------------------> (09143 CellGrowthDeath)
    #             |              (Edge implied by level C edge (01524,04115))             |
    #             |                                                                       |
    #             |                                                                       |
    # C:    (01524 PlatDrugRes) ----------------------------------------------> (04115 p53Signaling)
    #                             (Edge implied by level C pathway maps)
    level_b_adj = dict()
    for node_data1 in tqdm(level_b_dict.values(), total=len(level_b_dict), desc="Creating level B adjacency dict"):
        children1 = set(child_id for child_id in node_data1["children"])
        for node_data2 in level_b_dict.values():
            if node_data1 != node_data2:
                children2 = set(child_id for child_id in node_data2["children"])
                for child_id in children1:
                    if child_id in level_c_adj.keys():  # NB: not all children of level B are in level_c_adj.keys()
                        has_edge = len(level_c_adj[child_id].intersection(children2)) > 0
                        if has_edge:
                            if node_data1["id"] in level_b_adj.keys():
                                level_b_adj[node_data1["id"]].add(node_data2["id"])
                            else:
                                level_b_adj[node_data1["id"]] = {node_data2["id"]}
                            break
    level_b_adj = bpg.make_undirected(level_b_adj)
    # # Check which nodes have no neighbours
    # degree_zero = list(set(level_b_dict.keys()).difference(set(level_b_adj.keys())))
    # for node in degree_zero:
    #     print(level_b_dict[node])
    # # Nothing really important -- either irrelevant to humans or molecular families

    # Level A is fully connected, undirected
    level_a_adj = dict()
    for node in tqdm(level_a_dict.keys(), total=len(level_a_dict), desc="Creating level A adjacency dict"):
        neighbors = set(level_a_dict.keys())
        neighbors.remove(node)
        level_a_adj[node] = neighbors

    level_a = (level_a_dict, level_a_adj)
    level_b = (level_b_dict, level_b_adj)
    level_c = (level_c_dict, level_c_adj)
    level_d = (level_d_dict, level_d_adj)
    brite_graph = (level_a, level_b, level_c, level_d)

    with open(os.path.join(output_dir, "brite_graph.pkl"), "wb") as file_out:
        pickle.dump(brite_graph, file_out)


if __name__ == "__main__":
    main()
