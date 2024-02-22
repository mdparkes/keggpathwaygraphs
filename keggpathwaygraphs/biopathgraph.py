"""
Notes:

Some terms in the BRITE orthology, such as BR:09190 "Not Included in Pathway or Brite" and BR:09180 "Brite Hierarchies,"
could be excluded from the graph at the user's discretion. Arguments in favor of their removal include shrinking an
already very large graph, and adhering to a functional perspective on genes (BR:09180, for example, organizes protein
families whose members may not be functionally related to one another). Arguments against their removal include
retaining potentially predictive information, and retaining information that could enhance explanations about
the relationships between genes and prediction targets.

Level D has nodes (genes) with two identical outgoing edge types (e.g. hsa7086 -> hsa6888). In the case of hsa7086,
this is explained by the fact that pathway map hsa00030 relates hsa7086 (2.2.1.1) and hsa6888 (2.2.1.2) through two
different non- gene-encoded compounds that they both play a role in metabolizing (Arguably there should be no edge
between these two genes because there is no directed path between them through the metabolized compounds on the
rendered pathway map available online, but the KGML data nonetheless specify a directed edge between them. Consider
this a limitation of the KGML data.). The graph construction functions remove these duplicate edge types because they
do not provide unique information.
"""

from __future__ import annotations

import Bio.KEGG.REST
import os
import pickle
import re
import warnings
import xml.etree.ElementTree as ETree

from tqdm import tqdm
from typing import Any, Dict, List, Set, Tuple, Union


# Custom types for type hints
NodeInfoDict = Dict[str, Dict[str, Any]]
AdjacencyDict = Dict[str, Union[Set[str], Set[Tuple[str, Tuple[str]]]]]


def re_strip(string: str, pattern: str) -> str:
    """ A version of the str.strip() function that works with regular expressions"""
    if pattern is None:
        string = re.sub(r"^\s*", "", string)
        string = re.sub(r"\s*$", "", string)
    else:
        string = re.sub(pattern, "", string)
    return string


def _parse_brite_entry(line: str) -> dict:
    """
    Extracts structured information from an entry in the KEGG BRITE orthology. Each piece of information is
    returned in a dictionary object with the following keys:

        "level" - the entry's level in the orthology (A, B, C, or D)

        "id" - the numeric component of the BRITE ID (or KEGG ID if it exists)

        "gene_symbol" - The gene symbol associated with a level D entry

        "name" - The entry's descriptive name

        "kegg_orthology" - KEGG orthology (KO) IDs for level D entries

        "orthology_symbol" - The symbol of the associated KO entry

        "orthology_name" - The descriptive name of the associated KO entry

        "accession" - The KEGG hsa ID of a pathway or gene, if it exists

    :param line: A single line representing and entry in the KEGG hsa BRITE orthology (BR:hsa00001)
    :type line: str

    :returns: A dictionary of information extracted from the BRITE entry
    :rtype: dict
    """

    regexp = re.compile(
        r"(?P<level>^[A-D]) *"
        r"(?P<id>\d+) "
        r"(?P<gene_symbol>[A-Za-z\d\-]+(?=; ))?(?:; )?"
        r"(?P<name>[^\t\n\[]+(?!\[))(?:[\t\n]| \[)?"
        r"(?P<kegg_orthology>K\d+)? ?"
        r"(?P<orthology_symbol>[A-Za-z\d\-]+(?=; ))?(?:; )?"
        r"(?P<orthology_name>(?<!\[)[^\t\n\[]+(?!\[))?\s?\[?"
        r"(?P<accession>[^]]+(?=]))?]?"
    )
    entry_dict = regexp.search(line).groupdict()
    return entry_dict


def adjacency_from_path(
        identifier: str, gene_adjacency: AdjacencyDict = None, path_adjacency: AdjacencyDict = None
) -> Tuple[AdjacencyDict, AdjacencyDict]:
    """
        This function uses KGML data to create/update adjacency dictionaries for pathways (BRITE orthology level C)
    and genes (BRITE orthology level D). If adjacency dictionaries are supplied as arguments to `gene_adjacency` or
    `path_adjacency`, this function will update them with the set of edges listed in the KGML data for the KEGG
    pathway identified by the `identifier` parameter. If no adjacency dictionaries are supplied, new ones will be
    created. Edges in the level D adjacency dictionary are associated with one or more edge types; edges in the
    level C adjacency dictionary do not have different types. The edges are directed. In the returned object,
    any neighbor of a node is interpreted as a target for the edge. Therefore, undirected edges are represented as two
    directed edges. Sometimes the KGML data lists replicate edge types between the same two nodes; in this case the
    replicates will be removed so that the listed edge types are unique.

        KGML is an XML format for representing KEGG pathway maps. The "entry" and "relation" elements are used to create
    the adjacency lists. The "entry" element represents nodes in the pathway. Nodes are not restricted to genes, and can
    include other pathways, non- gene-encoded compounds, groups of nodes (e.g. representing protein  complexes), and
    more. Each entry has a numeric ID, and is associated with one or more KEGG IDs. Information about edges involving
    gene-encoded molecules is given by the "relation" elements. Each relation is between two entries. If a relation's
    entries are associated with multiple KEGG IDs, a directed edge is formed between all pairs of KEGG IDs when the
    adjacency list is being created/updated.

        It is worth noting that KGML is not used internally by KEGG to generate reference pathway drawings; it is simply
    a data exchange format intended for use by external investigators who are interested in using pathway data in
    their analyses. As such, some of the relations that are present in manually-drawn KEGG reference pathways are not
    represented by the KGML data.

        KGML has other characteristics that may be viewed as a limitation with regards to graphing biological
    relationships between genes. For example, not all entries represent individual molecules, but rather groups of
    molecules that belong to a family but are not necessarily functionally related. Unavoidably, edges are created
    from each member of the first group to all members of the second group, even if there is no true biological
    basis for an edge. Also, the human pathway KGML data may contain entries that are not present in humans,
    but these are never represented by identifiers that contain the hsa (Homo sapiens) organism code, and are ignored
    when the adjacency dictionaries are created/updated.

        For more information about the KGML format, refer to the documentation at https://www.genome.jp/kegg/xml/docs/.


        :param identifier: KEGG identifier that corresponds to a pathway associated with KGML data.
        :type identifier: str

        :param gene_adjacency: An adjacency dictionary with KEGG hsa IDs as keys and sets of KEGG hsa IDs as values. Defaults to None.
        :type gene_adjacency: dict, optional

        :param path_adjacency: An adjacency dictionary with KEGG hsa IDs as keys and sets of KEGG hsa IDs as values. Defaults to None.
        :type path_adjacency: dict, optional

        :return: Tuple of dictionaries of :class:`set` objects. The first item in the tuple is the adjacency list for the genes, and the second item is the adjacency list for the pathways.
        """

    gene_adjacency = dict() if gene_adjacency is None else {key: set(value) for key, value in gene_adjacency.items()}
    path_adjacency = dict() if path_adjacency is None else {key: set(value) for key, value in path_adjacency.items()}

    with Bio.KEGG.REST.kegg_get(identifier, option="kgml") as query:
        content = query.read()

    tree = ETree.fromstring(content)

    kgml_id_map = dict()
    for entry in tree.iter("entry"):  # map the KGML ID number to the KEGG ID number
        if entry.attrib["type"] == "group":  # Map the group id to its components' accessions instead of 'undefined'
            kegg_id_list = list()
            kgml_id = entry.attrib["id"]
            component_list = tree.findall(f"./entry[@id='{kgml_id}']/component")
            component_list = [cmp.attrib["id"] for cmp in component_list]
            for component_id in component_list:
                component_acc = tree.find(f"./entry[@id='{component_id}']").attrib["name"].split(" ")
                kegg_id_list.extend(component_acc)
        else:
            kegg_id_list = entry.attrib["name"].split(" ")
        kegg_id_list = [re.sub("^path", "PATH", kegg_id) for kegg_id in kegg_id_list]
        for kegg_id in kegg_id_list:
            # Make an edge to entry pathway from pathway denoted by `identifier` parameter. Only add edges to
            # neighboring nodes that are present in humans (i.e. prefixed by path:hsa as opposed to, say, path:map).
            if "PATH:hsa" in kegg_id and identifier not in kegg_id:  # Prevent self-loops. Necessary.
                key = f"PATH:{identifier}"
                if key in path_adjacency.keys():
                    path_adjacency[key].add(kegg_id)
                else:
                    path_adjacency[key] = {kegg_id}

        kgml_id_map[entry.attrib["id"]] = kegg_id_list

    if tree.find("relation") is not None:
        for relation in tree.iter("relation"):
            entry1 = relation.attrib["entry1"]  # KGML ID
            entry2 = relation.attrib["entry2"]  # KGML ID
            # There are sometimes multiple subtypes (edge types). For example, p53 pathway is referenced in hsa01524
            # "Platinum Drug Resistance" in the context of p53 loss of function. Thus, the first edge type from p53
            # to its neighbors is "expression" but the second edge type is "missing interaction" which represents edge
            # attenuation. Biologically speaking, "missing interaction" signifies non-functioning p53 which cannot
            # perform its normal function of inducing the expression of its neighboring gene nodes.
            xpath_query = f"./relation[@entry1='{entry1}'][@entry2='{entry2}']/subtype"
            # Record non-redundant edge types
            edge_types = tuple(set(edge_type.attrib["name"] for edge_type in tree.findall(xpath_query)))
            for kegg_id1 in kgml_id_map[entry1]:  # Create/add to set of neighboring genes' accession numbers
                if "PATH:" not in kegg_id1:
                    kegg_id1 = re.sub("hsa:", "hsa", kegg_id1)
                    if kegg_id1 in gene_adjacency.keys():
                        for kegg_id2 in kgml_id_map[entry2]:
                            if "PATH:" not in kegg_id2:
                                kegg_id2 = re.sub("hsa:", "hsa", kegg_id2)
                                gene_adjacency[kegg_id1].add((kegg_id2, edge_types))
                    else:
                        for kegg_id2 in kgml_id_map[entry2]:
                            if "PATH:" not in kegg_id2:
                                kegg_id2 = re.sub("hsa:", "hsa", kegg_id2)
                                gene_adjacency[kegg_id1] = {(kegg_id2, edge_types)}

    return gene_adjacency, path_adjacency


def make_undirected(adjacency: AdjacencyDict) -> AdjacencyDict:
    """
    Convert all directed edges in `adjacency` to undirected edges.

    :param dict adjacency: A dictionary of sets of neighboring nodes
    :returns: An undirected adjacency dictionary
    :rtype: dict[str, set]
    """
    for key, values in adjacency.copy().items():
        for value in values:
            if value in adjacency.keys():
                adjacency[value].add(key)
            else:
                adjacency[value] = {key}
    return adjacency


def _list_ids(entries: Dict[Dict] | List[Dict]) -> List[str]:
    if type(entries) is list:
        return [entry["id"] for entry in entries]
    elif type(entries) is dict:
        return [entry["id"] for entry in entries.values()]


def _child_id_set(entry: Dict, hierarchy: List[Dict[str, Any]]) -> Set[str]:
    """
    List the children of a BRITE entry.

    :param dict entry: The BRITE entry for which to list child entry IDs.
    :param list hierarchy: A BRITE hierarchy given as a list of entry dictionaries. The list must be ordered on the
    hierarchy in a depth-first fashion.
    :returns: A set of IDs for the children of ``entry`` if ``entry["level"]`` is A, B, or C, else None
    """
    children = set()
    entry_level = entry["level"]
    child_level_map = {"A": "B", "B": "C", "C": "D"}
    start_idx = hierarchy.index(entry) + 1
    if entry_level in child_level_map.keys() and start_idx < len(hierarchy):
        child_level = child_level_map[entry_level]
        search_candidates = hierarchy[start_idx:]
        i = 0
        while i < len(search_candidates):
            candidate_level = search_candidates[i]["level"]
            if candidate_level == child_level:
                if child_level == "C" and search_candidates[i]["accession"] is not None:
                    children.add(search_candidates[i]["accession"])  # Use a KEGG hsa ID if it exists
                elif child_level == "D":
                    children.add("hsa" + search_candidates[i]["id"])  # All level D entries have KEGG hsa IDs
                else:
                    children.add(search_candidates[i]["id"])  # Entries in levels A and B do not have KEGG hsa IDs
            elif candidate_level == entry_level:
                break
            i += 1
    return children


def list_from_brite(content: str) -> List[Dict]:
    """
    Creates a list of dictionaries containing structured information about each entry in a KEGG BRITE orthology.
    Each dictionary in the list corresponds to a single entry (line) in the BRITE orthology. The dictionary has the
    following keys:

        "level" - the entry's level in the orthology (A, B, C, or D)

        "id" - the numeric component of the BRITE ID (or KEGG ID if it exists)

        "gene_symbol" - The gene symbol associated with a level D entry

        "name" - The entry's descriptive name

        "kegg_orthology" - KEGG orthology (KO) IDs for level D entries

        "orthology_symbol" - The symbol of the associated KO entry

        "orthology_name" - The descriptive name of the associated KO entry

        "accession" - The KEGG hsa ID of a pathway or gene, if it exists


    :param content: A KEGG BRITE orthology downloaded as a string
    :type content: str

    :returns: A list of dictionaries containing information about each entry in the KEGG BRITE orthology
    :rtype: list[dict]
    """

    content = re_strip(content, r"\n!\n#\n#\[ KO \| BRITE \| KEGG2 \| KEGG ]\n#Last updated: \w+ \d{1,2}, \d{4}")
    content = content.strip("+D\tGENES\tKO\n!").split("\n")
    entries = list()
    for i, line in enumerate(content):
        try:
            entries.append(_parse_brite_entry(line))
        except AttributeError:
            warnings.warn(f"Could not parse line {i}: {line}")
            continue
    for i, entry in enumerate(entries):
        children = _child_id_set(entry, entries)
        entries[i]["children"] = children
    return entries


def brite_level_dict(entries: List[Dict], level: str) -> NodeInfoDict:
    """
    Return a dictionary of BRITE entries from the specified orthology level. Each entry is represented as a
    dictionary that contains information parsed from the BRITE orthology. Redundant entries are excluded from the
    returned dictionary.

    :param entries: a list of dictionaries containing structured information about entries in the KEGG BRITE orthology
    :type entries: list[dict]

    :param level: the level of the KEGG BRITE orthology to return entries from.  Must be "A", "B", "C", or "D".
    :type level: str

    :returns: A dictionary of dictionaries containing information about entries from the specified BRITE level keyed
    by the numeric component of entries' BRITE (or KEGG) IDs.
    :rtype: dict[dict]
    """

    entries_new = dict()
    if level == "D":  # D contains redundant entries; redundant entries must be excluded
        entries = {entry["id"]: entry for entry in entries if entry["level"] == "D"}
        # Create a set of all possible (unique) KEGG hsa gene IDs
        with Bio.KEGG.REST.kegg_list("hsa") as query:
            id_list = query.read().split("\n")
            if id_list[len(id_list) - 1] == "":
                id_list = id_list[0:-1]
            id_set = set(re.search(r"(?<=hsa:)\d+", item).group() for item in id_list)
        # Put non-redundant level D BRITE entries into the new dictionary
        while len(id_set) > 0:
            key = id_set.pop()
            try:
                entries_new[f"hsa{key}"] = entries[key]
            except KeyError:
                continue
    else:
        for entry in entries:  # Add entry to entries_new iff it is from the specified level of the BRITE orthology
            if entry["level"] != level:
                continue
            if level == "C":
                new_key = entry["accession"] if entry["accession"] is not None else entry["id"]
            else:
                new_key = entry["id"]
            entries_new[new_key] = entry
    return entries_new


def remove_node(target: str, nodes: NodeInfoDict, adjacency: AdjacencyDict) -> Tuple[NodeInfoDict, AdjacencyDict]:
    """
    Remove a node from the graph defined by `nodes` and `adjacency`. If the node slated for removal is the only node
    between two other nodes U and V, a new edge will be formed between U and V that respects the flow of information
    through the node slated for removal. The removed node will also be removed from the set of children of its
    parent nodes. The parents' children sets will receive the removed node's children.

    Note: When edge types are used in downstream applications, the edge removal may invalidate the graph. For
    example, consider A -[activates]-> B -[inhibits]-> C -[activates]-> D. Depending on the mechanism of inhibition
    of C by B, this may imply that A -[inhibits]-> D through the action of B, but if B and C are removed via remove_node
    the result would be A -[activates]-> D.

    :param target: The key that identifies the node slated for removal
    :param nodes: A dictionary containing information about the graph's nodes
    :param adjacency: A dictionary that contains information about edges between nodes in the graph
    :return: A tuple of updated NodeInfoDict and AdjacencyDict objects
    """
    # The set of children will be a string (the ID) or a tuple that contains the ID (at position 0) and edge types.
    adjacency_has_edge_types = all([isinstance(child, tuple) for children in adjacency.values() for child in children])
    children_set = set()
    if target in nodes.keys():
        nodes.pop(target)
    # Remove the target node from the adjacency dict but keep track of its children
    if target in adjacency.keys():
        if adjacency_has_edge_types:
            children_set.union(set(child for child in adjacency[target] if child[0] != target))
        else:
            children_set.union(adjacency[target])
            children_set.discard(target)  # In case of self-edge
        adjacency.pop(target)
    # Form edges between the parents of the target node and all the target node's children, remove target from children
    if adjacency_has_edge_types:
        for parent, children in adjacency.items():
            for child in children.copy():
                if child[0] == target:
                    adjacency[parent].remove(child)
                    adjacency[parent] = adjacency[parent].union(children_set)
    else:
        for parent, children in adjacency.items():
            if target in children:
                adjacency[parent].remove(target)
                adjacency[parent] = adjacency[parent].union(children_set)
    return nodes, adjacency


def update_children(nodes: NodeInfoDict, children: Set[str]) -> NodeInfoDict:
    """
    Update the set of a node's children to exclude nodes that do not appear in a user-supplied set of nodes

    :param nodes: A dictionary containing information about nodes in a graph, including information about their children
    :param children: Possible children that nodes in the updated NodeInfoDict could have
    :return: A NodeInfoDict whose nodes' children sets have been updated
    """
    for node in nodes.keys():
        children_set = nodes[node]["children"]
        children_set = children_set.intersection(children)
        nodes[node]["children"] = children_set
    return nodes


if __name__ == "__main__":

    # def construct_graph(nodes: Dict[Dict], adjacency: Dict[str, Set], **kwargs) -> nx.Graph:
    #     graph = nx.Graph(**kwargs)
    #     graph.add_nodes_from([(key, value) for key, value in nodes.items()])
    #     graph.add_edges_from([(v1, v2) for v1, v1_neighbors in adjacency.items() for v2 in v1_neighbors])
    #     return graph

    # # Tests
    # with Bio.KEGG.REST.kegg_get("hsa04650", option="kgml") as query:
    #     test_kgml1 = query.read()
    #
    # with Bio.KEGG.REST.kegg_get("hsa04658", option="kgml") as query:
    #     test_kgml2 = query.read()
    #
    # with Bio.KEGG.REST.kegg_get("hsa00510", option="kgml") as query:
    #     test_kgml3 = query.read()
    #
    # with Bio.KEGG.REST.kegg_get("hsa00010", option="kgml") as query:
    #     test_kgml4 = query.read()
    #
    # with Bio.KEGG.REST.kegg_get("hsa04115", option="kgml") as query:
    #     test_kgml5 = query.read()

    print("Downloading KEGG BRITE orthology", end="... ")
    with Bio.KEGG.REST.kegg_get("br:hsa00001") as kegg_query:
        brite_content = kegg_query.read()
    print("Done")

    # Create a lists of BRITE entries extracted from br:hsa00001
    print("Converting KEGG BRITE orthology to list object", end="... ")
    brite_list = list_from_brite(brite_content)
    print("Done")

    print("Assembling node dictionaries for BRITE orthology levels", end="... ")
    level_a_dict = brite_level_dict(brite_list, level="A")
    level_b_dict = brite_level_dict(brite_list, level="B")
    level_c_dict = brite_level_dict(brite_list, level="C")
    level_d_dict = brite_level_dict(brite_list, level="D")
    # Restrict level_c_dict to nodes that are associated with hsa pathway maps
    # NB: nodes in level C that are not associated with a KEGG hsa ID still have children in level D, but since these
    # level C nodes are not relevant to humans we don't care about their children.
    for node, node_data in level_c_dict.copy().items():
        if node_data["accession"] is None:
            level_c_dict.pop(node)
    print("Done")

    # Construct the adjacency dictionaries. Adjacency dictionaries are created for levels C and D first because the
    # edges at level B depend on the edges between nodes at level C. The adjacency dictionaries for levels C and D
    # are created at the same time because both require a GET call to the KEGG REST API for each pathway. It takes
    # several minutes to build the adjacency dictionaries for levels C and D from scratch, so it is wise to save these
    # adjacency dictionaries to disk once they have been created.
    if os.path.exists("data/level_c_adj.pkl") and os.path.exists("data/level_d_adj.pkl"):

        file_in = open("data/level_c_adj.pkl", "rb")
        level_c_adj = pickle.load(file_in)
        file_in.close()

        file_in = open("data/level_d_adj.pkl", "rb")
        level_d_adj = pickle.load(file_in)
        file_in.close()

    else:

        level_c_adj = dict()
        level_d_adj = dict()
        for node in tqdm(level_c_dict.values(), total=len(level_c_dict), desc="Creating level C and D adjacency dicts"):
            if node["accession"] is not None:
                # There are two types of accessions: those prefixed with PATH and those prefixed with BR. PATH denotes a
                # pathway map. BR denotes collections of related molecules, e.g. "Ion channels." This loop only adds
                # edges to PATH: nodes. BR: nodes have degree zero.
                if re.search(r"^PATH:", node["accession"]) is not None:
                    accession = re_strip(node["accession"], "^[A-Z]*:")
                    # print(f"Updating adjacency list from {accession}")
                    level_d_adj, level_c_adj = adjacency_from_path(accession, level_d_adj, level_c_adj)

        file_out = open("data/level_c_adj.pkl", "wb")
        pickle.dump(level_c_adj, file_out)
        file_out.close()

        file_out = open("data/level_d_adj.pkl", "wb")
        pickle.dump(level_d_adj, file_out)
        file_out.close()

    level_c_adj = make_undirected(level_c_adj)

    # The level D entries in BR:hsa00001 all have hsa gene IDs, but we need to restrict the adjacency dict to hsa gene
    # IDs because the adjacency dict is derived from KGML files, which contain all sorts of entities beyond genes,
    # such as compounds (cpd)
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

    # The code below creates an adjacency list for BRITE level B. The edges are directed. For example, the following
    # relationships will be created (if level C adjacency dict is undirected):
    #
    # B:    (09176 DrugResist) -----------------------------------------------> (09143 CellGrowthDeath)
    #             |              (Edge implied by level C edge (01524,04115))             |
    #             |                                                                       |
    #             |                                                                       |
    # C:    (01524 PlatDrugRes) ----------------------------------------------> (04115 p53Signaling)
    #                             (Edge implied by level C pathway maps)
    #
    level_b_adj = dict()
    for node_data1 in tqdm(level_b_dict.values(), total=len(level_b_dict), desc="Creating level B adjacency dict"):
        children1 = set(child_id for child_id in node_data1["children"])
        for node_data2 in level_b_dict.values():
            if node_data1 != node_data2:
                children2 = set(child_id for child_id in node_data2["children"])
                for child_id in children1:
                    if child_id in level_c_adj.keys():  # NB: not all children of level B are in level_c_adj.keys()
                        # make an edge if any child of node 1 is adjacent (in level C) to any child of node 2
                        has_edge = len(level_c_adj[child_id].intersection(children2)) > 0
                        if has_edge:
                            if node_data1["id"] in level_b_adj.keys():
                                level_b_adj[node_data1["id"]].add(node_data2["id"])
                            else:
                                level_b_adj[node_data1["id"]] = {node_data2["id"]}
                            break
    level_b_adj = make_undirected(level_b_adj)
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
