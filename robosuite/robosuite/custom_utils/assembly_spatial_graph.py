from collections import defaultdict, deque
import heapq
import re

from robosuite.custom_utils.voxposer_utils import get_clock_time, bcolors
from robosuite.custom_utils.assembly_utils import BASE_BLOCK_ALIAS


inverse_direction = {
    "right": "left",
    "left": "right",
    "front": "behind",
    "behind": "front",
    "top": "top",  # top relations are directional (asymmetric)
    "base": "base",  # base is a one-way relation
}

class AssemblySpatialGraph:
    def __init__(self):
        self.directions_2d = ["right", "left", "front", "behind"]
        self.top_alias = ["on top of", "directly supported by", "above"]
        
    def __call__(self, assembly_structure):
        print(f"{bcolors.OKBLUE}[assembly_spatial_graph.py] Generating a spatial graph for the assembly structure{bcolors.RESET}")
        directed_graph = self._get_spatial_graph(assembly_structure)
        print("Spatial graph: ", directed_graph)
        
        print(f"{bcolors.OKBLUE}[assembly_spatial_graph.py] Generating the assembly order{bcolors.RESET}")
        spatial_dag, assembly_order = self._topological_sort(directed_graph)
        print("Spatial DAG: ", spatial_dag)
        print("Assembly order: ", assembly_order)
        
        # return spatial_graph, assembly_order
        return spatial_dag, assembly_order
        
    def _get_spatial_graph(self, assembly_structure):
        """
        Parse the assembly structure into a structure graph
        """
        spatial_graph = defaultdict(list)  # key: block, value: list of (direction, reference_block)
        
        for i, instr in enumerate(assembly_structure):
            words = instr.strip('.').split()
            
            try:
                # Handle 2D direction relations
                if i > 0 and any(f" {d} " in instr for d in self.directions_2d):
                    pattern = fr"(block\d+).*?({'|'.join(self.directions_2d)}).*?(block\d+)"
                    match = re.search(pattern, instr)
                    block, direction, ref_block = match.groups()
                    spatial_graph[block].append((direction, ref_block))
                
                # Handle top relations
                elif "on top of" in instr or "directly supported by" in instr or "above" in instr:
                    pattern = fr"(block\d+).*?({'|'.join(self.top_alias)}).*?((?:block\d+(?:,?\s?(?:and\s)?))*block\d+)"
                    match = re.search(pattern, instr)
                    block, _, ref_blocks = match.groups()
                    supporters = [b.replace(",", "") for b in ref_blocks.split(' ') if b.startswith("block")]
                    spatial_graph[block].append(("top", *supporters))
                
                # Handle base placement
                elif any(b in instr for b in BASE_BLOCK_ALIAS):
                    # block = words[0]
                    blocks = [w.replace(",", "") for w in words if 'block' in w]
                    direction = "base"
                    ref_block = "none"
                    for block in blocks:
                        spatial_graph[block].append((direction, ref_block))
                
            except Exception as e:
                print(e)
        
        return spatial_graph
        
    def _topological_sort(self, spatial_graph):
        """
        Get the assembly order (topological sort)
        """
        visited = set()
        visiting = set()
        order = []
        dag = defaultdict(set)  # key: node, value: list of prerequisite nodes
        
        def has_inverse(ref_obj, obj, direction):
            inv_dir = inverse_direction.get(direction)
            if inv_dir is None:
                return False
            return any(
                edge[0] == inv_dir and obj in edge[1:]
                for edge in dag.get(ref_obj, [])
            )

        def dfs(obj):
            if obj in visited:
                return
            if obj in visiting:
                print(f"Cycle detected at {obj}")
                return
            
            visiting.add(obj)
            
            # Get all relationships for the current object
            relationships = spatial_graph.get(obj, [])
            
            for relationship in relationships:
                direction = relationship[0]
                ref_objs = relationship[1:]
                for ref_obj in ref_objs:
                    # Add 'base' object with supporter 'none' — it's not a real object, just a placeholder
                    if ref_obj == "none":
                        dag[obj].add(tuple([direction] + [ref_obj]))
                        continue    # Skip 'none' not to be added in order
                    # Build DAG: Skip if inverse edge already exists
                    elif not has_inverse(ref_obj, obj, direction):
                        dag[obj].add(tuple([direction] + [ob for ob in ref_objs]))
                        
                    dfs(ref_obj)
                    
                    if obj not in order:
                        order.append(obj)
                        
            visiting.remove(obj)
            visited.add(obj)
            # order.append(obj)
            if obj not in order:
                order.append(obj)
            
        # Keep the objects in the same order they appeared
        all_objs = list(spatial_graph.keys())
        
        # DFS to get the order
        for obj in all_objs:
            dfs(obj)

        return {k: list(v) for k, v in dag.items()}, order  # No reversal needed for 'order'


class ObjectAssemblySpatialGraph:
    def __init__(self):
        self.directions_2d = ["right", "left", "front", "behind"]
        self.top_alias = ["on top of", "directly supported by", "above"]
        
    def __call__(self, assembly_structure):
        print(f"{bcolors.OKBLUE}[assembly_spatial_graph.py] Generating a spatial graph for the assembly structure{bcolors.RESET}")
        spatial_graph = self._get_spatial_graph(assembly_structure)
        print("Spatial graph: ", spatial_graph)
        
        print(f"{bcolors.OKBLUE}[assembly_spatial_graph.py] Generating the assembly order{bcolors.RESET}")
        spatial_dag, assembly_order = self._topological_sort(spatial_graph)
        print("Spatial DAG: ", spatial_dag)
        print("Assembly order: ", assembly_order)
        
        # return spatial_graph, assembly_order
        return spatial_dag, assembly_order
        
    def _get_spatial_graph(self, assembly_structure):
        """
        Parse the assembly structure into a structure graph ('block' -> 'object')
        """
        spatial_graph = defaultdict(list)  # key: object, value: list of (direction, reference_object)
        
        for i, instr in enumerate(assembly_structure):
            words = instr.strip('.').split()
            
            try:
                # Handle base placement
                if any(b in instr for b in BASE_BLOCK_ALIAS):
                    # object = words[0]
                    objs = [w.replace(",", "") for w in words if 'object' in w]
                    direction = "base"
                    ref_obj = "none"
                    for obj in objs:
                        spatial_graph[obj].append((direction, ref_obj))
                
                # Handle top relations
                if "on top of" in instr or "directly supported by" in instr or "above" in instr:
                    pattern = fr"(object\d+).*?({'|'.join(self.top_alias)}).*?((?:object\d+(?:,?\s?(?:and\s)?))*object\d+)"
                    match = re.search(pattern, instr)
                    obj, _, ref_objs = match.groups()
                    supporters = [b.replace(",", "") for b in ref_objs.split(' ') if b.startswith("object")]
                    spatial_graph[obj].append(("top", *supporters))
                        
                # Handle 2D direction relations
                if i > 0 and any(f" {d} " in instr for d in self.directions_2d):
                    pattern = fr"(object\d+).*?({'|'.join(self.directions_2d)}).*?(object\d+)"
                    match = re.search(pattern, instr)
                    obj, direction, ref_obj = match.groups()
                    spatial_graph[obj].append((direction, ref_obj))
                
            except Exception as e:
                print(e)
        
        return spatial_graph

    def _topological_sort(self, spatial_graph):
        """
        Get the assembly order (topological sort)
        """
        visited = set()
        visiting = set()
        order = []
        dag = defaultdict(set)  # key: node, value: list of prerequisite nodes
        
        def has_inverse(ref_obj, obj, direction):
            inv_dir = inverse_direction.get(direction)
            if inv_dir is None:
                return False
            return any(
                edge[0] == inv_dir and obj in edge[1:]
                for edge in dag.get(ref_obj, [])
            )

        def dfs(obj):
            if obj in visited:
                return
            if obj in visiting:
                print(f"Cycle detected at {obj}")
                return
            
            visiting.add(obj)
            
            # Get all relationships for the current object
            relationships = spatial_graph.get(obj, [])
            
            for relationship in relationships:
                direction = relationship[0]
                ref_objs = relationship[1:]
                for ref_obj in ref_objs:
                    # Ignore 'none' — it's not a real object, just a placeholder
                    if ref_obj == "none":  # only follow support relationships
                        # Build DAG for the base block
                        dag[obj].add(tuple(['base', '']))
                        continue
                    
                    # Build DAG: Skip if inverse edge already exists
                    if not has_inverse(ref_obj, obj, direction):
                        dag[obj].add(tuple([direction] + [ob for ob in ref_objs]))
                        
                    dfs(ref_obj)
                    
                    if obj not in order:
                        order.append(obj)
                        
            visiting.remove(obj)
            visited.add(obj)
            # order.append(obj)
            if obj not in order:
                order.append(obj)
            
        # Keep the objects in the same order they appeared
        all_objs = list(spatial_graph.keys())
        
        # DFS to get the order
        for obj in all_objs:
            dfs(obj)

        return {k: list(v) for k, v in dag.items()}, order  # No reversal needed for 'order'


if __name__ == "__main__":
    g = AssemblySpatialGraph()
    
    assembly_structure = [
        "block1 is at the base.",
        "block2 is directly supported by block1.",
        "block3 is at the base right of block1.",
        "block4 is directly supported by block3.",
        "block0 is on top of block2 and block4."
    ]
    print("assembly_structure: ", assembly_structure)
    
    spatial_graph, assembly_order = g(assembly_structure)
    
    print("* * * * *")
    
    g_obj = ObjectAssemblySpatialGraph()
    
    assembly_structure = [
        "object0 is at the base.",
        "object2 is directly supported by object0.",
        "object3 is directly supported by object2.",
        "object7 is directly supported by object3.",
        "object4 is directly supported by object7.",
        "object5 is left of object6.",
        "object6 is directly supported by object4.",
        "object1 is on top of object6."
    ]
    print("Assembly Structure: ", assembly_structure)
    
    spatial_graph, assembly_order = g_obj(assembly_structure)
    
    print("* * * * *")
    
    assembly_structure = [
        "object0 is at the base.",
        "object4 is on top of object0.",
        "object3 is on top of object4.",
        "object2 is on top of object3.",
        "object5 is on top of object2.",
        "object6 is left of object7 and on top of object5.",
        "object7 is right of object6 and on top of object5.",
        "object1 is on top of object6 and object7."
    ]
    print("Assembly Structure: ", assembly_structure)
    
    spatial_graph, assembly_order = g_obj(assembly_structure)
    
    print("* * * * *")
    
    assembly_structure = [
        "object0 is at the bottom.",
        "object4 is on top of object0.",
        "object2 is on top of object4.",
        "object3 is on top of object2.",
        "object5 is on top of object3.",
        "object6 is left of object7 and on top of object5.",
        "object1 is on top of object6 and object7."
    ]
    print("Assembly Structure: ", assembly_structure)
    
    spatial_graph, assembly_order = g_obj(assembly_structure)
    
