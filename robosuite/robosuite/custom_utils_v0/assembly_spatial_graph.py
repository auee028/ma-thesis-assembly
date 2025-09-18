import re
from collections import defaultdict
from robosuite.custom_utils.voxposer_utils import get_clock_time, bcolors
from robosuite.custom_utils.assembly_utils import BASE_BLOCK_ALIAS


class AssemblySpatialGraph:
    def __init__(self):
        self.directions_2d = ["right", "left", "front", "behind"]
        
    def __call__(self, instructions):
        print(f"{bcolors.OKBLUE}[assembly_spatial_graph.py] Generating a spatial graph for the instructions{bcolors.RESET}")
        spatial_graph = self._spatial_graph(instructions)
        print("Spatial graph: ", spatial_graph)
        
        print(f"{bcolors.OKBLUE}[assembly_spatial_graph.py] Generating the assembly order{bcolors.RESET}")
        assembly_order = self._topological_sort(spatial_graph)
        print("Assembly order: ", assembly_order)
        
        return spatial_graph, assembly_order        
        
    def _spatial_graph(self, instructions):
        """
        Parse the instructions into a structure graph
        """
        spatial_graph = defaultdict(list)  # key: block, value: list of (direction, reference_block)
        
        for i, instr in enumerate(instructions):
            words = instr.strip('.').split()
            
            # "blockX is at the base.", "blockX is on the table.", "blockX is at the bottom."
            if any(b in instr for b in BASE_BLOCK_ALIAS) and i == 0:
                block = words[0]
                direction = "base"
                ref_block = "none"
                spatial_graph[block].append((direction, ref_block))
            
            # "blockX is [direction] of blockY."
            # "blockX is at the base [direction] of blockY."
            elif any(d in instr for d in self.directions_2d):
                pattern = fr"(block\d+).*?({'|'.join(self.directions_2d)}).*?(block\d+)"
                match = re.search(pattern, instr)
                block, direction, ref_block = match.groups()
                spatial_graph[block].append((direction, ref_block))
                
            # "blockX is directly supported by blockY."
            elif "directly supported by" in instr:  # there should be one supporter ideally
                block = words[0]
                direction = "top"
                supporter = words[-1]
                spatial_graph[block].append((direction, supporter))
                
            # "blockX is above blockY and blockZ."
            # "blockX is on top of blockY and blockZ."
            elif "above" in instr or "on top of" in instr:
                block = words[0]
                direction = "top"
                supporters = [w for w in words if w.startswith("block")][1:]
                spatial_graph[block].append((direction, *supporters))
        
        return spatial_graph

    def _topological_sort(self, spatial_graph):
        """
        Get the assembly order (topological sort)
        """
        visited = set()
        order = []

        def dfs(block):
            if block in visited:
                return
                
            # Get all relationships for the current block
            relationships = spatial_graph.get(block, [])
            
            for relationship in relationships:
                direction = relationship[0]
                if direction == "top":  # only follow support relationships
                    ref_blocks = relationship[1:]  # Get all reference blocks
                    for ref_block in ref_blocks:
                        dfs(ref_block)
            visited.add(block)
            order.append(block)
            
        # Keep the blocks in the same order they appeared
        all_blocks = list(spatial_graph.keys())
        
        # DFS to get the order
        for block in all_blocks:
            dfs(block)

        return order  # No reversal needed


if __name__ == "__main__":
    g = AssemblySpatialGraph()
    
    instructions = [
        "block1 is at the base.",
        "block2 is directly supported by block1.",
        "block3 is at the base right of block1.",
        "block4 is directly supported by block3.",
        "block0 is on top of block2 and block4."
    ]
    print("Instructions: ", instructions)
    
    spatial_graph, assembly_order = g(instructions)
    
