
from collections import defaultdict
import re

# 1. Parse the instructions into a structure graph

def get_spatial_graph(instructions):
    spatial_graph = defaultdict(list)  # key: block, value: list of (direction, reference_block)
    
    directions_2d = ["right", "left", "front", "behind"]
    
    for instr in instructions:
        words = instr.strip('.').split()
        
        # "blockX is at the base."
        if "at the base":
            block = words[0]
            direction = "base"
            ref_block = 'none'
            spatial_graph[block].append((direction, ref_block))
        
        # "blockX is [direction] of blockY."
        # "blockX is at the base [direction] of blockY."
        elif any(d in instr for d in directions_2d):
            match = re.search(r"(block\d+).*base (\w+) of (block\d+)", instr)
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

instructions = [
    "block1 is at the base.",
    "block2 is directly supported by block1.",
    "block3 is at the base right of block1.",
    "block4 is directly supported by block3.",
    "block0 is on top of block2 and block4."
]
spatial_graph = get_spatial_graph(instructions)

# 2. Get the assembly order (topological sort)
def topological_sort(spatial_graph):
    visited = set()
    order = []

    def dfs(block):
        if block in visited:
            return
        for direction, ref_block in spatial_graph.get(block, []):
            if direction == "top" and ref_block:  # only follow support relationships
                dfs(ref_block)
        visited.add(block)
        order.append(block)

    for block in spatial_graph:
        dfs(block)

    return order[::-1]  # reverse to get correct topological order

assembly_order = topological_sort(spatial_graph)
print("Assembly order:", assembly_order)
