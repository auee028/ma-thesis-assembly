from collections import defaultdict
import re

from robosuite.custom_utils.voxposer_utils import get_clock_time, bcolors
from robosuite.custom_utils.assembly_utils import BASE_BLOCK_ALIAS


inverse_direction = {
    "right": "left",
    "left": "right",
    "front": "back",
    "back": "front",
    "behind": "front",

    # Compound directions
    "front-left": "back-right",
    "front-right": "back-left",
    "back-left": "front-right",
    "back-right": "front-left",

    # Asymmetric relations
    "above": "above",
    "top": "top",

    "base": "base",
    "center": "center",
}


class AssemblySpatialGraph:
    def __init__(self):
        # IMPORTANT:
        # Put compound directions before their component directions.
        self.directions_2d = [
            "right",
            "left",
            "front",
            "back",
            "behind",
            "front-left",
            "front-right",
            "back-left",
            "back-right",
        ]

        self.top_alias = [
            "above",
            "on top of",
            "directly supported by",
        ]

        self.position_alias = [
            "left",
            "right",
            "front",
            "back",
            "above",
            "center",
            "front-left",
            "front-right",
            "back-left",
            "back-right",
        ]

    def __call__(self, assembly_structure):
        print(
            f"{bcolors.OKBLUE}"
            "[assembly_spatial_graph.py] Generating a spatial graph "
            f"for the assembly structure{bcolors.RESET}"
        )

        directed_graph = self._get_spatial_graph(assembly_structure)
        print("Spatial graph: ", dict(directed_graph))

        print(
            f"{bcolors.OKBLUE}"
            "[assembly_spatial_graph.py] Generating the assembly order"
            f"{bcolors.RESET}"
        )

        spatial_dag, assembly_order = self._topological_sort(
            directed_graph
        )

        print("Spatial DAG: ", spatial_dag)
        print("Assembly order: ", assembly_order)

        return spatial_dag, assembly_order

    def _get_spatial_graph(self, assembly_structure):
        """
        Parse assembly instructions into:

            object -> [(relation, reference_object, ...)]

        Examples:

            block1 is left of block0
                -> ("left", "block0")

            block1 is front-left of block0
                -> ("front-left", "block0")

            block3 is above block1 and block4
                -> ("above", "block1", "block4")
        """

        spatial_graph = defaultdict(list)

        for instr in assembly_structure:

            instr = instr.lower().strip(".")

            try:
                # ---------------------------------------------------------
                # Get main block
                # ---------------------------------------------------------
                block_match = re.search(r"(block\d+)", instr)

                if not block_match:
                    continue

                block = block_match.group(1)

                # ---------------------------------------------------------
                # Base / frame
                # ---------------------------------------------------------
                if "frame" in instr and "block0" in instr:
                    spatial_graph["block0"].append(
                        ("base", "none")
                    )

                # ---------------------------------------------------------
                # Above / top relation
                #
                # Example:
                #   block3 is above block1 and block4
                #
                # Result:
                #   ("above", "block1", "block4")
                # ---------------------------------------------------------
                top_pattern = (
                    r"(block\d+).*?"
                    r"(above|on top of|directly supported by)"
                    r"(.*)"
                )

                match = re.search(top_pattern, instr)

                if match:
                    obj = match.group(1)
                    relation = match.group(2)
                    remainder = match.group(3)

                    ref_objs = re.findall(
                        r"block\d+",
                        remainder
                    )

                    if ref_objs:
                        spatial_graph[obj].append(
                            (relation, *ref_objs)
                        )

                    # Do not continue here because an instruction
                    # could potentially contain other spatial relations.

                # ---------------------------------------------------------
                # Compound / 2D directions
                #
                # IMPORTANT:
                # directions are checked longest-first.
                # ---------------------------------------------------------
                direction_pattern = (
                    r"(block\d+).*?"
                    r"(front-left|front-right|back-left|back-right|"
                    r"right|left|front|back|behind)"
                    r".*?"
                    r"(block\d+)"
                )

                match = re.search(
                    direction_pattern,
                    instr
                )

                if match:
                    obj, direction, ref_obj = match.groups()

                    spatial_graph[obj].append(
                        (direction, ref_obj)
                    )

                # ---------------------------------------------------------
                # Center relation
                #
                # Example:
                #   block2 is at the center of block0
                #
                # Result:
                #   ("center", "block0")
                # ---------------------------------------------------------
                center_pattern = (
                    r"(block\d+).*?"
                    r"center.*?"
                    r"(block\d+)"
                )

                match = re.search(
                    center_pattern,
                    instr
                )

                if match:
                    obj, ref_obj = match.groups()

                    spatial_graph[obj].append(
                        ("center", ref_obj)
                    )

            except Exception as e:
                print(
                    "[Parser error]",
                    instr,
                    e
                )

        return spatial_graph

    def _topological_sort(self, spatial_graph):
        """
        Get the assembly order while preserving spatial relations.
        """

        visited = set()
        visiting = set()
        order = []

        dag = defaultdict(set)

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

            relationships = spatial_graph.get(obj, [])

            for relationship in relationships:

                direction = relationship[0]
                ref_objs = relationship[1:]

                # -----------------------------------------------------
                # Base relation
                # -----------------------------------------------------
                if ref_objs == ("none",):

                    dag[obj].add(
                        ("base", "")
                    )

                    continue

                # -----------------------------------------------------
                # Add complete relationship.
                #
                # For example:
                #
                # ("above", "block1", "block4")
                #
                # stays exactly as it is.
                # -----------------------------------------------------
                if not has_inverse(
                    ref_objs[0],
                    obj,
                    direction
                ):
                    dag[obj].add(
                        tuple(
                            [direction] + list(ref_objs)
                        )
                    )

                # -----------------------------------------------------
                # Recursively visit all reference blocks
                # -----------------------------------------------------
                for ref_obj in ref_objs:
                    dfs(ref_obj)

                    if obj not in order:
                        order.append(obj)

            visiting.remove(obj)
            visited.add(obj)

            if obj not in order:
                order.append(obj)

        for obj in spatial_graph.keys():
            dfs(obj)

        return (
            {k: list(v) for k, v in dag.items()},
            order
        )


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
    '''
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
    '''

    assembly_structure = [
        'The frame is the base of the assembly.',
        'block1 (pink) is inserted into the frame at the front-leftmost position.',
        'block2 (pink) is inserted into the frame, to the right of block1.',
        'block0 (yellow) is inserted into the frame, to the right of block2, occupying two adjacent holes in the front row.',
        'block3 (green) is inserted into the frame, behind block1, at the back-leftmost position.',
        'block4 (green) is inserted into the frame, to the right of block3.',
        'block5 (blue) is inserted into the frame, to the right of block4.',
        'block6 (blue) is inserted into the frame, to the right of block5.'
    ]

    print("Assembly Structure: ", assembly_structure)
    
    spatial_graph, assembly_order = g_obj(assembly_structure)
    print(spatial_graph)
    print(assembly_order)
