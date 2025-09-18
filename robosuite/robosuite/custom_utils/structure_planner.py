import os
import re
from datetime import datetime

from robosuite.custom_utils.assembly_utils import *
from robosuite.custom_utils.voxposer_utils import get_clock_time, bcolors

from llama_index.llms.openai import OpenAI
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock, ImageBlock

import openai


class StructurePlanner:
    """
    High-level structure planner using LLMs (here, GPT-4o)
    """
    def __init__(self, configs, use_primitives=True):
        self.config = configs
        self.use_prim = use_primitives
        
        self.llm_config = self.config.get("llm")
        self.llm_provider = self.llm_config.get("model_config").get("provider")
        self.llm_model = self.llm_config.get("model_config").get("model")

        self.which_img = self.config.get("which_img")
        
        self.target_config = self.config.get("task")
        
        self._register_api_key()
        self.prompt_files = self._get_prompt_files()
        
    def _register_api_key(self):
        """
        Set the api key
        """
        os.environ["OPENAI_API_KEY"] = self.llm_config.get("api_keys").get("openai")
    
    def _get_prompt_files(self):
        """
        Get file paths of prompt entries
        """
        prompt_files = {}
        
        prompt_files["system_prompt"] = self.config.get("relation").get("prim_prompt_files").get("system_prompt")
        prompt_files["examples_prompt"] = self.config.get("relation").get("prim_prompt_files").get("examples_prompt")
                
        prompt_files["examples_prompt_dist"] = self.config.get("relation").get("dist_prompt_files").get("examples_prompt")

        prompt_files["example_rgb_paths"] = self.config.get("relation").get("example_images").get("example_rgb_paths")
        prompt_files["example_depth_paths"] = self.config.get("relation").get("example_images").get("example_depth_paths")
        prompt_files["example_mask_paths"] = self.config.get("relation").get("example_images").get("example_mask_paths")
        prompt_files["example_rgbd_paths"] = self.config.get("relation").get("example_images").get("example_rgbd_paths")
        prompt_files["example_rgbseg_paths"] = self.config.get("relation").get("example_images").get("example_rgbseg_paths")
        
        target_dir = os.path.join(self.config.get("save_dir"), self.target_config.get("target_dir_tmp"))
        
        prompt_files["target_query"] = self.target_config.get("target_query_structure")  # Query structure
        
        prompt_files["target_rgb_path"] = os.path.join(target_dir, self.target_config.get("target_rgb_file")) \
            if self.which_img in ["rgb", "rgb_d", "rgb_m", "rgb_d_m"] else ""   # Color image to analyze
        prompt_files["target_depth_path"] = os.path.join(target_dir, self.target_config.get("target_depth_file")) \
            if self.which_img in ["rgb_d", "rgb_d_m"] else ""  # Depth image to analyze
        prompt_files["target_mask_path"] = os.path.join(target_dir, self.target_config.get("target_mask_file")) \
            if self.which_img in ["rgb_m", "rgb_d_m"] else ""  # Segmentation mask to analyze
        prompt_files["target_rgbd_path"] = os.path.join(target_dir, self.target_config.get("target_rgbd_file")) \
            if self.which_img == "fused_rgbd" else ""  # 4-Channel RGB-D to analyze
        prompt_files["target_rgbseg_path"] = os.path.join(target_dir, self.target_config.get("target_rgbseg_file")) \
            if self.which_img == "fused_rgbm" else ""  # blocks segmentation image to analyze
        
        return prompt_files

    def _process_examples(
            self,
            examples_prompt,
            example_rgbs,
            example_depths,
            example_masks,
            example_rgbds,
            example_rgbsegs):
        """
        Process raw examples into structured format
        """
        example_entries = examples_prompt.split('### Example')[1:]  # Skip first empty split
        example_rgb_paths = [path.strip() for path in example_rgbs.split('\n') if path.strip()]
        example_depth_paths = [path.strip() for path in example_depths.split('\n') if path.strip()]
        example_mask_paths = [path.strip() for path in example_masks.split('\n') if path.strip()]
        example_rgbd_paths = [path.strip() for path in example_rgbds.split('\n') if path.strip()]
        example_rgbseg_paths = [path.strip() for path in example_rgbsegs.split('\n') if path.strip()]
        
        examples = []
        for i, entry in enumerate(example_entries):
            if i >= len(example_rgb_paths):
                break  # Ensure we don't exceed available images
                
            # Extract components from each example
            parts = entry.split('Output:')
            input_part = parts[0]
            output_json = parts[1].strip()
            
            # Get image path and encode
            input_rgb_path = example_rgb_paths[i]
            input_depth_path = example_depth_paths[i] if example_depth_paths else ""
            input_mask_path = example_mask_paths[i] if example_mask_paths else ""
            input_rgbd_path = example_rgbd_paths[i] if example_rgbd_paths else ""
            input_rgbseg_path = example_rgbseg_paths[i] if example_rgbseg_paths else ""
            
            # Get query
            query = input_part.split('Query: "')[1].split('"')[0]
            
            examples.append({
                'query': query,
                'rgb_path': input_rgb_path,
                'depth_path': input_depth_path,
                'mask_path': input_mask_path,
                'rgbd_path': input_rgbd_path,
                'rgbseg_path': input_rgbseg_path,
                'output': output_json
            })
        
        return examples
    
    def _process_examples_for_distances(
            self,
            examples_prompt,
            example_rgbs,
            example_depths,
            example_masks,
            example_rgbds,
            example_rgbsegs):
        """
        Process raw examples into structured format
        """
        example_entries = re.split(f'### Example \d+\s*', examples_prompt)[1:]  # Skip first empty split
        example_rgb_paths = [path.strip() for path in example_rgbs.split('\n') if path.strip()]
        example_depth_paths = [path.strip() for path in example_depths.split('\n') if path.strip()]
        example_mask_paths = [path.strip() for path in example_masks.split('\n') if path.strip()]
        example_rgbd_paths = [path.strip() for path in example_rgbds.split('\n') if path.strip()]
        example_rgbseg_paths = [path.strip() for path in example_rgbsegs.split('\n') if path.strip()]
        
        examples = []
        for i, entry in enumerate(example_entries):
            if i >= len(example_rgb_paths):
                break  # Ensure we don't exceed available images
                
            # Extract components from each example
            entry = entry.strip()
            
            query = re.search(r"=\s*'([^']+)'", entry).group(1)
            input_rgb_path = example_rgb_paths[i] if example_rgb_paths else ""
            input_depth_path = example_depth_paths[i] if example_depth_paths else ""
            input_mask_path = example_mask_paths[i] if example_mask_paths else ""
            input_rgbd_path = example_rgbd_paths[i] if example_rgbd_paths else ""
            input_rgbseg_path = example_rgbseg_paths[i] if example_rgbseg_paths else ""
            output_part = '\n'.join(entry.splitlines()[1:])
            
            examples.append({
                'query': query,
                'rgb_path': input_rgb_path,
                'depth_path': input_depth_path,
                'mask_path': input_mask_path,
                'rgbd_path': input_rgbd_path,
                'rgbseg_path': input_rgbseg_path,
                'output': output_part,
            })
        
        return examples
    
    def _build_messages_openai(self):
        """
        Build complete message only for OpenAI API
        """
        # Load prompt entries
        system_prompt = read_file(self.prompt_files["system_prompt"])
        examples_prompt = read_file(self.prompt_files["examples_prompt"])
        rgb_image_paths = read_file(self.prompt_files["example_rgb_paths"])
        depth_image_paths = read_file(self.prompt_files["example_depth_paths"])
        mask_image_paths = read_file(self.prompt_files["example_mask_paths"])
        rgbd_image_paths = read_file(self.prompt_files["example_rgbd_paths"])
        rgbseg_image_paths = read_file(self.prompt_files["example_rgbseg_paths"])

        examples = self._process_examples(
            examples_prompt,
            rgb_image_paths,
            depth_image_paths,
            mask_image_paths,
            rgbd_image_paths,
            rgbseg_image_paths
        )

        target_query = self.prompt_files["target_query"]
        target_rgb_path = self.prompt_files["target_rgb_path"]
        target_depth_path = self.prompt_files["target_depth_path"]
        target_mask_path = self.prompt_files["target_mask_path"]
        target_rgbd_path = self.prompt_files["target_rgbd_path"]
        target_rgbseg_path = self.prompt_files["target_rgbseg_path"]
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        if self.which_img == "rgb":    # RGB without depth or mask
            assert len(target_rgb_path) != 0 and len(target_depth_path) == 0 and len(target_mask_path) == 0 \
                and len(target_rgbd_path) == 0 and len(target_rgbseg_path) == 0
            
            # Add few-shot examples
            for example in examples:
                # User message (image + query)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze these blocks. Query: {example['query']}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(example['rgb_path'])}"}}
                    ]
                })
                
                # Assistant response
                messages.append({
                    "role": "assistant",
                    "content": example['output']
                })
            
            # Add target query
            target_b64 = encode_image(target_rgb_path)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Analyze these blocks. Query: {target_query}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{target_b64}"}}
                ]
            })
        
        elif self.which_img == "rgb_d":    # separate RGB+D
            assert len(target_rgb_path) != 0 and len(target_depth_path) != 0 and len(target_mask_path) == 0 \
                and len(target_rgbd_path) == 0 and len(target_rgbseg_path) == 0
            
            # Add few-shot examples
            for example in examples:
                # User message (image + query)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze these blocks. Query: {example['query']}"},
                        {"type": "text", "text": f"RGB image: "},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(example['rgb_path'])}"}},
                        {"type": "text", "text": f"Depth image: "},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(example['depth_path'])}"}}
                    ]
                })
                
                # Assistant response
                messages.append({
                    "role": "assistant",
                    "content": example['output']
                })
            
            # Add target query
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Analyze these blocks. Query: {target_query}"},
                    {"type": "text", "text": f"RGB image: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_rgb_path)}"}},
                    {"type": "text", "text": f"Depth image: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_depth_path)}"}}
                ]
            })

        elif self.which_img == "rgb_m":    # separate RGB+M
            assert len(target_rgb_path) != 0 and len(target_depth_path) == 0 and len(target_mask_path) != 0 \
                and len(target_rgbd_path) == 0 and len(target_rgbseg_path) == 0
            
            # Add few-shot examples
            for example in examples:
                # User message (image + query)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze these blocks. Query: {example['query']}"},
                        {"type": "text", "text": f"RGB image: "},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(example['rgb_path'])}"}},
                        {"type": "text", "text": f"Segmentation mask: "},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(example['mask_path'])}"}}
                    ]
                })
                
                # Assistant response
                messages.append({
                    "role": "assistant",
                    "content": example['output']
                })
            
            # Add target query
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Analyze these blocks. Query: {target_query}"},
                    {"type": "text", "text": f"RGB image: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_rgb_path)}"}},
                    {"type": "text", "text": f"Segmentation mask: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_mask_path)}"}}
                ]
            })
        
        elif self.which_img == "rgb_d_m":    # separate RGB+D+M
            assert len(target_rgb_path) != 0 and len(target_depth_path) != 0 and len(target_mask_path) != 0 \
                and len(target_rgbd_path) == 0 and len(target_rgbseg_path) == 0
            
            # Add few-shot examples
            for example in examples:
                # User message (image + query)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze these blocks. Query: {example['query']}"},
                        {"type": "text", "text": f"RGB image: "},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(example['rgb_path'])}"}},
                        {"type": "text", "text": f"Depth image: "},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(example['depth_path'])}"}},
                        {"type": "text", "text": f"Segmentation mask: "},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(example['mask_path'])}"}}
                    ]
                })
                
                # Assistant response
                messages.append({
                    "role": "assistant",
                    "content": example['output']
                })
            
            # Add target query
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Analyze these blocks. Query: {target_query}"},
                    {"type": "text", "text": f"RGB image: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_rgb_path)}"}},
                    {"type": "text", "text": f"Depth image: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_depth_path)}"}},
                    {"type": "text", "text": f"Segmentation mask: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_mask_path)}"}}
                ]
            })

        elif self.which_img == "fused_rgbd":    # 4-channel RGB-D
            assert len(target_rgb_path) == 0 and len(target_depth_path) == 0 and len(target_mask_path) == 0 \
                and len(target_rgbd_path) != 0 and len(target_rgbseg_path) == 0
            
            # Add few-shot examples
            for example in examples:
                # User message (image + query)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze these blocks. Query: {example['query']}"},
                        {"type": "text", "text": f"4-channel RGB-D image: "},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(example['rgbd_path'])}"}}
                    ]
                })
                
                # Assistant response
                messages.append({
                    "role": "assistant",
                    "content": example['output']
                })
            
            # Add target query
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Analyze these blocks. Query: {target_query}"},
                    {"type": "text", "text": f"4-channel RGB-D image: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_rgbd_path)}"}}
                ]
            })
        
        elif self.which_img == "fused_rgbm":    # RGB block segmentation
            assert len(target_rgb_path) == 0 and len(target_depth_path) == 0 and len(target_mask_path) == 0 \
                and len(target_rgbd_path) == 0 and len(target_rgbseg_path) != 0
            
            # Add few-shot examples
            for example in examples:
                # User message (image + query)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze these blocks. Query: {example['query']}"},
                        {"type": "text", "text": f"RGB segmentation: "},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(example['rgbseg_path'])}"}}
                    ]
                })
                
                # Assistant response
                messages.append({
                    "role": "assistant",
                    "content": example['output']
                })
            
            # Add target query
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Analyze these blocks. Query: {target_query}"},
                    {"type": "text", "text": f"RGB segmentation: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_rgbseg_path)}"}},
                ]
            })
        
        else:
            raise Exception("Error: Check the image paths in config or in txt files")
        
        return messages
    
    def _build_messages_openai_for_distances(self):
        """
        Build complete message for numerical distances (only for OpenAI API)
        """
        # Set and load prompt entries
        examples_prompt = read_file(self.prompt_files["examples_prompt_dist"])
        rgb_image_paths = read_file(self.prompt_files["example_rgb_paths"])
        depth_image_paths = read_file(self.prompt_files["example_depth_paths"])
        mask_image_paths = read_file(self.prompt_files["example_mask_paths"])
        rgbd_image_paths = read_file(self.prompt_files["example_rgbd_paths"])
        rgbseg_image_paths = read_file(self.prompt_files["example_rgbseg_paths"])
        
        examples = self._process_examples_for_distances(
            examples_prompt,
            rgb_image_paths,
            depth_image_paths,
            mask_image_paths,
            rgbd_image_paths,
            rgbseg_image_paths
        )
        
        target_query = self.prompt_files["target_query"]
        target_rgb_path = self.prompt_files["target_rgb_path"]
        target_depth_path = self.prompt_files["target_depth_path"]
        target_mask_path = self.prompt_files["target_mask_path"]
        target_rgbd_path = self.prompt_files["target_rgbd_path"]
        target_rgbseg_path = self.prompt_files["target_rgbseg_path"]

        output_format = '''{\n  "detected_blocks": [\n    {\n      "block_id":  "string (must start with 'block' followed by number)",\n      "color": "string (basic color name)",\n      "size": "float (length of the block in meters)"\n    }\n  ],\n  "structure_relations": [\n    {\n      "from": "string (reference block ID)",\n      "to": "string (target block ID)",\n      "center_distance": {\n        "x": "float (distance along x-axis)",\n        "y": "float (distance along y-axis)",\n        "z": "float (distance along z-axis)"\n      }\n    }\n  ]\n}'''
        
        # Build messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant that pays attention to the user's instructions and writes a json form containing the elements that the user wants."}
        ]
        if self.which_img == "unpaired":    # Without any images (RGB only in target query)
            assert len(target_rgb_path) == 0 and len(target_depth_path) == 0 and len(target_mask_path) == 0 \
                and len(target_rgbd_path) == 0 and len(target_rgbseg_path) == 0
        
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"I want you to generate a directed graph that represents contact information as relationships between blocks on the table. This directed graph will encode the structural information needed to build the input query structure. You are given an RGB image and a query structure. Follow these steps: 1. Detect each building block on the table in the input image. 2. Assign a unique 'block_id' and 'color' to individual building blocks. 3. Generate a directed graph with nodes of the building blocks and edges of relative distances from one block to another. Note that x is back to front, y is left to right, and z is bottom to up. Expected output format is ```{output_format}```\nFor reference, I will give you some examples.\n\n"}
                ]
            })
        
            for ex_id, example in enumerate(examples):
                messages[-1]["content"] += [
                    {"type": "text", "text": f"### Example {ex_id + 1}:\n"},
                    {"type": "text", "text": f"Query structure: {example['query']}"},
                    {"type": "text", "text": f"Expected output: ```{example['output']}```\n\n"}
                ]
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"### Respond a correct output to this query:\n"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_rgb_path)}"}},
                    {"type": "text", "text": f"Query structure: {target_query}"}
                ],
            })
        
        elif self.which_img == "rgb":    # RGB without depth or mask in demonstrations
            assert len(target_rgb_path) != 0 and len(target_depth_path) == 0 and len(target_mask_path) == 0 \
                and len(target_rgbd_path) == 0 and len(target_rgbseg_path) == 0
        
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"I want you to generate a directed graph that represents contact information as relationships between blocks on the table. This directed graph will encode the structural information needed to build the input query structure. You are given an RGB image and a query structure. Follow these steps: 1. Detect each building block on the table in the input image. 2. Assign a unique 'block_id' and 'color' to individual building blocks. 3. Generate a directed graph with nodes of the building blocks and edges of relative distances from one block to another. Note that x is back to front, y is left to right, and z is bottom to up. Expected output format is ```{output_format}```\nFor reference, I will give you some examples.\n\n"}
                ]
            })
        
            for ex_id, example in enumerate(examples):
                messages[-1]["content"] += [
                    {"type": "text", "text": f"### Example {ex_id + 1}:\n"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(example['rgb_path'])}"}},
                    {"type": "text", "text": f"Query structure: {example['query']}"},
                    {"type": "text", "text": f"Expected output: ```{example['output']}```\n\n"}
                ]
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"### Respond a correct output to this query:\n"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_rgb_path)}"}},
                    {"type": "text", "text": f"Query structure: {target_query}"}
                ],
            })
        
        elif self.which_img == "rgb_d":    # separate RGB+D
            assert len(target_rgb_path) != 0 and len(target_depth_path) != 0 and len(target_mask_path) == 0 \
                and len(target_rgbd_path) == 0 and len(target_rgbseg_path) == 0
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"I want you to generate a directed graph that represents contact information as relationships between blocks on the table. This directed graph will encode the structural information needed to build the input query structure. You are given an RGB image, a depth image (in millimeters), and a query structure. Follow these steps: 1. Detect each building block on the table in the input image. 2. Assign a unique 'block_id' and 'color' to individual building blocks. 3. Generate a directed graph with nodes of the building blocks and edges of relative distances from one block to another. Note that x is back to front, y is left to right, and z is bottom to up. Expected output format is ```{output_format}```\nFor reference, I will give you some examples.\n\n"}
                ]
            })
        
            for ex_id, example in enumerate(examples):
                messages[-1]["content"] += [
                    {"type": "text", "text": f"Example {ex_id + 1}:\n"},
                    {"type": "text", "text": f"RGB image: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(example['rgb_path'])}"}},
                    {"type": "text", "text": f"Depth image: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(example['depth_path'])}"}},
                    {"type": "text", "text": f"Query structure: {example['query']}"},
                    {"type": "text", "text": f"Expected output: ```{example['output']}```\n\n"}
                ]
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"### Respond a correct output to this query:\n"},
                    {"type": "text", "text": f"RGB image: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_rgb_path)}"}},
                    {"type": "text", "text": f"Depth image: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_depth_path)}"}},
                    {"type": "text", "text": f"Query structure: {target_query}"}
                ],
            })
        
        elif self.which_img == "rgb_m":    # separate RGB+M
            assert len(target_rgb_path) != 0 and len(target_depth_path) == 0 and len(target_mask_path) != 0 \
                and len(target_rgbd_path) == 0 and len(target_rgbseg_path) == 0
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"I want you to generate a directed graph that represents contact information as relationships between blocks on the table. This directed graph will encode the structural information needed to build the input query structure. You are given an RGB image, a segmentation mask of the blocks, and a query structure. Follow these steps: 1. Detect each building block on the table in the input image. 2. Assign a unique 'block_id' and 'color' to individual building blocks. 3. Generate a directed graph with nodes of the building blocks and edges of relative distances from one block to another. Note that x is back to front, y is left to right, and z is bottom to up. Expected output format is ```{output_format}```\nFor reference, I will give you some examples.\n\n"}
                ]
            })
        
            for ex_id, example in enumerate(examples):
                messages[-1]["content"] += [
                    {"type": "text", "text": f"Example {ex_id + 1}:\n"},
                    {"type": "text", "text": f"RGB image: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(example['rgb_path'])}"}},
                    {"type": "text", "text": f"Segmentation mask: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(example['mask_path'])}"}},
                    {"type": "text", "text": f"Query structure: {example['query']}"},
                    {"type": "text", "text": f"Expected output: ```{example['output']}```\n\n"}
                ]
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"### Respond a correct output to this query:\n"},
                    {"type": "text", "text": f"RGB image: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_rgb_path)}"}},
                    {"type": "text", "text": f"Segmentation mask: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_mask_path)}"}},
                    {"type": "text", "text": f"Query structure: {target_query}"}
                ],
            })
        
        elif self.which_img == "rgb_d_m":    # separate RGB+D+M
            assert len(target_rgb_path) != 0 and len(target_depth_path) != 0 and len(target_mask_path) != 0 \
                and len(target_rgbd_path) == 0 and len(target_rgbseg_path) == 0
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"I want you to generate a directed graph that represents contact information as relationships between blocks on the table. This directed graph will encode the structural information needed to build the input query structure. You are given an RGB image, a depth image (in millimeters), a segmentation mask of the blocks, and a query structure. Follow these steps: 1. Detect each building block on the table in the input image. 2. Assign a unique 'block_id' and 'color' to individual building blocks. 3. Generate a directed graph with nodes of the building blocks and edges of relative distances from one block to another. Note that x is back to front, y is left to right, and z is bottom to up. Expected output format is ```{output_format}```\nFor reference, I will give you some examples.\n\n"}
                ]
            })
        
            for ex_id, example in enumerate(examples):
                messages[-1]["content"] += [
                    {"type": "text", "text": f"Example {ex_id + 1}:\n"},
                    {"type": "text", "text": f"RGB image: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(example['rgb_path'])}"}},
                    {"type": "text", "text": f"Depth image: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(example['depth_path'])}"}},
                    {"type": "text", "text": f"Segmentation mask: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(example['mask_path'])}"}},
                    {"type": "text", "text": f"Query structure: {example['query']}"},
                    {"type": "text", "text": f"Expected output: ```{example['output']}```\n\n"}
                ]
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"### Respond a correct output to this query:\n"},
                    {"type": "text", "text": f"RGB image: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_rgb_path)}"}},
                    {"type": "text", "text": f"Depth image: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_depth_path)}"}},
                    {"type": "text", "text": f"Segmentation mask: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_mask_path)}"}},
                    {"type": "text", "text": f"Query structure: {target_query}"}
                ],
            })
        
        elif self.which_img == "fused_rgbd":    # 4-channel RGB-D
            assert len(target_rgb_path) == 0 and len(target_depth_path) == 0 and len(target_mask_path) == 0 \
                and len(target_rgbd_path) != 0 and len(target_rgbseg_path) == 0
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"I want you to generate a directed graph that represents contact information as relationships between blocks on the table. This directed graph will encode the structural information needed to build the input query structure. You are given a 4-channel RGB-D image (depth in millimeters), and a query structure. Follow these steps: 1. Detect each building block on the table in the input image. 2. Assign a unique 'block_id' and 'color' to individual building blocks. 3. Generate a directed graph with nodes of the building blocks and edges of relative distances from one block to another. Note that x is back to front, y is left to right, and z is bottom to up. Expected output format is ```{output_format}```\nFor reference, I will give you some examples.\n\n"}
                ]
            })
        
            for ex_id, example in enumerate(examples):
                messages[-1]["content"] += [
                    {"type": "text", "text": f"Example {ex_id + 1}:\n"},
                    {"type": "text", "text": f"4-channel RGB-D image: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(example['rgbd_path'])}"}},
                    {"type": "text", "text": f"Query structure: {example['query']}"},
                    {"type": "text", "text": f"Expected output: ```{example['output']}```\n\n"}
                ]
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"### Respond a correct output to this query:\n"},
                    {"type": "text", "text": f"4-channel RGB-D image: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_rgbd_path)}"}},
                    {"type": "text", "text": f"Query structure: {target_query}"}
                ],
            })
        
        elif self.which_img == "fused_rgbm":    # RGB Segmentation of blocks
            assert len(target_rgb_path) == 0 and len(target_depth_path) == 0 and len(target_mask_path) == 0 \
                and len(target_rgbd_path) == 0 and len(target_rgbseg_path) != 0
        
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"I want you to generate a directed graph that represents contact information as relationships between blocks on the table. This directed graph will encode the structural information needed to build the input query structure. You are given an RGB segmentation of blocks, and a query structure. Follow these steps: 1. Detect each building block on the table in the input image. 2. Assign a unique 'block_id' and 'color' to individual building blocks. 3. Generate a directed graph with nodes of the building blocks and edges of relative distances from one block to another. Note that x is back to front, y is left to right, and z is bottom to up. Expected output format is ```{output_format}```\nFor reference, I will give you some examples.\n\n"}
                ]
            })
        
            for ex_id, example in enumerate(examples):
                messages[-1]["content"] += [
                    {"type": "text", "text": f"Example {ex_id + 1}:\n"},
                    {"type": "text", "text": f"RGB Segmentation: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(example['rgbseg_path'])}"}},
                    {"type": "text", "text": f"Query structure: {example['query']}"},
                    {"type": "text", "text": f"Expected output: ```{example['output']}```\n\n"}
                ]
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"### Respond a correct output to this query:\n"},
                    {"type": "text", "text": f"RGB Segmentation image: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_rgbseg_path)}"}},
                    {"type": "text", "text": f"Query structure: {target_query}"}
                ],
            })
        
        else:
            raise Exception("Error: Check the image paths in config or in txt files")
            
        
        return messages
    
    def run_llm(self):
        """
        Build LLM model and run the api 
        """
        # Set model
        if not self.llm_model:
            model = "gpt-4o"
        else:
            model = self.llm_model     

        # Build complete message list
        if self.use_prim:
            messages = self._build_messages_openai()
        else:
            messages = self._build_messages_openai_for_distances()
        
        # Make API call
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                top_p=self.llm_config.get("model_config").get("top_p"),
                response_format={"type": "json_object"}
            )
            print("\nResponse:\n", response.choices[0].message.content)

            # Log the token usage 
            ts = response.created
            model = response.model
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            assert input_tokens + output_tokens == total_tokens
            
            token_log_dir = self.llm_config.get("token_log_dir")
            if not os.path.exists(token_log_dir):
                os.makedirs(token_log_dir, exist_ok=True)
            log_token_usage(ts, model, input_tokens, output_tokens, total_tokens, token_log_dir=token_log_dir)

        except Exception as e:
            print(f"Error: {str(e)}")

        return response


    def __call__(self, log_res_path=None, num_iter=None):
        response = self.run_llm()
        content_str = response.choices[0].message.content
        
        if log_res_path is not None:
            target_query = self.prompt_files["target_query"]

            with open(log_res_path, 'a') as f:
                if num_iter is not None:
                    f.write(f"### {num_iter + 1}\n")
                ts = response.created
                f.write(f"{datetime.fromtimestamp(ts).strftime('%Y%m%d-%H:%M:%S')}\tQuery: {target_query}\n")
                
                if content_str is None:
                    f.write("None" + "\n\n")
                else:
                    f.write(content_str + "\n\n")
        
        if content_str == None:
            content = "None"
        else:
            content = string_to_json(content_str)
        
        return content


class AssemblyLLM:
    def __init__(self, configs):
        self.config = configs
        self.llm_config = self.config.get("llm")
        
        self.llm_provider = self.llm_config.get("model_config").get("provider")
        self.llm_model = self.llm_config.get("model_config").get("model")

        self.target_config = self.config.get("task")
        
        self._register_api_key()
        self.prompt_files = self._get_prompt_files()
        
        # Init model
        if self.llm_provider == "openai":
            if not self.llm_model:
                model = "gpt-4o"
            else:
                model = self.llm_model
            self.llm = OpenAI(model=model)

        elif self.llm_provider == "mistral":
            if not self.llm_model:
                model = "mistral-small-latest"
            else:
                model = self.llm_model
            self.llm = MistralAI(model=model)

        elif self.llm_provider == "google":
            if not self.llm_model:
                model = "gemini-2.0-flash"
            else:
                model = self.llm_model
            self.llm = GoogleGenAI(model=model)
        self.llm_model = model
        
    def _register_api_key(self):
        """
        Set the api key
        """
        os.environ["OPENAI_API_KEY"] = self.llm_config.get("api_keys").get("openai")
        os.environ["MISTRAL_API_KEY"] = self.llm_config.get("api_keys").get("mistral")
        os.environ["GOOGLE_API_KEY"] = self.llm_config.get("api_keys").get("google")
    
    def _get_prompt_files(self):
        """
        Get file paths of prompt entries
        """
        prompt_files = {}
        
        prompt_files["system_prompt"] = self.config.get("relation").get("prim_prompt_files").get("system_prompt")
        prompt_files["examples_prompt"] = self.config.get("relation").get("prim_prompt_files").get("examples_prompt")
        prompt_files["example_rgb_paths"] = self.config.get("relation").get("example_images").get("example_rgb_paths")
        prompt_files["example_depth_paths"] = self.config.get("relation").get("example_images").get("example_depth_paths")
        prompt_files["example_mask_paths"] = self.config.get("relation").get("example_images").get("example_mask_paths")

        target_dir = os.path.join(self.config.get("save_dir"), self.target_config.get("target_dir_tmp"))
        
        prompt_files["target_query"] = self.target_config.get("target_query_structure")  # Query structure
        prompt_files["target_rgb_path"] = os.path.join(target_dir, self.target_config.get("target_rgb_file"))  # Image to analyze
        
        return prompt_files

    def _process_examples(self, examples_prompt, example_images):
        """
        Process raw examples into structured format
        """
        example_image_paths = [path.strip() for path in example_images.split('\n') if path.strip()]
        example_entries = examples_prompt.split('### Example')[1:]  # Skip first empty split
        
        examples = []
        for i, entry in enumerate(example_entries):
            if i >= len(example_image_paths):
                break  # Ensure we don't exceed available images
                
            # Extract components from each example
            parts = entry.split('Output:')
            input_part = parts[0]
            output_json = parts[1].strip()
            
            # Get image path and encode
            image_path = example_image_paths[i]
            
            # Get query
            query = input_part.split('Query: "')[1].split('"')[0]
            
            examples.append({
                'image': image_path,
                'query': query,
                'output': output_json
            })
        
        return examples

    def _build_messages(self):
        """
        Build messages for assembly tasks to feed into LLM
        """
        # Load prompt entries
        system_prompt = read_file(self.prompt_files["system_prompt"])
        examples = self._process_examples(
            read_file(self.prompt_files["example_rgb_paths"]),
            read_file(self.prompt_files["examples_prompt"])
        )
        target_query = self.prompt_files["target_query"]
        target_rgb_path = self.prompt_files["target_rgb_path"]
        
        # Build messages
        messages = []
        
        # System message
        system_msg = ChatMessage(
            role=MessageRole.SYSTEM,
            blocks=[
                TextBlock(text=system_prompt),
            ]
        )
        messages.append(system_msg)

        # Few-shot examples
        for example in examples:
            # User message (image + query)
            user_msg = ChatMessage(
                role=MessageRole.USER,
                blocks=[
                    TextBlock(text=f"Analyze these blocks. Query: {example['query']}"),
                    # TextBlock(text=f"Query: {example['query']}"),
                    ImageBlock(path=example['image']),
                ]
            )
            messages.append(user_msg)

            # Assistant response
            assistant_msg = ChatMessage(
                role=MessageRole.ASSISTANT,
                blocks=[
                    TextBlock(text=example['output'])
                ]
            )
            messages.append(assistant_msg)
        
        # Add target query
        query_msg = ChatMessage(
            role=MessageRole.USER,
            blocks=[
                TextBlock(text=f"Analyze these blocks. Query: {target_query}"),
                # TextBlock(text=f"Query: {target_query}"),
                ImageBlock(path=target_rgb_path),
            ]
        )
        messages.append(query_msg)

        return messages

    def run_llm(self):
        """
        Build LLM model and run the api 
        """
        # Set model
        if not self.llm_model:
            model = "gpt-4o"
        else:
            model = self.llm_model 
            
        # Build complete message list
        messages = self._build_messages()
        
        # Make API call
        response = self.llm.chat(
            messages,
            top_p=self.llm_config.get("model_config").get("top_p"),
            response_format={
              "type": "json_object",
            }
        )
        ts_now = datetime.now().timestamp()
        
        print("Response:\n", response.message.content)

        # Log the token usage
        if self.llm_provider == "openai":
            ts = response.raw.created
            usage = response.raw.usage
            
            input_tokens = usage.prompt_tokens   # input token usage
            output_tokens = usage.completion_tokens    # output token usage
            total_tokens = usage.total_tokens
        elif self.llm_provider == "google":
            ts = response.raw.get("created", None)
            usage = response.raw.get("usage_metadata", {})
            
            input_tokens = usage.get("prompt_token_count", None)   # input token usage
            output_tokens = usage.get("candidates_token_count", None)    # output token usage
            total_tokens = usage.get("total_token_count", None)
        else:
            ts = response.raw.get("created", None)
            usage = response.raw.get("usage", {})
            
            input_tokens = usage.prompt_tokens   # input token usage
            output_tokens = usage.completion_tokens    # output token usage
            total_tokens = usage.total_tokens
        assert input_tokens + output_tokens == total_tokens
        
        if ts is None:
            ts = ts_now
        
        token_log_dir = self.llm_config.get("token_log_dir")
        if not os.path.exists(token_log_dir):
            os.makedirs(token_log_dir, exist_ok=True)
        log_token_usage(ts, model, input_tokens, output_tokens, total_tokens, token_log_dir=token_log_dir)

        return response

    def __call__(self, log_res_path=None, num_iter=None):
        print(f"{bcolors.OKCYAN}[structure_planner.py | {get_clock_time()}] Requesting LLM's response{bcolors.RESET}")

        response = self.run_llm()
        
        if log_res_path is not None:
            with open(log_res_path, 'a') as f:
                target_query = self.prompt_files["target_query"]

                if num_iter is not None:
                    f.write(f"### {num_iter + 1}\n")

                ts_now = datetime.now().timestamp()
                if self.llm_provider == "openai":
                    ts = response.raw.created
                else:
                    ts = response.raw.get("created", None)
                if ts is None:
                    ts = ts_now
                f.write(f"{datetime.fromtimestamp(ts).strftime('%Y%m%d-%H:%M:%S')}\tQuery: {target_query}\n")
                
                if response.message.content is None:
                    f.write("None" + "\n\n")
                else:
                    f.write(response.message.content + "\n\n")
        
        content_str = response.message.content
        if content_str == None:
            content = "None"
        else:
            content = string_to_json(content_str)
        
        return content