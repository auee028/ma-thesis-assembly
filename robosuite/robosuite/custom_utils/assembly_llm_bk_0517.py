import os
import re
from datetime import datetime

from robosuite.custom_utils.assembly_utils import *

from llama_index.llms.openai import OpenAI
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock, ImageBlock

import openai


class AssemblyLLM:
    def __init__(self, configs):
        self.config = configs
        self.llm_config = self.config.get("llm")
        
        self.llm_provider = self.llm_config.get("model_config").get("provider")
        self.llm_model = self.llm_config.get("model_config").get("model")
        
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
        
        prompt_files["system_prompt"] = self.llm_config.get("prompt_txt_files").get("system_prompt")
        prompt_files["examples_prompt"] = self.llm_config.get("prompt_txt_files").get("examples_prompt")
        prompt_files["example_image_paths"] = self.llm_config.get("prompt_txt_files").get("example_image_paths")
        
        prompt_files["target_query"] = self.config.get("task").get("target_query_structure")  # Query structure
        prompt_files["target_rgb_path"] = self.config.get("task").get("target_rgb_path")  # Image to analyze
        
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
            read_file(self.prompt_files["example_image_paths"]),
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
            os.mkdir(token_log_dir)
        log_token_usage(ts, model, input_tokens, output_tokens, total_tokens, token_log_dir=token_log_dir)

        return response


    def __call__(self, log_res_path=None):
        response = self.run_llm()
        
        if log_res_path is not None:
            with open(log_res_path, 'a') as f:
                ts_now = datetime.now().timestamp()
                if self.llm_provider == "openai":
                    ts = response.raw.created
                else:
                    ts = response.raw.get("created", None)
                if ts is None:
                    ts = ts_now
                f.write(f"{datetime.fromtimestamp(ts).strftime('%Y%m%d-%H:%M:%S')}")
                
                if response.message.content is None:
                    f.write("None" + "\n\n")
                else:
                    f.write(response.message.content + "\n\n")
        
        content = string_to_json(response.message.content)
        
        # detected_blocks = content["detected_blocks"]
        # assembly_structure = content["assembly_structure"]
        
        # return detected_blocks, assembly_structure
        return content


class AssemblyLLMOpenAI:
    def __init__(self, configs, distance_relations=False):
        self.config = configs
        self.distance_relations = distance_relations
        
        self.llm_config = self.config.get("llm")
        self.llm_provider = self.llm_config.get("model_config").get("provider")
        self.llm_model = self.llm_config.get("model_config").get("model")
        
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
        
        prompt_files["system_prompt"] = self.llm_config.get("prompt_txt_files").get("system_prompt")
        prompt_files["examples_prompt"] = self.llm_config.get("prompt_txt_files").get("examples_prompt")
        prompt_files["example_image_paths"] = self.llm_config.get("prompt_txt_files").get("example_image_paths")
        
        prompt_files["target_query"] = self.config.get("task").get("target_query_structure")  # Query structure
        prompt_files["target_rgb_path"] = self.config.get("task").get("target_rgb_path")  # Color image to analyze
        prompt_files["target_depth_path"] = self.config.get("task").get("target_depth_path")  # Depth to analyze
        prompt_files["target_seg_path"] = self.config.get("task").get("target_seg_path")  # Mask image to analyze
        
        prompt_files["examples_prompt_dist"] = self.config.get("relation").get("distances").get("examples_prompt")
        prompt_files["example_rgb_paths"] = self.config.get("relation").get("distances").get("example_rgb_paths")
        prompt_files["example_depth_paths"] = self.config.get("relation").get("distances").get("example_depth_paths")
        prompt_files["example_seg_paths"] = self.config.get("relation").get("distances").get("example_seg_paths")
        
        return prompt_files

    def _process_examples(self, examples_prompt, example_images):
        """
        Process raw examples into structured format
        """
        example_entries = examples_prompt.split('### Example')[1:]  # Skip first empty split
        example_image_paths = [path.strip() for path in example_images.split('\n') if path.strip()]
        
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
    
    def _process_examples_for_distances(self, examples_prompt, example_rgbs, example_depths, example_segs):
        """
        Process raw examples into structured format
        """
        example_entries = re.split(f'### Example \d+\s*', examples_prompt)[1:]  # Skip first empty split
        example_rgb_paths = [path.strip() for path in example_rgbs.split('\n') if path.strip()]
        example_depth_paths = [path.strip() for path in example_depths.split('\n') if path.strip()]
        example_seg_paths = [path.strip() for path in example_segs.split('\n') if path.strip()]
        
        examples = []
        for i, entry in enumerate(example_entries):
            if i >= len(example_rgb_paths):
                break  # Ensure we don't exceed available images
                
            # Extract components from each example
            entry = entry.strip()
            
            query = re.search(r"=\s*'([^']+)'", entry).group(1)
            input_rgb_path = example_rgb_paths[i] if example_rgb_paths else ""
            image_depth_path = example_depth_paths[i] if example_depth_paths else ""
            image_seg_path = example_seg_paths[i] if example_seg_paths else ""
            output_part = '\n'.join(entry.splitlines()[1:])
            
            examples.append({
                'query': query,
                'rgb_path': input_rgb_path,
                'depth_path': image_depth_path,
                'mask_path': image_seg_path,
                'output': output_part,
            })
        
        return examples
    
    def _build_messages_openai(self):
        """
        Build complete message only for OpenAI API
        """
        # Load prompt entries
        system_prompt = read_file(self.prompt_files["system_prompt"])
        examples = self._process_examples(
            read_file(self.prompt_files["example_image_paths"]),
            read_file(self.prompt_files["examples_prompt"])
        )
        target_query = self.prompt_files["target_query"]
        target_rgb_path = self.prompt_files["target_rgb_path"]
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add few-shot examples
        for example in examples:
            # User message (image + query)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Analyze these blocks. Query: {example['query']}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{example['image']}"}}
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
        
        return messages
    
    def _build_messages_openai_for_distances(self):
        """
        Build complete message for numerical distances (only for OpenAI API)
        """
        # Set and load prompt entries
        examples_prompt = read_file(self.prompt_files["examples_prompt_dist"])
        rgb_image_paths = read_file(self.prompt_files["example_rgb_paths"]) if self.prompt_files["example_rgb_paths"] else ""
        depth_image_paths = read_file(self.prompt_files["example_depth_paths"]) if self.prompt_files["example_depth_paths"] else ""
        seg_image_paths = read_file(self.prompt_files["example_seg_paths"]) if self.prompt_files["example_seg_paths"] else ""
        
        examples = self._process_examples_for_distances(examples_prompt, rgb_image_paths, depth_image_paths, seg_image_paths)
        
        target_query = self.prompt_files["target_query"]
        target_rgb_path = self.prompt_files["target_rgb_path"]
        target_depth_path = self.prompt_files["target_depth_path"]
        target_seg_path = self.prompt_files["target_seg_path"]
        
        # Build messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant that pays attention to the user's instructions and writes a json form containing the elements that the user wants."}
        ]
        
        if seg_image_paths:
            assert len(rgb_image_paths) != 0 and len(depth_image_paths) != 0 and len(seg_image_paths) != 0
            
            messages.append({
                "role": "user",
                "content": "I want you to generate a directed graph that represents contact information as relationships between blocks on the table shown in the input RGB image. This directed graph will encode the structural information needed to build the input query structure. You are given an RGB image, a depth image, and a segmentation mask of the blocks."
            })
            
            messages.append({
                "role": "assistant",
                "content": "Got it. Give me input images."
            })
        
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Input RGB image:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(target_rgb_path)}"}
                    },
                    {"type": "text", "text": f"Input depth image:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(target_depth_path)}"}
                    },
                    {"type": "text", "text": f"Input segmentation mask:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(target_seg_path)}"}
                    },
                    {"type": "text", "text": f"The query structure is {target_query}."},
                    {"type": "text", "text": f"Do it step by step. First, detect each building block on the table in the input image."},
                    {"type": "text", "text": f"Second, Assign a unique `block_id` and `color` to individual building blocks."},
                    {"type": "text", "text": f"Third, generate a directed graph with nodes of the building blocks and edges of relative distances from one block to another. Note that x is back to front, y is left to right, and z is bottom to up."},
                    {"type": "text", "text": f"For reference, I will give you some examples."}
                ],
            })
            
            for ex_id, example in enumerate(examples):
                messages[-1]["content"] += [
                    {"type": "text", "text": f"Example {ex_id + 1}: Query structure is {example['query']}"},
                    {"type": "text", "text": f"Example {ex_id + 1}: RGB image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(example['rgb_path'])}"}
                    },
                    {"type": "text", "text": f"Example {ex_id + 1}: Depth image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(example['depth_path'])}"}
                    },
                    {"type": "text", "text": f"Example {ex_id + 1}: Segmentation mask"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(example['mask_path'])}"}
                    },
                    {"type": "text", "text": f"Example {ex_id + 1}: Expected output elements"},
                    {"type": "text", "text": f"{example['output']}"}
                ]
            
        elif depth_image_paths:
            assert len(rgb_image_paths) != 0 and len(depth_image_paths) != 0 and len(seg_image_paths) == 0
            
            messages.append({
                "role": "user",
                "content": "I want you to generate a directed graph that represents contact information as relationships between blocks on the table shown in the input RGB image. This directed graph will encode the structural information needed to build the input query structure. You are given an RGB image and a depth image."
            })
            
            messages.append({
                "role": "assistant",
                "content": "Got it. Give me input images."
            })
        
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Input RGB image:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(target_rgb_path)}"}
                    },
                    {"type": "text", "text": f"Input depth image:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(target_depth_path)}"}
                    },
                    {"type": "text", "text": f"The query structure is {target_query}."},
                    {"type": "text", "text": f"Do it step by step. First, detect each building block on the table in the input image."},
                    {"type": "text", "text": f"Second, Assign a unique `block_id` and `color` to individual building blocks."},
                    {"type": "text", "text": f"Third, generate a directed graph with nodes of the building blocks and edges of relative distances from one block to another. Note that x is back to front, y is left to right, and z is bottom to up."},
                    {"type": "text", "text": f"For reference, I will give you some examples."}
                ],
            })
            
            for ex_id, example in enumerate(examples):
                messages[-1]["content"] += [
                    {"type": "text", "text": f"Example {ex_id + 1}: Query structure is {example['query']}"},
                    {"type": "text", "text": f"Example {ex_id + 1}: RGB image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(example['rgb_path'])}"}
                    },
                    {"type": "text", "text": f"Example {ex_id + 1}: Depth image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(example['depth_path'])}"}
                    },
                    {"type": "text", "text": f"Example {ex_id + 1}: Expected output elements"},
                    {"type": "text", "text": f"{example['output']}"}
                ]
            
        elif rgb_image_paths:
            assert len(rgb_image_paths) != 0 and len(depth_image_paths) == 0 and len(seg_image_paths) == 0
        
            messages.append({
                "role": "user",
                "content": "I want you to generate a directed graph that represents contact information as relationships between blocks on the table shown in the input image. This directed graph will encode the structural information needed to build the input query structure."
            })
            
            messages.append({
                "role": "assistant",
                "content": "Got it. Give me input images."
            })
        
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_rgb_path)}"}},
                    {"type": "text", "text": f"The query structure is {target_query}."},
                    {"type": "text", "text": f"Do it step by step. First, detect each building block on the table in the input image."},
                    {"type": "text", "text": f"Second, Assign a unique `block_id` and `color` to individual building blocks."},
                    {"type": "text", "text": f"Third, generate a directed graph with nodes of the building blocks and edges of relative distances from one block to another. Note that x is back to front, y is left to right, and z is bottom to up."},
                    {"type": "text", "text": f"For reference, I will give you some examples."}
                ],
            })
            
            for ex_id, example in enumerate(examples):
                messages[-1]["content"] += [
                    {"type": "text", "text": f"Example {ex_id + 1}: Query structure is {example['query']}"},
                    {"type": "text", "text": f"Example {ex_id + 1}: RGB image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(example['rgb_path'])}"}
                    },
                    {"type": "text", "text": f"Example {ex_id + 1}: Expected output elements"},
                    {"type": "text", "text": f"{example['output']}"}
                ]
        
        else:
            assert len(rgb_image_paths) == 0 and len(depth_image_paths) == 0 and len(seg_image_paths) == 0
            
            messages.append({
                "role": "user",
                "content": "I want you to generate a directed graph that represents contact information as relationships between blocks on the table shown in the input image. This directed graph will encode the structural information needed to build the input query structure."
            })
            
            messages.append({
                "role": "assistant",
                "content": "Got it. Give me input images."
            })
        
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(target_rgb_path)}"}},
                    {"type": "text", "text": f"The query structure is {target_query}."},
                    {"type": "text", "text": f"Do it step by step. First, detect each building block on the table in the input image."},
                    {"type": "text", "text": f"Second, Assign a unique `block_id` and `color` to individual building blocks."},
                    {"type": "text", "text": f"Third, generate a directed graph with nodes of the building blocks and edges of relative distances from one block to another. Note that x is back to front, y is left to right, and z is bottom to up."},
                    {"type": "text", "text": f"For reference, I will give you some examples."}
                ],
            })
            
            for ex_id, example in enumerate(examples):
                messages[-1]["content"] += [
                    {"type": "text", "text": f"Example {ex_id + 1}: Query structure is {example['query']}"},
                    {"type": "text", "text": f"Example {ex_id + 1}: Expected output elements"},
                    {"type": "text", "text": f"{example['output']}"}
                ]
        
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
        if self.distance_relations is False:
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
                os.mkdir(token_log_dir)
            log_token_usage(ts, model, input_tokens, output_tokens, total_tokens, token_log_dir=token_log_dir)

        except Exception as e:
            print(f"Error: {str(e)}")

        return response


    def __call__(self, log_res_path=None):
        response = self.run_llm()
        content_str = response.choices[0].message.content
        
        if log_res_path is not None:
            with open(log_res_path, 'a') as f:
                ts = response.created
                f.write(f"{datetime.fromtimestamp(ts).strftime('%Y%m%d-%H:%M:%S')}")
                
                if content_str is None:
                    f.write("None" + "\n\n")
                else:
                    f.write(content_str + "\n\n")
        
        content = string_to_json(content_str)
        
        return content_build_messages_openai_for_distances
