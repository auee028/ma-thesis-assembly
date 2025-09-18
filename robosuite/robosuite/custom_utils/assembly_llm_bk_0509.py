import os
from datetime import datetime

from robosuite.custom_utils.assembly_utils import read_file, string_to_json, log_token_usage

from llama_index.llms.openai import OpenAI
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock, ImageBlock


class AssemblyLLM:
    def __init__(self, configs):
        self.config = configs
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
        prompt_files["target_image_path"] = self.config.get("task").get("target_rgb_path")  # Image to analyze
        
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
        target_image_path = self.prompt_files["target_image_path"]
        
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
                ImageBlock(path=target_image_path),
            ]
        )
        messages.append(query_msg)

        return messages

    def run_llm(self):
        """
        Build LLM model and run the api 
        """
        # Init model
        if self.llm_provider == "openai":
            if not self.llm_model:
                model = "gpt-4o"
            else:
                model = self.llm_model
            llm = OpenAI(model=model)

        elif self.llm_provider == "mistral":
            if not self.llm_model:
                model = "mistral-small-latest"
            else:
                model = self.llm_model
            llm = MistralAI(model=model)

        elif self.llm_provider == "google":
            if not self.llm_model:
                model = "gemini-2.0-flash"
            else:
                model = self.llm_model
            llm = GoogleGenAI(model=model)        

        # Build complete message list
        messages = self._build_messages()
        
        # Make API call
        response = llm.chat(
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
                
                f.write(response.message.content + "\n\n")
        
        content = string_to_json(response.message.content)
        
        detected_blocks = content["detected_blocks"]
        assembly_structure = content["assembly_structure"]
        
        return detected_blocks, assembly_structure
                
