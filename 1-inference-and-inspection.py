#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

logging.basicConfig(level=logging.INFO)

MODEL_ID = "Qwen/Qwen3.5-4B"
HF_TOKEN_STATUS = False
try:
    login()
    HF_TOKEN_STATUS = True
except Exception as e:
    logging.error(f"Huggingface login failed with exception {e}")

logging.info(f"Inspecting LLM Model {MODEL_ID} with HF_TOKEN status {HF_TOKEN_STATUS}")


# The tokenizer translates raw strings into integer arrays (token IDs)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load the core execution engine (the model graph) into VRAM
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    dtype=torch.bfloat16, 
    device_map = "cuda" # device_map="auto" # Automatically maps layers to available GPUs
)

# Put the model in evaluation mode (disables dropout, fixes batch norm, etc.)
model.eval()

# This prints the structural representation of the network
print(model)

# Accessing a specific layer deeply nested in the architecture
layer_15 = model.model.layers[15]
print(f"Layer 15 attention module: {layer_15.self_attn}")

prompt = "I am"

# 1. Encode: String -> Tensor of IDs
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(f"Input shape: {inputs['input_ids'].shape}") 
# Shape: [batch_size=1, sequence_length]

with torch.no_grad(): # Disable gradient tracking to save memory during pure inference
    
    # 2. Forward Pass: Push the tensor through the layer stack
    # By passing output_hidden_states=True, we capture the memory state after every single layer.
    outputs = model(
        **inputs, 
        output_hidden_states=True 
    )
    
    # 3. Analyze the output Logits
    logits = outputs.logits
    print(f"Logits shape: {logits.shape}") 
    # Shape: [batch_size=1, sequence_length, vocab_size]
    
    # We only care about the model's prediction for the *very next* token, 
    # which is the last slice in the sequence dimension.
    next_token_logits = logits[0, -1, :]
    
    # 4. Decode: Find the token ID with the highest probability (Greedy decoding)
    next_token_id = torch.argmax(next_token_logits)
    
    # 5. Detokenize: Integer ID -> String
    next_word = tokenizer.decode(next_token_id)
    print(f"Predicted next token: '{next_word}'")

    # hidden_states[0] is the output of the embedding layer
    # hidden_states[-1] is the output of the final decoder layer
    final_hidden_state = outputs.hidden_states[-1]

    print(f"Hidden state shape: {final_hidden_state.shape}") 
    # Shape: [batch_size=1, sequence_length, hidden_dimension]

    # You can now analyze these vectors mathematically. 
    # For example, extracting the vector representation of the final token:
    # last_token_vector = final_hidden_state[0, -1, :]


# Instantiate streamer, explicitly telling it NOT to print the input prompt
streamer = TextStreamer(tokenizer, skip_prompt=True)
with torch.no_grad(): # Disable gradient tracking to save memory
    print(f"Chat with {MODEL_ID}\nTemplate:\n{tokenizer.chat_template}\n")
    print("-" * 40)
    
    conversation = []
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        # 1. Update state
        conversation.append({"role": "user", "content": user_input})
        
        # 2. Template & Tokenize (CRITICAL: return_dict=True)
        # This returns a dict with 'input_ids' and 'attention_mask'
        inputs = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device)

        raw_input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        print("\n[DEBUG] --- Raw Input to Model ---\n")
        print(raw_input_text)
        print("[DEBUG] --------------------------\n")
        print("Assistant: ", end="") # Streamer will append to this line
        
        # 3. Generate
        # Now **inputs safely unpacks input_ids and attention_mask
        output_ids = model.generate(
            **inputs, 
            streamer=streamer, 
            max_new_tokens=1337,
            pad_token_id=tokenizer.eos_token_id # Prevents annoying warning logs
        )

        # 4. Extract only the newly generated tokens
        input_length = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_length:]
        
        # 5. Decode using single decode, not batch_decode
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # 6. Append to state
        conversation.append({"role": "assistant", "content": response_text})
        print("\n" + "-" * 40)