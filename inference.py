import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from platform_mps import install_mps_signal_handler

def load_inference_model(base_model_id, lora_dir):
    """
    Load the base model in 4-bit or 16-bit, and load the LoRA weights on top of it.
    """
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)
    
    print("Loading base model (using GPU if available)...")
    # For inference, if VRAM is 16GB, we can load the base model in 16-bit float for highest accuracy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading LoRA adapter from {lora_dir}...")
    model = PeftModel.from_pretrained(model, lora_dir)
    model.eval()
    
    return model, tokenizer, device

def predict_performance(model, tokenizer, device, recipe):
    """
    Formulates the user recipe into the chat template, runs the model,
    and returns the predicted photovoltaic properties.
    """
    # System instruction must match the training phase
    messages = [
        {"role": "system", "content": "You are a materials science AI assistant specialized in perovskite solar cells."},
        {"role": "user", "content": recipe}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer([text], return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,  # Greedy decoding for consistent prediction values
            pad_token_id=tokenizer.eos_token_id
        )
        
    # Exclude input prompt tokens from decoding
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

if __name__ == "__main__":
    install_mps_signal_handler()

    base_model = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
    lora_path = os.environ.get("LORA_DIR", "./output/perovskite_qwen_lora")
    
    if not os.path.exists(lora_path):
        print(f"Error: LoRA adapter directory '{lora_path}' does not exist.")
        print("Please run 'python train_lora.py' to train and save the model first.")
        exit(1)
        
    model, tokenizer, device = load_inference_model(base_model, lora_path)
    
    print("\n==================================================")
    print("Perovskite Solar Cell LLM Predictor Ready!")
    print("==================================================")
    
    # Example recipe input (Standard CsFAMA Pb IBr perovskite recipe)
    example_recipe = (
        "Predict the photovoltaic performance of the following perovskite solar cell device.\n"
        "Device Recipe:\n"
        "- Perovskite Composition: Cs0.05(FA0.83MA0.17)0.95Pb(I0.83Br0.17)3 (CsFAMAPbBrI)\n"
        "- Electron Transport Layer (ETL): SnO2-np\n"
        "- Hole Transport Layer (HTL): Spiro-MeOTAD\n"
        "- Backcontact: Au\n"
        "- Deposition Solvents: DMF:DMSO\n"
        "- Thermal Annealing: 100 C for 30 min"
    )
    
    print("\n--- Running Prediction for Example Recipe ---")
    print(f"Recipe Details:\n{example_recipe}\n")
    prediction = predict_performance(model, tokenizer, device, example_recipe)
    print("--- Model Prediction Response ---")
    print(prediction)
    print("==================================================")
    
    # Enter interactive mode
    print("\nEntering interactive mode. Press Ctrl+C to exit.")
    while True:
        try:
            print("\nPlease specify the details of the device recipe (or press Enter to run the example again):")
            user_input = input("Recipe details: ").strip()
            if not user_input:
                user_input = example_recipe
                
            print("\nGenerating prediction...")
            prediction = predict_performance(model, tokenizer, device, user_input)
            print("\n--- Prediction Output ---")
            print(prediction)
            print("--------------------------------------------------")
        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break
