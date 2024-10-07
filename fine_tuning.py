
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset, load_dataset


def fine_tuning(n):
    
    from huggingface_hub import login

    curr_model = "danigambit/llama-2-7b-chat-bnb-4bit_unsloth_"+str(n)
    curr_doc_gen = "danigambit/doc_gen_" + str(n)


    hf_token = "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"
    login(token = hf_token) #writeToken


    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = curr_model,
        max_seq_length = 512,
        dtype = None,
        load_in_4bit = True,
        device_map = {"": 0})

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama")


    def formatting_prompts_func(prompts):
        convs = prompts["conversations"]
        texts = [tokenizer.apply_chat_template(conv, tokenize = False, add_generation_prompt = False) for conv in convs]
        return { "text" : texts, }

    #########################################


    
    doc_gen = load_dataset(curr_doc_gen, split="train")

    convs = {"conversations" :[[{"role": "user", "content": doc_gen["prompt"][k] },
                                {"role": "assistant", "content": doc_gen["doc"][k] }] for k in range(2050)]}


    train_ds = Dataset.from_dict(convs)
    new_dataset = train_ds.map(formatting_prompts_func, batched = True)


    ########################################

   
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = curr_model,
        max_seq_length = 512,
        dtype = None,
        load_in_4bit = True,
        device_map = {"": 0})


    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = True,
        random_state = 3407,
        max_seq_length = 512,
        use_rslora = False,  # Rank stabilized LoRA
        loftq_config = None, # LoftQ
    )

    trainer = SFTTrainer(
        model = model,
        train_dataset = new_dataset,
        dataset_text_field = "text",
        max_seq_length = 512,
        tokenizer = tokenizer,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            max_steps = 100,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            output_dir = "outputs",
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
        ),
    )
    trainer.train()

    ############################################################################


    FastLanguageModel.for_inference(model)

    next_model = "danigambit/llama-2-7b-chat-bnb-4bit_unsloth_"+str(n+1)

    from huggingface_hub import login, HfApi, create_repo, Repository

    repo_name = next_model
    api = HfApi()
    api.create_repo(repo_name, private=False)

    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)    


