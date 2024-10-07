


def prova(n):
    print(n)

from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
from huggingface_hub import login

def gen_docs(n, run):

    from huggingface_hub import login
    curr_model = "danigambit/llama-2-7b-chat-bnb-4bit_unsloth_"+str(n)
    summ_df_url = "danigambit/summ_df"

    hf_token = "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"

    login(token = hf_token) #writeToken

    summ_df = load_dataset(summ_df_url, split="train")


    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = curr_model,
        max_seq_length = 512,
        dtype = None,
        load_in_4bit = True,
        device_map = {"": 0})


    def generate_text(text):
        inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
        outputs = model.generate(**inputs, max_new_tokens=500)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    FastLanguageModel.for_inference(model)


    dict_docs = {"id": [],
                "prompt":[],
                "doc": []}


    for i in range(2050):
        print("----------------------")
        print(run, n, i)
        dict_docs["id"].append(i)

        prompt = "<user>: You are a writer that generates a meaningful, beautiful documents and an English reader wants to read a new document generated based on the key points of the given referenced document. Generate a new document for me in the same context with the following document: \n'''" +summ_df["summ"][i] + "\n'''<assistant>: "
        gen_text = generate_text(prompt).split("<assistant>: ", 1)[1]

        if i%100 == 0:
            print(i)
            print(prompt)
            print(gen_text)

        dict_docs["prompt"].append(prompt)
        dict_docs["doc"].append(gen_text)


    new_dataset = Dataset.from_dict(dict_docs)

    new_dataset.push_to_hub("danigambit/doc_gen_"+str(n)+"_"+run)