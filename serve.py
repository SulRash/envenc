import torch
import torchvision
from math import sqrt

from tinyllava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from tinyllava.conversation import Conversation, SeparatorStyle
from tinyllava.utils import disable_torch_init
from tinyllava.mm_utils import process_images, tokenizer_image_token

cached_out = {}

def preprocess(tensor, hidden_size):
    min_value = tensor.min()
    max_value = tensor.max()
    normalized_tensor = (tensor - min_value) / (max_value - min_value) * 255

    batch_size = tensor.shape[0]
    box = int(sqrt(hidden_size))

    return normalized_tensor.reshape(batch_size, box, box)


def load_model(vlm: str = 'idefics', load_4bit: bool = False, device: str = "cuda"):
    vlms = ['tinyllava', 'idefics']

    if vlm == 'tinyllava':
        from tinyllava.model.builder import load_pretrained_model
        from tinyllava.mm_utils import get_model_name_from_path

        model_path = "bczhou/TinyLLaVA-2.0B"

        if load_4bit:
            processor, model, vision_tower, _ = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=get_model_name_from_path(model_path),
                load_4bit=True
            )
            image_processor = vision_tower.image_processor

        else:
            processor, model, vision_tower, _ = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=get_model_name_from_path(model_path),
                load_4bit=False
            )
            image_processor = torch.compile(vision_tower).image_processor
            model = torch.compile(model)
        
        hidden_size = 2048

    elif vlm == 'idefics':
        from transformers import AutoProcessor, AutoModelForVision2Seq

        processor = AutoProcessor.from_pretrained(
            'HuggingFaceM4/idefics2-8b',
            size={"longest_edge": 84, "shortest_edge": 84},
            do_image_splitting=False
        )

        model = AutoModelForVision2Seq.from_pretrained(
            'HuggingFaceM4/idefics2-8b',
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2"
        ).to(device)
        image_processor = None
        model = torch.compile(model)
        hidden_size = 4096

    else:
        raise(f"Invalid VLM choice! Possible options: {vlms}")

    return {
        'processor': processor,
        'model': model,
        'image_processor': image_processor,
        'hidden_size': hidden_size
    }


def infer_tinyllava(image_arrays, **kwargs):
    disable_torch_init()

    tokenizer, model, image_processor = kwargs['processor'], kwargs['model'], kwargs['image_processor']

    input_ids = torch.zeros((image_arrays.shape[0], 49), dtype=torch.long).to(model.device)
    i = 0

    inp = 'Describe the image in detail.'

    conv = Conversation(
        system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=('USER', 'ASSISTANT'),
        version='phi',
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.TWO,
        sep=' ',
        sep2='<|endoftext|>',
    )

    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)

    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    for i in range(len(input_ids)):
        input_ids[i] = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')[0].unsqueeze(0).to(model.device)

    images = []
    for image_array in image_arrays:
        images.append(torchvision.transforms.ToPILImage()(image_array))
    image_tensor = process_images(images, image_processor, model.config)

    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    print(input_ids.shape)
    print(image_tensor.shape)

    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False) and torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
            use_cache=True
        )

        # Accesses first generated token's last layer's hidden state
        hidden_states = output_ids.hidden_states[0][-1]
        print(len(output_ids.hidden_states))
        print(len(output_ids.hidden_states[0]))
        merged_out = torch.zeros((len(hidden_states), 2048)).half().to(model.device)

        print(output_ids.hidden_states[0][-1].shape)
        print(output_ids.hidden_states[1][-1].shape)
        print(output_ids.sequences.shape)
        print(output_ids.sequences)

        outputs = tokenizer.decode(output_ids.sequences[1]).strip()
        print(outputs)
        exit()
        for i in range(len(hidden_states)):
            merged_out[i] = hidden_states[i].mean(dim=0)

    return merged_out

def hash_tensor(tensor):
    # Convert the tensor to a numpy array
    array = tensor.cpu().numpy()
    
    # Compute the hash using numpy's built-in hash function
    hash_value = hash(array.tobytes())
    
    return hash_value

def infer_idefics(image_arrays, device='cuda', **kwargs):
    processor = kwargs['processor']
    model = kwargs['model']

    hashed_arrays = [hash_tensor(tensor) for tensor in image_arrays]

    new_batch_arrays = []
    new_batch_indices = []
    cached_batch_embeddings = []
    cached_batch_indices = []

    for i, hash_value in enumerate(hashed_arrays):
        if hash_value in cached_out:
            cached_batch_embeddings.append(cached_out[hash_value])
            cached_batch_indices.append(i)
        else:
            new_batch_arrays.append(image_arrays[i])
            new_batch_indices.append(i)

    if len(new_batch_arrays) > 0:
        images = [[torchvision.transforms.ToPILImage()(image_array)] for image_array in new_batch_arrays]
        messages = [[
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': 'What is the best move for the player in the image of this atari game in one word?'},
                ]
            }
        ]] * len(images)

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        input_ids = processor(text=prompt, images=images, return_tensors='pt')
        input_ids = {k: v.to(device) for k, v in input_ids.items()}

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False) and torch.no_grad():
            output_ids = model.generate(
                **input_ids,
                do_sample=False,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_hidden_states=True,
                use_cache=True
            )

        hidden_states = output_ids.hidden_states[-1][-1]
        idx_of_the_last_non_padding_token = input_ids['attention_mask'].bool().sum(1) - 1
        new_batch_embeddings = hidden_states[torch.arange(hidden_states.shape[0]), idx_of_the_last_non_padding_token]

        for i, embedding in zip(new_batch_indices, new_batch_embeddings):
            hash_value = hashed_arrays[i]
            cached_out[hash_value] = embedding

    else:
        new_batch_embeddings = []

    output_embeddings = torch.empty(len(image_arrays), new_batch_embeddings.shape[1] if len(new_batch_embeddings) > 0 else cached_batch_embeddings[0].shape[0]).half().to(device)

    if len(cached_batch_embeddings) > 0:
        output_embeddings[cached_batch_indices] = torch.stack(cached_batch_embeddings)
    if len(new_batch_embeddings) > 0:
        output_embeddings[new_batch_indices] = new_batch_embeddings

    return output_embeddings