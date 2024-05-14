import torch
import torchvision
import requests

from tinyllava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from tinyllava.conversation import Conversation, SeparatorStyle
from tinyllava.utils import disable_torch_init
from tinyllava.mm_utils import process_images, tokenizer_image_token

from PIL import Image
from io import BytesIO


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def infer_no_lm(tokenizer, model, image_processor, image_arrays):
    disable_torch_init()
    
    input_ids = torch.zeros((image_arrays.shape[0], 49), dtype=torch.long).to(model.device)
    i = 0

    inp = 'Describe the image in detail.'

    conv = Conversation(
        system='A chat between a curious user and an artificial intelligence assistant. '
           'The assistant gives helpful, detailed, and polite answers to the user's questions.',
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


# Qwen-VL implementation for VLM
# Struggling with the performance issue of saving picture -> getting path -> querying -> deleting image loop.
# Switching over to minigpt-4
# def serve_qwen(
#         tokenizer,
#         model,
#         image_arrays,
#         device: str = 'cuda',
#         system: str = 'You are a helpful AI game guide, assist the user with any queries they ask helpfully and intelligently.',
#         decode: bool = False
#     ):

#     generation_config = generation_config if generation_config is not None else model.generation_config
#     print(generation_config)

#     history = []
#     stop_words_ids = []

#     query = tokenizer.from_list_format([
#         {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
#         {'text': '这是什么'},
#     ])

#     max_window_size = generation_config.max_window_size
#     raw_text, context_tokens = make_context(
#         tokenizer,
#         query,
#         history=history,
#         system=system,
#         max_window_size=max_window_size,
#         chat_format=generation_config.chat_format,
#     )

#     stop_words_ids.extend(get_stop_words_ids(
#         generation_config.chat_format, tokenizer
#     ))
#     input_ids = torch.tensor([context_tokens]).to(device)
#     outputs = model.generate(
#                 input_ids,
#                 stop_words_ids=stop_words_ids,
#                 return_dict_in_generate=True,
#                 generation_config=generation_config,
#             )

#     if decode:
#         response = decode_tokens(
#             outputs[0],
#             tokenizer,
#             raw_text_len=len(raw_text),
#             context_length=len(context_tokens),
#             chat_format=generation_config.chat_format,
#             verbose=False,
#             errors='replace'
#         )
#         print(response)

#     print(outputs)



def infer_idefics(image_arrays, device = 'cuda', **kwargs):

    processor = kwargs['processor']
    model = kwargs['model']

    # Note that passing the image urls (instead of the actual pil images) to the processor is also possible
    images = []
    for image_array in image_arrays:
        images.append(torchvision.transforms.ToPILImage()(image_array))

    
    input_ids = []
    for image in images:
        # Create inputs
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': 'What is the best move for the player in this image in one word?'},
                ]
            } 
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

        input_id = processor(text=prompt, images=[image], return_tensors='pt')
        input_ids = {k: v.to(device) for k, v in input_id.items()}
        print(input_ids)
        exit()

        # Generate
        generated_ids = model.generate(**input_ids, max_new_tokens=500)
        generated_texts = processor.batch_decode([generated_ids], skip_special_tokens=True)

    print(generated_texts)
    # ['User: What do we see in this image? \nAssistant: In this image, we can see the city of New York, and more specifically the Statue of Liberty. \nUser: And how about this image? \nAssistant: In this image we can see buildings, trees, lights, water and sky.']

def model_loader(vlm: str = 'idefics', load_4bit: bool = False, device: str = "cuda"):
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
        
    elif vlm == 'idefics':
        from transformers import AutoProcessor, AutoModelForVision2Seq

        processor = AutoProcessor.from_pretrained('HuggingFaceM4/idefics2-8b')
        model = AutoModelForVision2Seq.from_pretrained(
            'HuggingFaceM4/idefics2-8b',
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2"
        ).to(device)
        image_processor = None
        model = torch.compile(model)
    
    else:
        raise(f"Invalid VLM choice! Possible options: {vlms}")

