import torch
import torchvision
import requests

from tinyllava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from tinyllava.conversation import Conversation, SeparatorStyle
from tinyllava.utils import disable_torch_init
from tinyllava.mm_utils import process_images, tokenizer_image_token

from PIL import Image
from io import BytesIO

from qwenvl.qwen_generation_utils import make_context, get_stop_words_ids, decode_tokens

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

    inp = "Describe the image in detail."

    conv = Conversation(
        system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        version="phi",
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.TWO,
        sep=" ",
        sep2="<|endoftext|>",
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
#         device: str = "cuda",
#         system: str = "You are a helpful AI game guide, assist the user with any queries they ask helpfully and intelligently.",
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
