import torch
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
    image_tensors = torch.zeros((image_arrays.shape[0], 3, 384, 384), dtype=torch.float16).to(model.device)
    i = 0

    for image_array in image_arrays:

        inp = "What is the best move for the player in this image?"

        conv = Conversation(
            system="You are a helpful game guide, you will be given a frame of a video game and you should predict the best move for the player.",
            roles=("USER", "ASSISTANT"),
            version="phi",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="<|endoftext|>",
        )

        if image_array is not None:
            # image = Image.fromarray(image_array)

            # For grayscale tensor to image?
            import torchvision
            image = torchvision.transforms.ToPILImage()(image_array)
            image_tensor = process_images([image], image_processor, model.config)

            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            image_tensors[i] = image_tensor[0]
        else:
            image = None
        if image is not None:
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
            image_tensor = None
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids[i] = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')[0].unsqueeze(0).to(model.device)
        i += 1

    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False) and torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensors,
            do_sample=False,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True,
            use_cache=True
        )

        # Accesses first generated token's last layer's hidden state
        hidden_states = output_ids.hidden_states[0][-1]
        merged_out = torch.zeros((len(hidden_states), 2048))
        for i in range(len(hidden_states)):
            merged_out[i] = hidden_states[i].mean(dim=0)
        merged_out = hidden_states[0][0].mean(dim=0)
    
    return merged_out

# def infer(tokenizer, model, image_processor, inp, image_file):
#     disable_torch_init()

#     max_new_tokens = 256
    
#     debug = False
#     temperature = 0
#     conv_mode = 'phi'
#     conv = conv_templates[conv_mode].copy()

#     if image_file is not None:
#         image = load_image(image_file)
#         image_tensor = process_images([image], image_processor, model.config)
#         if type(image_tensor) is list:
#             image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
#         else:
#             image_tensor = image_tensor.to(model.device, dtype=torch.float16)
#     else:
#         image = None
#     if image is not None:
#         if model.config.mm_use_im_start_end:
#             inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
#         else:
#             inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
#         conv.append_message(conv.roles[0], inp)
#         image = None
#     else:
#         # later messages
#         conv.append_message(conv.roles[0], inp)
#         image_tensor = None
#     conv.append_message(conv.roles[1], None)
#     prompt = conv.get_prompt()

#     input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
#     stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
#     keywords = [stop_str]
#     stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
#     streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

#     with torch.inference_mode():
#         output_ids = model.generate(
#             input_ids,
#             images=image_tensor,
#             do_sample=True if temperature > 0 else False,
#             temperature=temperature,
#             max_new_tokens=max_new_tokens,
#             streamer=streamer,
#             use_cache=True,
#             stopping_criteria=[stopping_criteria]
#         )

#     outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
#     conv.messages[-1][-1] = outputs

#     if debug:
#         print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
    
#     return outputs, output_ids