import argparse
import os
import time
import torch
import cv2
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token
from torchvision import transforms

def load_model(args, device):
    """Load pretrained LLaVA model."""
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        model_name=args.conv_mode,
        model_base=args.model_base,
        load_8bit=False,
        load_4bit=False,
        attn_implementation="flash_attention_2",
        device_map="cuda:0"
    )
    model.to(device)
    model.to(torch.float32)
    model.eval()

    return tokenizer, model, image_processor


def preprocess_image(image_path, device):
    """Load and preprocess image without using image_processor.preprocess."""
    # breakpoint()
    image = Image.open(image_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    return image_tensor


def build_prompt(prompt_text, conv_mode):
    """Build conversation prompt."""
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), conv


def generate_response(model, tokenizer, input_ids, image_tensor, args):
    """Run model inference."""
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            output_hidden_states=args.output_hidden_states,
            return_dict_in_generate=args.return_dict_in_generate,
            use_cache=True
        )
        # breakpoint()
        output_ids = output['out_seq']
        mask  = output['sam_pred'].cpu().numpy()*255
        os.makedirs(args.save_res_path,exist_ok=True)
        cv2.imwrite(f"{args.save_res_path}/demo.png",mask.transpose(1,2,0))
    return tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )[0]


def eval_model(args):
    disable_torch_init()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, model, image_processor = load_model(args, device)

    with open(args.question_file,"r") as src:
        prompt_text = src.readlines()[0]
    prompt, conv = build_prompt(prompt_text, args.conv_mode)

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).to(device)

    image_tensor = preprocess_image(
        args.image_folder,
        device
    ).type(torch.float32)

    start_time = time.time()

    try:
        outputs = generate_response(
            model,
            tokenizer,
            input_ids,
            image_tensor,
            args
        )
        print(outputs)

    except Exception as e:
        print(f"[Error during inference] {e}")

    print(f"Inference time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="./checkpoints/chesapeake_weights")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="./chesapeake_landcover/test/img/m_3807504_ne_18_1_naip-new__512__0___384.png")
    parser.add_argument("--question_file", type=str, default="./question.txt")

    parser.add_argument("--conv_mode", type=str, default="qwen_1_5")

    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    parser.add_argument("--output_hidden_states", default=True)
    parser.add_argument("--return_dict_in_generate", default=True)

    parser.add_argument("--mask_threshold", type=float, default=0.0)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--save_res_path", type=str, default="./vision_output")

    args = parser.parse_args()

    eval_model(args)