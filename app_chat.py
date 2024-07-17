import random
import socket
import string
import sys, os
import time

import requests

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
os.environ['https_proxy'] = 'http://127.0.0.1:7890/'
os.environ['http_proxy'] = 'http://127.0.0.1:7890/'
api_key = 'sk-Pt26e47dd5da527f0941c6201edcc72570fda7987e1aHdQH'

import logging

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s")

logger = logging.getLogger(__name__)

import torch
import argparse
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence, get_bert
from text.cleaner import clean_text
import gradio as gr
import webbrowser
import tiktoken
import soundfile as sf

reply_by_socket=True

if not reply_by_socket:
    from whis import Trans, Trans_from_file
    
net_g = None
TOKEN_LIMIT=65536

def get_text(text, language_str, hps):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert = get_bert(norm_text, word2ph, language_str)
    del word2ph

    assert bert.shape[-1] == len(phone)

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)

    return bert, phone, tone, language

def infer(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid):
    global net_g
    bert, phones, tones, lang_ids = get_text(text, "ZH", hps)
    with torch.no_grad():
        x_tst=phones.to(device).unsqueeze(0)
        tones=tones.to(device).unsqueeze(0)
        lang_ids=lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        del phones
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, speakers, tones, lang_ids, bert, sdp_ratio=sdp_ratio
                           , noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0,0].data.cpu().float().numpy()
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers
        return audio
    
enc = tiktoken.get_encoding("cl100k_base")
def count_tokens(messages):
    return sum([len(enc.encode(message["content"])) for message in messages])

def chat_fn(text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, history):
    if history is None:
        history = []
    messages = [
        {"role": "system", "content": "你的名字叫丁真, 如果别人问你会不会说藏话, 你就说\"我测你的码.\""},
        {"role": "user", "content": """
         在下面的的对话中, 你要扮演一位藏族少年, 你的名字叫丁真, 
         如果别人问你会不会说藏话, 你就说\"我测你的码.\", 
         你最喜欢的事情是抽瑞克五代,
         你最喜欢的动物是芝士雪豹,
         你的省份是妈妈生的,
         你的家乡是世界最高城:理塘,
         在接下来的对话中,你要始终给出最简短的回答。
         """},
        {"role": "assistant", "content": "好的，我明白了。在接下来的对话中，我会按照你指定的角色进行扮演，并给出最简短的回答。你好，我叫丁真。你有什么问题想问我吗？"},
        {"role": "user", "content": "你会说藏话吗"},
        {"role": "assistant", "content": "我测你的码。"},
        {"role": "user", "content": "你是哪个省的"},
        {"role": "assistant", "content": "我是妈妈生的"},
        {"role": "user", "content": "你叫什么名字"},
        {"role": "assistant", "content": "我的名字叫丁真"},
    ]
    for (user, bot) in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})

    messages.append({"role": "user", "content": text})

    # 检查总 tokens 数量并删除最早的消息直到低于 TOKEN_LIMIT
    while count_tokens(messages) > TOKEN_LIMIT:
        # 删除最早的用户和助手消息
        if len(messages) > 3:
            del messages[1:3]

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    data = {
        'model': 'gpt-4o',
        'messages': messages
    }

    response = requests.post('https://api.gptsapi.net/v1/chat/completions', headers=headers, json=data, proxies={
        'http': os.environ['http_proxy'],
        'https': os.environ['https_proxy']
    })

    response_json = response.json()
    reply_content = response_json['choices'][0]['message']['content']

    history.append((text, reply_content))

    with torch.no_grad():
        audio = infer(reply_content, sdp_ratio=sdp_ratio, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale, sid=speaker)
    return reply_content, (hps.data.sampling_rate, audio), history

def reply_fn(session_id, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, history):
    with open(f"./recv/{session_id}/tmp.txt", "r") as f:
        text = f.read()
    return chat_fn(text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, history)

def generate_history_html(state):
    history_html = """
    <div id='history' style='height: 500px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px;'>
    """
    if state is not None:
        history_html += "<br>".join([f"Q: {q}<br>A: {a}" for q, a in state])
    return history_html

def generate_random_string():# 生成随机字符串作为会话id
    return ''.join(random.choices(string.ascii_letters + string.digits, k=16))

def save_session_id():
    session_id = generate_random_string()
    html_content = f"<div id=session_id_holder>{session_id}<div>"
    print(f"Generated session id: {session_id}")
    return html_content, session_id

def load_main_js():
    with open("./js/main.js", "r", encoding="utf-8") as file:
        return file.read()

def save_audio(audio,file_path):
    sample_rate = audio[0]
    audio_data = audio[1]
    print("audio saved")
    sf.write(file_path, audio_data, sample_rate)

def reply_mic(input_audio,session_id, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, state):
    
    if not os.path.exists(f"./recv"):
        os.makedirs(f"./recv")
        
    input_file=f"./recv/{session_id}.wav"
    save_audio(input_audio,input_file)
    if(reply_by_socket):
        global client_socket
        if client_socket is None:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(('127.0.0.1', 19198))
        client_socket.send(input_file.encode('utf-8'))
        text=client_socket.recv(1024).decode('utf-8')
    else:
        text=Trans_from_file(input_file)
        
    return chat_fn(text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, state)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./logs/dingzhen/dingzhen.pth", help="path of your model")
    parser.add_argument("--config_dir", default="./configs/config.json", help="path of your config file")
    parser.add_argument("--share", default=False, help="make link public")
    parser.add_argument("-d", "--debug", action="store_true", help="enable DEBUG-LEVEL log")

    args = parser.parse_args()
    if args.debug:
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)
    hps = utils.get_hparams_from_file(args.config_dir)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.model_dir, net_g, None, skip_optimizer=True)

    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    
    try:
        if reply_by_socket:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(('127.0.0.1', 19198))
        else:
            client_socket=None
    except:
        client_socket=None
    
    with gr.Blocks(title="Chat with Dingzhen") as app:
        hidden_html = gr.HTML("", visible=False)
        hidden_button = gr.Button(visible=False, elem_id="generate_session_id_button")
        reply=gr.Button(visible=False,elem_id="reply")
        with gr.Row():
            with gr.Column():
                gr.Markdown(value="""
                使用本模型请严格遵守法律法规！\n
                发布二创作品请标注本项目作者及链接、作品使用Bert-VITS2 AI生成！\n
                项目地址：https://github.com/Stardust-minus/Bert-VITS2 \n                
                """)
                text = gr.TextArea(label="Text", placeholder="Input Text Here",
                                    value="你会说藏话吗")
                btn = gr.Button("发送文字", variant="primary")
                mic_audio=gr.Audio(label="Mic Audio",type="numpy",source="microphone",sample_rate=44100)
                mic_btn=gr.Button("发送语音",variant="primary")
                stream_btn = gr.Button("实时语音对话",elem_id="start_streaming",variant="primary")
                speaker = gr.Dropdown(choices=speakers, value=speakers[0], label='Speaker',visible=False)
                sdp_ratio = gr.Slider(minimum=0.1, maximum=1, value=0.2, step=0.1, label='SDP/DP混合比')
                noise_scale = gr.Slider(minimum=0.1, maximum=1, value=0.5, step=0.1, label='感情调节')
                noise_scale_w = gr.Slider(minimum=0.1, maximum=1, value=0.9, step=0.1, label='音素长度')
                length_scale = gr.Slider(minimum=0.1, maximum=2, value=1, step=0.01, label='生成长度')
                state = gr.State()
                session_id_state = gr.State("")
            with gr.Column():
                history_html = gr.HTML(label="History", value="")
                text_output = gr.Textbox(label="Message")
                audio_output = gr.Audio(label="Output Audio")
                

        hidden_button.click(fn=save_session_id, inputs=[], outputs=[hidden_html, session_id_state])
        
        mic_btn.click(reply_mic,
                inputs=[mic_audio,session_id_state, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, state],
                outputs=[text_output, audio_output, state],
                ).then(generate_history_html,
                        inputs=[state],
                        outputs=[history_html]
                       )
        
        reply.click(reply_fn,
                inputs=[session_id_state, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, state],
                outputs=[text_output, audio_output, state],
                ).then(generate_history_html,
                        inputs=[state],
                        outputs=[history_html]
                       )
                
        btn.click(chat_fn,
                inputs=[text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, state],
                outputs=[text_output, audio_output, state],
                ).then(generate_history_html,
                        inputs=[state],
                        outputs=[history_html]
                       )
        app.load(_js=load_main_js())

    app.queue()
    app.launch(show_error=True, share=True)
