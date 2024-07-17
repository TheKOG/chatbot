import gradio as gr

# 定义一个函数来加载HTML文件内容
def load_audio_js():
    with open("./js/audio.js", "r", encoding="utf-8") as file:
        return file.read()
    
# def load_main_js():
#     with open("./js/main.js", "r", encoding="utf-8") as file:
#         return file.read()


with gr.Blocks(title="Chat with Dingzhen") as app:
    html = gr.HTML(value="test_id", elem_id="session_id_holder")
    button=gr.Button("start streaming",elem_id="start_streaming")
    app.load(_js=load_audio_js())
    
app.launch()
