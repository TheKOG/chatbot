import asyncio
import json
import os
import shutil
import socket
import threading
import time
import wave
import websockets
from pydub import AudioSegment
from whis import Trans_from_file
from pydub.silence import detect_nonsilent, detect_silence

def save_audio_file(data, counter, session_id):
    output_dir = f'recv/{session_id}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"{output_dir}/{counter:02d}.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(data)

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' has been deleted.")
    else:
        print(f"Folder '{folder_path}' does not exist.")

merge_num=30
def merge(st, ed, audio_dir):
    combined = AudioSegment.empty()
    st_=(ed-merge_num+1+100)%100
    ptr=ed
    tot=0
    print("merge start")
    while True:
        print(f"{ptr} {st} {st_} {ed}")
        file_name = f"{ptr:02d}.wav"
        file_path = os.path.join(audio_dir, file_name)
        if not os.path.exists(file_path):
            return 0
        audio = AudioSegment.from_wav(file_path)
        combined = audio+combined
        tot+=1
        if ptr==st_ or ptr==st:
            break
        ptr=(ptr-1+100)%100
        
    combined.export(f"{audio_dir}/tmp.wav", format="wav")
    return 1

def is_sentence_finished(audio_path, silence_thresh=-40, silence_len=750):
    audio = AudioSegment.from_file(audio_path)
    
    # 检测非静音部分
    nonsilent = detect_nonsilent(audio, min_silence_len=silence_len, silence_thresh=silence_thresh)
    if not nonsilent:
        return -1  # 没有检测到说话
    
    # 检测静音部分
    silence = detect_silence(audio, min_silence_len=silence_len, silence_thresh=silence_thresh)
    if silence:
        last_silence_start = silence[-1][0]
        if len(audio) - last_silence_start > silence_len:
            return 1  # 句子说完了
    
    return 0  # 句子没说完

async def reg_and_send(audio_path, audio_dir, websocket):
    try:
        text = Trans_from_file(audio_path)
    except:
        text=" "
    with open(os.path.join(audio_dir, "tmp.txt"), "w") as f:
        f.write(text)
    print(text)
    await websocket.send("sentence_finished")

def recogize(websocket, start_id, cnt,flags, audio_dir):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def inner_recogize():
        while start_id[0] != -1:
            st = start_id[0]
            ed = cnt[0]
            if ed == -1:
                await asyncio.sleep(1)
                continue
            flag=merge(st=st, ed=ed, audio_dir=audio_dir)
            if flag==0:
                await asyncio.sleep(1)
                continue
            merge_file_path = f"{audio_dir}/tmp.wav"
            flag = is_sentence_finished(merge_file_path)
            if flag == 1:
                await asyncio.create_task(
                    reg_and_send(merge_file_path, audio_dir, websocket)
                )
                start_id[0] = (ed+1) % 100
                flags[0]=True
            elif flag == -1:
                start_id[0] = (ed+1) % 100
                flags[0]=True
                
            await asyncio.sleep(1)
    
    loop.run_until_complete(inner_recogize())

async def handle_client(websocket, path):
    if not os.path.exists('recv'):
        os.makedirs('recv')
        
    print("Client connected")
    session_id = await websocket.recv()
    print(f"Session ID received: {session_id}")


    start_id=[0]# 列表保存单值 用于在外部函数内修改
    cnt = [-1]# 列表保存单值 用于在外部函数内修改
    flags=[True]
    thread = threading.Thread(target=recogize, args=(websocket,start_id,cnt,flags,f'recv/{session_id}'))
    thread.daemon = True  # 设置为守护线程
    
    thread.start()
    start_time = None
    buffer = bytearray()

    # try:
    async for message in websocket:
        if start_time is None:
            start_time = time.time()
        if message:
            buffer.extend(message)
            current_time = time.time()
            if current_time - start_time >= 1.0:  # 每隔一秒保存一次
                cnt_=(cnt[0]+1)%100
                st_=start_id[0]
                if(cnt_==st_):
                    if flags[0]:
                        flags[0]=False
                    else:
                        st_=(st_+1)%100
                save_audio_file(buffer, cnt_, session_id)
                cnt[0]=cnt_
                start_id[0]=st_
                    
                buffer = bytearray()
                start_time = current_time
    # except websockets.exceptions.ConnectionClosed:
    #     print("Connection closed")
    # finally:
    start_id[0]=-1
    delete_folder(f"recv/{session_id}")

async def web_main():
    async with websockets.serve(handle_client, "127.0.0.1", 6006):
        await asyncio.Future()  # run forever

def local_client(client_socket):
    while True:
        try:
            audio_file = client_socket.recv(1024).decode('utf-8')
            if not audio_file:
                break
            reply = Trans_from_file(audio_file)
            client_socket.send(reply.encode('utf-8'))
            if os.path.exists(audio_file):
                os.remove(audio_file)
        except ConnectionResetError:
            break
    client_socket.close()
    print("Connection closed")

def local_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('127.0.0.1', 19198))
    server_socket.listen(5)
    print("Server listening on port 19198")
    while True:
        client_socket, addr = server_socket.accept()
        print(f"Accepted connection from {addr}")
        client_handler = threading.Thread(target=local_client, args=(client_socket,))
        client_handler.start()
        
def local_main():
    local_thread = threading.Thread(target=local_server)
    local_thread.daemon = True
    local_thread.start()
    
if __name__ == "__main__":
    local_main()
    asyncio.run(web_main())
