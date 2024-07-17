function main() {
    async function startStreaming(sessionId) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const audioContext = new AudioContext();
            const source = audioContext.createMediaStreamSource(stream);
            const processor = audioContext.createScriptProcessor(512, 1, 1); // 调整缓冲区大小

            const ws = new WebSocket("ws://localhost:19198");
            ws.binaryType = "arraybuffer";

            source.connect(processor);
            processor.connect(audioContext.destination);

            ws.onopen = function() {
                console.log("WebSocket connection established");
                ws.send(sessionId);
            };

            ws.onclose = function() {
                console.log("WebSocket connection closed");
                processor.disconnect();
                source.disconnect();
            };

            ws.onerror = function(error) {
                console.error("WebSocket error:", error);
            };

            processor.onaudioprocess = function(e) {
                if (ws.readyState === WebSocket.OPEN) {
                    const input = e.inputBuffer.getChannelData(0);
                    const buffer = new ArrayBuffer(input.length * 2);
                    const view = new DataView(buffer);
                    for (let i = 0; i < input.length; i++) {
                        view.setInt16(i * 2, input[i] * 0x7FFF, true);
                    }
                    ws.send(buffer);
                }
            };

            setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send('');
                }
            }, 5000);
        } catch (error) {
            console.error('Error accessing media devices.', error);
        }
    }
    let session_id = document.getElementById('session_id_holder').innerText;
    console.log("Generated session id: ", session_id);
    sessionStorage.setItem('session_id', session_id);
    document.getElementById('start_streaming').onclick=startStreaming(session_id);
}
