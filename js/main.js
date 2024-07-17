function main() {
    let previousHref = "";

    function checkAndPlayAudio() {
        const downloadLink = document.querySelector('a[href*="file="]');

        if (downloadLink) {
            if (downloadLink.href !== previousHref) {
                previousHref = downloadLink.href;
                const audioElement = new Audio(downloadLink.href);

                audioElement.play().catch(error => {
                    console.error('Error playing audio:', error);
                });
            }
        }
    }

    let previousHeight = 0;

    function scrollToBottom() {
        const historyDiv = document.getElementById('history');
        if (historyDiv) {
            const currentHeight = historyDiv.scrollHeight;
            if (currentHeight !== previousHeight) {
                historyDiv.scrollTop = currentHeight;
                previousHeight = currentHeight;
            }
        }
    }

    function startCheckingAudio() {
        setInterval(checkAndPlayAudio, 100);
    }

    function startCheckingScroll() {
        setInterval(scrollToBottom, 100);
    }

    async function startStreaming(sessionId) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const audioContext = new AudioContext();
            const source = audioContext.createMediaStreamSource(stream);
            const processor = audioContext.createScriptProcessor(512, 1, 1);

            // const ws = new WebSocket('wss://frp-end.top:52752', { agent });
            const ws = new WebSocket('wss://www.u15428.nyat.app:52752');
            // const ws = new WebSocket("ws://localhost:6006");

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

            ws.onmessage = function(event) {
                if (event.data === "sentence_finished") {
                    document.getElementById("reply").click();
                }
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

    console.log('Debug: Interface loaded');
    startCheckingAudio();
    startCheckingScroll();

    try {
        document.getElementById('generate_session_id_button').click();
    } catch (e) {
        console.error('Error clicking button.', e);
    }
    let flag = false;
    const checkSessionId = setInterval(function() {
        let session_id = document.getElementById('session_id_holder').innerText;
        if (session_id) {
            console.log("Generated session id: ", session_id);
            sessionStorage.setItem('session_id', session_id);
            document.getElementById('start_streaming').onclick = function() {
                if (flag) {
                    return;
                }
                startStreaming(session_id);
                flag = true;
            };
            clearInterval(checkSessionId); // 停止检查
        }
    }, 100); // 每100毫秒检查一次
    
}
