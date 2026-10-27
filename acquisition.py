import streamlit as st
import streamlit.components.v1 as html_component

def run():
    st.header("Transcription Data Acquisition Mode (Canvas & Webcam)")
    st.write("You can write on the canvas using a stylus or S-Pen. The recorded sensor coordinate data can be downloaded as a CSV file, and the handwriting image can be saved directly from the canvas.")

    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * { box-sizing: border-box; }
            body { font-family: Arial, sans-serif; background: #f8f9fa; margin: 0; padding: 10px; width: 100%; }
            .container { background: white; padding: 20px; border-radius: 8px; border: 1px solid #ddd; width: 100%; }
            .canvas-area { display: flex; gap: 20px; align-items: flex-start; margin-top: 10px; width: 100%; }
            .canvas-wrapper { flex: 1; min-width: 300px; }
            canvas { background: white; border: 2px solid #333; cursor: crosshair; touch-action: none; border-radius: 4px; width: 100%; height: 380px; }
            .webcam-wrapper { width: 320px; flex-shrink: 0; }
            video { width: 100%; height: 240px; background: #000; border: 2px solid #333; border-radius: 4px; object-fit: cover; }
            .action-bar { margin-top: 15px; padding-top: 10px; border-top: 1px solid #eee; display: flex; gap: 10px; flex-wrap: wrap; }
            .btn { padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 14px; }
            .btn-clear { background: #ff4d4d; color: white; }
            .btn-clear:hover { background: #e60000; }
            .btn-save { background: #28a745; color: white; }
            .btn-save:hover { background: #218838; }
            .btn-img { background: #007bff; color: white; }
            .btn-img:hover { background: #0056b3; }
            #statusInfo { font-size: 12px; margin-top: 8px; font-family: monospace; }
        </style>
    </head>
    <body>

    <div class="container">
        <h3 style="margin-top:0;">Kanvas Sensor Stylus / Hardware Pen</h3>
        <p style="font-size: 12px; color: #555;">Use a stylus (S-Pen, Apple Pencil, or Wacom). The sensor field automatically turns <b>OFF (0)</b> if a mouse or standard touch input is used.</p>

        <div class="canvas-area">
            <div class="canvas-wrapper">
                <canvas id="paintCanvas"></canvas>
            </div>
            <div class="webcam-wrapper">
                <video id="webcam" autoplay playsinline muted></video>
                <div id="statusInfo">
                    Status Pen Sensor: <span id="penSensorStatus" style="color: red; font-weight: bold;">OFF (0 DETECTED)</span>
                </div>
            </div>
        </div>

        <div class="action-bar">
            <button class="btn btn-clear" onclick="clearCanvas()">Delete Canvas</button>
            <button class="btn btn-save" onclick="downloadCSV()">Download Sensor CSV</button>
            <button class="btn btn-img" onclick="saveCanvasImage()">Save Canvas Image (PNG)</button>
        </div>
    </div>

    <script>
    const canvas = document.getElementById('paintCanvas');
    const ctx = canvas.getContext('2d');
    const video = document.getElementById('webcam');
    const statusDisplay = document.getElementById('penSensorStatus');

    let isDrawing = false;
    let records = [];

    // Fungsi Inisialisasi Latar Belakang Putih Kanvas
    function setCanvasBackground() {
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    function resizeCanvas() {
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        setCanvasBackground();
    }
    window.addEventListener('load', resizeCanvas);

    async function setupWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            video.srcObject = stream;
        } catch (err) {
            console.log("Webcam error: " + err.message);
        }
    }
    setupWebcam();

    function getStylusMetrics(e) {
        const isPen = e.pointerType === 'pen';
        let pressure = 0, altitude = 0, azimuth = 0;

        if (isPen) {
            statusDisplay.innerText = "ACTIVE (HARDWARE DETECTED)";
            statusDisplay.style.color = "green";

            pressure = Math.round(e.pressure * 500);

            const tiltX = e.tiltX || 0;
            const tiltY = e.tiltY || 0;

            if (tiltX !== 0 || tiltY !== 0) {
                const radX = tiltX * (Math.PI / 180);
                const radY = tiltY * (Math.PI / 180);

                const altRad = Math.atan(1 / Math.sqrt(Math.pow(Math.tan(radX), 2) + Math.pow(Math.tan(radY), 2)));
                altitude = Math.round(altRad * (180 / Math.PI));

                let aziRad = Math.atan2(Math.tan(radY), Math.tan(radX));
                if (aziRad < 0) aziRad += 2 * Math.PI;
                azimuth = Math.round(aziRad * (180 / Math.PI));
            }
        } else {
            statusDisplay.innerText = "OFF (MOUSE/TOUCH DETECTED)";
            statusDisplay.style.color = "red";
        }

        return { altitude, azimuth, pressure };
    }

    function processPoint(e, penStatus) {
        const rect = canvas.getBoundingClientRect();
        const x = Math.round((e.clientX - rect.left) * 50);
        const y = Math.round((e.clientY - rect.top) * 30);
        const timestampInMs = Date.now();
        const metrics = getStylusMetrics(e);

        records.push([x, y, timestampInMs, penStatus, metrics.altitude, metrics.azimuth, metrics.pressure]);
    }

    function startDrawing(e) {
        isDrawing = true;
        const rect = canvas.getBoundingClientRect();
        ctx.beginPath();
        ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
        processPoint(e, 1);
    }

    function draw(e) {
        if (!isDrawing) return;
        const rect = canvas.getBoundingClientRect();
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.strokeStyle = '#000000';
        ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
        ctx.stroke();
        processPoint(e, 1);
    }

    function stopDrawing(e) {
        if (isDrawing) {
            isDrawing = false;
            ctx.closePath();
            processPoint(e, 0);
        }
    }

    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        setCanvasBackground();
        records = [];
    }

    function downloadCSV() {
        if (records.length === 0) {
            alert("There is no data on the canvas yet!");
            return;
        }

        let csvContent = "data:text/csv;charset=utf-8,X,Y,Timestamp,Status,Altitude,Azimuth,Pressure\\n";
        records.forEach(row => {
            csvContent += row.join(",") + "\\n";
        });

        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "sample_sensor_data.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    // Perbaikan: Mengunduh Hasil Goresan Tulisan pada Kanvas sebagai PNG
    function saveCanvasImage() {
        if (records.length === 0) {
            alert("Canvas is empty! Please write something first.");
            return;
        }

        const imageURI = canvas.toDataURL("image/png");
        const link = document.createElement("a");
        link.setAttribute("href", imageURI);
        link.setAttribute("download", "handwriting_canvas.png");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    canvas.addEventListener('pointerdown', startDrawing);
    canvas.addEventListener('pointermove', draw);
    canvas.addEventListener('pointerup', stopDrawing);
    canvas.addEventListener('pointerleave', stopDrawing);
    </script>
    </body>
    </html>
    """

    html_component.html(html_code, height=650, scrolling=False)

if __name__ == "__main__":
    run()