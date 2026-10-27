const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// --------------------------------------------------
// CANVAS SETTINGS
// --------------------------------------------------

ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.lineWidth = 13;
ctx.strokeStyle = "#eafff8";

let drawing = false;
let history = [];
let mode = "doodle";
let ocrTimer = null;

// --------------------------------------------------
// CANVAS POSITION
// --------------------------------------------------

function pos(e) {
    const r = canvas.getBoundingClientRect();

    return [
        (e.clientX - r.left) * canvas.width / r.width,
        (e.clientY - r.top) * canvas.height / r.height
    ];
}

// --------------------------------------------------
// SAVE CANVAS FOR UNDO
// --------------------------------------------------

function save() {
    history.push(
        ctx.getImageData(
            0,
            0,
            canvas.width,
            canvas.height
        )
    );

    if (history.length > 20) {
        history.shift();
    }
}

// --------------------------------------------------
// STATUS
// --------------------------------------------------

function setStatus(text) {
    const status = document.getElementById("status");

    if (status) {
        status.textContent = text;
    }
}

// --------------------------------------------------
// CANVAS IMAGE
// --------------------------------------------------

function image() {
    return canvas.toDataURL("image/png");
}

// --------------------------------------------------
// CHECK WHETHER CANVAS HAS DRAWING
// --------------------------------------------------

function canvasHasDrawing() {
    const pixels = ctx.getImageData(
        0,
        0,
        canvas.width,
        canvas.height
    ).data;

    for (let i = 3; i < pixels.length; i += 4) {
        if (pixels[i] !== 0) {
            return true;
        }
    }

    return false;
}

// --------------------------------------------------
// DRAWING
// --------------------------------------------------

canvas.addEventListener("pointerdown", function (e) {

    drawing = true;

    save();

    const [x, y] = pos(e);

    ctx.beginPath();
    ctx.moveTo(x, y);

    canvas.setPointerCapture(e.pointerId);

    if (mode === "write") {
        setStatus("WRITING");
    } else {
        setStatus("DRAWING");
    }

    // Cancel any previous automatic OCR
    if (ocrTimer) {
        clearTimeout(ocrTimer);
        ocrTimer = null;
    }
});


canvas.addEventListener("pointermove", function (e) {

    if (!drawing) return;

    const [x, y] = pos(e);

    ctx.lineTo(x, y);
    ctx.stroke();
});


function finishDrawing() {

    if (!drawing) return;

    drawing = false;

    if (mode === "write") {

        setStatus("PROCESSING");

        // Wait until the user stops writing
        // before sending the canvas to OCR.
        ocrTimer = setTimeout(function () {

            if (canvasHasDrawing()) {
                runOCR();
            }

        }, 1200);

    } else {

        setStatus("READY");
    }
}


canvas.addEventListener("pointerup", finishDrawing);
canvas.addEventListener("pointercancel", finishDrawing);
canvas.addEventListener("pointerleave", function (e) {
    // Don't finish drawing when leaving because
    // pointer capture normally keeps the stroke alive.
});

// Prevent browser scrolling while writing
canvas.style.touchAction = "none";


// --------------------------------------------------
// UNDO
// --------------------------------------------------

document.getElementById("undo").onclick = function () {

    if (history.length > 0) {

        const previous = history.pop();

        ctx.putImageData(previous, 0, 0);

        setStatus("UNDO");
    }
};


// --------------------------------------------------
// CLEAR
// --------------------------------------------------

document.getElementById("clear").onclick = function () {

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    history = [];

    if (ocrTimer) {
        clearTimeout(ocrTimer);
        ocrTimer = null;
    }

    document.getElementById("textOutput").value = "";

    document.getElementById("prediction").textContent = "Waiting";

    document.getElementById("confidence").textContent =
        "Your CNN result will appear here.";

    setStatus("READY");
};


// --------------------------------------------------
// MODE SWITCHING
// --------------------------------------------------

document.querySelectorAll(".mode").forEach(function (btn) {

    btn.onclick = function () {

        document
            .querySelectorAll(".mode")
            .forEach(function (x) {
                x.classList.remove("active");
            });

        btn.classList.add("active");

        mode = btn.dataset.mode;

        const modeLabel = document.getElementById("modeLabel");
        const hint = document.getElementById("hint");

        const recognize = document.getElementById("recognize");
        const convert = document.getElementById("convert");

        const doodleResult = document.getElementById("doodleResult");
        const writeResult = document.getElementById("writeResult");

        if (mode === "doodle") {

            modeLabel.textContent = "DOODLE MODE";

            hint.textContent =
                "Draw a supported object — cat, house, tree, flower, star or apple.";

            recognize.classList.remove("hidden");
            convert.classList.add("hidden");

            doodleResult.classList.remove("hidden");
            writeResult.classList.add("hidden");

        } else {

            modeLabel.textContent = "HANDWRITING MODE";

            hint.textContent =
                "Write clearly across the canvas. OCR will read your handwriting automatically.";

            recognize.classList.add("hidden");
            convert.classList.remove("hidden");

            doodleResult.classList.add("hidden");
            writeResult.classList.remove("hidden");
        }

        if (ocrTimer) {
            clearTimeout(ocrTimer);
            ocrTimer = null;
        }

        setStatus("READY");
    };
});


// --------------------------------------------------
// CNN DOODLE RECOGNITION
// --------------------------------------------------

document.getElementById("recognize").onclick = async function () {

    if (mode !== "doodle") {
        return;
    }
    if (!canvasHasDrawing()) {
        setStatus("DRAW SOMETHING");
        return;
    }
    setStatus("CNN RUNNING");
    document.getElementById("prediction").textContent =
        "Thinking...";
document.getElementById("confidence").textContent =
        "Running neural network...";
    try {
        const response = await fetch("/predict-doodle", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },
body: JSON.stringify({
                image: image()
            })
        });
        const data = await response.json();
  if (!response.ok) {
            throw new Error(
                data.message || "CNN request failed."
            );
        }
        document.getElementById("prediction").textContent =
            String(data.prediction || "Unknown").toUpperCase();
document.getElementById("confidence").textContent =
            `CNN confidence: ${data.confidence}%`;
const bars = document.getElementById("bars");
 if (data.scores) {
bars.innerHTML =
                Object.entries(data.scores)
                    .sort(function (a, b) {
                        return b[1] - a[1];
                    })
                    .map(function ([key, value]) {

                        return `
                            <div class="bar-row">

                                <div class="bar-top">
                                    <span>${key.toUpperCase()}</span>
                                    <span>${value}%</span>
                                </div>

                                <div class="bar">
                                    <i style="width:${value}%"></i>
                                </div>

                            </div>
                        `;
                    })
                    .join("");
        }
        setStatus("COMPLETE");

    } catch (error) {

        console.error("CNN ERROR:", error);

        document.getElementById("prediction").textContent =
            "MODEL NOT READY";

        document.getElementById("confidence").textContent =
            error.message;
        setStatus("CHECK MODEL");
    }
};


// --------------------------------------------------
// HANDWRITING OCR
// --------------------------------------------------

async function runOCR() {

    if (mode !== "write") {
        return;
    }

    if (!canvasHasDrawing()) {
        setStatus("READY");
        return;
    }

    setStatus("NEURAL OCR RUNNING");

    const output = document.getElementById("textOutput");

    output.value = "Reading handwriting...";


    try {

        const response = await fetch("/ocr", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                image: image()
            })
        });


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.message || "OCR request failed."
            );
        }


        output.value =
            data.text || "No handwriting detected.";


        setStatus("COMPLETE");

    } catch (error) {

        console.error("OCR ERROR:", error);

        output.value =
            "OCR ERROR: " + error.message;

        setStatus("CHECK OCR");
    }
}


// --------------------------------------------------
// MANUAL OCR BUTTON
// --------------------------------------------------

document.getElementById("convert").onclick = async function () {

    if (mode !== "write") {
        return;
    }

    if (ocrTimer) {
        clearTimeout(ocrTimer);
        ocrTimer = null;
    }

    await runOCR();
};


// --------------------------------------------------
// EXPORT WORD DOCUMENT
// --------------------------------------------------

document.getElementById("export").onclick = async function () {

    const text =
        document.getElementById("textOutput").value;


    if (!text.trim()) {

        alert("Nothing to export.");

        return;
    }


    try {

        const response = await fetch("/export", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                text: text
            })
        });


        if (!response.ok) {
            throw new Error("Export failed.");
        }


        const blob = await response.blob();

        const url =
            URL.createObjectURL(blob);


        const link =
            document.createElement("a");

        link.href = url;

        link.download =
            "neurocanvas-document.docx";

        document.body.appendChild(link);

        link.click();

        link.remove();

        URL.revokeObjectURL(url);

    } catch (error) {

        console.error(error);

        alert("Export failed: " + error.message);
    }
};


// --------------------------------------------------
// SAVE CANVAS AS PNG
// --------------------------------------------------

document.getElementById("saveImage").onclick = async function () {

    if (!canvasHasDrawing()) {

        alert("Nothing to save.");

        return;
    }

    setStatus("EXPORTING");


    try {

        const response = await fetch("/save-doodle", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                image: image()
            })
        });


        if (!response.ok) {

            throw new Error(
                "Image export failed."
            );
        }


        const blob =
            await response.blob();


        const url =
            URL.createObjectURL(blob);


        const link =
            document.createElement("a");

        link.href = url;

        link.download =
            "neurocanvas-doodle.png";

        document.body.appendChild(link);

        link.click();

        link.remove();

        URL.revokeObjectURL(url);


        setStatus("PNG SAVED");

    } catch (error) {

        console.error(error);

        setStatus("EXPORT FAILED");

        alert(error.message);
    }
};


// --------------------------------------------------
// INITIAL STATE
// --------------------------------------------------

setStatus("READY");

console.log("Canvaru Neural Creative Studio loaded.");