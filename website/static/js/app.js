document.getElementById("start-btn").addEventListener("click", function() {
    fetch("/start", { method: "POST" })
        .then(response => response.json())
        .then(data => {
            document.getElementById("status").innerText = data.message;
            document.getElementById("start-btn").style.display = "none";
            document.getElementById("stop-btn").style.display = "inline";
        });
});

document.getElementById("stop-btn").addEventListener("click", function() {
    fetch("/stop", { method: "POST" })
        .then(response => response.json())
        .then(data => {
            document.getElementById("status").innerText = data.message;
            document.getElementById("stop-btn").style.display = "none";
            document.getElementById("start-btn").style.display = "inline";
        });
});
