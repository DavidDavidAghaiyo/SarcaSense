async function detectSarcasm() {
    var userInput = document.getElementById("userInput").value;


    //Send user input to your backend for sarcasm detection
    const response = await fetch('/detect-sarcasm', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: userInput }),
    });

    const result = await response.json();

    var resultBubble = document.createElement("div");
    resultBubble.className = result.isSarcastic ? "bubble sarcastic" : "bubble not-sarcastic";
    resultBubble.textContent = result.isSarcastic ? "Sarcastic" : "Not Sarcastic";

    document.getElementById("inputBox").appendChild(resultBubble);
}