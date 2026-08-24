const analyzeBtn = document.getElementById("analyzeBtn");
const status = document.getElementById("status");

analyzeBtn.addEventListener("click", async () => {

    status.innerText = "Reading conversation...";

    try {

        // Get current tab
        const tabs = await chrome.tabs.query({
            active: true,
            currentWindow: true
        });

        const tab = tabs[0];

        // Ask content.js for conversation
        const response = await chrome.tabs.sendMessage(
            tab.id,
            {
                action: "getConversation"
            }
        );

        const conversation = response.conversation;

        console.log("Conversation:", conversation);

        status.innerText = "Sending to backend...";

        // Send conversation to backend
        const backendResponse = await fetch(
            "http://localhost:8000/analyze",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    conversation: conversation
                })
            }
        );

        const result = await backendResponse.json();

        console.log("Backend result:", result);

        displayResult(result);

        status.innerText = "Analysis complete";

    } catch (error) {

        console.error(error);

        status.innerText =
            "Error: " + error.message;
    }
});


function displayResult(result) {

    document
        .getElementById("result")
        .classList.remove("hidden");

    document.getElementById("score").innerText =
        result.health_score;

    document.getElementById("contradictions").innerText =
        result.contradictions;

    document.getElementById("drift").innerText =
        result.topic_drift;

    document.getElementById("constraints").innerText =
        result.constraint_violations;

    document.getElementById("recommendation").innerText =
        result.recommendation;
}