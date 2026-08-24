function extractConversation() {
    const messages = [];

    // ChatGPT messages
    const elements = document.querySelectorAll(
        '[data-message-author-role]'
    );

    elements.forEach(element => {
        const role = element.getAttribute(
            'data-message-author-role'
        );

        const content = element.innerText.trim();

        if (content) {
            messages.push({
                role: role,
                content: content
            });
        }
    });

    return messages;
}

chrome.runtime.onMessage.addListener(
    (request, sender, sendResponse) => {

        if (request.action === "getConversation") {

            const conversation = extractConversation();

            sendResponse({
                conversation: conversation
            });
        }

        return true;
    }
);