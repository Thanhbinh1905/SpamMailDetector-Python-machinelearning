
document.querySelectorAll('input[name="correct_classification"]').forEach((elem) => {
    elem.addEventListener("change", function(event) {
        var value = event.target.value;
        var correctLabelDiv = document.getElementById('correct_label_div');
        correctLabelDiv.style.display = value === "no" ? "block" : "none";
    });
});

// Show the notification if it exists
window.addEventListener('load', function() {
    const notification = document.getElementById('notification');
    if (notification) {
        notification.style.display = 'block';
        setTimeout(function() {
            notification.style.display = 'none';
        }, 5000);  // Hide the notification after 5 seconds
    }
});

const messageInput = document.getElementById('message');
const checkButton = document.getElementById('check-button');
const resetButton = document.getElementById('reset-button');

function toggleCheckButton() {
    if (messageInput.value.trim() === '') {
        checkButton.disabled = true;
        checkButton.classList.add('disabled');
    } else {
        checkButton.disabled = false;
        checkButton.classList.remove('disabled');
    }
}

messageInput.addEventListener('input', toggleCheckButton);

// Initial check in case there's already content
toggleCheckButton();

