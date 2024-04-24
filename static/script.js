document.addEventListener('DOMContentLoaded', function() {
    const taskCards = document.querySelectorAll('.info');

    taskCards.forEach(card => {
        const contentWidth = card.querySelector('.info_content').offsetHeight;
        if (contentWidth <= 128) {
            card.classList.toggle('closed');
            card.classList.toggle('always_open');
        }
        else {
            card.addEventListener('click', function() {
                card.classList.toggle('closed');
                card.classList.toggle('open'); // Переключить класс для открытия/закрытия описания
            });
        }
    });
});

function showNotification(message) {
    var notification = document.createElement('div');
    notification.classList.add('notification');
    notification.textContent = message;
    document.body.appendChild(notification);
    notification.style.display = 'block';
    setTimeout(function() {
        notification.style.display = 'none';
        document.body.removeChild(notification);
    }, 3000);
}

document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.complaint-form').forEach(function(form) {
        form.onsubmit = function(event) {
            event.preventDefault();
            fetch('/submit_complaint', {
                method: 'POST',
                body: new FormData(form)
            })
            .then(response => response.json())
            .then(data => {
                showNotification(data.message);
                form.parentNode.classList.remove('sent');
            });
        };
    });
});